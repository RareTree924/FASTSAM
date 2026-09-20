import CoreGraphics
import CoreML
import Foundation

/// One bounding box in 480x480 pixels of the cropped photo.
struct Box {
    var x: Int
    var y: Int
    var w: Int
    var h: Int
}

protocol Detector {
    func detect(_ image: CGImage) async throws -> [Box]
}

/// Placeholder used only if the real model can't be loaded.
struct StubDetector: Detector {
    func detect(_ image: CGImage) async throws -> [Box] {
        return [
            Box(x: 120, y: 120, w: 240, h: 240),
            Box(x: 20,  y: 20,  w: 100, h: 60),
        ]
    }
}

enum DetectorError: LocalizedError {
    case modelMissing
    case badModel(String)

    var errorDescription: String? {
        switch self {
        case .modelMissing:      return "FastSAM-s model not found in the app bundle"
        case .badModel(let why): return "Unexpected model layout: \(why)"
        }
    }
}

/// Runs FastSAM-s (a YOLOv8s-seg model) through Core ML and returns bounding
/// boxes only. Output layout: [1, 37, 4725] = 4 box values (center x, center y,
/// width, height) + 1 score + 32 mask coefficients, for each of 4725 candidates.
final class FastSAMDetector: Detector {
    // ---- Tunables ----
    private let minScore: Float = 0.4    // drop candidates below this confidence
    private let nmsIoU: Float = 0.7      // boxes overlapping more than this count as duplicates
    private let minSide: Float = 12      // drop boxes smaller than this many pixels
    private let maxBoxes = 25            // must match CAM_MAX_BOXES on the ESP32 side
    private let size: Float = 480        // model input size in pixels

    private struct Cand {
        var x1: Float
        var y1: Float
        var x2: Float
        var y2: Float
        var score: Float
    }

    private let model: MLModel
    private let inputName: String
    private let constraint: MLImageConstraint

    init() throws {
        let url = try FastSAMDetector.locateModel()
        let config = MLModelConfiguration()
        config.computeUnits = .all       // lets Core ML use the Neural Engine
        let m = try MLModel(contentsOf: url, configuration: config)

        guard let input = m.modelDescription.inputDescriptionsByName.first(where: { $0.value.type == .image }),
              let c = input.value.imageConstraint else {
            throw DetectorError.badModel("no image input")
        }
        model = m
        inputName = input.key
        constraint = c
    }

    /// Xcode compiles the .mlpackage into FastSAM-s.mlmodelc during the build.
    /// If only the raw package is in the bundle, compile it on the phone instead.
    private static func locateModel() throws -> URL {
        if let compiled = Bundle.main.url(forResource: "FastSAM-s", withExtension: "mlmodelc") {
            return compiled
        }
        if let package = Bundle.main.url(forResource: "FastSAM-s", withExtension: "mlpackage") {
            return try MLModel.compileModel(at: package)
        }
        throw DetectorError.modelMissing
    }

    func detect(_ image: CGImage) async throws -> [Box] {
        let feature = try MLFeatureValue(cgImage: image, constraint: constraint, options: nil)
        let input = try MLDictionaryFeatureProvider(dictionary: [inputName: feature])

        // Run the model off the main thread so the UI stays responsive.
        let output = try await Task.detached(priority: .userInitiated) { [model] in
            try model.prediction(from: input)
        }.value

        return try decode(output)
    }

    private func decode(_ output: MLFeatureProvider) throws -> [Box] {
        // Pick the 3-D output ([1, 37, 4725]); the mask prototypes are 4-D and ignored.
        var found: MLMultiArray?
        for name in output.featureNames {
            if let a = output.featureValue(for: name)?.multiArrayValue, a.shape.count == 3 {
                found = a
            }
        }
        guard let arr = found else { throw DetectorError.badModel("no [1, channels, anchors] output") }
        guard arr.shape[1].intValue >= 5 else { throw DetectorError.badModel("too few channels") }

        let anchors = arr.shape[2].intValue
        let s1 = arr.strides[1].intValue     // Core ML may pad rows, so always use the strides
        let s2 = arr.strides[2].intValue
        let isFloat32 = (arr.dataType == .float32)
        let ptr = arr.dataPointer.assumingMemoryBound(to: Float.self)

        func read(_ ch: Int, _ i: Int) -> Float {
            if isFloat32 { return ptr[ch * s1 + i * s2] }
            return arr[[NSNumber(value: 0), NSNumber(value: ch), NSNumber(value: i)]].floatValue
        }

        // 1. Keep confident candidates and convert center/size to corners.
        var cands: [Cand] = []
        var maxCoord: Float = 0
        for i in 0..<anchors {
            let score = read(4, i)
            if score < minScore { continue }
            let cx = read(0, i), cy = read(1, i), w = read(2, i), h = read(3, i)
            maxCoord = max(maxCoord, cx + w / 2, cy + h / 2)
            cands.append(Cand(x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2, score: score))
        }

        // Some exports give 0..1 coordinates instead of pixels; detect that and scale up.
        let k: Float = (maxCoord > 0 && maxCoord <= 2.0) ? size : 1

        // 2. Clamp to the image and drop tiny boxes.
        var boxes: [Cand] = []
        for c in cands {
            let x1 = max(0, c.x1 * k), y1 = max(0, c.y1 * k)
            let x2 = min(size, c.x2 * k), y2 = min(size, c.y2 * k)
            if x2 - x1 < minSide || y2 - y1 < minSide { continue }
            boxes.append(Cand(x1: x1, y1: y1, x2: x2, y2: y2, score: c.score))
        }
        boxes.sort { $0.score > $1.score }

        // 3. Non-maximum suppression: keep the best box, skip near-duplicates of it.
        var kept: [Cand] = []
        for c in boxes {
            if kept.count >= maxBoxes { break }
            if kept.contains(where: { iou($0, c) > nmsIoU }) { continue }
            kept.append(c)
        }

        return kept.map {
            Box(x: Int($0.x1.rounded()),
                y: Int($0.y1.rounded()),
                w: Int(($0.x2 - $0.x1).rounded()),
                h: Int(($0.y2 - $0.y1).rounded()))
        }
    }

    private func iou(_ a: Cand, _ b: Cand) -> Float {
        let ix = max(0, min(a.x2, b.x2) - max(a.x1, b.x1))
        let iy = max(0, min(a.y2, b.y2) - max(a.y1, b.y1))
        let inter = ix * iy
        let union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
        return union > 0 ? inter / union : 0
    }
}
