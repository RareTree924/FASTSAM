import Accelerate
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

/// Segmentation tuning. Set on the phone, or sent by the S3 with each photo.
struct SamSettings: Equatable {
    var minScore: Float = 0.25   // outline an object only if YOLOE is at least this sure (0...1)
    var mergeIoU: Float = 0.70   // objects overlapping more than this are merged into one (0...1)
    var prompt: String = ""      // what to outline, e.g. "mug"; empty = everything
}

/// What goes back to the S3: where the objects are and their borders. No class names.
struct Detection {
    static let outlineBytes = 480 * 480 / 8   // CAM_OUTLINE_BYTES

    var boxes: [Box]
    /// 1 bit per pixel, row-major, 60 bytes per row, most significant bit = leftmost pixel.
    /// A set bit is a border pixel. nil = boxes only.
    var outline: Data?
}

protocol Detector {
    func detect(_ image: CGImage, settings: SamSettings) async throws -> Detection
}

/// Placeholder used only if the real model can't be loaded.
struct StubDetector: Detector {
    func detect(_ image: CGImage, settings: SamSettings) async throws -> Detection {
        Detection(boxes: [Box(x: 120, y: 120, w: 240, h: 240), Box(x: 20, y: 20, w: 100, h: 60)], outline: nil)
    }
}

enum DetectorError: LocalizedError {
    case modelMissing(String)
    case badModel(String)

    var errorDescription: String? {
        switch self {
        case .modelMissing(let name): return "\(name) not found in the app bundle"
        case .badModel(let why):      return "Unexpected model layout: \(why)"
        }
    }
}

/// Reads an MLMultiArray of any float type through its strides.
private struct FloatReader {
    let array: MLMultiArray
    private let f32: UnsafeMutablePointer<Float>?
    private let f16: UnsafeMutablePointer<Float16>?

    init(_ a: MLMultiArray) {
        array = a
        f32 = a.dataType == .float32 ? a.dataPointer.assumingMemoryBound(to: Float.self) : nil
        f16 = a.dataType == .float16 ? a.dataPointer.assumingMemoryBound(to: Float16.self) : nil
    }

    /// Element at a flat offset computed from the strides.
    func at(_ offset: Int) -> Float {
        if let p = f32 { return p[offset] }
        if let p = f16 { return Float(p[offset]) }
        return array[offset].floatValue
    }
}

/// YOLOE-11L-seg through Core ML (see tools/export_yoloe.py).
///   det   [1, 38, 4725]: cx, cy, w, h (pixels), text score, object score, 32 mask coefficients
///   proto [1, 32, 120, 120]: mask prototypes
/// With a prompt, objects are scored against the typed words (text score); without
/// one, YOLOE's prompt-free object score is used to outline everything.
final class YOLOEDetector: Detector {
    private let size: Float = 480
    private let minSide: Float = 8       // drop objects smaller than this many pixels
    private let maxObjects = 25          // CAM_MAX_BOXES on the ESP32 side

    private let model: MLModel
    private let textModel: MLModel
    private let tokenizer: CLIPTokenizer
    private let constraint: MLImageConstraint
    private let noText: MLMultiArray
    private var textCache: [String: MLMultiArray] = [:]

    private struct Cand {
        var x1: Float, y1: Float, x2: Float, y2: Float
        var score: Float
        var index: Int
    }

    init() throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all       // lets Core ML use the Neural Engine
        model = try MLModel(contentsOf: try Self.locate("YOLOE-seg"), configuration: config)
        textModel = try MLModel(contentsOf: try Self.locate("YOLOE-text"), configuration: config)
        guard let t = CLIPTokenizer() else { throw DetectorError.modelMissing("clip_merges.txt") }
        tokenizer = t
        guard let c = model.modelDescription.inputDescriptionsByName["image"]?.imageConstraint else {
            throw DetectorError.badModel("no image input")
        }
        constraint = c
        noText = try MLMultiArray(shape: [1, 1, 512], dataType: .float32)   // zeros: unused without a prompt
        for i in 0..<noText.count { noText[i] = 0 }
    }

    /// Xcode compiles each .mlpackage into a .mlmodelc during the build.
    private static func locate(_ name: String) throws -> URL {
        if let compiled = Bundle.main.url(forResource: name, withExtension: "mlmodelc") { return compiled }
        if let package = Bundle.main.url(forResource: name, withExtension: "mlpackage") {
            return try MLModel.compileModel(at: package)
        }
        throw DetectorError.modelMissing(name)
    }

    func detect(_ image: CGImage, settings: SamSettings) async throws -> Detection {
        // Run everything off the main thread so the UI stays responsive.
        try await Task.detached(priority: .userInitiated) { [self] in
            try self.run(image, settings: settings)
        }.value
    }

    private func run(_ image: CGImage, settings: SamSettings) throws -> Detection {
        let prompt = settings.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let text = prompt.isEmpty ? noText : try textEmbedding(prompt)
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "image": try MLFeatureValue(cgImage: image, constraint: constraint, options: nil),
            "text": MLFeatureValue(multiArray: text),
        ])
        let out = try model.prediction(from: input)
        guard let det = out.featureValue(for: "det")?.multiArrayValue,
              let proto = out.featureValue(for: "proto")?.multiArrayValue else {
            throw DetectorError.badModel("missing det / proto output")
        }
        return try decode(det: det, proto: proto, scoreChannel: prompt.isEmpty ? 5 : 4, settings: settings)
    }

    /// Typed words -> YOLOE text embedding [1, 1, 512]. Cached: the same word is typed often.
    private func textEmbedding(_ prompt: String) throws -> MLMultiArray {
        let key = prompt.lowercased()
        if let hit = textCache[key] { return hit }
        let ids = tokenizer.tokenize(prompt)
        let tokens = try MLMultiArray(shape: [1, NSNumber(value: CLIPTokenizer.contextLength)], dataType: .int32)
        for (i, id) in ids.enumerated() { tokens[i] = NSNumber(value: id) }
        let out = try textModel.prediction(from: MLDictionaryFeatureProvider(dictionary: ["tokens": tokens]))
        guard let t = out.featureValue(for: "text")?.multiArrayValue, t.count == 512 else {
            throw DetectorError.badModel("text encoder output")
        }
        // Copy into a plain float32 array for the image model.
        let r = FloatReader(t)
        let emb = try MLMultiArray(shape: [1, 1, 512], dataType: .float32)
        let last = t.strides.last!.intValue
        for i in 0..<512 { emb[i] = NSNumber(value: r.at(i * last)) }
        if textCache.count > 32 { textCache.removeAll() }
        textCache[key] = emb
        return emb
    }

    private func decode(det: MLMultiArray, proto: MLMultiArray, scoreChannel: Int, settings: SamSettings) throws -> Detection {
        guard det.shape.count == 3, det.shape[1].intValue >= 7, proto.shape.count == 4 else {
            throw DetectorError.badModel("det \(det.shape) proto \(proto.shape)")
        }
        let anchors = det.shape[2].intValue
        let nm = det.shape[1].intValue - 6
        let d1 = det.strides[1].intValue, d2 = det.strides[2].intValue   // Core ML may pad rows: use strides
        let dr = FloatReader(det)
        func read(_ ch: Int, _ i: Int) -> Float { dr.at(ch * d1 + i * d2) }

        // 1. Confident candidates, as clamped corners.
        var cands: [Cand] = []
        for i in 0..<anchors {
            let s = read(scoreChannel, i)
            if s < settings.minScore { continue }
            let cx = read(0, i), cy = read(1, i), w = read(2, i), h = read(3, i)
            let c = Cand(x1: max(0, cx - w / 2), y1: max(0, cy - h / 2),
                         x2: min(size, cx + w / 2), y2: min(size, cy + h / 2), score: s, index: i)
            if c.x2 - c.x1 >= minSide && c.y2 - c.y1 >= minSide { cands.append(c) }
        }
        cands.sort { $0.score > $1.score }

        // 2. Non-maximum suppression: keep the best, skip near-duplicates of it.
        var kept: [Cand] = []
        for c in cands {
            if kept.count >= maxObjects { break }
            if kept.contains(where: { iou($0, c) > settings.mergeIoU }) { continue }
            kept.append(c)
        }
        if kept.isEmpty { return Detection(boxes: [], outline: Data(count: Detection.outlineBytes)) }

        // 3. Masks: coefficients (k x nm) times prototypes (nm x P*P) -> k masks of P x P logits.
        let p = proto.shape[2].intValue
        let pp = p * p
        let pr = FloatReader(proto)
        let ps1 = proto.strides[1].intValue, ps2 = proto.strides[2].intValue, ps3 = proto.strides[3].intValue
        var protos = [Float](repeating: 0, count: nm * pp)
        for m in 0..<nm {
            for y in 0..<p {
                for x in 0..<p { protos[m * pp + y * p + x] = pr.at(m * ps1 + y * ps2 + x * ps3) }
            }
        }
        var coeffs = [Float](repeating: 0, count: kept.count * nm)
        for (j, c) in kept.enumerated() {
            for m in 0..<nm { coeffs[j * nm + m] = read(6 + m, c.index) }
        }
        var logits = [Float](repeating: 0, count: kept.count * pp)
        vDSP_mmul(coeffs, 1, protos, 1, &logits, 1, vDSP_Length(kept.count), vDSP_Length(pp), vDSP_Length(nm))

        // 4. Each mask, upsampled (bilinear, like Ultralytics) inside its box; its border goes into the outline.
        var outline = [UInt8](repeating: 0, count: Detection.outlineBytes)
        var boxes: [Box] = []
        let n = Int(size)
        let scale = Float(p) / size
        for (j, c) in kept.enumerated() {
            let bx1 = Int(c.x1.rounded(.down)), by1 = Int(c.y1.rounded(.down))
            let bx2 = min(n, Int(c.x2.rounded(.up))), by2 = min(n, Int(c.y2.rounded(.up)))
            let w = bx2 - bx1, h = by2 - by1
            guard w > 0, h > 0 else { continue }

            // Source columns for bilinear sampling (align_corners = false).
            var sx0 = [Int](repeating: 0, count: w), sx1 = sx0, fx = [Float](repeating: 0, count: w)
            for x in 0..<w {
                let s = min(max((Float(bx1 + x) + 0.5) * scale - 0.5, 0), Float(p - 1))
                sx0[x] = Int(s); sx1[x] = min(sx0[x] + 1, p - 1); fx[x] = s - Float(sx0[x])
            }
            var inside = [Bool](repeating: false, count: w * h)
            var minX = w, minY = h, maxX = -1, maxY = -1
            logits.withUnsafeBufferPointer { lg in
                let base = j * pp
                for y in 0..<h {
                    let s = min(max((Float(by1 + y) + 0.5) * scale - 0.5, 0), Float(p - 1))
                    let y0 = Int(s), y1 = min(y0 + 1, p - 1), fy = s - Float(y0)
                    let r0 = base + y0 * p, r1 = base + y1 * p
                    for x in 0..<w {
                        let top = lg[r0 + sx0[x]] * (1 - fx[x]) + lg[r0 + sx1[x]] * fx[x]
                        let bot = lg[r1 + sx0[x]] * (1 - fx[x]) + lg[r1 + sx1[x]] * fx[x]
                        if top * (1 - fy) + bot * fy > 0 {   // logit > 0  <=>  mask probability > 0.5
                            inside[y * w + x] = true
                            minX = min(minX, x); maxX = max(maxX, x); minY = min(minY, y); maxY = max(maxY, y)
                        }
                    }
                }
            }
            guard maxX >= 0 else { continue }   // empty mask: nothing to outline

            // Border = inside pixels with a 4-neighbour outside (or at the box edge).
            for y in minY...maxY {
                for x in minX...maxX where inside[y * w + x] {
                    let edge = x == 0 || y == 0 || x == w - 1 || y == h - 1 ||
                        !inside[y * w + x - 1] || !inside[y * w + x + 1] ||
                        !inside[(y - 1) * w + x] || !inside[(y + 1) * w + x]
                    if edge {
                        let px = bx1 + x, py = by1 + y
                        outline[py * (n / 8) + px / 8] |= 0x80 >> UInt8(px % 8)
                    }
                }
            }
            boxes.append(Box(x: bx1 + minX, y: by1 + minY, w: maxX - minX + 1, h: maxY - minY + 1))   // tight to the mask
        }
        return Detection(boxes: boxes, outline: Data(outline))
    }

    private func iou(_ a: Cand, _ b: Cand) -> Float {
        let ix = max(0, min(a.x2, b.x2) - max(a.x1, b.x1))
        let iy = max(0, min(a.y2, b.y2) - max(a.y1, b.y1))
        let inter = ix * iy
        let union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
        return union > 0 ? inter / union : 0
    }
}
