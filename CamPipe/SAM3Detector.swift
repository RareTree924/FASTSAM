import CoreAIImageSegmenter
import CoreGraphics
import Foundation
import os

/// SAM 3 (Meta) through Apple's Core AI for typed words: it finds the named thing far more
/// reliably than YOLOE's text prompt. SAM 3 always needs a word, so photos with nothing
/// typed go to `fallback` (YOLOE's prompt-free "everything"), as does every photo if the
/// SAM 3 model can't be loaded.
///
/// The model is the SAM3/ folder in the app: Apple's lite iOS export of facebook/sam3
/// (336x336, apple/coreai-models models/sam3), made by the Build IPA workflow.
final class SAM3Detector: Detector {
    private let fallback: Detector
    private let model: Task<SAM3Model, Error>

    init(fallback: Detector) {
        self.fallback = fallback
        model = Task.detached(priority: .utility) { try await SAM3Model.load() }
    }

    /// Waits for the model to load; says whether it did.
    func status() async -> String {
        do {
            _ = try await model.value
            return "SAM 3 loaded"
        } catch {
            return "SAM 3 not loaded (\(error.localizedDescription)); YOLOE does typed words too"
        }
    }

    func detect(_ image: CGImage, settings: SamSettings) async throws -> Detection {
        let prompt = settings.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty, let sam = try? await model.value else {
            return try await fallback.detect(image, settings: settings)
        }
        return try await sam.detect(image, prompt: prompt, settings: settings)
    }
}

private actor SAM3Model {
    private let segmenter: ImageSegmenter
    private let maxObjects = 25   // CAM_MAX_BOXES on the ESP32 side

    private init(_ segmenter: ImageSegmenter) {
        self.segmenter = segmenter
    }

    static func load() async throws -> SAM3Model {
        guard let url = Bundle.main.url(forResource: "SAM3", withExtension: nil) else {
            throw DetectorError.modelMissing("SAM3")
        }
        let freeMB = os_proc_available_memory() / 1_000_000
        let segmenter: ImageSegmenter
        do {
            segmenter = try await ImageSegmenter(resourcesAt: url.path)   // specializes the model for this phone
        } catch {
            throw LoadError(step: "loading", error: error, freeMB: freeMB)
        }
        do {
            try await segmenter.warmup()   // first run on a dummy photo, so a real one doesn't wait for it
        } catch {
            throw LoadError(step: "first run", error: error, freeMB: freeMB)
        }
        return SAM3Model(segmenter)
    }

    /// Core AI's own messages are just "AIModelError error N", so this spells out
    /// everything the error carries, plus the phone and its free memory.
    private struct LoadError: LocalizedError {
        let step: String
        let error: Error
        let freeMB: Int

        var errorDescription: String? {
            let ns = error as NSError
            var s = "\(step) failed: \(String(reflecting: error)) [\(ns.domain) \(ns.code)]"
            if let d = ns.userInfo[NSDebugDescriptionErrorKey] as? String { s += " \(d)" }
            if let u = ns.userInfo[NSUnderlyingErrorKey] as? NSError {
                s += " <- \(u.domain) \(u.code): \(u.localizedDescription)"
            }
            var u = utsname()
            uname(&u)
            let phone = withUnsafeBytes(of: &u.machine) { String(decoding: $0.prefix { $0 != 0 }, as: UTF8.self) }
            return s + "; \(phone), \(freeMB) MB free before loading"
        }
    }

    func detect(_ image: CGImage, prompt: String, settings: SamSettings) async throws -> Detection {
        var params = SegmentationParameters.default
        params.maxSegments = maxObjects
        // Segments come back best first, each with a mask the size of the photo
        // (upsampled bilinearly from the model's 96x96).
        let found = try await segmenter.segment(image: image, prompt: prompt, parameters: params)

        let n = Detection.size
        var outline = [UInt8](repeating: 0, count: Detection.outlineBytes)
        var boxes: [Box] = []
        for s in found.segments where s.score >= settings.minScore {
            guard s.maskWidth == n, s.maskHeight == n else {
                throw DetectorError.badModel("mask \(s.maskWidth)x\(s.maskHeight), photo \(n)x\(n)")
            }
            var minX = n, minY = n, maxX = -1, maxY = -1
            for y in 0..<n {
                for x in 0..<n where s.mask[y * n + x] {
                    minX = min(minX, x); maxX = max(maxX, x); minY = min(minY, y); maxY = max(maxY, y)
                }
            }
            guard maxX >= 0 else { continue }   // empty mask: nothing to outline
            let box = Box(x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1)   // tight to the mask

            // Skip near-duplicates of a surer object, as YOLOE does.
            if boxes.contains(where: { iou($0, box) > settings.mergeIoU }) { continue }
            Detection.addBorder(of: s.mask, w: n, h: n, at: 0, 0, within: minX, minY, maxX, maxY, to: &outline)
            boxes.append(box)
        }
        return Detection(boxes: boxes, outline: Data(outline), model: "SAM 3")
    }

    private func iou(_ a: Box, _ b: Box) -> Float {
        let ix = max(0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
        let iy = max(0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
        let inter = ix * iy
        let union = a.w * a.h + b.w * b.h - inter
        return union > 0 ? Float(inter) / Float(union) : 0
    }
}
