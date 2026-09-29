import SwiftUI
import UIKit

private let outSize = 480          // must match CAM_OUT_SIZE in pipeline_msgs.h
private let maxBoxes = 25          // must match CAM_MAX_BOXES

private func le16(_ v: Int) -> [UInt8] {
    let c = UInt16(max(0, min(65535, v)))
    return [UInt8(c & 0xFF), UInt8(c >> 8)]
}

/// Builds the bytes of cam_boxes_msg_t: type, frame_id, img_w, img_h, count, boxes.
private func encodeBoxes(frameId: UInt8, boxes: [Box]) -> Data {
    let list = Array(boxes.prefix(maxBoxes))
    var bytes: [UInt8] = [0x02, frameId]          // CAM_MSG_BOXES
    bytes += le16(outSize)
    bytes += le16(outSize)
    bytes.append(UInt8(list.count))
    for b in list {
        bytes += le16(b.x)
        bytes += le16(b.y)
        bytes += le16(b.w)
        bytes += le16(b.h)
    }
    return Data(bytes)
}

/// The outline bitmap as a green-on-transparent picture for the preview (each
/// border pixel drawn 2x2 so it shows up at phone size).
private func outlineImage(_ bits: Data) -> UIImage? {
    let n = outSize
    var rgba = [UInt8](repeating: 0, count: n * n * 4)
    bits.withUnsafeBytes { (b: UnsafeRawBufferPointer) in
        for y in 0..<n {
            for x in 0..<n where b[y * (n / 8) + x / 8] & (0x80 >> UInt8(x % 8)) != 0 {
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] where x + dx < n && y + dy < n {
                    let o = ((y + dy) * n + x + dx) * 4
                    rgba[o] = 0; rgba[o + 1] = 255; rgba[o + 2] = 0; rgba[o + 3] = 255
                }
            }
        }
    }
    guard let provider = CGDataProvider(data: Data(rgba) as CFData),
          let cg = CGImage(width: n, height: n, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: n * 4,
                           space: CGColorSpaceCreateDeviceRGB(),
                           bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                           provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
    else { return nil }
    return UIImage(cgImage: cg)
}

@MainActor
final class Pipeline: ObservableObject {
    @Published var baseURL = "http://192.168.4.1"
    @Published var running = false
    @Published var status = "Idle"
    @Published var preview: UIImage?
    @Published var boxes: [Box] = []
    @Published var outline: UIImage?    // the last photo's borders, drawn over the preview

    private var task: Task<Void, Never>?
    private let detector: Detector
    @Published var detectorNote = ""

    // ---- segmentation settings ----
    // The phone's own values (sliders + what to outline), remembered between launches.
    @Published var phoneSettings: SamSettings {
        didSet {
            UserDefaults.standard.set(phoneSettings.minScore, forKey: "samMinScore")
            UserDefaults.standard.set(phoneSettings.mergeIoU, forKey: "samMergeIoU")
            UserDefaults.standard.set(phoneSettings.prompt, forKey: "samPrompt")
        }
    }
    /// On: use what the S3 sends with each photo (its Tools menu + what you typed after
    /// taking the photo). Off, or when the photo carries none (older firmware, mock
    /// camera): use the phone's own settings.
    @Published var useS3Settings: Bool {
        didSet { UserDefaults.standard.set(useS3Settings, forKey: "samUseS3") }
    }
    @Published var s3Settings: SamSettings?        // from the last photo, if it had any
    @Published var lastUsedSettings: SamSettings?  // what the last detection actually ran with

    init() {
        let d = UserDefaults.standard
        d.register(defaults: ["samMinScore": SamSettings().minScore,
                              "samMergeIoU": SamSettings().mergeIoU,
                              "samPrompt": "",
                              "samUseS3": true])
        phoneSettings = SamSettings(minScore: d.float(forKey: "samMinScore"),
                                    mergeIoU: d.float(forKey: "samMergeIoU"),
                                    prompt: d.string(forKey: "samPrompt") ?? "")
        useS3Settings = d.bool(forKey: "samUseS3")

        // YOLOE outlines everything when nothing is typed; SAM 3 handles typed words.
        let yoloe: Detector
        do {
            yoloe = try YOLOEDetector()
            detectorNote = "YOLOE-11L loaded"
        } catch {
            yoloe = StubDetector()
            detectorNote = "YOLOE not loaded (\(error.localizedDescription)); using fixed test boxes"
        }
        let sam = SAM3Detector(fallback: yoloe)
        detector = sam
        let yoloeNote = detectorNote
        detectorNote = yoloeNote + " · SAM 3 loading..."
        Task { [weak self] in
            let note = await sam.status()
            self?.detectorNote = yoloeNote + " · " + note
        }
    }

    func start() {
        guard !running else { return }
        running = true
        UIApplication.shared.isIdleTimerDisabled = true   // keep the screen on
        task = Task { await self.loop() }
    }

    func stop() {
        task?.cancel()
        task = nil
        running = false
        UIApplication.shared.isIdleTimerDisabled = false
        status = "Stopped"
    }

    /// X-Sam-Min-Score / X-Sam-Merge-Iou (whole percents) from the CAM, if both are present,
    /// plus X-Sam-Prompt (percent-encoded; missing = outline everything).
    private static func samSettings(from http: HTTPURLResponse) -> SamSettings? {
        guard let s = http.value(forHTTPHeaderField: "X-Sam-Min-Score").flatMap(Int.init),
              let m = http.value(forHTTPHeaderField: "X-Sam-Merge-Iou").flatMap(Int.init) else { return nil }
        let prompt = http.value(forHTTPHeaderField: "X-Sam-Prompt")?.removingPercentEncoding ?? ""
        return SamSettings(minScore: Float(min(max(s, 0), 100)) / 100,
                           mergeIoU: Float(min(max(m, 0), 100)) / 100,
                           prompt: prompt)
    }

    private func pause(_ ms: Int) async {
        try? await Task.sleep(nanoseconds: UInt64(ms) * 1_000_000)
    }

    private func loop() async {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 4
        let session = URLSession(configuration: config)
        defer { session.invalidateAndCancel() }   // a new session is made on every Start

        while !Task.isCancelled {
            do {
                let base = baseURL
                    .trimmingCharacters(in: .whitespaces)
                    .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
                guard let frameURL = URL(string: base + "/frame"),
                      let resultURL = URL(string: base + "/result") else {
                    status = "Bad URL"
                    await pause(1000)
                    continue
                }

                // 1. Ask for a photo. 204 means "nothing yet".
                let (data, response) = try await session.data(from: frameURL)
                guard let http = response as? HTTPURLResponse else {
                    await pause(500)
                    continue
                }
                if http.statusCode == 204 {
                    status = "Waiting for a capture request..."
                    await pause(200)
                    continue
                }
                guard http.statusCode == 200,
                      let idText = http.value(forHTTPHeaderField: "X-Frame-Id"),
                      let frameId = UInt8(idText),
                      let photo = UIImage(data: data)?.cgImage else {
                    status = "Bad /frame response (HTTP \(http.statusCode))"
                    await pause(500)
                    continue
                }

                // 2. Center-crop the 480x480 square (x offset 80 for a 640x480 photo; the
                //    OV2640 sends 480x480, where this is a no-op).
                let cx = (photo.width - outSize) / 2
                let cy = (photo.height - outSize) / 2
                guard cx >= 0, cy >= 0,
                      let square = photo.cropping(to: CGRect(x: cx, y: cy, width: outSize, height: outSize)) else {
                    status = "Photo too small: \(photo.width)x\(photo.height)"
                    await pause(500)
                    continue
                }
                preview = UIImage(cgImage: square)
                outline = nil

                // 3. Outline, with the S3's settings and words if this photo carries them.
                s3Settings = Self.samSettings(from: http)
                let settings = (useS3Settings ? s3Settings : nil) ?? phoneSettings
                lastUsedSettings = settings
                status = settings.prompt.isEmpty ? "Outlining everything..." : "Looking for \"\(settings.prompt)\"..."
                let t0 = Date()
                let found = try await detector.detect(square, settings: settings)
                boxes = found.boxes
                outline = found.outline.flatMap(outlineImage)
                let ms = Int(Date().timeIntervalSince(t0) * 1000)

                // 4. Send boxes + borders back to the CAM (it relays them to the S3).
                //    Only the geometry goes back - no class names.
                var body = encodeBoxes(frameId: frameId, boxes: found.boxes)
                if let bits = found.outline, bits.count == Detection.outlineBytes { body.append(bits) }
                var req = URLRequest(url: resultURL)
                req.httpMethod = "POST"
                req.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")
                let (_, resp) = try await session.upload(for: req, from: body)
                let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
                if Task.isCancelled { break }   // don't overwrite "Stopped"
                let what = settings.prompt.isEmpty ? "everything" : "\"\(settings.prompt)\""
                status = "frame \(frameId): \(found.boxes.count) outlined (\(what), \(found.model)), \(ms) ms, POST -> HTTP \(code)"
            } catch {
                if Task.isCancelled { break }
                status = "Error: \(error.localizedDescription)"
                await pause(1000)
            }
        }
    }
}
