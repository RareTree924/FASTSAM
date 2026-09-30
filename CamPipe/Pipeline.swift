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

/// One neon colour per object, in box order - the same list as the S3's viewer
/// (k_object_colors in its main.c). No pink or purple.
private let objectColors: [(UInt8, UInt8, UInt8)] = [
    (57, 255, 20),    // neon green
    (255, 110, 0),    // neon orange
    (0, 240, 255),    // cyan
    (255, 40, 40),    // neon red
    (40, 120, 255),   // electric blue
    (0, 255, 170),    // mint
]

/// The outline bitmap on a transparent picture for the preview, each object in its own
/// colour (each border pixel drawn 2x2 so it shows up at phone size). Like the S3, a
/// border pixel belongs to the smallest box holding it.
private func outlineImage(_ bits: Data, boxes: [Box]) -> UIImage? {
    let n = outSize
    let order = boxes.indices.sorted { boxes[$0].w * boxes[$0].h < boxes[$1].w * boxes[$1].h }   // smallest first
    var rgba = [UInt8](repeating: 0, count: n * n * 4)
    bits.withUnsafeBytes { (b: UnsafeRawBufferPointer) in
        for y in 0..<n {
            for x in 0..<n where b[y * (n / 8) + x / 8] & (0x80 >> UInt8(x % 8)) != 0 {
                let i = order.first { let r = boxes[$0]; return x >= r.x && x < r.x + r.w && y >= r.y && y < r.y + r.h } ?? 0
                let (r, g, bl) = objectColors[i % objectColors.count]
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] where x + dx < n && y + dy < n {
                    let o = ((y + dy) * n + x + dx) * 4
                    rgba[o] = r; rgba[o + 1] = g; rgba[o + 2] = bl; rgba[o + 3] = 255
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
            UserDefaults.standard.set(phoneSettings.backgroundShare, forKey: "samBackground")
            UserDefaults.standard.set(phoneSettings.surroundShare, forKey: "samSurround")
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
                              "samBackground": SamSettings().backgroundShare,
                              "samSurround": SamSettings().surroundShare,
                              "samPrompt": "",
                              "samUseS3": true])
        phoneSettings = SamSettings(minScore: d.float(forKey: "samMinScore"),
                                    mergeIoU: d.float(forKey: "samMergeIoU"),
                                    backgroundShare: d.float(forKey: "samBackground"),
                                    surroundShare: d.float(forKey: "samSurround"),
                                    prompt: d.string(forKey: "samPrompt") ?? "")
        useS3Settings = d.bool(forKey: "samUseS3")

        do {
            detector = try YOLOEDetector()
            detectorNote = "YOLOE-11L loaded"
        } catch {
            detector = StubDetector()
            detectorNote = "YOLOE not loaded (\(error.localizedDescription)); using fixed test boxes"
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
    /// plus X-Sam-Prompt (percent-encoded; missing = outline everything) and the background
    /// filter X-Sam-Background / X-Sam-Surround (percents; older CAM firmware sends none,
    /// and then `phone`'s are used).
    private static func samSettings(from http: HTTPURLResponse, phone: SamSettings) -> SamSettings? {
        func pct(_ name: String) -> Float? {
            http.value(forHTTPHeaderField: name).flatMap(Int.init).map { Float(min(max($0, 0), 100)) / 100 }
        }
        guard let s = pct("X-Sam-Min-Score"), let m = pct("X-Sam-Merge-Iou") else { return nil }
        let prompt = http.value(forHTTPHeaderField: "X-Sam-Prompt")?.removingPercentEncoding ?? ""
        return SamSettings(minScore: s, mergeIoU: m,
                           backgroundShare: pct("X-Sam-Background") ?? phone.backgroundShare,
                           surroundShare: pct("X-Sam-Surround") ?? phone.surroundShare,
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

                // 2. Center-crop the square: the OV2640 sends a square already (640x640 photos,
                //    480x480 from older firmware); other sensors' 640x480 lose 80 px a side.
                //    Whatever its size, the results come back in 480x480 for the S3.
                let side = min(photo.width, photo.height)
                guard side >= outSize,
                      let square = photo.cropping(to: CGRect(x: (photo.width - side) / 2, y: (photo.height - side) / 2,
                                                             width: side, height: side)) else {
                    status = "Photo too small: \(photo.width)x\(photo.height)"
                    await pause(500)
                    continue
                }
                preview = UIImage(cgImage: square)
                outline = nil

                // 3. Outline, with the S3's settings and words if this photo carries them.
                s3Settings = Self.samSettings(from: http, phone: phoneSettings)
                let settings = (useS3Settings ? s3Settings : nil) ?? phoneSettings
                lastUsedSettings = settings
                status = settings.prompt.isEmpty ? "Outlining everything..." : "Looking for \"\(settings.prompt)\"..."
                let t0 = Date()
                let found = try await detector.detect(square, settings: settings)
                boxes = found.boxes
                outline = found.outline.flatMap { outlineImage($0, boxes: found.boxes) }
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
                status = "frame \(frameId): \(found.boxes.count) outlined (\(what)), \(ms) ms, POST -> HTTP \(code)"
            } catch {
                if Task.isCancelled { break }
                status = "Error: \(error.localizedDescription)"
                await pause(1000)
            }
        }
    }
}
