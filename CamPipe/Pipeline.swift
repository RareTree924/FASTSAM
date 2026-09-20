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

@MainActor
final class Pipeline: ObservableObject {
    @Published var baseURL = "http://192.168.4.1"
    @Published var running = false
    @Published var status = "Idle"
    @Published var preview: UIImage?
    @Published var boxes: [Box] = []

    private var task: Task<Void, Never>?
    private let detector: Detector
    @Published var detectorNote = ""

    init() {
        do {
            detector = try FastSAMDetector()
            detectorNote = "FastSAM-s loaded"
        } catch {
            detector = StubDetector()
            detectorNote = "FastSAM not loaded (\(error.localizedDescription)); using fixed test boxes"
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

    private func pause(_ ms: Int) async {
        try? await Task.sleep(nanoseconds: UInt64(ms) * 1_000_000)
    }

    private func loop() async {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 4
        let session = URLSession(configuration: config)

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
                guard let http = response as? HTTPURLResponse else { continue }
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

                // 2. Center-crop the 480x480 square (x offset 80 for a 640x480 photo).
                let cx = (photo.width - outSize) / 2
                let cy = (photo.height - outSize) / 2
                guard cx >= 0, cy >= 0,
                      let square = photo.cropping(to: CGRect(x: cx, y: cy, width: outSize, height: outSize)) else {
                    status = "Photo too small: \(photo.width)x\(photo.height)"
                    await pause(500)
                    continue
                }
                preview = UIImage(cgImage: square)

                // 3. Detect.
                let t0 = Date()
                let found = try await detector.detect(square)
                boxes = found
                let ms = Int(Date().timeIntervalSince(t0) * 1000)

                // 4. Send the boxes back to the CAM (it relays them to the S3).
                var req = URLRequest(url: resultURL)
                req.httpMethod = "POST"
                req.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")
                let (_, resp) = try await session.upload(for: req, from: encodeBoxes(frameId: frameId, boxes: found))
                let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
                status = "frame \(frameId): \(found.count) boxes, \(ms) ms detect, POST -> HTTP \(code)"
            } catch {
                if Task.isCancelled { break }
                status = "Error: \(error.localizedDescription)"
                await pause(1000)
            }
        }
    }
}
