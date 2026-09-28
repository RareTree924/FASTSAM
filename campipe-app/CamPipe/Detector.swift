import CoreGraphics

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

/// Placeholder so the whole ESP32 -> iPhone -> S3 path can be tested
/// before FastSAM is wired in.
struct StubDetector: Detector {
    func detect(_ image: CGImage) async throws -> [Box] {
        return [
            Box(x: 120, y: 120, w: 240, h: 240),
            Box(x: 20,  y: 20,  w: 100, h: 60),
        ]
    }
}
