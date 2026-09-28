import SwiftUI

struct ContentView: View {
    @StateObject private var pipe = Pipeline()

    var body: some View {
        ScrollView {
            VStack(spacing: 12) {
                TextField("Camera URL", text: $pipe.baseURL)
                    .textFieldStyle(.roundedBorder)
                    .textInputAutocapitalization(.never)
                    .disableAutocorrection(true)
                    .keyboardType(.URL)
                    .disabled(pipe.running)

                Button(pipe.running ? "Stop" : "Start") {
                    if pipe.running { pipe.stop() } else { pipe.start() }
                }
                .buttonStyle(.borderedProminent)

                Text(pipe.detectorNote)
                    .font(.caption)
                    .foregroundColor(.secondary)

                Text(pipe.status)
                    .font(.footnote)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)

                if let img = pipe.preview {
                    Image(uiImage: img)
                        .resizable()
                        .aspectRatio(1, contentMode: .fit)
                        .overlay(
                            GeometryReader { geo in
                                let s = geo.size.width / 480
                                ForEach(Array(pipe.boxes.enumerated()), id: \.offset) { _, b in
                                    Rectangle()
                                        .stroke(Color.green, lineWidth: 2)
                                        .frame(width: CGFloat(b.w) * s, height: CGFloat(b.h) * s)
                                        .position(x: (CGFloat(b.x) + CGFloat(b.w) / 2) * s,
                                                  y: (CGFloat(b.y) + CGFloat(b.h) / 2) * s)
                                }
                            }
                        )
                }

                settingsPanel
            }
            .padding()
        }
    }

    private func pct(_ v: Float) -> String { "\(Int((v * 100).rounded()))%" }

    private var settingsPanel: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("FastSAM settings").font(.headline)

            Toggle("Use the S3's settings", isOn: $pipe.useS3Settings)
            Text(pipe.s3Settings.map { "S3 sent: confidence \(pct($0.minScore)), merge overlap \(pct($0.mergeIoU))" }
                 ?? "The last photo carried no S3 settings, so the sliders below are used.")
                .font(.caption)
                .foregroundColor(.secondary)

            Text("Min confidence (phone): \(pct(pipe.phoneSettings.minScore))")
                .font(.subheadline)
            Slider(value: $pipe.phoneSettings.minScore, in: 0.05...0.95, step: 0.05)
            Text("Show a box only if FastSAM is at least this sure. Lower = more boxes.")
                .font(.caption2)
                .foregroundColor(.secondary)

            Text("Merge overlap (phone): \(pct(pipe.phoneSettings.mergeIoU))")
                .font(.subheadline)
            Slider(value: $pipe.phoneSettings.mergeIoU, in: 0.05...1.0, step: 0.05)
            Text("Boxes overlapping more than this are merged into one. Lower = merge more.")
                .font(.caption2)
                .foregroundColor(.secondary)

            if let used = pipe.lastUsedSettings {
                Text("Last detection used \(pct(used.minScore)) / \(pct(used.mergeIoU))")
                    .font(.caption)
            }
        }
        .padding()
        .background(RoundedRectangle(cornerRadius: 10).fill(Color(.secondarySystemBackground)))
    }
}
