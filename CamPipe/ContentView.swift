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
                            Group {
                                if let outline = pipe.outline {   // the borders, exactly as the S3 gets them
                                    Image(uiImage: outline)
                                        .resizable()
                                        .interpolation(.none)
                                } else {                          // boxes only (stub detector)
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
            Text("YOLOE settings").font(.headline)

            Toggle("Use the S3's settings", isOn: $pipe.useS3Settings)
            Text(pipe.s3Settings.map { s3 in
                "S3 sent: \(s3.prompt.isEmpty ? "outline everything" : "find \"\(s3.prompt)\""), "
                    + "confidence \(pct(s3.minScore)), merge overlap \(pct(s3.mergeIoU)), "
                    + "background max \(pct(s3.backgroundShare)), surround max \(pct(s3.surroundShare))"
            } ?? "The last photo carried no S3 settings, so the phone's settings below are used.")
                .font(.caption)
                .foregroundColor(.secondary)

            Text("What to outline (phone)").font(.subheadline)
            TextField("Blank = everything", text: $pipe.phoneSettings.prompt)
                .textFieldStyle(.roundedBorder)
                .textInputAutocapitalization(.never)
                .disableAutocorrection(true)
            Text("Type a thing, e.g. \"mug\", to outline only that. On the S3 you type it right after taking the photo.")
                .font(.caption2)
                .foregroundColor(.secondary)

            Text("Min confidence (phone): \(pct(pipe.phoneSettings.minScore))")
                .font(.subheadline)
            Slider(value: $pipe.phoneSettings.minScore, in: 0.05...0.95, step: 0.05)
            Text("Outline an object only if YOLOE is at least this sure. Lower = more outlines.")
                .font(.caption2)
                .foregroundColor(.secondary)

            Text("Merge overlap (phone): \(pct(pipe.phoneSettings.mergeIoU))")
                .font(.subheadline)
            Slider(value: $pipe.phoneSettings.mergeIoU, in: 0.05...1.0, step: 0.05)
            Text("Objects overlapping more than this count as one. Lower = merge more.")
                .font(.caption2)
                .foregroundColor(.secondary)

            Text("Background max (phone): \(pct(pipe.phoneSettings.backgroundShare))")
                .font(.subheadline)
            Slider(value: $pipe.phoneSettings.backgroundShare, in: 0.05...1.0, step: 0.05)
            Text("An outline covering more of the photo than this is taken for the floor, a wall or a table, and dropped. Lower = drop more; 100% = off.")
                .font(.caption2)
                .foregroundColor(.secondary)

            Text("Surround max (phone): \(pct(pipe.phoneSettings.surroundShare))")
                .font(.subheadline)
            Slider(value: $pipe.phoneSettings.surroundShare, in: 0.05...1.0, step: 0.05)
            Text("An outline covering more than this with a smaller object sitting in a hole of it (the table under a mug) is dropped too. Lower = drop more; 100% = off.")
                .font(.caption2)
                .foregroundColor(.secondary)

            if let used = pipe.lastUsedSettings {
                Text("Last run: \(used.prompt.isEmpty ? "everything" : "\"\(used.prompt)\""), "
                     + "\(pct(used.minScore)) / \(pct(used.mergeIoU)) / "
                     + "\(pct(used.backgroundShare)) / \(pct(used.surroundShare))")
                    .font(.caption)
            }
        }
        .padding()
        .background(RoundedRectangle(cornerRadius: 10).fill(Color(.secondarySystemBackground)))
    }
}
