import SwiftUI

struct ContentView: View {
    @StateObject private var pipe = Pipeline()

    var body: some View {
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
            Spacer()
        }
        .padding()
    }
}
