import Foundation
import SwiftUI

struct ContentView: View {
    @State private var state = "Loading model…"
    @State private var output = ""
    @State private var latency = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Mini Edge Runtime")
                .font(.title.bold())
            Text(state)
                .foregroundStyle(state == "Inference complete" ? .green : .primary)
            infoRow("Backend", value: "C++ CPU")
            infoRow("Input", value: "[1.0, 2.0, 3.0]")
            infoRow("Output", value: output.isEmpty ? "—" : output)
            infoRow("Latency", value: latency.isEmpty ? "—" : latency)
        }
        .padding(24)
        .task { runBundledModel() }
    }

    private func infoRow(_ label: String, value: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label).foregroundStyle(.secondary)
            Spacer()
            Text(value).multilineTextAlignment(.trailing)
        }
    }

    @MainActor
    private func runBundledModel() {
        do {
            guard let modelURL = Bundle.main.url(
                forResource: "tiny_mlp",
                withExtension: "mer"
            ) else {
                throw MiniOrtRuntimeError(
                    message: "tiny_mlp.mer is missing from the app bundle"
                )
            }

            let modelData = try Data(contentsOf: modelURL)
            let session = try MiniOrtRuntimeSession(modelData: modelData)
            let start = ProcessInfo.processInfo.systemUptime
            let values = try session.run(input: [1, 2, 3])
            let elapsedMilliseconds =
                (ProcessInfo.processInfo.systemUptime - start) * 1_000

            output = values.map { String(format: "%.2f", $0) }
                .joined(separator: ", ")
            output = "[\(output)]"
            latency = String(format: "%.3f ms", elapsedMilliseconds)
            state = "Inference complete"
        } catch {
            state = error.localizedDescription
        }
    }
}
