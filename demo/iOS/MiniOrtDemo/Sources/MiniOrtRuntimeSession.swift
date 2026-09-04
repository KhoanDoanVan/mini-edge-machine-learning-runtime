import Foundation
import MiniOrt

struct MiniOrtRuntimeError: LocalizedError {
    let message: String

    var errorDescription: String? { message }
}

final class MiniOrtRuntimeSession {
    private var handle: OpaquePointer?

    let inputFeatures: Int
    let outputFeatures: Int

    init(modelData: Data) throws {
        var session: OpaquePointer?
        let status = modelData.withUnsafeBytes { bytes in
            MiniOrtCreateSessionFromBuffer(
                bytes.baseAddress,
                bytes.count,
                &session
            )
        }
        try Self.throwIfError(status)
        guard let session else {
            throw MiniOrtRuntimeError(
                message: "The native runtime returned no session"
            )
        }

        handle = session
        inputFeatures = MiniOrtGetInputFeatureCount(session)
        outputFeatures = MiniOrtGetOutputFeatureCount(session)
        if inputFeatures == 0 || outputFeatures == 0 {
            MiniOrtReleaseSession(session)
            handle = nil
            throw MiniOrtRuntimeError(
                message: "The model has invalid input or output metadata"
            )
        }
    }

    deinit {
        if let handle {
            MiniOrtReleaseSession(handle)
        }
    }

    func run(input: [Float], batch: Int = 1) throws -> [Float] {
        guard let handle else {
            throw MiniOrtRuntimeError(message: "The session is closed")
        }
        let (expectedInputCount, inputOverflow) =
            batch.multipliedReportingOverflow(by: inputFeatures)
        guard batch > 0, !inputOverflow, input.count == expectedInputCount else {
            throw MiniOrtRuntimeError(
                message: "Input shape does not match the model"
            )
        }

        let (outputCount, overflow) = batch.multipliedReportingOverflow(
            by: outputFeatures
        )
        guard !overflow else {
            throw MiniOrtRuntimeError(message: "Output size overflow")
        }

        let inputShape = [Int64(batch), Int64(inputFeatures)]
        var inputValue: OpaquePointer?
        let inputStatus = input.withUnsafeBufferPointer { values in
            inputShape.withUnsafeBufferPointer { shape in
                MiniOrtCreateFloatTensor(
                    values.baseAddress,
                    values.count,
                    shape.baseAddress,
                    shape.count,
                    &inputValue
                )
            }
        }
        try Self.throwIfError(inputStatus)
        guard let inputValue else {
            throw MiniOrtRuntimeError(
                message: "The native runtime returned no input Tensor"
            )
        }
        defer { MiniOrtReleaseValue(inputValue) }

        let outputStorage = [Float](repeating: 0, count: outputCount)
        let outputShape = [Int64(batch), Int64(outputFeatures)]
        var outputValue: OpaquePointer?
        let outputStatus = outputStorage.withUnsafeBufferPointer { values in
            outputShape.withUnsafeBufferPointer { shape in
                MiniOrtCreateFloatTensor(
                    values.baseAddress,
                    values.count,
                    shape.baseAddress,
                    shape.count,
                    &outputValue
                )
            }
        }
        try Self.throwIfError(outputStatus)
        guard let outputValue else {
            throw MiniOrtRuntimeError(
                message: "The native runtime returned no output Tensor"
            )
        }
        defer { MiniOrtReleaseValue(outputValue) }

        try Self.throwIfError(MiniOrtRunInto(handle, inputValue, outputValue))
        guard let outputData = MiniOrtGetTensorData(outputValue) else {
            throw MiniOrtRuntimeError(
                message: "The native runtime returned no output data"
            )
        }
        return Array(
            UnsafeBufferPointer(start: outputData, count: outputCount)
        )
    }

    private static func throwIfError(_ status: OpaquePointer?) throws {
        guard let status else { return }
        defer { MiniOrtReleaseStatus(status) }
        let message = MiniOrtGetErrorMessage(status).map(String.init(cString:))
            ?? "Unknown native runtime error"
        throw MiniOrtRuntimeError(message: message)
    }
}
