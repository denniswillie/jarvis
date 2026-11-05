import Darwin
import Foundation

/// Opt-in, recording-worker-only probe. One private file, at most 20 seconds.
/// This observes the current filter; it never changes playback or microphone PCM.
final class JarvisRecordingEchoDiagnostic {
    private let file: FileHandle
    private let directory: URL
    private let maximumFrames: Int
    private let channelNames: [String]
    private var origin: Int64?
    private var written = 0
    private var gaps: [[String: Int64]] = []
    private var closed = false

    static func configured(directoryName: String = "JarvisEchoDiagnostic",
                           channelNames: [String] = ["microphone_input", "current_filter_playback_reference", "filtered_microphone", "selected_digital_mix"]) -> JarvisRecordingEchoDiagnostic? {
        #if DEBUG
        guard ProcessInfo.processInfo.environment["JARVIS_RECORDING_ECHO_DIAGNOSTICS"] == "1",
              let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return nil }
        return JarvisRecordingEchoDiagnostic(directory: documents.appendingPathComponent(directoryName), channelNames: channelNames)
        #else
        return nil
        #endif
    }

    init?(directory: URL, maximumFrames: Int = 20 * 48_000,
          channelNames: [String] = ["microphone_input", "current_filter_playback_reference", "filtered_microphone", "selected_digital_mix"]) {
        guard (1...(20 * 48_000)).contains(maximumFrames), channelNames.count == 4 else { return nil }
        self.directory = directory; self.maximumFrames = maximumFrames
        self.channelNames = channelNames
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true,
                                                    attributes: [.posixPermissions: 0o700])
        } catch { return nil }
        let path = directory.appendingPathComponent("capture.wav").path
        let descriptor = open(path, O_WRONLY | O_CREAT | O_EXCL, 0o600)
        guard descriptor >= 0 else { return nil } // Never overwrite an earlier probe.
        file = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
        do { try file.write(contentsOf: Self.header(frames: 0)) }
        catch { try? file.close(); return nil }
    }

    deinit { finish() }

    func append(startFrame: Int64, microphone: [Float], reference: [Float],
                cleaned: [Float], direct: [Float], count: Int) {
        guard !closed, count > 0, count <= microphone.count, count <= reference.count,
              count <= cleaned.count, count <= direct.count else { return }
        if origin == nil { origin = startFrame }
        guard let origin, startFrame >= origin + Int64(written) else { return }
        do {
            let gap = Int(min(Int64(maximumFrames - written), startFrame - origin - Int64(written)))
            if gap > 0 {
                gaps.append(["start_frame": origin + Int64(written), "frames": Int64(gap)])
                var remaining = gap
                while remaining > 0 {
                    let part = min(remaining, 4800)
                    try file.write(contentsOf: Data(count: part * 16))
                    written += part; remaining -= part
                }
            }
            let accepted = min(count, maximumFrames - written)
            var data = Data(capacity: accepted * 16)
            for i in 0..<accepted {
                for sample in [microphone[i], reference[i], cleaned[i], direct[i]] {
                    Self.put(sample.bitPattern, into: &data)
                }
            }
            try file.write(contentsOf: data)
            written += accepted
            if written == maximumFrames { finish() }
        } catch { finish() } // A diagnostic failure must not fail the recording.
    }

    func finish() {
        guard !closed else { return }
        closed = true
        defer { try? file.close() }
        do {
            try file.seek(toOffset: 0)
            try file.write(contentsOf: Self.header(frames: written))
            try file.synchronize()
            let metadata: [String: Any] = [
                "format": "float32_le_interleaved_wav", "sample_rate": 48_000,
                "channels": channelNames,
                "first_recording_frame": origin ?? 0, "frames": written, "gaps": gaps,
                "maximum_seconds": 20, "finished_at": ISO8601DateFormatter().string(from: Date()),
                "reference_basis": "Rendered Voice plus gain-adjusted consumed original music, aligned by the existing recorder. Not an independent post-mix hardware tap.",
                "scope": "Explicit diagnostic only; does not prove acoustic playback or input clock alignment."
            ]
            let bytes = try JSONSerialization.data(withJSONObject: metadata, options: [.prettyPrinted, .sortedKeys])
            try bytes.write(to: directory.appendingPathComponent("capture.json"), options: .atomic)
        } catch { }
    }

    private static func put<T: FixedWidthInteger>(_ value: T, into data: inout Data) {
        var little = value.littleEndian
        withUnsafeBytes(of: &little) { data.append(contentsOf: $0) }
    }

    private static func header(frames: Int) -> Data {
        var data = Data("RIFF".utf8)
        put(UInt32(36 + frames * 16), into: &data)
        data.append(contentsOf: "WAVEfmt ".utf8)
        put(UInt32(16), into: &data); put(UInt16(3), into: &data) // IEEE Float32
        put(UInt16(4), into: &data); put(UInt32(48_000), into: &data)
        put(UInt32(48_000 * 16), into: &data); put(UInt16(16), into: &data)
        put(UInt16(32), into: &data); data.append(contentsOf: "data".utf8)
        put(UInt32(frames * 16), into: &data)
        return data
    }
}

/// Native-rate, post-mix I/O probe. Used only by the off-I/O capture queue.
/// Receipt times describe arrival at the phone adapter, not microphone capture.
final class JarvisRecordingIODiagnostic {
    private let files: [FileHandle]
    private let sampleRate: Double
    private let channels: [Int]
    private let maximumFrames: Int
    private let openedAt = ProcessInfo.processInfo.systemUptime
    private var frames = [0, 0]
    private var bytesWritten = 0
    private var eventsWritten = 0
    private var closed = false

    static func configured(sampleRate: Double, inputChannels: Int, outputChannels: Int,
                           context: [String: Any]) -> JarvisRecordingIODiagnostic? {
        #if DEBUG
        guard ProcessInfo.processInfo.environment["JARVIS_RECORDING_IO_DIAGNOSTICS"] == "1",
              let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return nil }
        return JarvisRecordingIODiagnostic(directory: documents.appendingPathComponent("JarvisEchoIODiagnostic"),
            sampleRate: sampleRate, inputChannels: inputChannels, outputChannels: outputChannels, context: context)
        #else
        return nil
        #endif
    }

    init?(directory: URL, sampleRate: Double, inputChannels: Int, outputChannels: Int,
          context: [String: Any] = [:], frameLimit: Int? = nil) {
        guard sampleRate.isFinite, (8000...192000).contains(sampleRate),
              (1...2).contains(inputChannels), (1...2).contains(outputChannels) else { return nil }
        let limit = frameLimit ?? Int(sampleRate * 20)
        guard limit > 0, limit <= Int(sampleRate * 20), mkdir(directory.path, 0o700) == 0 else { return nil }
        var opened: [FileHandle] = []
        for name in ["microphone.pcm", "playback.pcm", "packets.jsonl"] {
            let descriptor = open(directory.appendingPathComponent(name).path, O_WRONLY | O_CREAT | O_EXCL, 0o600)
            guard descriptor >= 0 else { for file in opened { try? file.close() }; return nil }
            opened.append(FileHandle(fileDescriptor: descriptor, closeOnDealloc: true))
        }
        files = opened; self.sampleRate = sampleRate
        channels = [inputChannels, outputChannels]; maximumFrames = limit
        var timebase = mach_timebase_info_data_t()
        mach_timebase_info(&timebase)
        event(["type": "header", "format": "signed_pcm16_le_interleaved", "sample_rate": sampleRate,
               "channels": channels, "maximum_frames_per_track": limit,
               "host_tick_numer": timebase.numer, "host_tick_denom": timebase.denom,
               "created_at": ISO8601DateFormatter().string(from: Date()), "context": context,
               "tracks": ["microphone_before_network_ducking", "post_mix_output_submission"],
               "scope": "Output is the exact submitted mix, not acoustic playback. External submit times are phone-adapter arrival times, not DAT capture timestamps. No resampling or timeline alignment is applied."])
    }

    deinit { finish(reason: "released") }

    func append(track: Int, pcm16: Data, frameCount: Int, metadata: [String: Any]) {
        guard !closed else { return }
        guard (0...1).contains(track), (1...8192).contains(frameCount),
              pcm16.count == frameCount * channels[track] * 2 else { finish(reason: "invalid_packet"); return }
        guard ProcessInfo.processInfo.systemUptime - openedAt < 22 else { finish(reason: "wall_time_cap"); return }
        let accepted = min(frameCount, maximumFrames - frames[track])
        guard accepted > 0 else { return }
        let count = accepted * channels[track] * 2
        guard bytesWritten + count <= 40 * 1024 * 1024 else { finish(reason: "byte_cap"); return }
        do {
            try files[track].write(contentsOf: pcm16.prefix(count))
            var row = metadata
            row["type"] = "packet"; row["track"] = track
            row["byte_offset"] = frames[track] * channels[track] * 2
            row["frames_written"] = accepted; row["packet_frames"] = frameCount
            frames[track] += accepted; bytesWritten += count
            event(row)
            if frames.allSatisfy({ $0 == maximumFrames }) { finish(reason: "frame_cap") }
        } catch { finish(reason: "write_failed") }
    }

    func event(_ value: [String: Any]) {
        guard !closed else { return }
        guard eventsWritten < 10000 else { finish(reason: "event_cap"); return }
        do {
            let data = try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]) + Data([10])
            guard data.count <= 16384, bytesWritten + data.count <= 40 * 1024 * 1024 else {
                finish(reason: "metadata_cap"); return
            }
            try files[2].write(contentsOf: data)
            bytesWritten += data.count; eventsWritten += 1
        } catch { finish(reason: "metadata_failed") }
    }

    func finish(reason: String) {
        guard !closed else { return }
        closed = true
        if let data = try? JSONSerialization.data(withJSONObject: ["type": "end", "reason": reason,
            "frames": frames, "sample_rate": sampleRate, "events": eventsWritten,
            "elapsed_seconds": ProcessInfo.processInfo.systemUptime - openedAt], options: [.sortedKeys]) {
            try? files[2].write(contentsOf: data + Data([10]))
        }
        for file in files { try? file.close() }
    }
}
