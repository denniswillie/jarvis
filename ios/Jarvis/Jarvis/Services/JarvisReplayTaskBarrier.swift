#if DEBUG
import CoreFoundation
import CryptoKit
import Darwin
import Foundation

enum JarvisReplayTaskError: Error, Equatable {
    case invalidControl, invalidScope, staleReceipt, sequenceRejected, bindingChanged
    case claimUnavailable, incompleteStimuli, retired
}

enum JarvisReplayTaskFailureStage: String, Sendable {
    case pollState = "poll_state", controlRead = "control_read", receiptConsume = "receipt_consume"
    case stimulusCompletion = "stimulus_completion", stimulusPublish = "stimulus_publish"
}

/// Static guard categories only. Never retain the rejected field value or filesystem path.
enum JarvisReplayTaskFailurePredicate: String, Sendable {
    case directoryAvailable = "directory_available", directoryIdentity = "directory_identity"
    case readName = "read_name", readOpen = "read_open", readMetadata = "read_metadata"
    case readStableBytes = "read_stable_bytes"
    case publishEnvelope = "publish_envelope", publishOpen = "publish_open"
    case publishWrite = "publish_write", publishExclusive = "publish_exclusive"
    case receiptJSON = "receipt_json", receiptKeys = "receipt_keys", receiptScope = "receipt_scope"
    case receiptSequence = "receipt_sequence", receiptTime = "receipt_time"
    case admissionObject = "admission_object", admissionKeys = "admission_keys"
    case admissionScope = "admission_scope", admissionIdentity = "admission_identity"
    case admissionRevision = "admission_revision", admissionProviderCall = "admission_provider_call"
    case admissionDigest = "admission_digest", admissionEncoding = "admission_encoding"
    case admissionStable = "admission_stable", receiptStatus = "receipt_status"
    case receiptReason = "receipt_reason", claimObject = "claim_object", claimKeys = "claim_keys"
    case claimIdentity = "claim_identity", claimRevision = "claim_revision", claimScope = "claim_scope"
    case claimEncoding = "claim_encoding", claimStable = "claim_stable", blockedShape = "blocked_shape"
    case completionEvidence = "completion_evidence"
}

struct JarvisReplayTaskFailureDiagnostic: Sendable {
    let stage: JarvisReplayTaskFailureStage
    let predicate: JarvisReplayTaskFailurePredicate
    let generation: UUID
    let observedAt: TimeInterval
    let deliveredAt: TimeInterval
    var isValid: Bool {
        observedAt.isFinite && observedAt >= 0 && deliveredAt.isFinite && deliveredAt >= observedAt
    }
    var json: [String: Any] {
        ["schema_version": 1, "error_code": "invalidControl", "stage": stage.rawValue,
         "predicate": predicate.rawValue, "barrier_generation": generation.uuidString.lowercased(),
         "observed_monotonic_s": observedAt, "delivered_monotonic_s": deliveredAt]
    }
}

/// Small strict JSON boundary. No provider text, credentials or commands are retained.
enum JarvisReplayTaskJSON {
    static func object(_ data: Data, limit: Int) throws -> [String: Any] {
        guard !data.isEmpty, data.count <= limit,
              let result = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw JarvisReplayTaskError.invalidControl
        }
        // Foundation accepts duplicate keys. Reject them, including escaped aliases.
        var scanner = Keys(bytes: Array(data))
        try scanner.value(depth: 0)
        scanner.space()
        guard scanner.index == scanner.bytes.count else { throw JarvisReplayTaskError.invalidControl }
        return result
    }
    private struct Keys {
        let bytes: [UInt8]
        var index = 0
        mutating func space() { while index < bytes.count, [9, 10, 13, 32].contains(bytes[index]) { index += 1 } }
        mutating func string() throws -> String {
            let start = index
            guard index < bytes.count, bytes[index] == 34 else { throw JarvisReplayTaskError.invalidControl }
            index += 1
            while index < bytes.count {
                let byte = bytes[index]; index += 1
                if byte == 92 { index += 1 }
                else if byte == 34 { return try JSONDecoder().decode(String.self, from: Data(bytes[start..<index])) }
            }
            throw JarvisReplayTaskError.invalidControl
        }
        mutating func value(depth: Int) throws {
            space()
            guard depth <= 24, index < bytes.count else { throw JarvisReplayTaskError.invalidControl }
            if bytes[index] == 34 { _ = try string(); return }
            if bytes[index] == 123 || bytes[index] == 91 {
                let object = bytes[index] == 123, end: UInt8 = bytes[index] == 123 ? 125 : 93
                index += 1; space()
                var keys = Set<String>()
                if index < bytes.count, bytes[index] == end { index += 1; return }
                while index < bytes.count {
                    if object {
                        let key = try string()
                        guard keys.insert(key).inserted else { throw JarvisReplayTaskError.invalidControl }
                        space()
                        guard index < bytes.count, bytes[index] == 58 else { throw JarvisReplayTaskError.invalidControl }
                        index += 1
                    }
                    try value(depth: depth + 1); space()
                    guard index < bytes.count else { throw JarvisReplayTaskError.invalidControl }
                    if bytes[index] == end { index += 1; return }
                    guard bytes[index] == 44 else { throw JarvisReplayTaskError.invalidControl }
                    index += 1; space()
                }
                throw JarvisReplayTaskError.invalidControl
            }
            let start = index
            while index < bytes.count, ![9, 10, 13, 32, 44, 93, 125].contains(bytes[index]) { index += 1 }
            guard index > start else { throw JarvisReplayTaskError.invalidControl }
        }
    }
    static func keys(_ value: [String: Any], _ expected: Set<String>) throws {
        guard Set(value.keys) == expected else { throw JarvisReplayTaskError.invalidControl }
    }
    static func string(_ value: Any?, max: Int = 200) throws -> String {
        guard let value = value as? String, !value.isEmpty, value.utf8.count <= max else { throw JarvisReplayTaskError.invalidControl }
        return value
    }
    static func positive(_ value: Any?) throws -> Int {
        guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
              number.doubleValue.isFinite, number.doubleValue > 0, number.doubleValue <= 9_007_199_254_740_991,
              number.doubleValue == Double(number.int64Value) else { throw JarvisReplayTaskError.invalidControl }
        return number.intValue
    }
    static func uuid(_ value: Any?) throws -> UUID {
        var text = try string(value, max: 36)
        if text.count == 32 {
            let chars = Array(text)
            text = [String(chars[0..<8]), String(chars[8..<12]), String(chars[12..<16]),
                    String(chars[16..<20]), String(chars[20..<32])].joined(separator: "-")
        }
        guard let result = UUID(uuidString: text) else { throw JarvisReplayTaskError.invalidControl }
        return result
    }
    static func time(_ value: Any?) throws -> Date {
        let text = try string(value, max: 64)
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let date = formatter.date(from: text) { return date }
        formatter.formatOptions = [.withInternetDateTime]
        guard let date = formatter.date(from: text) else { throw JarvisReplayTaskError.invalidControl }
        return date
    }
    static func timestamp(_ date: Date) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter.string(from: date)
    }
    static func encode(_ object: [String: Any], limit: Int = 8 * 1024) throws -> Data {
        let data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        guard data.count <= limit else { throw JarvisReplayTaskError.invalidControl }
        return data
    }
    static func hash(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
}

struct JarvisReplayTaskScope: Equatable {
    static let fields: Set<String> = ["run_id", "barrier_id", "owner_id", "database_schema", "worker_id",
                                      "runtime_host_id", "model", "host_expires_at"]
    let runID: UUID
    let barrierID: UUID
    let ownerID: String
    let schema: String
    let workerID: String
    let runtimeHostID: String
    let model: String
    let expires: Date

    init(_ value: [String: Any]) throws {
        typealias J = JarvisReplayTaskJSON
        try J.keys(value, Self.fields)
        runID = try J.uuid(value["run_id"]); barrierID = try J.uuid(value["barrier_id"])
        ownerID = try J.string(value["owner_id"]); schema = try J.string(value["database_schema"])
        workerID = try J.string(value["worker_id"], max: 100)
        runtimeHostID = try J.string(value["runtime_host_id"]); model = try J.string(value["model"], max: 100)
        expires = try J.time(value["host_expires_at"])
        let prefix = "jarvis_sb_eval_" + runID.uuidString.replacingOccurrences(of: "-", with: "").lowercased() + "_"
        guard schema.hasPrefix(prefix), !schema.dropFirst(prefix.count).isEmpty,
              schema.dropFirst(prefix.count).allSatisfy({ $0.isASCII && $0.isNumber }) else { throw JarvisReplayTaskError.invalidScope }
    }
}

struct JarvisReplayTaskConfiguration {
    let scope: JarvisReplayTaskScope
    let manifestHash: String
    let scopeHash: String
    let controls: URL

    static func load(scopeData: Data, manifestData: Data, inputsRoot: URL,
                     runID: UUID, ownerID: String) throws -> Self {
        var value = try JarvisReplayTaskJSON.object(scopeData, limit: 8 * 1024)
        guard value.removeValue(forKey: "format") as? String == "jarvis.native_task_scope.v1" else {
            throw JarvisReplayTaskError.invalidScope
        }
        let scope = try JarvisReplayTaskScope(value)
        let inputs = inputsRoot.standardizedFileURL
        guard inputs.lastPathComponent == "inputs", scope.runID == runID, scope.ownerID == ownerID,
              scope.expires > Date() else { throw JarvisReplayTaskError.invalidScope }
        let controls = inputs.deletingLastPathComponent().resolvingSymlinksInPath().appendingPathComponent("controls")
        let directory = try JarvisReplayTaskControlDirectory(controls)
        directory.close()
        return Self(scope: scope, manifestHash: JarvisReplayTaskJSON.hash(manifestData),
                    scopeHash: JarvisReplayTaskJSON.hash(scopeData), controls: controls)
    }

    func envelope(format: String, sessionID: UUID, now: Date) -> [String: Any] {
        ["format": format, "run_id": scope.runID.uuidString.lowercased(),
         "barrier_id": scope.barrierID.uuidString.lowercased(), "owner_id": scope.ownerID,
         "session_id": sessionID.uuidString.lowercased(), "manifest_file_sha256": manifestHash,
         "scope_file_sha256": scopeHash, "recorded_at": JarvisReplayTaskJSON.timestamp(now)]
    }
}

/// Watermarks and the canonical task/attempt are pinned for the whole session, including reconnects.
struct JarvisReplayTaskReceiptConsumer {
    let scope: JarvisReplayTaskScope
    let sessionID: UUID
    private(set) var sequence = 0
    private(set) var admission: Data?
    private(set) var claimIdentity: Data?
    private var previousBytes: Data?
    private(set) var failurePredicate: JarvisReplayTaskFailurePredicate = .receiptJSON
    private var eligible: (observed: Date, expires: Date, uptimeExpiry: TimeInterval)?
    static let reasons: Set<String> = ["active_claim_observed", "admission_not_committed", "admission_mismatch",
        "ambiguous_tasks", "source_ineligible", "source_replaced", "task_revision_replaced", "task_not_active",
        "claim_ineligible", "claim_replaced", "lease_expired", "worker_ineligible", "schema_mismatch",
        "borrowed_transaction", "canonical_unavailable"]

    init(scope: JarvisReplayTaskScope, sessionID: UUID) { self.scope = scope; self.sessionID = sessionID }
    mutating func unavailable() { eligible = nil }
    func eligibleSequence(now: Date, uptime: TimeInterval) -> Int? {
        guard let eligible, uptime.isFinite, now >= eligible.observed, now < eligible.expires,
              now < scope.expires, uptime < eligible.uptimeExpiry else { return nil }
        return sequence
    }

    @discardableResult mutating func consume(_ data: Data, now: Date, uptime: TimeInterval) throws -> Bool {
        if data == previousBytes { return false }
        eligible = nil
        typealias J = JarvisReplayTaskJSON
        failurePredicate = .receiptJSON
        let value = try J.object(data, limit: 16 * 1024)
        failurePredicate = .receiptKeys
        try J.keys(value, ["format", "scope", "admission", "sequence", "observed_at", "expires_at", "status", "reason", "claim"])
        failurePredicate = .receiptScope
        guard value["format"] as? String == "jarvis.native_task_barrier.v1",
              var receivedScope = value["scope"] as? [String: Any],
              try J.uuid(receivedScope.removeValue(forKey: "session_id")) == sessionID,
              try JarvisReplayTaskScope(receivedScope) == scope else { throw JarvisReplayTaskError.invalidScope }
        failurePredicate = .receiptSequence
        let next = try J.positive(value["sequence"])
        guard next > sequence else { throw JarvisReplayTaskError.sequenceRejected }
        failurePredicate = .receiptTime
        let observed = try J.time(value["observed_at"]), expires = try J.time(value["expires_at"])
        guard uptime.isFinite, uptime >= 0, observed <= now, now < expires, expires <= scope.expires,
              expires > observed, expires.timeIntervalSince(observed) <= 5 else { throw JarvisReplayTaskError.staleReceipt }
        failurePredicate = .admissionObject
        guard var witness = value["admission"] as? [String: Any] else { throw JarvisReplayTaskError.invalidControl }
        failurePredicate = .admissionKeys
        try J.keys(witness, ["owner_id", "session_id", "request_event_id", "request_revision", "request_id", "task_id",
                             "accepted_revision", "parameters_digest", "provider_call_id", "operation"])
        failurePredicate = .admissionScope
        guard try J.string(witness["owner_id"]) == scope.ownerID,
              try J.uuid(witness["session_id"]) == sessionID, witness["operation"] as? String == "create" else {
            throw JarvisReplayTaskError.bindingChanged
        }
        failurePredicate = .admissionIdentity
        for key in ["session_id", "request_event_id", "request_id", "task_id"] {
            witness[key] = try J.uuid(witness[key]).uuidString.lowercased()
        }
        failurePredicate = .admissionRevision
        for key in ["request_revision", "accepted_revision"] { witness[key] = try J.positive(witness[key]) }
        failurePredicate = .admissionProviderCall
        _ = try J.string(witness["provider_call_id"], max: 512)
        failurePredicate = .admissionDigest
        let digest = try J.string(witness["parameters_digest"], max: 64)
        guard digest.utf8.count == 64, digest.allSatisfy({ "0123456789abcdef".contains($0) }) else { throw JarvisReplayTaskError.invalidControl }
        failurePredicate = .admissionEncoding
        let witnessBytes = try J.encode(witness)
        failurePredicate = .admissionStable
        guard admission == nil || admission == witnessBytes else { throw JarvisReplayTaskError.bindingChanged }
        failurePredicate = .receiptStatus
        let reason = try J.string(value["reason"]), status = try J.string(value["status"])
        failurePredicate = .receiptReason
        guard Self.reasons.contains(reason) else { throw JarvisReplayTaskError.invalidControl }
        if status == "eligible" {
            failurePredicate = .claimObject
            guard reason == "active_claim_observed", var claim = value["claim"] as? [String: Any] else {
                throw JarvisReplayTaskError.invalidControl
            }
            failurePredicate = .claimKeys
            try J.keys(claim, ["task_id", "task_revision", "attempt_id", "attempt_number", "fence", "worker_id",
                              "runtime_host_id", "lease_until", "worker_expires_at", "task_status", "attempt_status", "provider_state"])
            failurePredicate = .claimIdentity
            for key in ["task_id", "attempt_id"] { claim[key] = try J.uuid(claim[key]).uuidString.lowercased() }
            failurePredicate = .claimRevision
            for key in ["task_revision", "attempt_number", "fence"] { claim[key] = try J.positive(claim[key]) }
            failurePredicate = .claimScope
            guard claim["task_id"] as? String == witness["task_id"] as? String,
                  claim["task_revision"] as? Int == witness["accepted_revision"] as? Int,
                  try J.string(claim["worker_id"], max: 100) == scope.workerID,
                  try J.string(claim["runtime_host_id"]) == scope.runtimeHostID,
                  claim["task_status"] as? String == "running", claim["attempt_status"] as? String == "running",
                  ["not_started", "submitting", "live", "terminal"].contains(try J.string(claim["provider_state"])),
                  expires <= (try J.time(claim["lease_until"])),
                  expires <= (try J.time(claim["worker_expires_at"])) else { throw JarvisReplayTaskError.bindingChanged }
            for key in ["lease_until", "worker_expires_at", "task_status", "attempt_status", "provider_state"] {
                claim.removeValue(forKey: key)
            }
            failurePredicate = .claimEncoding
            let identity = try J.encode(claim)
            failurePredicate = .claimStable
            guard claimIdentity == nil || claimIdentity == identity else { throw JarvisReplayTaskError.bindingChanged }
            claimIdentity = identity
            eligible = (observed, expires, uptime + expires.timeIntervalSince(now))
        } else {
            failurePredicate = .blockedShape
            guard status == "blocked", reason != "active_claim_observed", value["claim"] is NSNull else {
                throw JarvisReplayTaskError.invalidControl
            }
        }
        sequence = next; admission = witnessBytes; previousBytes = data
        return true
    }
}

/// Fixed private directory, regular single-link files only. No socket, FIFO, symlink or unbounded read.
final class JarvisReplayTaskControlDirectory {
    private(set) var failurePredicate: JarvisReplayTaskFailurePredicate = .directoryIdentity
    private var fd: Int32
    private let root: URL
    private let device: dev_t
    private let inode: ino_t
    private let afterReadOpen: ((String) -> Void)?
    init(_ root: URL, afterReadOpen: ((String) -> Void)? = nil) throws {
        self.root = root
        self.afterReadOpen = afterReadOpen
        fd = Darwin.open(root.path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        var info = stat()
        guard fd >= 0, fstat(fd, &info) == 0, info.st_uid == geteuid(),
              info.st_mode & 0o077 == 0, info.st_mode & S_IFMT == S_IFDIR else {
            if fd >= 0 { Darwin.close(fd) }
            throw JarvisReplayTaskError.invalidControl
        }
        device = info.st_dev; inode = info.st_ino
    }
    deinit { close() }
    func close() { if fd >= 0 { Darwin.close(fd); fd = -1 } }
    private func check() throws {
        failurePredicate = .directoryIdentity
        var info = stat()
        guard fd >= 0, lstat(root.path, &info) == 0, info.st_dev == device, info.st_ino == inode,
              info.st_mode & S_IFMT == S_IFDIR, info.st_mode & 0o077 == 0, info.st_uid == geteuid() else {
            throw JarvisReplayTaskError.invalidControl
        }
    }
    func readBarrier() throws -> Data? { try read("task-barrier.json", limit: 16 * 1024) }
    func read(_ name: String, limit: Int) throws -> Data? {
        try check()
        failurePredicate = .readName
        guard ["task-barrier.json", "session-binding.json", "stimulus-completion.json"].contains(name) else {
            throw JarvisReplayTaskError.invalidControl
        }
        failurePredicate = .readOpen
        let file = openat(fd, name, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
        if file < 0 {
            if errno == ENOENT { return nil }
            throw JarvisReplayTaskError.invalidControl
        }
        defer { Darwin.close(file) }
        failurePredicate = .readMetadata
        afterReadOpen?(name)
        var before = stat()
        guard fstat(file, &before) == 0, before.st_mode & S_IFMT == S_IFREG,
              before.st_uid == geteuid(), before.st_mode & 0o077 == 0,
              before.st_size > 0, before.st_size <= limit else { throw JarvisReplayTaskError.invalidControl }
        // Atomic replacement can unlink the old inode after openat. Do not
        // consume its bytes or retain claim eligibility; poll the current name
        // again on the next ordinary tick, just as for a missing receipt.
        if before.st_nlink == 0 {
            try check()
            return nil
        }
        guard before.st_nlink == 1 else { throw JarvisReplayTaskError.invalidControl }
        failurePredicate = .readStableBytes
        var bytes = [UInt8](repeating: 0, count: limit + 1)
        let count = bytes.withUnsafeMutableBytes { Darwin.read(file, $0.baseAddress, $0.count) }
        var after = stat()
        guard count == before.st_size, count <= limit, fstat(file, &after) == 0,
              before.st_size == after.st_size, before.st_mtimespec.tv_sec == after.st_mtimespec.tv_sec,
              before.st_mtimespec.tv_nsec == after.st_mtimespec.tv_nsec else { throw JarvisReplayTaskError.invalidControl }
        try check()
        return Data(bytes.prefix(count))
    }
    func publish(_ name: String, data: Data) throws {
        try check()
        failurePredicate = .publishEnvelope
        guard ["session-binding.json", "stimulus-completion.json"].contains(name), !data.isEmpty, data.count <= 8 * 1024 else {
            throw JarvisReplayTaskError.invalidControl
        }
        failurePredicate = .publishOpen
        let temporary = ".app-" + UUID().uuidString
        let file = openat(fd, temporary, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, mode_t(0o600))
        guard file >= 0 else { throw JarvisReplayTaskError.invalidControl }
        defer { Darwin.close(file); unlinkat(fd, temporary, 0) }
        failurePredicate = .publishWrite
        let written = data.withUnsafeBytes { Darwin.write(file, $0.baseAddress, $0.count) }
        guard written == data.count, fsync(file) == 0 else { throw JarvisReplayTaskError.invalidControl }
        try check()
        // A one-shot observation must never overwrite an earlier run or completion.
        failurePredicate = .publishExclusive
        guard renameatx_np(fd, temporary, fd, name, UInt32(RENAME_EXCL)) == 0, fsync(fd) == 0 else {
            throw JarvisReplayTaskError.invalidControl
        }
    }
}

/// Only metadata from actual SDK callbacks is eligible for the completion observation.
enum JarvisReplayTaskCompletion {
    static func make(configuration: JarvisReplayTaskConfiguration, sessionID: UUID, plan: JarvisReplayPlan,
                     evidence: JarvisReplayAudioEvidence, consumer: JarvisReplayTaskReceiptConsumer,
                     now: Date, uptime: TimeInterval) throws -> Data? {
        guard plan.runID == configuration.scope.runID, consumer.scope == configuration.scope,
              consumer.sessionID == sessionID, evidence.callbackFailure == nil, !evidence.inputMappingTruncated,
              evidence.utterances.count == plan.utterances.count,
              !evidence.utterances.contains(where: { $0.failureCode != nil || $0.status == "failed" }) else {
            throw JarvisReplayTaskError.incompleteStimuli
        }
        guard evidence.utterances.allSatisfy({ $0.status == "sdk_input_complete" }),
              let currentSequence = consumer.eligibleSequence(now: now, uptime: uptime),
              consumer.claimIdentity != nil else { return nil }
        var inputs: [[String: Any]] = []
        var previousEnd: TimeInterval = 0
        for (ordinal, item) in evidence.utterances.enumerated() {
            guard item.ordinal == ordinal, item.sampleCount == plan.utterances[ordinal].pcm16.count / 2,
                  item.sampleCount > 0, item.acceptedSamples == item.sampleCount,
                  item.pcmSHA256 == JarvisReplayTaskJSON.hash(plan.utterances[ordinal].pcm16),
                  let first = item.firstCallbackStart, let end = item.lastCallbackEnd,
                  first.isFinite, end.isFinite, first >= previousEnd, end >= first, end <= uptime else {
                throw JarvisReplayTaskError.incompleteStimuli
            }
            if plan.utterances[ordinal].trigger.requiresTaskClaim {
                guard let selected = item.taskClaimSequenceAtEnqueue, selected > 0, selected <= currentSequence else {
                    throw JarvisReplayTaskError.incompleteStimuli
                }
            } else if item.taskClaimSequenceAtEnqueue != nil { throw JarvisReplayTaskError.incompleteStimuli }
            inputs.append(["input_ordinal": ordinal, "status": "sdk_input_complete", "sample_count": item.sampleCount,
                "accepted_samples": item.acceptedSamples, "first_sdk_callback_monotonic": first,
                "last_sdk_callback_end_monotonic": end,
                "task_claim_sequence_at_enqueue": item.taskClaimSequenceAtEnqueue as Any? ?? NSNull()])
            previousEnd = end
        }
        var value = configuration.envelope(format: "jarvis.native_task_stimulus_completion.v1", sessionID: sessionID, now: now)
        value["inputs"] = inputs
        return try JarvisReplayTaskJSON.encode(value)
    }
}

/// Dedicated serial I/O queue, never the audio callback or frame loop. Stop joins it.
final class JarvisReplayTaskBarrier: @unchecked Sendable {
    typealias FailureDelivery = @MainActor @Sendable () -> Void
    var onFailure: (@MainActor (Error) -> Void)?
    private let configuration: JarvisReplayTaskConfiguration
    private let plan: JarvisReplayPlan
    private let evidence: () -> JarvisReplayAudioEvidence
    private let scheduleFailure: @Sendable (@escaping FailureDelivery) -> Void
    private let queue = DispatchQueue(label: "jarvis.debug-replay.task-controls")
    private let lock = NSLock()
    private var directory: JarvisReplayTaskControlDirectory?
    private var timer: DispatchSourceTimer?
    private var consumer: JarvisReplayTaskReceiptConsumer?
    private var retired = false
    private var failed = false
    private var published = false
    private var failureGeneration = UUID()
    private var failureDelivered = false
    private var deliveredDiagnostic: JarvisReplayTaskFailureDiagnostic?
    var completionPublished: Bool { lock.withLock { published } }
    var failureDiagnostic: JarvisReplayTaskFailureDiagnostic? { lock.withLock { deliveredDiagnostic } }

    init(configuration: JarvisReplayTaskConfiguration, plan: JarvisReplayPlan,
         evidence: @escaping () -> JarvisReplayAudioEvidence,
         scheduleFailure: @escaping @Sendable (@escaping FailureDelivery) -> Void = { delivery in
             Task { @MainActor in delivery() }
         }) {
        self.configuration = configuration; self.plan = plan; self.evidence = evidence
        self.scheduleFailure = scheduleFailure
    }
    func bind(sessionID: String) async throws {
        let session = try JarvisReplayTaskJSON.uuid(sessionID)
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            queue.async {
                do {
                    try self.lock.withLock {
                        guard !self.retired, !self.failed, self.consumer == nil,
                              Date() < self.configuration.scope.expires else { throw JarvisReplayTaskError.retired }
                    }
                    let directory = try JarvisReplayTaskControlDirectory(self.configuration.controls)
                    let binding = self.configuration.envelope(format: "jarvis.native_task_session_binding.v1", sessionID: session, now: Date())
                    try directory.publish("session-binding.json", data: JarvisReplayTaskJSON.encode(binding))
                    try self.lock.withLock {
                        guard !self.retired else { throw JarvisReplayTaskError.retired }
                        self.failureGeneration = UUID()
                        self.consumer = JarvisReplayTaskReceiptConsumer(scope: self.configuration.scope, sessionID: session)
                    }
                    self.directory = directory
                    let timer = DispatchSource.makeTimerSource(queue: self.queue)
                    timer.schedule(deadline: .now(), repeating: .milliseconds(100))
                    timer.setEventHandler { [weak self] in self?.poll() }
                    self.timer = timer; timer.resume()
                    continuation.resume()
                } catch { continuation.resume(throwing: error) }
            }
        }
    }
    func eligibleSequence() -> Int? {
        lock.withLock {
            guard !retired, !failed else { return nil }
            return consumer?.eligibleSequence(now: Date(), uptime: ProcessInfo.processInfo.systemUptime)
        }
    }
    func enqueue(requiresClaim: Bool, _ body: (Int?) throws -> Void) throws {
        try lock.withLock {
            guard !retired, !failed, consumer != nil else { throw JarvisReplayTaskError.retired }
            let sequence = consumer?.eligibleSequence(now: Date(), uptime: ProcessInfo.processInfo.systemUptime)
            if requiresClaim && sequence == nil { throw JarvisReplayTaskError.claimUnavailable }
            try body(requiresClaim ? sequence : nil)
        }
    }
    private func poll() {
        guard lock.withLock({ !retired && !failed }) else { return }
        var stage = JarvisReplayTaskFailureStage.pollState
        do {
            guard let directory else { throw JarvisReplayTaskError.invalidControl }
            stage = .controlRead
            let data = try directory.readBarrier()
            stage = .receiptConsume
            try lock.withLock {
                guard !retired, !failed else { return }
                guard Date() < configuration.scope.expires else { throw JarvisReplayTaskError.staleReceipt }
                if let data { try consumer?.consume(data, now: Date(), uptime: ProcessInfo.processInfo.systemUptime) }
                else { consumer?.unavailable() }
            }
            stage = .stimulusCompletion
            let snapshot = evidence()
            let completion: Data? = try lock.withLock {
                guard !retired, !failed, !published, let consumer else { return nil }
                return try JarvisReplayTaskCompletion.make(configuration: configuration, sessionID: consumer.sessionID,
                    plan: plan, evidence: snapshot, consumer: consumer, now: Date(), uptime: ProcessInfo.processInfo.systemUptime)
            }
            if let completion {
                stage = .stimulusPublish
                try directory.publish("stimulus-completion.json", data: completion)
                lock.withLock { published = true }
            }
        } catch {
            let observedAt = ProcessInfo.processInfo.systemUptime
            let pending = lock.withLock { () -> (UUID, JarvisReplayTaskFailurePredicate)? in
                guard !retired, !failed else { return nil }
                let predicate: JarvisReplayTaskFailurePredicate
                switch stage {
                case .pollState: predicate = .directoryAvailable
                case .controlRead, .stimulusPublish: predicate = directory?.failurePredicate ?? .directoryAvailable
                case .receiptConsume: predicate = consumer?.failurePredicate ?? .receiptJSON
                case .stimulusCompletion: predicate = .completionEvidence
                }
                failed = true; consumer?.unavailable()
                return (failureGeneration, predicate)
            }
            timer?.cancel()
            if let (generation, predicate) = pending {
                let observedStage = stage
                scheduleFailure { [weak self] in
                    guard let self else { return }
                    let deliver = self.lock.withLock { () -> Bool in
                        guard !self.retired, self.failed, !self.failureDelivered,
                              self.failureGeneration == generation else { return false }
                        self.failureDelivered = true
                        if error as? JarvisReplayTaskError == .invalidControl {
                            self.deliveredDiagnostic = .init(stage: observedStage, predicate: predicate,
                                generation: generation, observedAt: observedAt,
                                deliveredAt: ProcessInfo.processInfo.systemUptime)
                        }
                        return true
                    }
                    if deliver { self.onFailure?(error) }
                }
            }
        }
    }
    func stop() {
        lock.withLock {
            if !retired { failureGeneration = UUID() }
            retired = true; consumer?.unavailable()
        }
        queue.sync { timer?.cancel(); timer = nil; directory?.close(); directory = nil }
    }
}
#endif
