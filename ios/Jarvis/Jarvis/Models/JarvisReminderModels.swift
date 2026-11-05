import Foundation

struct JarvisReminderSummary: Codable, Identifiable, Equatable {
    let id: String
    let title: String
    let details: String
    let remindAt: String?
    let condition: String?
    let status: String
    let updatedAt: String

    enum CodingKeys: String, CodingKey {
        case id
        case title
        case details
        case remindAt = "remind_at"
        case condition
        case status
        case updatedAt = "updated_at"
    }

    var timingLabel: String {
        if let remindAt {
            return JarvisReminderDateFormatter.display(remindAt)
        }
        return condition ?? "Active"
    }
}

struct JarvisReminderEvent: Codable, Identifiable, Equatable {
    let id: String
    let reminderId: String
    let sessionId: String?
    let eventType: String
    let message: String
    let providerResponseId: String?
    let createdAt: String

    enum CodingKeys: String, CodingKey {
        case id
        case reminderId = "reminder_id"
        case sessionId = "session_id"
        case eventType = "event_type"
        case message
        case providerResponseId = "provider_response_id"
        case createdAt = "created_at"
    }

    var statusLabel: String {
        switch eventType {
        case "dispatched_to_session": return "Speaking"
        case "delivered_to_session": return "Delivered"
        case "missed": return "Waiting for you"
        case "failed": return "Needs attention"
        case "interrupted": return "Interrupted"
        default: return "Prepared"
        }
    }

    var createdLabel: String {
        JarvisReminderDateFormatter.display(createdAt)
    }
}

private enum JarvisReminderDateFormatter {
    static let fallbackISO8601: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter
    }()

    static func display(_ value: String) -> String {
        let date = ISO8601DateFormatter.jarvisFormatter.date(from: value)
            ?? fallbackISO8601.date(from: value)
        guard let date else { return value }
        return date.formatted(date: .abbreviated, time: .shortened)
    }
}

struct ActiveRemindersResponse: Codable, Equatable {
    let reminders: [JarvisReminderSummary]
}

struct ReminderEventsResponse: Codable, Equatable {
    let events: [JarvisReminderEvent]
}
