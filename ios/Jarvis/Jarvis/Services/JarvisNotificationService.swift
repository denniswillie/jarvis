import UIKit
import UserNotifications

@MainActor
final class JarvisNotificationService: NSObject, JarvisPushPlatform, UNUserNotificationCenterDelegate {
    static let shared = JarvisNotificationService()
    var onToken: ((String) -> Void)?
    var onRegistrationFailure: (() -> Void)?
    var onOpenInbox: (() -> Void)?
    private(set) var pendingLink: JarvisNotificationLink?
    var presentationEnabled = false

    var environment: String? {
        switch Bundle.main.object(forInfoDictionaryKey: "JarvisAPNsEnvironment") as? String {
        case "development": return "sandbox"
        case "production": return "production"
        default: return nil
        }
    }
    var topic: String { Bundle.main.bundleIdentifier ?? "" }

    func start() {
        guard environment != nil else { return }
        UNUserNotificationCenter.current().delegate = self
    }

    func permission() async -> JarvisPushPermission {
        let settings = await UNUserNotificationCenter.current().notificationSettings()
        switch settings.authorizationStatus {
        case .authorized: return .authorized
        case .provisional: return .provisional
        case .ephemeral: return .ephemeral
        case .denied: return .denied
        case .notDetermined: return .notDetermined
        @unknown default: return .denied
        }
    }

    func requestPermission() async throws -> JarvisPushPermission {
        guard environment != nil else { return await permission() }
        if await permission() == .notDetermined {
            _ = try await UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound])
        }
        return await permission()
    }

    func register() {
        guard environment != nil else { return }
        UIApplication.shared.registerForRemoteNotifications()
    }
    func unregister() { UIApplication.shared.unregisterForRemoteNotifications() }

    func registered(_ data: Data) {
        onToken?(data.map { String(format: "%02x", $0) }.joined())
    }

    func takePendingLink() -> JarvisNotificationLink? {
        defer { pendingLink = nil }
        return pendingLink
    }

    nonisolated func userNotificationCenter(
        _ center: UNUserNotificationCenter, didReceive response: UNNotificationResponse,
        withCompletionHandler completionHandler: @escaping () -> Void
    ) {
        let link = response.actionIdentifier == UNNotificationDefaultActionIdentifier
            ? JarvisNotificationLink.parse(response.notification.request.content.userInfo) : nil
        Task { @MainActor in
            if let link {
                self.pendingLink = link
                self.onOpenInbox?()
            }
            completionHandler()
        }
    }

    nonisolated func userNotificationCenter(
        _ center: UNUserNotificationCenter, willPresent notification: UNNotification,
        withCompletionHandler completionHandler: @escaping (UNNotificationPresentationOptions) -> Void
    ) {
        let valid = JarvisNotificationLink.parse(notification.request.content.userInfo) != nil
        Task { @MainActor in
            // Never add notification sound over the foreground voice/TTS channel.
            completionHandler(valid && self.presentationEnabled ? [.banner, .list] : [])
        }
    }
}

@MainActor
final class JarvisNotificationAppDelegate: NSObject, UIApplicationDelegate {
    func application(_ application: UIApplication,
                     didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]? = nil) -> Bool {
        JarvisNotificationService.shared.start()
        return true
    }
    func application(_ application: UIApplication, didRegisterForRemoteNotificationsWithDeviceToken deviceToken: Data) {
        JarvisNotificationService.shared.registered(deviceToken)
    }
    func application(_ application: UIApplication, didFailToRegisterForRemoteNotificationsWithError error: Error) {
        JarvisNotificationService.shared.onRegistrationFailure?()
    }
}
