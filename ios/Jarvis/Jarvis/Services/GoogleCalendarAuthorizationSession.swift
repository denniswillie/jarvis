import AuthenticationServices
import UIKit

@MainActor
final class GoogleCalendarAuthorizationSession: NSObject, ASWebAuthenticationPresentationContextProviding {
    private var activeSession: ASWebAuthenticationSession?

    func authorize(at authorizationURL: URL) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            let session = ASWebAuthenticationSession(
                url: authorizationURL,
                callbackURLScheme: "jarvis"
            ) { [weak self] callbackURL, error in
                self?.activeSession = nil
                if let authenticationError = error as? ASWebAuthenticationSessionError,
                   authenticationError.code == .canceledLogin {
                    continuation.resume(throwing: CancellationError())
                    return
                }
                if let error {
                    continuation.resume(throwing: error)
                    return
                }
                guard callbackURL?.scheme == "jarvis",
                      callbackURL?.host == "oauth",
                      callbackURL?.path == "/google/connected" else {
                    continuation.resume(throwing: JarvisAPIError(
                        code: "google_oauth_callback_invalid",
                        message: "Google Calendar returned an invalid callback."
                    ))
                    return
                }
                continuation.resume()
            }
            session.presentationContextProvider = self
            session.prefersEphemeralWebBrowserSession = false
            activeSession = session
            if !session.start() {
                activeSession = nil
                continuation.resume(throwing: JarvisAPIError(
                    code: "google_oauth_start_failed",
                    message: "Google Calendar authorization could not be opened."
                ))
            }
        }
    }

    func presentationAnchor(for session: ASWebAuthenticationSession) -> ASPresentationAnchor {
        for scene in UIApplication.shared.connectedScenes {
            guard let windowScene = scene as? UIWindowScene else { continue }
            if let keyWindow = windowScene.windows.first(where: \.isKeyWindow) {
                return keyWindow
            }
        }
        return ASPresentationAnchor()
    }
}
