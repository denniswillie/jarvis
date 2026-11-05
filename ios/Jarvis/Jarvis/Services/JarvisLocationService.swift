import CoreLocation
import Foundation

@MainActor
final class JarvisLocationService: NSObject, ObservableObject, @preconcurrency CLLocationManagerDelegate {
    @Published private(set) var authorizationStatus: CLAuthorizationStatus
    @Published private(set) var latestPlaceLabel: String?

    private let manager = CLLocationManager()
    private let geocoder = CLGeocoder()
    private var api: JarvisAPIClient?
    private var lastLocation: CLLocation?
    private var lastGeocodedLocation: CLLocation?
    private var lastGeocodedAt: Date?
    private var cachedPlaceName: String?
    private var cachedLocality: String?
    private var cachedCountry: String?
    private var pushTask: Task<Void, Never>?

    override init() {
        self.authorizationStatus = manager.authorizationStatus
        super.init()
        manager.delegate = self
        manager.desiredAccuracy = kCLLocationAccuracyHundredMeters
        manager.distanceFilter = 75
        manager.pausesLocationUpdatesAutomatically = true
    }

    func activate(api: JarvisAPIClient) {
        self.api = api
        authorizationStatus = manager.authorizationStatus
        switch authorizationStatus {
        case .notDetermined:
            manager.requestWhenInUseAuthorization()
        case .authorizedWhenInUse, .authorizedAlways:
            manager.startUpdatingLocation()
            manager.requestLocation()
            if let lastLocation {
                schedulePush(for: lastLocation, forceGeocode: false)
            }
        case .denied, .restricted:
            break
        @unknown default:
            break
        }
    }

    func syncNow() {
        guard api != nil else { return }
        authorizationStatus = manager.authorizationStatus
        switch authorizationStatus {
        case .notDetermined:
            manager.requestWhenInUseAuthorization()
        case .authorizedWhenInUse, .authorizedAlways:
            manager.startUpdatingLocation()
            if let lastLocation {
                schedulePush(for: lastLocation, forceGeocode: false)
            } else {
                manager.requestLocation()
            }
        default:
            break
        }
    }

    func deactivate() {
        pushTask?.cancel()
        pushTask = nil
        geocoder.cancelGeocode()
        manager.stopUpdatingLocation()
        api = nil
    }

    func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        authorizationStatus = manager.authorizationStatus
        guard api != nil else { return }
        if authorizationStatus == .authorizedWhenInUse || authorizationStatus == .authorizedAlways {
            manager.startUpdatingLocation()
            manager.requestLocation()
        } else {
            manager.stopUpdatingLocation()
        }
    }

    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard let newest = locations.last(where: { $0.horizontalAccuracy >= 0 }) else { return }
        lastLocation = newest
        schedulePush(for: newest, forceGeocode: false)
    }

    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        // Transient CLError.locationUnknown is normal indoors; keep monitoring quietly.
    }

    private func schedulePush(for location: CLLocation, forceGeocode: Bool) {
        guard let api else { return }
        pushTask?.cancel()
        pushTask = Task { [weak self] in
            guard let self, !Task.isCancelled else { return }
            await self.resolveGeocodeIfNeeded(for: location, force: forceGeocode)
            guard !Task.isCancelled else { return }
            let formatter = ISO8601DateFormatter()
            formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
            let payload = JarvisLiveLocationUpdate(
                latitude: location.coordinate.latitude,
                longitude: location.coordinate.longitude,
                horizontalAccuracyMeters: location.horizontalAccuracy >= 0 ? location.horizontalAccuracy : nil,
                placeName: self.cachedPlaceName,
                locality: self.cachedLocality,
                country: self.cachedCountry,
                timezone: TimeZone.current.identifier,
                capturedAt: formatter.string(from: location.timestamp)
            )
            try? await api.updateLiveLocation(payload)
        }
    }

    private func resolveGeocodeIfNeeded(for location: CLLocation, force: Bool) async {
        let now = Date()
        let movedEnough: Bool
        if let lastGeocodedLocation {
            movedEnough = location.distance(from: lastGeocodedLocation) >= 75
        } else {
            movedEnough = true
        }
        let staleGeocode = lastGeocodedAt.map { now.timeIntervalSince($0) >= 300 } ?? true
        guard force || movedEnough || staleGeocode else { return }

        do {
            let placemarks = try await geocoder.reverseGeocodeLocation(location)
            guard let first = placemarks.first else { return }
            let area = [first.subLocality, first.locality].compactMap { $0?.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
            let deduplicatedArea = area.reduce(into: [String]()) { acc, part in
                if !acc.contains(part) { acc.append(part) }
            }
            if !deduplicatedArea.isEmpty {
                cachedPlaceName = deduplicatedArea.joined(separator: ", ")
            } else if let name = first.name, !name.isEmpty {
                cachedPlaceName = name
            }
            cachedLocality = first.locality
            cachedCountry = first.country
            latestPlaceLabel = cachedPlaceName ?? cachedLocality
            lastGeocodedLocation = location
            lastGeocodedAt = now
        } catch {
            // Keep coordinates even if offline/throttled reverse-geocode fails.
        }
    }
}
