import Foundation

class SignalService: ObservableObject {
    // REPLACE THIS URL WITH YOUR ACTUAL GITHUB RAW URL
    private let GITHUB_RAW_URL = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/signals.json"
    
    @Published var data: SignalResponse?
    @Published var isLoading = false
    @Published var errorMessage: String?
    
    @MainActor
    func fetchSignals() async {
        // Prevent redundant calls if already loading
        guard