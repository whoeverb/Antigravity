import Foundation

class SignalService: ObservableObject {
    // REPLACE THIS URL WITH YOUR ACTUAL GITHUB RAW URL
    private let GITHUB_RAW_URL = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/signals.json"
    
    @Published var data: SignalResponse?
    @Published var isLoading = false
    @Published var errorMessage: String?
    
    @MainActor
    func fetchSignals() async {
        isLoading = true
        errorMessage = nil
        
        guard let url = URL(string: GITHUB_RAW_URL) else {
            errorMessage = "Invalid URL"
            isLoading = false
            return
        }
        
        do {
            let (data, _) = try await URLSession.shared.data(from: url)
            let decoder = JSONDecoder()
            decoder.keyDecodingStrategy = .convertFromSnakeCase
            self.data = try decoder.decode(SignalResponse.self, from: data)
        } catch {
            errorMessage = "Failed to load signals: \(error.localizedDescription)"
        }
        
        isLoading = false
    }
}
