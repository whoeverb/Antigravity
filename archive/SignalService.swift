import Foundation

class SignalService: ObservableObject {
    private let GITHUB_RAW_URL = "https://raw.githubusercontent.com/whoeverb/Antigravity/main/signals.json"
    
    @Published var data: SignalResponse?
    @Published var isLoading = false
    @Published var errorMessage: String?
    
    @MainActor
    func fetchSignals() async {
        // Prevent redundant calls if already loading
        guard !isLoading else { return }
        
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
            // No keyDecodingStrategy needed as we use CodingKeys in models
            self.data = try decoder.decode(SignalResponse.self, from: data)
        } catch {
            errorMessage = "Failed to load signals: \(error.localizedDescription)"
        }
        
        isLoading = false
    }
}
