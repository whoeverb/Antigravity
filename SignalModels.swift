import Foundation

struct SignalResponse: Codable {
    let generatedAt: String
    let signals: [String: SignalData]
    let topCandidates: [Candidate]
}

struct SignalData: Codable {
    let type: String
    let signal: String
    let price: Double?
    let changePct: Double?
    let regime: String
    let confidence: String
    let reasons: String
}

struct Candidate: Codable, Identifiable {
    var id: String { ticker }
    let ticker: String
    let score: Int
    let signal: String
}
