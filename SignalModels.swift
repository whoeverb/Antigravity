import Foundation

struct SignalResponse: Codable {
    let generatedAt: String
    let signals: [String: SignalData]
    let topCandidates: [Candidate]

    enum CodingKeys: String, CodingKey {
        case generatedAt = "generated_at"
        case signals
        case topCandidates = "top_candidates"
    }
}

struct SignalData: Codable {
    let type: String
    let signal: String
    let price: Double?
    let changePct: Double?
    let regime: String
    let confidence: String
    let reasons: String

    enum CodingKeys: String, CodingKey {
        case type, signal, price, regime, confidence, reasons
        case changePct = "change_pct"
    }
}

struct Candidate: Codable, Identifiable {
    var id: String { ticker }
    let ticker: String
    let score: Int
    let signal: String
}
