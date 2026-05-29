import SwiftUI

struct ContentView: View {
    @StateObject private var service = SignalService()
    
    var body: some View {
        TabView {
            SignalListView(service: service)
                .tabItem {
                    Label("Signals", systemImage: "chart.bar.fill")
                }
            
            CandidateListView(service: service)
                .tabItem {
                    Label("Top Picks", systemImage: "star.fill")
                }
        }
        .task {
            await service.fetchSignals()
        }
    }
}

struct SignalListView: View {
    @ObservedObject var service: SignalService
    
    var body: some View {
        NavigationView {
            List {
                if let data = service.data {
                    Section(header: Text("Last Updated: \(data.generatedAt)")) {
                        ForEach(data.signals.keys.sorted(), id: \.self) { ticker in
                            if let info = data.signals[ticker] {
                                HStack {
                                    VStack(alignment: .leading) {
                                        Text(ticker).font(.headline)
                                        Text(info.type).font(.caption).foregroundColor(.secondary)
                                    }
                                    Spacer()
                                    VStack(alignment: .trailing) {
                                        Text(String(format: "$%.2f", info.price ?? 0.0))
                                        Text(String(format: "%.2f%%", info.changePct ?? 0.0))
                                            .foregroundColor((info.changePct ?? 0) >= 0 ? .green : .red)
                                    }
                                    SignalPill(signal: info.signal)
                                }
                            }
                        }
                    }
                }
            }
            .navigationTitle("Market Signals")
            .overlay { if service.isLoading { ProgressView() } }
            .refreshable { await service.fetchSignals() }
        }
    }
}

struct CandidateListView: View {
    @ObservedObject var service: SignalService
    
    var body: some View {
        NavigationView {
            List(service.data?.topCandidates ?? []) { candidate in
                HStack {
                    Text(candidate.ticker).font(.headline)
                    Spacer()
                    Text("Score: \(candidate.score)").font(.subheadline)
                    SignalPill(signal: candidate.signal)
                }
            }
            .navigationTitle("Top Candidates")
            .overlay { if service.isLoading { ProgressView() } }
        }
    }
}

struct SignalPill: View {
    let signal: String
    
    var color: Color {
        switch signal {
        case "BUY": return .green
        case "SELL": return .red
        case "DCA": return .blue
        case "WAIT": return .orange
        default: return .gray
        }
    }
    
    var body: some View {
        Text(signal)
            .font(.caption2.bold())
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(color.opacity(0.2))
            .foregroundColor(color)
            .cornerRadius(8)
    }
}
