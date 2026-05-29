import 'dart:convert';

class SignalResponse {
  final String generatedAt;
  final Map<String, SignalData> signals;
  final List<Candidate> topCandidates;

  SignalResponse({
    required this.generatedAt,
    required this.signals,
    required this.topCandidates,
  });

  factory SignalResponse.fromJson(Map<String, dynamic> json) {
    return SignalResponse(
      generatedAt: json['generated_at'] ?? '',
      signals: (json['signals'] as Map<String, dynamic>).map(
        (key, value) => MapEntry(key, SignalData.fromJson(value)),
      ),
      topCandidates: (json['top_candidates'] as List)
          .map((i) => Candidate.fromJson(i))
          .toList(),
    );
  }
}

class SignalData {
  final String type;
  final String signal;
  final double? price;
  final double? changePct;
  final String regime;
  final String confidence;
  final String reasons;

  SignalData({
    required this.type,
    required this.signal,
    this.price,
    this.changePct,
    required this.regime,
    required this.confidence,
    required this.reasons,
  });

  factory SignalData.fromJson(Map<String, dynamic> json) {
    return SignalData(
      type: json['type'] ?? '',
      signal: json['signal'] ?? 'HOLD',
      price: (json['price'] as num?)?.toDouble(),
      changePct: (json['change_pct'] as num?)?.toDouble(),
      regime: json['regime'] ?? '',
      confidence: json['confidence'] ?? '',
      reasons: json['reasons'] ?? '',
    );
  }
}

class Candidate {
  final String ticker;
  final int score;
  final String signal;

  Candidate({required this.ticker, required this.score, required this.signal});

  factory Candidate.fromJson(Map<String, dynamic> json) {
    return Candidate(
      ticker: json['ticker'] ?? '',
      score: json['score'] ?? 0,
      signal: json['signal'] ?? '',
    );
  }
}
