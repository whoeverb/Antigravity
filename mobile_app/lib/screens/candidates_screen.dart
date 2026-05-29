import 'package:flutter/material.dart';
import '../models/signal_model.dart';
import '../services/signal_service.dart';

class CandidatesScreen extends StatefulWidget {
  const CandidatesScreen({super.key});

  @override
  State<CandidatesScreen> createState() => _CandidatesScreenState();
}

class _CandidatesScreenState extends State<CandidatesScreen> {
  final SignalService _service = SignalService();
  late Future<SignalResponse> _futureSignals;

  @override
  void initState() {
    super.initState();
    _futureSignals = _service.fetchSignals();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Top Candidates")),
      body: FutureBuilder<SignalResponse>(
        future: _futureSignals,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator());
          } else if (snapshot.hasError) {
            return Center(child: Text("Error: ${snapshot.error}"));
          }

          final candidates = snapshot.data!.topCandidates;

          return ListView.builder(
            itemCount: candidates.length,
            itemBuilder: (context, index) {
              final item = candidates[index];
              return Card(
                margin: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                child: ListTile(
                  leading: CircleAvatar(child: Text("${index + 1}")),
                  title: Text(item.ticker, style: const TextStyle(fontWeight: FontWeight.bold)),
                  subtitle: Text("Score: ${item.score}"),
                  trailing: Chip(label: Text(item.signal)),
                ),
              );
            },
          );
        },
      ),
    );
  }
}
