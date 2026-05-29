import 'package:flutter/material.dart';
import '../models/signal_model.dart';
import '../services/signal_service.dart';

class SignalsScreen extends StatefulWidget {
  const SignalsScreen({super.key});

  @override
  State<SignalsScreen> createState() => _SignalsScreenState();
}

class _SignalsScreenState extends State<SignalsScreen> {
  final SignalService _service = SignalService();
  late Future<SignalResponse> _futureSignals;

  @override
  void initState() {
    super.initState();
    _futureSignals = _service.fetchSignals();
  }

  Future<void> _refresh() async {
    setState(() {
      _futureSignals = _service.fetchSignals();
    });
  }

  Color _getSignalColor(String signal) {
    switch (signal) {
      case 'BUY': return Colors.green;
      case 'SELL': return Colors.red;
      case 'DCA': return Colors.blue;
      case 'WAIT': return Colors.orange;
      default: return Colors.grey;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Market Signals")),
      body: RefreshIndicator(
        onRefresh: _refresh,
        child: FutureBuilder<SignalResponse>(
          future: _futureSignals,
          builder: (context, snapshot) {
            if (snapshot.connectionState == ConnectionState.waiting) {
              return const Center(child: CircularProgressIndicator());
            } else if (snapshot.hasError) {
              return Center(child: Text("Error: ${snapshot.error}"));
            }

            final data = snapshot.data!;
            final tickers = data.signals.keys.toList()..sort();

            return ListView.builder(
              itemCount: tickers.length,
              itemBuilder: (context, index) {
                final ticker = tickers[index];
                final info = data.signals[ticker]!;
                
                return Card(
                  margin: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                  child: ListTile(
                    title: Text(ticker, style: const TextStyle(fontWeight: FontWeight.bold)),
                    subtitle: Text("${info.type} • ${info.regime}"),
                    trailing: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      crossAxisAlignment: CrossAxisAlignment.end,
                      children: [
                        Text("\$${info.price?.toStringAsFixed(2) ?? 'N/A'}"),
                        Text("${info.changePct?.toStringAsFixed(2) ?? '0.00'}%", 
                             style: TextStyle(color: (info.changePct ?? 0) >= 0 ? Colors.green : Colors.red)),
                      ],
                    ),
                    leading: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                      decoration: BoxDecoration(
                        color: _getSignalColor(info.signal).withOpacity(0.2),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: Text(info.signal, style: TextStyle(color: _getSignalColor(info.signal), fontWeight: FontWeight.bold)),
                    ),
                    // Future: Add onTap to navigate to a DetailScreen with charts
                  ),
                );
              },
            );
          },
        ),
      ),
    );
  }
}
