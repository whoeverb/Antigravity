import 'dart:convert';
import 'package:http/http.dart' as http;
import '../models/signal_model.dart';

class SignalService {
  // Updated to point to the new location in signal_engine/
  static const String _url = "https://raw.githubusercontent.com/whoeverb/Antigravity/main/signals.json";

  Future<SignalResponse> fetchSignals() async {
    final response = await http.get(Uri.parse(_url));
    
    if (response.statusCode == 200) {
      return SignalResponse.fromJson(json.decode(response.body));
    } else {
      throw Exception('Failed to load signals: Status Code ${response.statusCode}');
    }
  }
}
