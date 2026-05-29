import json
import datetime
import math
import analysis_engine
import sys

PORTFOLIO_ETFS   = ["SCHG", "SMH", "QTUM", "VOO", "XT", "SCHD", "VUG"]
PORTFOLIO_STOCKS = ["AMZN", "ORCL", "LRN", "NBIS", "NVDA", "ASML", "TSM", "EVGO", "BRK.B", "BSM", "INTA", "KO", "SNDL", "TEM"]
OUTPUT_FILE      = "signals.json"

def clean_data(obj):
    """Recursively replace NaN with None for JSON serialization."""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: clean_data(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_data(v) for v in obj]
    return obj

def run():
    try:
        print("Starting signal generation...")
        macro = analysis_engine.get_macro_overlay()
        results = {}
        
        # Process ETFs
        for sym in PORTFOLIO_ETFS:
            data = analysis_engine.etf_signal(sym, macro)
            results[sym] = {"type": "ETF", **data}

        # Process Stocks
        for sym in PORTFOLIO_STOCKS:
            data = analysis_engine.stock_signal(sym, macro)
            results[sym] = {"type": "STOCK", **data}

        # Generate Top Candidates
        score_map = {"BUY": 3, "DCA": 2, "HOLD": 1, "WAIT": 0, "SELL": -1}
        candidates = []
        for sym, data in results.items():
            score = score_map.get(data["signal"], 0)
            candidates.append({"ticker": sym, "score": score, "signal": data["signal"]})
        
        candidates.sort(key=lambda x: x["score"], reverse=True)
        
        output = {
            "generated_at": str(datetime.datetime.now()),
            "signals": clean_data(results),
            "top_candidates": clean_data(candidates[:5])
        }
        
        with open(OUTPUT_FILE, "w") as f:
            json.dump(output, f, indent=2)
        
        print(f"Successfully generated {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"Error generating signals: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run()
