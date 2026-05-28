import json
import datetime
import analysis_engine

PORTFOLIO_ETFS   = ["SCHG", "SMH", "QTUM", "VOO", "XT", "SCHD", "VUG"]
PORTFOLIO_STOCKS = ["AMZN", "ORCL", "LRN", "NBIS", "NVDA", "ASML", "TSM", "EVGO", "BRK.B", "BSM", "INTA", "KO", "SNDL", "TEM"]

def run():
    # 1. Get Macro Overlay ONCE
    macro = analysis_engine.get_macro_overlay()
    results = {}
    
    # 2. Process ETFs
    for sym in PORTFOLIO_ETFS:
        sig, price, chg, reason = analysis_engine.etf_signal(sym, macro)
        df, _ = analysis_engine._quick_load(sym)
        regime, _, conf = analysis_engine.calculate_market_regime(df)
        
        results[sym] = {
            "type": "ETF",
            "signal": sig,
            "price": round(price, 2) if price else None,
            "change_pct": round(chg, 2) if chg else None,
            "regime": regime,
            "confidence": conf,
            "reasons": reason
        }

    # 3. Process Stocks
    for sym in PORTFOLIO_STOCKS:
        sig, price, chg, reason = analysis_engine.stock_signal(sym, macro)
        df, _ = analysis_engine._quick_load(sym)
        regime, _, conf = analysis_engine.calculate_market_regime(df)
        
        results[sym] = {
            "type": "STOCK",
            "signal": sig,
            "price": round(price, 2) if price else None,
            "change_pct": round(chg, 2) if chg else None,
            "regime": regime,
            "confidence": conf,
            "reasons": reason
        }

    # 4. Generate Top Candidates
    score_map = {"BUY": 3, "DCA": 2, "HOLD": 1, "WAIT": 0, "SELL": -1}
    candidates = []
    for sym, data in results.items():
        score = score_map.get(data["signal"], 0)
        candidates.append({"ticker": sym, "score": score, "signal": data["signal"]})
    
    candidates.sort(key=lambda x: x["score"], reverse=True)
    
    # 5. Output JSON
    output = {
        "generated_at": str(datetime.datetime.now()),
        "signals": results,
        "top_candidates": candidates[:5]
    }
    
    with open("signals.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print("Successfully generated signals.json")

if __name__ == "__main__":
    run()
