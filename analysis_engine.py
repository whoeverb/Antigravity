import datetime
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import time

# ─── Data Fetching ────────────────────────────────────────────────────────────
def load_stock_data(sym, start, end):
    # Retry logic for CI stability
    for attempt in range(3):
        try:
            df = yf.download(sym, start=start, end=end, progress=False)
            if df.empty: 
                if attempt < 2: 
                    time.sleep(2)
                    continue
                return pd.DataFrame(), f"No data for '{sym}'."
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            for c in ['Open','High','Low','Close','Volume']:
                if c not in df.columns: return pd.DataFrame(), f"Missing '{c}'."
            return df.sort_index(), None
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
                continue
            return pd.DataFrame(), str(e)
    return pd.DataFrame(), "Failed to fetch data after 3 attempts."

def _quick_load(sym, days=300):
    end   = datetime.date.today()
    start = end - datetime.timedelta(days=days)
    return load_stock_data(sym, start, end)

# ─── Macro Overlay Engine ──────────────────────────────────────────────────────
def get_macro_overlay():
    tickers = ["SPY", "QQQ", "^VIX", "^TNX"]
    for attempt in range(3):
        try:
            df = yf.download(tickers, period="1y", progress=False)['Close']
            spy = df['SPY'].iloc[-1]
            spy_sma200 = df['SPY'].rolling(200).mean().iloc[-1]
            qqq = df['QQQ'].iloc[-1]
            qqq_sma200 = df['QQQ'].rolling(200).mean().iloc[-1]
            vix = df['^VIX'].iloc[-1]
            tnx = df['^TNX'].iloc[-1]
            
            score = 0
            if spy > spy_sma200: score += 2
            if qqq > qqq_sma200: score += 1
            if vix < 20: score += 2
            elif vix > 30: score -= 3
            if tnx < 4.5: score += 1
            else: score -= 1
            
            if score >= 4: regime, risk = "Bullish", "Low"
            elif score <= 0: regime, risk = "Risk-Off", "High"
            else: regime, risk = "Neutral", "Moderate"
            
            return {"regime": regime, "confidence": min(100, max(0, score * 15 + 50)), "risk_level": risk}
        except Exception:
            if attempt < 2:
                time.sleep(2)
                continue
    return {"regime": "Neutral", "confidence": 50, "risk_level": "Moderate"}

# ─── Market Regime Engine ──────────────────────────────────────────────────────
def calculate_market_regime(df):
    if df.empty or len(df) < 200:
        return "Neutral", 0, "Low"

    close = df['Close'].squeeze().astype(float)
    price = float(close.iloc[-1])
    
    rsi = float(ta.rsi(close, length=14).iloc[-1])
    sma50 = float(ta.sma(close, length=50).iloc[-1])
    sma200 = float(ta.sma(close, length=200).iloc[-1])
    bb = ta.bbands(close, length=20, std=2)
    bbu = float(bb[[c for c in bb.columns if c.startswith('BBU')][0]].iloc[-1])
    bbl = float(bb[[c for c in bb.columns if c.startswith('BBL')][0]].iloc[-1])
    vol = float(close.pct_change().std() * np.sqrt(252) * 100)
    
    score = 0
    if rsi > 78: score += 3
    elif rsi > 70: score += 1
    elif rsi < 35: score -= 3
    elif rsi < 45: score -= 1
    if price > sma200 * 1.12: score += 2
    elif price < sma200: score -= 2
    if sma50 > sma200: score += 2
    else: score -= 2
    if price >= bbu: score += 1
    elif price <= bbl: score -= 1
    if vol > 30: score += 2
    
    if score >= 7: regime = "Overheated"
    elif score >= 4: regime = "Strong Uptrend"
    elif score >= 1: regime = "Bullish"
    elif score <= -7: regime = "Oversold Opportunity"
    elif score <= -4: regime = "Pullback"
    else: regime = "Neutral"
    
    if vol < 25 and abs(score) >= 4: confidence = "High"
    elif vol > 40 or abs(score) < 2: confidence = "Low"
    else: confidence = "Medium"
    
    return regime, score, confidence

# ─── Signal Engines ────────────────────────────────────────────────────────────
def etf_signal(sym, macro):
    df, err = _quick_load(sym)
    if err or df.empty: 
        return {"signal": "DCA", "price": 0.0, "change_pct": 0.0, "regime": "Neutral", "confidence": "Low", "reasons": "Data unavailable."}
    
    regime, _, conf = calculate_market_regime(df)
    close = df['Close'].squeeze().astype(float)
    price = float(close.iloc[-1])
    chg_pct = float((close.iloc[-1]-close.iloc[-2])/close.iloc[-2]*100) if len(close)>1 else 0.0
    score = 0
    reasons = []

    rsi = float(ta.rsi(close, length=14).iloc[-1])
    if rsi > 73: score += 3; reasons.append("Momentum stretched")
    elif rsi < 40: score -= 2; reasons.append("Oversold")

    sma200 = ta.sma(close, length=min(200,len(close)-1))
    if sma200 is not None and not sma200.empty:
        s200 = float(sma200.iloc[-1])
        prem = (price - s200) / s200 * 100
        if prem > 10: score += 3; reasons.append("Above long-term avg")
        elif prem < 0: score -= 4; reasons.append("Below long-term avg")

    sig = "WAIT" if score >= 5 else ("BUY" if score <= -3 else "DCA")
    if macro['regime'] == "Risk-Off":
        if sig == "BUY": sig = "DCA"
        elif sig == "DCA": sig = "WAIT"
        
    return {
        "signal": sig,
        "price": round(price, 2),
        "change_pct": round(chg_pct, 2),
        "regime": regime,
        "confidence": conf,
        "reasons": ", ".join(reasons) if reasons else "Steady"
    }

def stock_signal(sym, macro):
    df, err = _quick_load(sym)
    if err or df.empty: 
        return {"signal": "HOLD", "price": 0.0, "change_pct": 0.0, "regime": "Neutral", "confidence": "Low", "reasons": "Data unavailable."}
    
    regime, _, conf = calculate_market_regime(df)
    close = df['Close'].squeeze().astype(float)
    price = float(close.iloc[-1])
    chg_pct = float((close.iloc[-1]-close.iloc[-2])/close.iloc[-2]*100) if len(close)>1 else 0.0

    bb = ta.bbands(close, length=20, std=2)
    sma50 = ta.sma(close, length=min(50,len(close)-1))
    rsi_s = ta.rsi(close, length=14)

    bbl = bbu = sma50_val = rsi_val = None
    if bb is not None and not bb.empty:
        bbl = float(bb[[c for c in bb.columns if c.startswith('BBL')][0]].iloc[-1])
        bbu = float(bb[[c for c in bb.columns if c.startswith('BBU')][0]].iloc[-1])
    if sma50 is not None and not sma50.empty: sma50_val = float(sma50.iloc[-1])
    if rsi_s is not None: rsi_val = float(rsi_s.iloc[-1])

    sell_reasons = []
    if rsi_val and rsi_val > 78: sell_reasons.append("Momentum exhausted")
    if bbu and price > bbu * 1.08: sell_reasons.append("Stretched volatility")
    if sell_reasons: 
        return {"signal": "SELL", "price": round(price, 2), "change_pct": round(chg_pct, 2), "regime": regime, "confidence": conf, "reasons": ", ".join(sell_reasons)}

    buy_reasons = []
    if bbl and price <= bbl * 1.015: buy_reasons.append("Support found")
    if sma50_val and price <= sma50_val: buy_reasons.append("Below 50d trend")
    if buy_reasons: 
        return {"signal": "BUY", "price": round(price, 2), "change_pct": round(chg_pct, 2), "regime": regime, "confidence": conf, "reasons": ", ".join(buy_reasons)}

    if macro['regime'] == "Risk-Off": 
        return {"signal": "HOLD", "price": round(price, 2), "change_pct": round(chg_pct, 2), "regime": regime, "confidence": conf, "reasons": "Macro caution"}
    
    return {"signal": "HOLD", "price": round(price, 2), "change_pct": round(chg_pct, 2), "regime": regime, "confidence": conf, "reasons": "Steady"}
