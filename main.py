import datetime
import json
import os
import re
import pandas as pd
import streamlit as st
import yfinance as yf
import pandas_ta as ta
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Research & Projection App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Portfolio Definition ──────────────────────────────────────────────────────
PORTFOLIO_ETFS   = ["SCHG", "SMH", "QTUM", "VOO", "XT", "SCHD", "VUG"]
PORTFOLIO_STOCKS = ["AMZN", "ORCL", "LRN", "NBIS", "NVDA", "ASML", "TSM", "EVGO", "BRK.B", "BSM", "INTA", "KO", "SNDL", "TEM"]

# ─── Persistent P&L Storage ───────────────────────────────────────────────────
PNL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pnl_data.json")

def _load_pnl_from_disk():
    try:
        if os.path.exists(PNL_FILE):
            with open(PNL_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _save_pnl_to_disk(data: dict):
    try:
        with open(PNL_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

# Load from disk into session state on first run
if "pnl_data" not in st.session_state:
    st.session_state["pnl_data"] = _load_pnl_from_disk()
if "pnl_loaded" not in st.session_state:
    st.session_state["pnl_loaded"] = True

# ─── Global CSS (Professional Financial Dashboard Aesthetic) ───────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
* { font-family: 'Inter', sans-serif !important; }

/* Main app background */
.stApp { background-color: #0F172A; }

/* High Contrast Text */
h1, h2, h3, h4, h5, h6 { color: #F8FAFC !important; font-weight: 700 !important; }
p, li, div { color: #E2E8F0 !important; }

/* Metric components */
div[data-testid="stMetricValue"] { font-size:1.8rem !important; font-weight:800 !important; color:#FFFFFF !important; }
div[data-testid="stMetricLabel"] { font-size:0.9rem !important; color:#94A3B8 !important; font-weight:600 !important; text-transform:uppercase; }
[data-testid="stMetric"] {
    background: rgba(30, 41, 59, 0.8);
    border: 1px solid rgba(71, 85, 105, 0.5);
    border-radius: 12px; padding:20px;
}

/* Inputs - High Contrast */
.stNumberInput > label { color: #F8FAFC !important; font-weight: 600 !important; font-size: 0.9rem !important; }
.stNumberInput > div > div > input {
    background-color: #1E293B !important;
    border: 2px solid #475569 !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}

/* Pulse Animation */
@keyframes pulse {
    0%   { transform:scale(0.95); box-shadow:0 0 0 0 rgba(16,185,129,0.7); }
    70%  { transform:scale(1);    box-shadow:0 0 0 6px rgba(16,185,129,0); }
    100% { transform:scale(0.95); box-shadow:0 0 0 0 rgba(16,185,129,0); }
}
.status-pulse {
    width:10px; height:10px; background-color:#10b981;
    border-radius:50%; display:inline-block;
    animation: pulse 2s infinite;
}

/* Section Panels */
.etf-panel {
    background: rgba(99, 102, 241, 0.05);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 24px;
}
.stock-panel {
    background: rgba(245, 158, 11, 0.05);
    border: 1px solid rgba(245, 158, 11, 0.2);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 24px;
}
.regime-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    margin-right: 4px;
}
</style>
""", unsafe_allow_html=True)

# ─── Inline Style Constants ────────────────────────────────────────────────────
SIG_STYLES = {
    "BUY":  {
        "card_bg":    "linear-gradient(135deg,rgba(16,185,129,0.15),rgba(5,150,105,0.2))",
        "card_bdr":   "1px solid rgba(16,185,129,0.4)",
        "pill_bg":    "rgba(16,185,129,0.2)",
        "pill_color": "#34D399",
        "pill_bdr":   "1px solid rgba(16,185,129,0.4)",
        "pill_label": "🟢 BUY",
    },
    "DCA":  {
        "card_bg":    "linear-gradient(135deg,rgba(99,102,241,0.15),rgba(79,70,229,0.2))",
        "card_bdr":   "1px solid rgba(99,102,241,0.4)",
        "pill_bg":    "rgba(99,102,241,0.2)",
        "pill_color": "#818CF8",
        "pill_bdr":   "1px solid rgba(99,102,241,0.4)",
        "pill_label": "💙 DCA",
    },
    "WAIT": {
        "card_bg":    "linear-gradient(135deg,rgba(245,158,11,0.15),rgba(234,179,8,0.2))",
        "card_bdr":   "1px solid rgba(245,158,11,0.4)",
        "pill_bg":    "rgba(245,158,11,0.2)",
        "pill_color": "#FBBF24",
        "pill_bdr":   "1px solid rgba(245,158,11,0.4)",
        "pill_label": "🟡 WAIT",
    },
    "HOLD": {
        "card_bg":    "linear-gradient(135deg,rgba(71,85,105,0.15),rgba(51,65,85,0.2))",
        "card_bdr":   "1px solid rgba(100,116,139,0.3)",
        "pill_bg":    "rgba(100,116,139,0.2)",
        "pill_color": "#94A3B8",
        "pill_bdr":   "1px solid rgba(100,116,139,0.3)",
        "pill_label": "⚪ HOLD",
    },
    "SELL": {
        "card_bg":    "linear-gradient(135deg,rgba(239,68,68,0.15),rgba(220,38,38,0.2))",
        "card_bdr":   "1px solid rgba(239,68,68,0.4)",
        "pill_bg":    "rgba(239,68,68,0.2)",
        "pill_color": "#F87171",
        "pill_bdr":   "1px solid rgba(239,68,68,0.4)",
        "pill_label": "🔴 SELL",
    },
}

# ─── Helpers ───────────────────────────────────────────────────────────────────
def _fmt_age(pub_ts):
    try:
        if isinstance(pub_ts, (int, float)):
            pub_dt = datetime.datetime.fromtimestamp(pub_ts, tz=datetime.timezone.utc)
        else:
            pub_dt = pub_ts
        if pub_dt.tzinfo is None:
            pub_dt = pub_dt.replace(tzinfo=datetime.timezone.utc)
        delta  = datetime.datetime.now(datetime.timezone.utc) - pub_dt
        hours  = int(delta.total_seconds() // 3600)
        if hours < 1:  return "just now"
        if hours < 24: return f"{hours}h ago"
        return f"{hours // 24}d ago"
    except Exception:
        return ""

def _days_until(d):
    return (d - datetime.date.today()).days

def _card_wrap(inner_html, sig):
    s = SIG_STYLES.get(sig, SIG_STYLES["HOLD"])
    # Fixed height container (approx 210px to accommodate regime badge)
    return (
        f'<div style="background:{s["card_bg"]};border:{s["card_bdr"]};'
        f'border-radius:12px;padding:16px;margin-bottom:12px;height:210px;display:flex;flex-direction:column;">'
        f'{inner_html}</div>'
    )

def _pill(sig):
    s = SIG_STYLES.get(sig, SIG_STYLES["HOLD"])
    return (f'<span style="display:inline-block;padding:2px 10px;border-radius:999px;'
            f'font-size:0.7rem;font-weight:700;letter-spacing:0.05em;text-transform:uppercase;'
            f'background:{s["pill_bg"]};color:{s["pill_color"]};border:{s["pill_bdr"]};">'
            f'{s["pill_label"]}</span>')

def _type_badge(is_etf):
    if is_etf:
        return ('<span style="font-size:0.65rem;color:#818CF8;background:rgba(99,102,241,0.1);'
                'border:1px solid rgba(99,102,241,0.2);padding:1px 6px;border-radius:4px;margin-left:6px;">ETF</span>')
    return ('<span style="font-size:0.65rem;color:#C084FC;background:rgba(192,132,252,0.1);'
            'border:1px solid rgba(192,132,252,0.2);padding:1px 6px;border-radius:4px;margin-left:6px;">STOCK</span>')

# ─── Market Regime Engine ──────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def calculate_market_regime(df):
    if df.empty or len(df) < 200:
        return "Neutral", 0, "Low", "Insufficient data for analysis.", {}

    close = df['Close'].squeeze().astype(float)
    price = float(close.iloc[-1])
    
    # Indicators
    rsi = float(ta.rsi(close, length=14).iloc[-1])
    sma50 = float(ta.sma(close, length=50).iloc[-1])
    sma200 = float(ta.sma(close, length=200).iloc[-1])
    bb = ta.bbands(close, length=20, std=2)
    bbu = float(bb[[c for c in bb.columns if c.startswith('BBU')][0]].iloc[-1])
    bbl = float(bb[[c for c in bb.columns if c.startswith('BBL')][0]].iloc[-1])
    vol = float(close.pct_change().std() * np.sqrt(252) * 100)
    
    # Trend Slope (20-day)
    y = close.iloc[-20:].values
    x = np.arange(len(y))
    slope = np.polyfit(x, y, 1)[0]
    
    # Scoring
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
    
    if vol > 30: score += 2 # High risk/volatility
    
    # Regime Mapping
    if score >= 7: regime, color = "Overheated", "#ef4444"
    elif score >= 4: regime, color = "Strong Uptrend", "#10b981"
    elif score >= 1: regime, color = "Bullish", "#10b981"
    elif score <= -7: regime, color = "Oversold Opportunity", "#34d399"
    elif score <= -4: regime, color = "Pullback", "#f59e0b"
    else: regime, color = "Neutral", "#94a3b8"
    
    # Confidence
    if vol < 25 and abs(score) >= 4: confidence = "High"
    elif vol > 40 or abs(score) < 2: confidence = "Low"
    else: confidence = "Medium"
    
    # Explanation
    explanation = f"The market is currently in a {regime} state. "
    if score > 0: explanation += "Momentum is positive, but monitor for overextension."
    else: explanation += "Conditions suggest caution or potential accumulation opportunities."
    
    return regime, score, confidence, explanation, {"color": color}

# ─── Cached Loaders ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_stock_data(sym, start, end):
    try:
        df = yf.download(sym, start=start, end=end, progress=False)
        if df.empty: return pd.DataFrame(), f"No data for '{sym}'."
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        for c in ['Open','High','Low','Close','Volume']:
            if c not in df.columns: return pd.DataFrame(), f"Missing '{c}'."
        return df.sort_index(), None
    except Exception as e:
        return pd.DataFrame(), str(e)

@st.cache_data(show_spinner=False)
def fetch_company_info(sym):
    try:
        info = yf.Ticker(sym).info
        if not info or not isinstance(info,dict): return None, f"No profile for '{sym}'."
        return info, None
    except Exception as e:
        return None, str(e)

@st.cache_data(show_spinner=False, ttl=900)
def fetch_news(sym):
    try:
        news = yf.Ticker(sym).news or []
        return [{'title':i.get('title',''),'link':i.get('link','#'),'ts':i.get('providerPublishTime')}
                for i in news[:5] if i.get('title')],