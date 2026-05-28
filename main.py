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
p, li, span, div { color: #E2E8F0 !important; }

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
    # Fixed height container (approx 180px)
    return (
        f'<div style="background:{s["card_bg"]};border:{s["card_bdr"]};'
        f'border-radius:12px;padding:16px;margin-bottom:12px;height:180px;display:flex;flex-direction:column;">'
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
                for i in news[:5] if i.get('title')], None
    except Exception as e:
        return [], str(e)

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_earnings_date(sym):
    try:
        t   = yf.Ticker(sym)
        cal = t.calendar
        if cal is None: return None, None
        if isinstance(cal, dict):
            raw = cal.get('Earnings Date') or cal.get('earningsDate')
            if raw:
                if isinstance(raw,(list,tuple)): raw = raw[0]
                return (raw.date() if hasattr(raw,'date') else pd.Timestamp(raw).date()), None
        elif isinstance(cal, pd.DataFrame):
            if 'Earnings Date' in cal.index:
                return pd.Timestamp(cal.loc['Earnings Date'].values[0]).date(), None
        ed = t.earnings_dates
        if ed is not None and not ed.empty:
            fut = ed[ed.index > pd.Timestamp.now()]
            if not fut.empty: return fut.index[-1].date(), None
        return None, None
    except Exception as e:
        return None, str(e)

@st.cache_data(show_spinner=False)
def generate_statistical_forecast(series_vals, series_idx, horizon=14):
    try:
        s = pd.Series(list(series_vals), index=list(series_idx)).squeeze().astype(float)
        if len(s) < 10: return pd.DataFrame(), "Need ≥10 days."
        fitted = ExponentialSmoothing(s, trend="add", seasonal=None,
                                      initialization_method="estimated").fit()
        fcast  = fitted.forecast(steps=horizon)
        dates  = pd.date_range(start=s.index[-1]+pd.Timedelta(days=1), periods=horizon, freq='D')
        std    = fitted.resid.std() or 0.0
        lo = [fcast.values[h-1] - 1.28*std*np.sqrt(h) for h in range(1,horizon+1)]
        hi = [fcast.values[h-1] + 1.28*std*np.sqrt(h) for h in range(1,horizon+1)]
        return pd.DataFrame({'ds':dates,'yhat':fcast.values,'yhat_lower':lo,'yhat_upper':hi}), None
    except Exception as e:
        return pd.DataFrame(), str(e)

# ─── Signal Engines ────────────────────────────────────────────────────────────
def _quick_load(sym, days=300):
    end   = datetime.date.today()
    start = end - datetime.timedelta(days=days)
    return load_stock_data(sym, start, end)

def etf_signal(sym):
    df, err = _quick_load(sym)
    if err or df.empty:
        return "DCA", "Data unavailable.", None, None
    close   = df['Close'].squeeze().astype(float)
    price   = float(close.iloc[-1])
    chg_pct = float((close.iloc[-1]-close.iloc[-2])/close.iloc[-2]*100) if len(close)>1 else 0.0
    score   = 0
    reasons = []

    rsi_s = ta.rsi(close, length=14)
    rsi   = float(rsi_s.iloc[-1]) if rsi_s is not None and not rsi_s.empty else 50.0
    if rsi > 73:
        score += 3; reasons.append("Price is looking overextended (RSI high).")
    elif rsi < 40:
        score -= 2; reasons.append("Price is looking cheap (RSI low).")

    sma200 = ta.sma(close, length=min(200,len(close)-1))
    if sma200 is not None and not sma200.empty:
        s200 = float(sma200.iloc[-1])
        prem = (price - s200) / s200 * 100
        if prem > 10:
            score += 3; reasons.append("Price is significantly above its long-term average.")
        elif prem < 0:
            score -= 4; reasons.append("Price is trading below its long-term average.")

    bb = ta.bbands(close, length=20, std=2)
    if bb is not None and not bb.empty:
        bbu = float(bb[[c for c in bb.columns if c.startswith('BBU')][0]].iloc[-1])
        bbl = float(bb[[c for c in bb.columns if c.startswith('BBL')][0]].iloc[-1])
        if price >= bbu * 0.98:
            score += 2; reasons.append("Price is hitting the upper range of normal volatility.")
        elif price <= bbl * 1.03:
            score -= 3; reasons.append("Price is hitting the lower range of normal volatility.")

    sig = "WAIT" if score >= 5 else ("BUY" if score <= -3 else "DCA")
    reason_str = "  ·  ".join(reasons) if reasons else "Trading within a normal range."
    return sig, reason_str, price, chg_pct

def stock_signal(sym):
    df, err = _quick_load(sym)
    if err or df.empty:
        return "HOLD", "Data unavailable.", None, None
    close   = df['Close'].squeeze().astype(float)
    price   = float(close.iloc[-1])
    chg_pct = float((close.iloc[-1]-close.iloc[-2])/close.iloc[-2]*100) if len(close)>1 else 0.0

    bb    = ta.bbands(close, length=20, std=2)
    sma50 = ta.sma(close, length=min(50,len(close)-1))
    rsi_s = ta.rsi(close, length=14)

    bbl = bbu = sma50_val = rsi_val = None
    rsi_3d = []

    if bb is not None and not bb.empty:
        bbl = float(bb[[c for c in bb.columns if c.startswith('BBL')][0]].iloc[-1])
        bbu = float(bb[[c for c in bb.columns if c.startswith('BBU')][0]].iloc[-1])
    if sma50 is not None and not sma50.empty:
        sma50_val = float(sma50.iloc[-1])
    if rsi_s is not None and len(rsi_s) >= 3:
        rsi_val = float(rsi_s.iloc[-1])
        rsi_3d  = list(rsi_s.iloc[-3:].values)

    sell_reasons = []
    if rsi_val and rsi_3d and all(r > 78 for r in rsi_3d):
        sell_reasons.append("Price momentum is exhausted (RSI very high).")
    if bbu and price > bbu * 1.08:
        sell_reasons.append("Price is extremely high compared to recent volatility.")
    if sma50_val and price > sma50_val * 1.25:
        sell_reasons.append("Price is significantly above its 50-day average.")
    if sell_reasons:
        return "SELL", "  ·  ".join(sell_reasons), price, chg_pct

    buy_reasons = []
    if bbl and price <= bbl * 1.015:
        buy_reasons.append("Price is at a support level (lower volatility band).")
    if sma50_val and price <= sma50_val:
        buy_reasons.append("Price is trading below its 50-day average.")
    if buy_reasons:
        return "BUY", "  ·  ".join(buy_reasons), price, chg_pct

    return "HOLD", "Price is stable, no extreme signals.", price, chg_pct

# ─── Portfolio Card Renderer ──────────────────────────────────────────────────
def _render_portfolio_card(sym, sig, reason, price, chg_pct, is_etf=False):
    s         = SIG_STYLES.get(sig, SIG_STYLES["HOLD"])
    price_str = f"${price:,.2f}" if price is not None else "—"
    
    # Price color logic: Green if >= 0, Red if < 0
    chg_color = "#10b981" if (chg_pct or 0) >= 0 else "#ef4444"
    arrow     = "▲" if (chg_pct or 0) >= 0 else "▼"
    chg_html  = (f'<span style="color:{chg_color};font-size:0.75rem;">'
                 f'{arrow} {abs(chg_pct):.2f}%</span>') if chg_pct is not None else ""

    pnl_html = ""
    saved    = st.session_state["pnl_data"].get(sym, {})
    shares   = saved.get("shares", 0.0)
    cost_avg = saved.get("cost",   0.0)
    if shares > 0 and cost_avg > 0 and price:
        unrealized  = shares * price - shares * cost_avg
        unreal_pct  = unrealized / (shares * cost_avg) * 100
        # P&L color logic: Green for positive, Red for negative
        u_color     = "#10b981" if unrealized >= 0 else "#ef4444"
        sign        = "+" if unrealized >= 0 else ""
        pnl_html = (
            f'<div style="font-size:0.75rem;margin-top:auto;padding-top:8px;'
            f'border-top:1px solid rgba(255,255,255,0.05);color:#94A3B8;">'
            f'{shares:g} sh @ ${cost_avg:,.2f} &nbsp; <span style="color:{u_color};font-weight:600;">'
            f'{sign}${unrealized:,.0f} ({sign}{unreal_pct:.1f}%)</span></div>'
        )

    # Truncate reason text to 2 lines
    truncated_reason = (reason[:65] + '...') if len(reason) > 65 else reason

    inner = f"""<div style="display:flex;justify-content:space-between;align-items:flex-start;">
    <div>
        <span style="font-size:1rem;font-weight:700;color:#F8FAFC;">{sym}</span>
        {_type_badge(is_etf)}
        <div style="font-size:0.85rem;color:#94A3B8;margin-top:2px;">{price_str} &nbsp;{chg_html}</div>
    </div>
    <div>{_pill(sig)}</div>
</div>
<div style="font-size:0.80rem;color:#CBD5E1;margin-top:10px;line-height:1.3;font-weight:500;flex-grow:1;overflow:hidden;">{truncated_reason}</div>
{pnl_html}"""
    return _card_wrap(inner, sig)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluent/96/000000/line-chart.png", width=50)
    st.markdown("### **Navigation**")
    ticker = st.text_input("🔍 Analyze Ticker", value="SCHG").upper().strip()

    today        = datetime.date.today()
    one_year_ago = today - datetime.timedelta(days=365)
    start_date   = st.date_input("Start Date", value=one_year_ago)
    end_date     = st.date_input("End Date",   value=today)

    st.markdown("---")
    st.markdown("### 📊 **Indicators**")
    show_ema  = st.checkbox("Short-term EMA", value=False)
    ema_period = st.number_input("EMA Period",  min_value=2, max_value=200, value=20, disabled=not show_ema)
    show_sma  = st.checkbox("Long-term SMA",   value=False)
    sma_period = st.number_input("SMA Period",  min_value=2, max_value=500, value=50, disabled=not show_sma)
    show_rsi_indicator = st.checkbox("RSI",    value=False)
    rsi_period = st.number_input("RSI Period",  min_value=2, max_value=100, value=14, disabled=not show_rsi_indicator)

    st.markdown("---")
    st.markdown("### 🔮 **Forecast**")
    show_forecast    = st.checkbox("Enable 14-Day Forecast", value=False)

    if ticker:
        st.markdown("---")
        st.markdown("### 📅 **Earnings**")
        earnings_date, earn_err = fetch_earnings_date(ticker)
        if earn_err: st.caption(f"⚠️ {earn_err}")
        elif earnings_date:
            days_away = _days_until(earnings_date)
            if days_away >= 0:
                urg = "#ef4444" if days_away<=7 else ("#fbbf24" if days_away<=21 else "#34d399")
                st.markdown(
                    f'<div style="background:rgba(30,41,59,0.5);border:1px solid rgba(71,85,105,0.3);'
                    f'border-radius:8px;padding:12px;text-align:center;">'
                    f'<div style="color:#94a3b8;font-size:0.7rem;text-transform:uppercase;">Next Earnings</div>'
                    f'<div style="font-size:1.5rem;font-weight:700;color:{urg};">{"Today" if days_away==0 else f"{days_away} Days"}</div>'
                    f'<div style="color:#94a3b8;font-size:0.75rem;">{earnings_date.strftime("%b %d, %Y")}</div>'
                    f'</div>', unsafe_allow_html=True)
            else: st.caption(f"Last: {earnings_date.strftime('%b %d, %Y')}")

    if ticker:
        st.markdown("---")
        st.markdown("### 📰 **News**")
        news_items, _ = fetch_news(ticker)
        for item in news_items:
            st.markdown(f'<div style="font-size:0.8rem;margin-bottom:8px;"><a href="{item["link"]}" target="_blank">{item["title"]}</a></div>', unsafe_allow_html=True)

# ─── Main Header ──────────────────────────────────────────────────────────────
h1, h2 = st.columns([3,1])
with h1:
    st.markdown('<h1 style="font-weight:700;font-size:2.2rem;margin-bottom:0;">📈 Stock Research</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#94a3b8;">Portfolio signals & deep-dive analysis.</p>', unsafe_allow_html=True)
with h2:
    st.markdown('<div style="text-align:right;padding-top:15px;"><span class="status-pulse"></span> Engine Active</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PORTFOLIO DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 📌 My Portfolio")

if "show_editor" not in st.session_state:
    st.session_state["show_editor"] = False

if st.button("⚙️ Edit Holdings"):
    st.session_state["show_editor"] = not st.session_state["show_editor"]

if st.session_state["show_editor"]:
    st.markdown("### 📝 Edit Holdings")
    all_tickers = PORTFOLIO_ETFS + PORTFOLIO_STOCKS
    
    # Prepare data for editor
    editor_data = []
    for sym in all_tickers:
        saved = st.session_state["pnl_data"].get(sym, {"shares": 0.0, "cost": 0.0})
        editor_data.append({
            "Ticker": sym,
            "Type": "ETF" if sym in PORTFOLIO_ETFS else "STOCK",
            "Shares": float(saved.get("shares", 0.0)),
            "Cost Basis": float(saved.get("cost", 0.0))
        })
    
    df_editor = pd.DataFrame(editor_data)
    
    # Display editor
    edited_df = st.data_editor(
        df_editor,
        column_config={
            "Ticker": st.column_config.TextColumn(disabled=True),
            "Type": st.column_config.TextColumn(disabled=True),
            "Shares": st.column_config.NumberColumn(min_value=0.0, format="%.2f"),
            "Cost Basis": st.column_config.NumberColumn(min_value=0.0, format="%.2f")
        },
        hide_index=True,
        use_container_width=True
    )
    
    if st.button("💾 Save Holdings"):
        new_pnl_data = {}
        for _, row in edited_df.iterrows():
            new_pnl_data[row["Ticker"]] = {
                "shares": row["Shares"],
                "cost": row["Cost Basis"]
            }
        st.session_state["pnl_data"] = new_pnl_data
        _save_pnl_to_disk(new_pnl_data)
        st.success("✅ Portfolio updated successfully!")
        st.session_state["show_editor"] = False
        st.rerun()

# ── ETF Cards ─────────────────────────────────────────────────────────────────
st.markdown("### 📦 ETFs")
etf_cols = st.columns(len(PORTFOLIO_ETFS))
for i, sym in enumerate(PORTFOLIO_ETFS):
    sig, reason, price, chg_pct = etf_signal(sym)
    with etf_cols[i]:
        st.markdown(_render_portfolio_card(sym, sig, reason, price, chg_pct, is_etf=True), unsafe_allow_html=True)

# ── Stock Cards ───────────────────────────────────────────────────────────────
st.markdown("### 📈 Stocks")
stock_rows = [PORTFOLIO_STOCKS[i:i+3] for i in range(0,len(PORTFOLIO_STOCKS),3)]
for row in stock_rows:
    cols = st.columns(3)
    for j, sym in enumerate(row):
        sig, reason, price, chg_pct = stock_signal(sym)
        with cols[j]:
            st.markdown(_render_portfolio_card(sym, sig, reason, price, chg_pct, is_etf=False), unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DEEP-DIVE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"## 🔬 Deep-Dive: `{ticker}`")

if not ticker:
    st.warning("⚠️ Enter a ticker in the sidebar.")
else:
    df, data_error = load_stock_data(ticker, start_date, end_date)
    if data_error:
        st.error(f"❌ {data_error}")
    elif df.empty:
        st.error(f"❌ No data.")
    else:
        close_series   = df['Close'].squeeze().astype(float)
        is_etf_ticker  = ticker in PORTFOLIO_ETFS

        if show_ema: df['EMA'] = ta.ema(close_series, length=ema_period)
        if show_sma: df['SMA'] = ta.sma(close_series, length=sma_period)
        if show_rsi_indicator: df['RSI'] = ta.rsi(close_series, length=rsi_period)

        forecast_df = pd.DataFrame()
        if show_forecast:
            fraw, ferr = generate_statistical_forecast(tuple(close_series.values), tuple(close_series.index), 14)
            if not ferr: forecast_df = fraw

        # Metrics
        latest_close = float(close_series.iloc[-1])
        m1,m2,m3,m4 = st.columns(4)
        with m1: st.metric("Latest Close", f"${latest_close:,.2f}")
        with m2: st.metric("High",  f"${float(df['High'].max()):,.2f}")
        with m3: st.metric("Low",   f"${float(df['Low'].min()):,.2f}")
        with m4: st.metric("Volatility", f"{float(close_series.pct_change().std()*np.sqrt(252)*100):.1f}%")

        # Chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index,open=df['Open'],high=df['High'],low=df['Low'],close=df['Close'],name="Price"))
        if show_ema: fig.add_trace(go.Scatter(x=df.index,y=df['EMA'],name="EMA",line=dict(color='#facc15')))
        if show_sma: fig.add_trace(go.Scatter(x=df.index,y=df['SMA'],name="SMA",line=dict(color='#22d3ee')))
        
        fig.update_layout(
            plot_bgcolor='#0F172A', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E2E8F0'),
            margin=dict(t=20,b=20,l=20,r=20),
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
        )
        st.plotly_chart(fig, use_container_width=True)

        # Fundamentals
        info, fund_err = fetch_company_info(ticker)
        if info:
            st.markdown("### 🏢 Fundamentals")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Market Cap", f"{info.get('marketCap',0)/1e9:.1f}B")
            c2.metric("P/E", f"{info.get('trailingPE', 'N/A')}")
            c3.metric("P/B", f"{info.get('priceToBook', 'N/A')}")
            c4.metric("Yield", f"{info.get('dividendYield', 0)*100:.2f}%")
