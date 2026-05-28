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
PORTFOLIO_ETFS   = ["SCHG", "SMH", "QTUM", "VOO", "XT"]
PORTFOLIO_STOCKS = ["AMZN", "ORCL", "LRN", "NBIS", "NVDA", "ASML", "TSM", "GOOG", "EVGO"]

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

# ─── Global CSS (only structural animations; all card styles will be inlined) ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
* { font-family: 'Outfit', sans-serif !important; }

@keyframes pulse {
    0%   { transform:scale(0.95); box-shadow:0 0 0 0   rgba(16,185,129,0.7); }
    70%  { transform:scale(1);    box-shadow:0 0 0 6px rgba(16,185,129,0);   }
    100% { transform:scale(0.95); box-shadow:0 0 0 0   rgba(16,185,129,0);   }
}
.status-pulse {
    width:8px; height:8px; background-color:#10b981;
    border-radius:50%; display:inline-block;
    animation: pulse 2s infinite;
}
div[data-testid="stMetricValue"] { font-size:1.8rem !important; font-weight:700 !important; color:#f8fafc !important; }
div[data-testid="stMetricLabel"] { font-size:0.9rem !important; color:#94a3b8 !important; text-transform:uppercase; letter-spacing:0.05em; }
[data-testid="stMetric"] {
    background: linear-gradient(135deg,rgba(30,41,59,0.4),rgba(15,23,42,0.5)) !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
    border-radius:12px !important; padding:12px 18px !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Inline Style Constants ────────────────────────────────────────────────────
# Card backgrounds & borders per signal — all inlined to avoid Streamlit CSS sandboxing
SIG_STYLES = {
    "BUY":  {
        "card_bg":    "linear-gradient(135deg,rgba(16,185,129,0.13),rgba(5,150,105,0.2))",
        "card_bdr":   "1px solid rgba(16,185,129,0.45)",
        "pill_bg":    "rgba(16,185,129,0.2)",
        "pill_color": "#10b981",
        "pill_bdr":   "1px solid rgba(16,185,129,0.45)",
        "pill_label": "🟢 BUY",
    },
    "DCA":  {
        "card_bg":    "linear-gradient(135deg,rgba(99,102,241,0.12),rgba(79,70,229,0.18))",
        "card_bdr":   "1px solid rgba(99,102,241,0.4)",
        "pill_bg":    "rgba(99,102,241,0.2)",
        "pill_color": "#818cf8",
        "pill_bdr":   "1px solid rgba(99,102,241,0.4)",
        "pill_label": "💙 DCA",
    },
    "WAIT": {
        "card_bg":    "linear-gradient(135deg,rgba(245,158,11,0.1),rgba(234,179,8,0.15))",
        "card_bdr":   "1px solid rgba(245,158,11,0.45)",
        "pill_bg":    "rgba(245,158,11,0.2)",
        "pill_color": "#fbbf24",
        "pill_bdr":   "1px solid rgba(245,158,11,0.45)",
        "pill_label": "🟡 WAIT",
    },
    "HOLD": {
        "card_bg":    "linear-gradient(135deg,rgba(71,85,105,0.15),rgba(51,65,85,0.22))",
        "card_bdr":   "1px solid rgba(148,163,184,0.28)",
        "pill_bg":    "rgba(100,116,139,0.2)",
        "pill_color": "#94a3b8",
        "pill_bdr":   "1px solid rgba(100,116,139,0.38)",
        "pill_label": "⚪ HOLD",
    },
    "SELL": {
        "card_bg":    "linear-gradient(135deg,rgba(239,68,68,0.12),rgba(220,38,38,0.18))",
        "card_bdr":   "1px solid rgba(239,68,68,0.45)",
        "pill_bg":    "rgba(239,68,68,0.2)",
        "pill_color": "#f87171",
        "pill_bdr":   "1px solid rgba(239,68,68,0.45)",
        "pill_label": "🔴 SELL",
    },
}

# ─── Helpers ───────────────────────────────────────────────────────────────────
def _fmt_age(pub_ts):
    try:
        # For Python 3.9+ utcfromtimestamp is deprecated.
        # Use fromtimestamp with timezone awareness.
        if isinstance(pub_ts, (int, float)):
            # Ensure timestamp is interpreted as UTC to match utcnow()
            pub_dt = datetime.datetime.fromtimestamp(pub_ts, tz=datetime.timezone.utc)
        else:
            pub_dt = pub_ts # Assume already a datetime object, potentially timezone naive
        
        # If pub_dt is timezone-naive, make it timezone-aware UTC for consistent comparison
        if pub_dt.tzinfo is None:
            pub_dt = pub_dt.replace(tzinfo=datetime.timezone.utc)

        delta  = datetime.datetime.now(datetime.timezone.utc) - pub_dt
        hours  = int(delta.total_seconds() // 3600)
        if hours < 1:  return "just now"
        if hours < 24: return f"{hours}h ago"
        return f"{hours // 24}d ago"
    except Exception: # Catch specific exceptions if possible, but broad for robustness here
        return ""

def _days_until(d):
    return (d - datetime.date.today()).days

def _hl(v):
    """Inline-styled highlight span — no CSS class dependency."""
    return (f'<span style="display:inline-block;background:rgba(99,102,241,0.18);'
            f'border:1px solid rgba(99,102,241,0.3);padding:1px 7px;border-radius:6px;'
            f'font-weight:600;color:#a5b4fc;font-size:0.92em;">{v}</span>')

def _card_wrap(inner_html, sig):
    """Wrap content in a fully inlined signal card."""
    s = SIG_STYLES.get(sig, SIG_STYLES["HOLD"])
    return (
        f'<div style="background:{s["card_bg"]};border:{s["card_bdr"]};'
        f'border-radius:14px;padding:14px 16px;margin-bottom:10px;">'
        f'{inner_html}</div>'
    )

def _pill(sig):
    s = SIG_STYLES.get(sig, SIG_STYLES["HOLD"])
    return (f'<span style="display:inline-block;padding:3px 11px;border-radius:999px;'
            f'font-size:0.7rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;'
            f'background:{s["pill_bg"]};color:{s["pill_color"]};border:{s["pill_bdr"]};">'
            f'{s["pill_label"]}</span>')

def _type_badge(is_etf):
    if is_etf:
        return ('<span style="font-size:0.68rem;color:#6366f1;background:rgba(99,102,241,0.12);'
                'border:1px solid rgba(99,102,241,0.25);padding:1px 7px;border-radius:999px;'
                'margin-left:6px;">ETF</span>')
    return ('<span style="font-size:0.68rem;color:#a855f7;background:rgba(168,85,247,0.12);'
            'border:1px solid rgba(168,85,247,0.25);padding:1px 7px;border-radius:999px;'
            'margin-left:6px;">STOCK</span>')

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
        return "DCA", "Could not fetch data.", None, None
    close   = df['Close'].squeeze().astype(float)
    price   = float(close.iloc[-1])
    chg_pct = float((close.iloc[-1]-close.iloc[-2])/close.iloc[-2]*100) if len(close)>1 else 0.0
    score   = 0
    reasons = []

    rsi_s = ta.rsi(close, length=14)
    rsi   = float(rsi_s.iloc[-1]) if rsi_s is not None and not rsi_s.empty else 50.0
    if rsi > 73:
        score += 3; reasons.append(f"RSI {rsi:.0f} — overbought territory (>73)")
    elif rsi < 40:
        score -= 2; reasons.append(f"RSI {rsi:.0f} — oversold, momentum reset (<40)")

    sma200 = ta.sma(close, length=min(200,len(close)-1))
    if sma200 is not None and not sma200.empty:
        s200 = float(sma200.iloc[-1])
        prem = (price - s200) / s200 * 100
        if prem > 10:
            score += 3; reasons.append(f"{prem:.1f}% above 200-SMA — premium priced")
        elif prem > 5:
            score += 1; reasons.append(f"{prem:.1f}% above 200-SMA — mildly extended")
        elif prem < 0:
            score -= 4; reasons.append(f"{abs(prem):.1f}% below 200-SMA — genuine dip")

    bb = ta.bbands(close, length=20, std=2)
    if bb is not None and not bb.empty:
        bbu = float(bb[[c for c in bb.columns if c.startswith('BBU')][0]].iloc[-1])
        bbl = float(bb[[c for c in bb.columns if c.startswith('BBL')][0]].iloc[-1])
        if price >= bbu * 0.98:
            score += 2; reasons.append(f"Price near upper Bollinger Band (${bbu:.2f})")
        elif price <= bbl * 1.03:
            score -= 3; reasons.append(f"Price near/below lower Bollinger Band (${bbl:.2f})")

    sig = "WAIT" if score >= 5 else ("BUY" if score <= -3 else "DCA")
    reason_str = "  ·  ".join(reasons) if reasons else "Normal trading range — stay on DCA schedule."
    return sig, reason_str, price, chg_pct

def stock_signal(sym):
    df, err = _quick_load(sym)
    if err or df.empty:
        return "HOLD", "Could not fetch data.", None, None
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
        sell_reasons.append(f"RSI {rsi_val:.0f} — 3-session momentum exhaustion (>78)")
    if bbu and price > bbu * 1.08:
        sell_reasons.append(f"{(price/bbu-1)*100:.1f}% above upper Bollinger Band — severely overextended")
    if sma50_val and price > sma50_val * 1.25:
        sell_reasons.append(f"{(price/sma50_val-1)*100:.1f}% above 50-SMA — unsustainable extension")
    if sell_reasons:
        return "SELL", "  ·  ".join(sell_reasons), price, chg_pct

    buy_reasons = []
    if bbl and price <= bbl * 1.015:
        buy_reasons.append(f"Price ${price:.2f} at/below lower BB ${bbl:.2f} — oversold floor")
    if sma50_val and price <= sma50_val:
        buy_reasons.append(f"{(sma50_val-price)/sma50_val*100:.1f}% below 50-SMA ${sma50_val:.2f} — structural discount")
    if buy_reasons:
        return "BUY", "  ·  ".join(buy_reasons), price, chg_pct

    hold_notes = []
    if sma50_val: hold_notes.append(f"{(price-sma50_val)/sma50_val*100:.1f}% above 50-SMA")
    if bbu:       hold_notes.append(f"{(bbu-price)/bbu*100:.1f}% from upper BB")
    return "HOLD", "Premium-priced but not extreme.  " + ("  ·  ".join(hold_notes) if hold_notes else ""), price, chg_pct

# ─── Portfolio Card Renderer (fully inlined styles) ──────────────────────────
def _render_portfolio_card(sym, sig, reason, price, chg_pct, is_etf=False):
    s         = SIG_STYLES.get(sig, SIG_STYLES["HOLD"])
    price_str = f"${price:,.2f}" if price is not None else "—"
    chg_color = "#10b981" if (chg_pct or 0) >= 0 else "#ef4444"
    arrow     = "▲" if (chg_pct or 0) >= 0 else "▼"
    chg_html  = (f'<span style="color:{chg_color};font-size:0.8rem;">'
                 f'{arrow} {abs(chg_pct):.2f}% today</span>') if chg_pct is not None else ""

    # P&L row
    pnl_html = ""
    saved    = st.session_state["pnl_data"].get(sym, {})
    shares   = saved.get("shares", 0.0)
    cost_avg = saved.get("cost",   0.0)
    if shares > 0 and cost_avg > 0 and price:
        unrealized  = shares * price - shares * cost_avg
        unreal_pct  = unrealized / (shares * cost_avg) * 100
        u_color     = "#10b981" if unrealized >= 0 else "#ef4444"
        sign        = "+" if unrealized >= 0 else ""
        pnl_html = (
            f'<div style="font-size:0.82rem;margin-top:7px;padding-top:7px;'
            f'border-top:1px solid rgba(255,255,255,0.07);">'
            f'<span style="color:#64748b;">{shares:g} sh @ ${cost_avg:,.2f}</span>'
            f'&nbsp;&nbsp;'
            f'<span style="color:{u_color};font-weight:600;">'
            f'{sign}${unrealized:,.2f} ({sign}{unreal_pct:.1f}%)</span>'
            f'&nbsp;<span style="color:#475569;font-size:0.75rem;">unrealized</span>'
            f'</div>'
        )

    inner = f"""<div style="display:flex;justify-content:space-between;align-items:flex-start;">
    <div>
        <span style="font-size:1.05rem;font-weight:700;color:#f1f5f9;">{sym}</span>
        {_type_badge(is_etf)}
        <div style="font-size:0.88rem;color:#94a3b8;margin-top:2px;">{price_str} &nbsp;{chg_html}</div>
    </div>
    <div style="margin-top:2px;">{_pill(sig)}</div>
</div>
{pnl_html}
<div style="font-size:0.76rem;color:#64748b;margin-top:6px;line-height:1.5;">{reason}</div>"""
    return _card_wrap(inner, sig)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluent/96/000000/line-chart.png", width=60)
    st.markdown("### **Navigation & Inputs**")
    ticker = st.text_input("🔍 Analyze Ticker", value="SCHG").upper().strip()

    today        = datetime.date.today()
    one_year_ago = today - datetime.timedelta(days=365)
    start_date   = st.date_input("Start Date", value=one_year_ago, max_value=today)
    end_date     = st.date_input("End Date",   value=today,        max_value=today)

    st.markdown("---")
    st.markdown("### 📊 **Indicators**")
    show_ema  = st.checkbox("Short-term EMA", value=False)
    ema_period = st.number_input("EMA Period",  min_value=2, max_value=200, value=20, step=1, disabled=not show_ema)
    show_sma  = st.checkbox("Long-term SMA",   value=False)
    sma_period = st.number_input("SMA Period",  min_value=2, max_value=500, value=50, step=1, disabled=not show_sma)
    show_rsi_indicator = st.checkbox("RSI",    value=False)
    rsi_period = st.number_input("RSI Period",  min_value=2, max_value=100, value=14, step=1, disabled=not show_rsi_indicator)

    st.markdown("---")
    st.markdown("### 🔮 **Forecast**")
    show_forecast    = st.checkbox("Enable 14-Day Forecast", value=False)
    forecast_horizon = 14

    # Earnings Countdown
    if ticker:
        st.markdown("---")
        st.markdown("### 📅 **Earnings Countdown**")
        earnings_date, earn_err = fetch_earnings_date(ticker)
        if earn_err:
            st.caption(f"⚠️ {earn_err}")
        elif earnings_date:
            days_away = _days_until(earnings_date)
            if days_away >= 0:
                urg = "#ef4444" if days_away<=7 else ("#fbbf24" if days_away<=21 else "#34d399")
                warn_txt = ("⚠️ Earnings near — expect volatility. Consider waiting." if days_away<=14
                            else "📌 Earnings approaching. Watch for swings." if days_away<=30 else "")
                st.markdown(
                    f'<div style="background:linear-gradient(135deg,rgba(245,158,11,0.1),rgba(234,179,8,0.07));'
                    f'border:1px solid rgba(245,158,11,0.35);border-radius:12px;padding:14px 16px;'
                    f'margin-top:8px;text-align:center;">'
                    f'<div style="color:#94a3b8;font-size:0.72rem;text-transform:uppercase;letter-spacing:.06em;">Next Earnings</div>'
                    f'<div style="font-size:2rem;font-weight:700;color:{urg};">{"Today" if days_away==0 else f"In {days_away} Day{chr(115) if days_away!=1 else ""}"}</div>'
                    f'<div style="color:#94a3b8;font-size:0.8rem;">{earnings_date.strftime("%B %d, %Y")}</div>'
                    f'{"<div style=color:#fcd34d;font-size:0.75rem;margin-top:5px;font-style:italic;>"+warn_txt+"</div>" if warn_txt else ""}'
                    f'</div>',
                    unsafe_allow_html=True)
            else:
                st.caption(f"Last reported: {earnings_date.strftime('%b %d, %Y')}")
        else:
            st.caption("No upcoming earnings date found.")

    # Live News
    if ticker:
        st.markdown("---")
        st.markdown("### 📰 **Latest News**")
        news_items, _ = fetch_news(ticker)
        if not news_items:
            st.caption("No recent headlines found.")
        else:
            for item in news_items:
                age = _fmt_age(item['ts']) if item['ts'] else ""
                st.markdown(
                    f'<div style="padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.07);'
                    f'font-size:0.82rem;line-height:1.45;">'
                    f'<a href="{item["link"]}" target="_blank" style="color:#a5b4fc;text-decoration:none;">{item["title"]}</a>'
                    f'{"<div style=color:#475569;font-size:0.73rem;margin-top:2px;>"+age+"</div>" if age else ""}'
                    f'</div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("💡 *Powered by Python · Streamlit · yFinance · Pandas*")

# ─── Main Header ──────────────────────────────────────────────────────────────
h1, h2 = st.columns([3,1])
with h1:
    st.markdown(
        '<h1 style="background:linear-gradient(to right,#6366f1,#a855f7,#ec4899);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
        'font-weight:700;font-size:2.75rem;margin-bottom:0.25rem;">📈 Stock Research & Projection App</h1>',
        unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#94a3b8;font-size:1.1rem;">Portfolio signals, deep-dive analysis, forecasting & accumulation strategy.</p>',
        unsafe_allow_html=True)
with h2:
    st.markdown(
        '<div style="text-align:right;padding-top:15px;">'
        '<div style="display:inline-flex;align-items:center;gap:8px;background:rgba(16,185,129,0.1);'
        'color:#10b981;border:1px solid rgba(16,185,129,0.2);padding:6px 14px;border-radius:9999px;'
        'font-weight:500;font-size:0.875rem;">'
        '<span class="status-pulse"></span> Engine Active</div></div>',
        unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PORTFOLIO DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 📌 My Portfolio Dashboard")
st.markdown(
    '<p style="color:#64748b;font-size:0.88rem;margin-top:-10px;margin-bottom:16px;">'
    'Live signals for all holdings. ETF logic: RSI + 200-SMA + Bollinger score. '
    'Stock logic: BB + 50-SMA buy threshold / RSI exhaustion + extension sell triggers.</p>',
    unsafe_allow_html=True)

# ── P&L Input Expander (saves to disk on every change) ────────────────────────
with st.expander("✏️ Enter / Update Shares & Cost Basis", expanded=False):
    st.markdown(
        '<p style="color:#64748b;font-size:0.82rem;margin-bottom:12px;">'
        'Your entries are saved to <code>pnl_data.json</code> next to this script and reloaded on every restart.</p>',
        unsafe_allow_html=True)
    all_tickers = PORTFOLIO_ETFS + PORTFOLIO_STOCKS
    pnl_cols    = st.columns(3)
    changed     = False
    for idx, sym in enumerate(all_tickers):
        with pnl_cols[idx % 3]:
            saved      = st.session_state["pnl_data"].get(sym, {})
            st.markdown(f'<div style="display:flex;flex-direction:column;gap:8px;">', unsafe_allow_html=True)
            shares_val = st.number_input(f"{sym} — Shares", min_value=0.0,
                                         value=float(saved.get("shares", 0.0)),
                                         step=0.1, format="%.2f", key=f"sh_{sym}", label_visibility="collapsed")
            cost_val   = st.number_input(f"{sym} — Avg Cost ($)", min_value=0.0,
                                         value=float(saved.get("cost", 0.0)),
                                         step=0.01, format="%.2f", key=f"co_{sym}", label_visibility="collapsed")
            st.markdown(f'<p style="color:#94a3b8;font-size:0.82rem;margin-top:-8px;margin-bottom:12px;">{sym} — Shares / Avg Cost ($)</p>', unsafe_allow_html=True)
            new_entry  = {"shares": shares_val, "cost": cost_val}
            if new_entry != saved:
                changed = True
            st.session_state["pnl_data"][sym] = new_entry
    if changed:
        _save_pnl_to_disk(st.session_state["pnl_data"])
        st.success("✅ Saved to disk.", icon="💾")

# ── ETF Cards ─────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="font-size:0.72rem;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;'
    'color:#475569;margin:18px 0 10px 2px;">📦 ETFs — Long-Term DCA Holdings</div>',
    unsafe_allow_html=True)
etf_cols = st.columns(len(PORTFOLIO_ETFS))
for i, sym in enumerate(PORTFOLIO_ETFS):
    sig, reason, price, chg_pct = etf_signal(sym)
    with etf_cols[i]:
        st.markdown(_render_portfolio_card(sym, sig, reason, price, chg_pct, is_etf=True),
                    unsafe_allow_html=True)

# ── Stock Cards (3 per row) ───────────────────────────────────────────────────
st.markdown(
    '<div style="font-size:0.72rem;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;'
    'color:#475569;margin:18px 0 10px 2px;">📈 Individual Stocks</div>',
    unsafe_allow_html=True)
stock_rows = [PORTFOLIO_STOCKS[i:i+3] for i in range(0,len(PORTFOLIO_STOCKS),3)]
for row in stock_rows:
    cols = st.columns(3)
    for j, sym in enumerate(row):
        sig, reason, price, chg_pct = stock_signal(sym)
        with cols[j]:
            st.markdown(_render_portfolio_card(sym, sig, reason, price, chg_pct, is_etf=False),
                        unsafe_allow_html=True)

# ── Portfolio Summary Bar ─────────────────────────────────────────────────────
all_sigs  = [etf_signal(s)[0]   for s in PORTFOLIO_ETFS] + \
            [stock_signal(s)[0] for s in PORTFOLIO_STOCKS]
buy_n  = sum(1 for s in all_sigs if s in ("BUY","DCA"))
wait_n = sum(1 for s in all_sigs if s == "WAIT")
hold_n = sum(1 for s in all_sigs if s == "HOLD")
sell_n = sum(1 for s in all_sigs if s == "SELL")

def _summary_chip(count, color, label):
    return (f'<div style="background:rgba({color},0.1);border:1px solid rgba({color},0.3);'
            f'padding:8px 20px;border-radius:10px;text-align:center;">'
            f'<div style="font-size:1.4rem;font-weight:700;color:rgb({color});">{count}</div>'
            f'<div style="font-size:0.72rem;color:#64748b;text-transform:uppercase;">{label}</div>'
            f'</div>')

st.markdown(
    f'<div style="display:flex;gap:12px;flex-wrap:wrap;margin:16px 0 6px 0;">'
    f'{_summary_chip(buy_n,  "16,185,129", "Buy / DCA")}'
    f'{_summary_chip(hold_n, "100,116,139","Hold")}'
    f'{_summary_chip(wait_n, "245,158,11", "Wait (ETF)")}'
    f'{_summary_chip(sell_n, "239,68,68",  "Sell Alert")}'
    f'<div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);'
    f'padding:8px 16px;border-radius:10px;font-size:0.73rem;color:#475569;align-self:center;">'
    f'⏱ Signals recalculate on each page load</div>'
    f'</div>',
    unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DEEP-DIVE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"## 🔬 Deep-Dive Analysis — `{ticker}`")

if not ticker:
    st.warning("⚠️ Enter a ticker in the sidebar.")
else:
    df, data_error = load_stock_data(ticker, start_date, end_date)
    if data_error:
        st.error(f"❌ {data_error}")
    elif df.empty:
        st.error(f"❌ No data for '{ticker}' in range.")
    else:
        close_series   = df['Close'].squeeze().astype(float)
        is_etf_ticker  = ticker in PORTFOLIO_ETFS

        # Indicators
        if show_ema: df['EMA'] = ta.ema(close_series, length=ema_period)
        if show_sma: df['SMA'] = ta.sma(close_series, length=sma_period)
        if show_rsi_indicator: df['RSI'] = ta.rsi(close_series, length=rsi_period)

        # Forecast
        forecast_df = pd.DataFrame()
        if show_forecast:
            fraw, ferr = generate_statistical_forecast(
                tuple(close_series.values), tuple(close_series.index), forecast_horizon)
            if ferr: st.error(f"❌ Forecast error: {ferr}")
            else: forecast_df = fraw

        # Base indicators always needed
        bb    = ta.bbands(close_series, length=20, std=2)
        sma50 = ta.sma(close_series, length=min(50, len(close_series)-1))
        current_price = float(close_series.iloc[-1])
        has_bb    = bb    is not None and not bb.empty
        has_sma50 = sma50 is not None and not sma50.empty

        current_bbl = current_bbu = current_sma50_val = None
        if has_bb:
            current_bbl = float(bb[[c for c in bb.columns if c.startswith('BBL')][0]].iloc[-1])
            current_bbu = float(bb[[c for c in bb.columns if c.startswith('BBU')][0]].iloc[-1])
        if has_sma50:
            current_sma50_val = float(sma50.iloc[-1])

        # ── Determine signal for this ticker ──────────────────────────────────
        if is_etf_ticker:
            deep_sig, _, _, _ = etf_signal(ticker)
            signal_source = None
        else:
            rsi_full = ta.rsi(close_series, length=14)
            sell_triggered = False
            sell_reasons_deep = []
            if rsi_full is not None and len(rsi_full) >= 3:
                rsi_3d_v   = list(rsi_full.iloc[-3:].values)
                rsi_now    = float(rsi_full.iloc[-1])
                if all(r > 78 for r in rsi_3d_v):
                    sell_triggered = True
                    sell_reasons_deep.append(f"RSI {rsi_now:.0f} — 3-session exhaustion above 78")
            if current_bbu and current_price > current_bbu * 1.08:
                sell_triggered = True
                sell_reasons_deep.append(f"{(current_price/current_bbu-1)*100:.1f}% above upper BB")
            if current_sma50_val and current_price > current_sma50_val * 1.25:
                sell_triggered = True
                sell_reasons_deep.append(f"{(current_price/current_sma50_val-1)*100:.1f}% above 50-SMA")

            signal_source = None
            if not sell_triggered:
                if current_bbl and current_price <= current_bbl * 1.015:
                    signal_source = "bb"
                elif current_sma50_val and current_price <= current_sma50_val:
                    signal_source = "sma50"

            deep_sig = ("SELL" if sell_triggered else ("BUY" if signal_source else "HOLD"))

        # ── Forecast direction ─────────────────────────────────────────────────
        forecast_is_bearish = (
            show_forecast and not forecast_df.empty and
            float(forecast_df['yhat'].iloc[-1]) < float(forecast_df['yhat'].iloc[0])
        )

        # ── Strategy card body ─────────────────────────────────────────────────
        def _li(text):
            return (f'<li style="color:#cbd5e1;font-size:0.95rem;line-height:1.75;margin-bottom:4px;">'
                    f'{text}</li>')

        def _ul(items):
            lis = "".join(_li(i) for i in items)
            return f'<ul style="margin:14px 0 0 22px;padding:0;">{lis}</ul>'

        if is_etf_ticker:
            rsi_etf  = None
            sma200_e = None
            rsi_e    = ta.rsi(close_series, length=14)
            sma200_s = ta.sma(close_series, length=min(200,len(close_series)-1))
            if rsi_e   is not None and not rsi_e.empty:   rsi_etf  = float(rsi_e.iloc[-1])
            if sma200_s is not None and not sma200_s.empty: sma200_e = float(sma200_s.iloc[-1])
            etf_prem = ((current_price-sma200_e)/sma200_e*100) if sma200_e else None

            sig_header_colors = {
                "WAIT": ("#fbbf24", "🟡 ETF SIGNAL: HOLD EXTRA CASH — Overheated"),
                "BUY":  ("#10b981", "🟢 ETF SIGNAL: ADD EXTRA — Real Dip Detected"),
                "DCA":  ("#818cf8", "💙 ETF SIGNAL: STAY ON SCHEDULE — Normal Range"),
            }
            hdr_color, hdr_text = sig_header_colors.get(deep_sig, sig_header_colors["DCA"])

            if deep_sig == "WAIT":
                bullets = [
                    "This ETF is showing signs of being <strong style='color:#f1f5f9;'>temporarily overvalued</strong> relative to its historical norms — you would be paying a premium above the long-term trend.",
                    *([ f"RSI is at {_hl(f'{rsi_etf:.0f}')} — pushing into overbought territory above 73, indicating short-term buying has been excessive." ] if rsi_etf and rsi_etf>68 else []),
                    *([ f"Price is {_hl(f'{etf_prem:.1f}% above')} its 200-day moving average of {_hl('$'+f'{sma200_e:,.2f}')} — a meaningful premium to long-term fair value." ] if etf_prem and etf_prem>5 else []),
                    "Since you DCA bi-weekly to monthly anyway, <strong style='color:#f1f5f9;'>continue your scheduled buys</strong> — but do <em>not</em> deploy any extra discretionary cash right now. Save it for the next pullback.",
                    "Even great ETFs have cooling-off periods. Paying 10%+ above trend rarely produces good near-term results, even if the long-term trajectory remains intact.",
                ]
            elif deep_sig == "BUY":
                bullets = [
                    "This ETF has pulled back into a <strong style='color:#f1f5f9;'>statistically significant discount zone</strong> — exactly the kind of dip you want to catch on top of your regular DCA schedule.",
                    *([ f"RSI has reset to {_hl(f'{rsi_etf:.0f}')} — well into oversold territory, meaning the short-term selling has been overdone." ] if rsi_etf and rsi_etf<42 else []),
                    *([ f"Price is now {_hl(f'{abs(etf_prem):.1f}% below')} the 200-day moving average — trading at a genuine long-term discount." ] if etf_prem and etf_prem<0 else []),
                    "This is a <strong style='color:#f1f5f9;'>high-conviction opportunity to deploy extra capital</strong>. For an ETF you're already committed to holding long-term, catching it at a discount materially improves your future cost basis and compounding.",
                    "Suggested approach: deploy 50–100% of any discretionary cash earmarked for this position, on top of your scheduled DCA buy.",
                ]
            else:
                bullets = [
                    "This ETF is trading in its normal range — not stretched to the upside, not beaten down. <strong style='color:#f1f5f9;'>Your regular DCA schedule is the right call here.</strong>",
                    *([ f"RSI is at {_hl(f'{rsi_etf:.0f}')} — a balanced reading, consistent with normal buying and selling pressure." ] if rsi_etf else []),
                    *([ f"Price is {_hl(f'{abs(etf_prem):.1f}% {'above' if etf_prem>=0 else 'below'}')} the 200-day SMA — within a normal fluctuation band." ] if etf_prem is not None else []),
                    "No urgency to rush in extra capital, but no reason to delay your regular investment either.",
                    "Stay disciplined: <strong style='color:#f1f5f9;'>consistency over timing</strong> is what builds long-term wealth in diversified ETFs.",
                ]

            guide_html = (
                f'<strong style="color:{hdr_color};font-size:1.2rem;">{hdr_text}</strong>'
                f'{_ul(bullets)}'
            )
            etf_badge = (' <span style="font-size:0.78rem;color:#6366f1;background:rgba(99,102,241,0.12);'
                         'border:1px solid rgba(99,102,241,0.25);padding:2px 10px;border-radius:999px;'
                         'margin-left:8px;font-weight:500;">ETF Mode</span>')

        else:
            pct_below_sma = ((current_sma50_val-current_price)/current_sma50_val*100) if current_sma50_val else None

            if deep_sig == "SELL":
                guide_html = (
                    f'<strong style="color:#f87171;font-size:1.2rem;">🔴 SELL SIGNAL — Momentum Exhaustion</strong>'
                    + _ul([
                        f"The stock has reached a zone of <strong style='color:#f1f5f9;'>statistical overextension</strong>. Exhaustion indicators fired: {_hl('  ·  '.join(sell_reasons_deep))}.",
                        "When momentum oscillators hold above 78 for multiple sessions, it historically marks a point where short-term buyers run out of steam and the stock corrects.",
                        "This is <em>not necessarily</em> a long-term thesis change — it signals that the <strong style='color:#f1f5f9;'>risk/reward for new money is poor right now</strong>.",
                        f"<strong style='color:#f1f5f9;'>Recommended action:</strong> Consider trimming 25–50% of the position to lock in gains. Set a re-entry alert at the 50-SMA {(_hl('$'+f'{current_sma50_val:,.2f}')) if current_sma50_val else ''}.",
                    ])
                )
            elif deep_sig == "BUY" and signal_source == "bb":
                guide_html = (
                    f'<strong style="color:#10b981;font-size:1.2rem;">🛒 BUY SIGNAL — Lower Bollinger Band Breach</strong>'
                    + _ul([
                        f"Current price {_hl('$'+f'{current_price:,.2f}')} has touched the <strong style='color:#f1f5f9;'>Lower Bollinger Band</strong> at {_hl('$'+f'{current_bbl:,.2f}')} — the statistical price floor set 2 standard deviations below the 20-day average.",
                        "Think of it like a <strong style='color:#f1f5f9;'>rubber band being stretched</strong>: the further price deviates downward from the mean, the stronger the mathematical tendency to snap back.",
                        "This signals the stock is <strong style='color:#f1f5f9;'>temporarily oversold</strong> — sellers have pushed price to an extreme and buying pressure typically builds from here.",
                        "<strong style='color:#f1f5f9;'>Recommended action:</strong> Begin accumulating a partial position. Scale in across 2–3 sessions to reduce timing risk.",
                    ])
                )
            elif deep_sig == "BUY" and signal_source == "sma50":
                pct_str   = f"{pct_below_sma:.1f}" if pct_below_sma else "?"
                sma50_fmt = f"{current_sma50_val:,.2f}" if current_sma50_val else "N/A"
                guide_html = (
                    f'<strong style="color:#10b981;font-size:1.2rem;">🛒 BUY SIGNAL — Below 50-Day Moving Average</strong>'
                    + _ul([
                        f"Trading at {_hl('$'+f'{current_price:,.2f}')} — {_hl(pct_str+'% below')} its 50-day SMA of {_hl('$'+sma50_fmt)}. You're cutting out short-term hype and buying at a structural discount.",
                        "This type of pullback is <em>normal and expected</em> in healthy bull trends — it shakes out weak hands and reloads the stock for its next leg up.",
                        f"The further a quality stock falls below its 50-day average, the <strong style='color:#f1f5f9;'>wider your margin of safety</strong>.",
                        "<strong style='color:#f1f5f9;'>Recommended action:</strong> Build a position here, with a plan to add more if price extends further below the average.",
                    ])
                )
            else:  # HOLD
                bbu_ref  = _hl('$'+f'{current_bbu:,.2f}')  if current_bbu       else ""
                sma_ref  = _hl('$'+f'{current_sma50_val:,.2f}') if current_sma50_val else ""
                bbl_ref  = _hl('$'+f'{current_bbl:,.2f}')  if current_bbl       else ""
                guide_html = (
                    f'<strong style="color:#94a3b8;font-size:1.2rem;">⏳ HOLD — Stock Trading at Premium</strong>'
                    + _ul([
                        f"At {_hl('$'+f'{current_price:,.2f}')}, the stock is above its 50-SMA {('('+sma_ref+')') if sma_ref else ''} and near the upper Bollinger Band {('('+bbu_ref+')') if bbu_ref else ''}. You are in <em>premium territory.</em>",
                        "Buying here means paying a price that already reflects optimism — <strong style='color:#f1f5f9;'>you're not getting a deal, you're paying full price at peak excitement.</strong>",
                        "Markets breathe. Even the strongest stocks pull back 5–15% before continuing higher. Your goal is to <strong style='color:#f1f5f9;'>buy those dips, not chase the peaks.</strong>",
                        f"<strong style='color:#f1f5f9;'>Recommended action:</strong> Set a price alert at the 50-SMA {(sma_ref+' ') if sma_ref else ''}or lower BB {(bbl_ref) if bbl_ref else ''} and wait for the market to come to you.",
                    ])
                )
            etf_badge = ""

        # Forecast dip explainer
        forecast_explainer_html = ""
        if forecast_is_bearish:
            f_start  = float(forecast_df['yhat'].iloc[0])
            f_end    = float(forecast_df['yhat'].iloc[-1])
            drop_pct = (f_start-f_end)/f_start*100
            forecast_explainer_html = (
                f'<div style="margin-top:18px;padding:14px 18px;'
                f'background:rgba(245,158,11,0.08);border:1px dashed rgba(245,158,11,0.4);border-radius:10px;">'
                f'<p style="color:#fbbf24;font-size:1rem;font-weight:600;margin:0 0 8px 0;">🔮 Why the Dip Warning?</p>'
                f'<p style="color:#cbd5e1;font-size:0.93rem;line-height:1.65;margin:0;">'
                f'The mathematical momentum embedded in recent closing prices shows a '
                f'<strong style="color:#fcd34d;">short-term downward trajectory</strong> — a cooling-off period '
                f'the Exponential Smoothing model projects will continue over the next 14 days '
                f'(estimated drop of ~{_hl(f"{drop_pct:.1f}%")} from current level).<br><br>'
                f'This is a <strong style="color:#fcd34d;">reversion to the mean in motion</strong>, not a crash signal. '
                f'If you are watching for a buy entry, <strong style="color:#f1f5f9;">exercise patience</strong>: '
                f'an even better window may open over the next fortnight.</p></div>'
            )

        # Render the guide card
        s = SIG_STYLES.get(deep_sig if not is_etf_ticker else
                           ("BUY" if deep_sig=="BUY" else ("WAIT" if deep_sig=="WAIT" else "DCA")),
                           SIG_STYLES["HOLD"])
        etf_badge = etf_badge if is_etf_ticker else ""
        st.markdown(
            f'<div style="background:{s["card_bg"]};border:{s["card_bdr"]};border-radius:16px;'
            f'padding:24px;box-shadow:0 4px 30px rgba(0,0,0,0.2);margin-bottom:24px;">'
            f'<h3 style="color:#f8fafc;font-size:1.35rem;font-weight:600;margin-top:0;margin-bottom:12px;">'
            f'🎯 Accumulation Strategy Guide{etf_badge}</h3>'
            f'{guide_html}'
            f'{forecast_explainer_html}'
            f'</div>',
            unsafe_allow_html=True)

        # ── Metrics ───────────────────────────────────────────────────────────
        latest_close = float(close_series.iloc[-1])
        if len(df) > 1:
            prev  = float(close_series.iloc[-2])
            dv    = latest_close-prev
            dp    = dv/prev*100
            dstr  = f"${dv:+,.2f} ({dp:+.2f}%)"
        else:
            dstr = "N/A"
        avg_vol = int(df['Volume'].mean())
        ann_vol = float(close_series.pct_change().dropna().std()*np.sqrt(252)*100)

        m1,m2,m3,m4,m5 = st.columns(5)
        with m1: st.metric(f"Latest Close ({df.index[-1].strftime('%b %d')})", f"${latest_close:,.2f}", dstr)
        with m2: st.metric("Period High",  f"${float(df['High'].max()):,.2f}")
        with m3: st.metric("Period Low",   f"${float(df['Low'].min()):,.2f}")
        with m4: st.metric("Avg Daily Vol",
                           f"{avg_vol/1_000_000:.1f}M" if avg_vol>=1_000_000 else f"{avg_vol/1_000:.0f}K")
        with m5: st.metric("Ann. Volatility", f"{ann_vol:.1f}%")

        # ── Chart ─────────────────────────────────────────────────────────────
        show_rsi = show_rsi_indicator and 'RSI' in df.columns
        num_rows = 3 if show_rsi else 2
        row_h    = [0.55,0.15,0.30] if show_rsi else [0.70,0.30]

        fig = make_subplots(rows=num_rows, cols=1, shared_xaxes=True,
                            vertical_spacing=0.04, row_heights=row_h)
        up_c, dn_c = '#10b981','#ef4444'

        fig.add_trace(go.Candlestick(x=df.index,open=df['Open'],high=df['High'],
            low=df['Low'],close=df['Close'],name="Price",
            increasing_line_color=up_c,decreasing_line_color=dn_c,
            increasing_fillcolor=up_c,decreasing_fillcolor=dn_c), row=1,col=1)

        if show_ema and 'EMA' in df.columns:
            fig.add_trace(go.Scatter(x=df.index,y=df['EMA'],name=f"EMA ({ema_period})",
                line=dict(color='#facc15',width=1.5),mode='lines'), row=1,col=1)
        if show_sma and 'SMA' in df.columns:
            fig.add_trace(go.Scatter(x=df.index,y=df['SMA'],name=f"SMA ({sma_period})",
                line=dict(color='#22d3ee',width=1.5),mode='lines'), row=1,col=1)

        if has_bb:
            bbl_c = [c for c in bb.columns if c.startswith('BBL')][0]
            bbm_c = [c for c in bb.columns if c.startswith('BBM')][0]
            bbu_c = [c for c in bb.columns if c.startswith('BBU')][0]
            fig.add_trace(go.Scatter(x=df.index,y=bb[bbu_c],name='BB Upper',
                line=dict(color='rgba(99,102,241,0.5)',width=1,dash='dot'),mode='lines'),row=1,col=1)
            fig.add_trace(go.Scatter(x=df.index,y=bb[bbm_c],name='BB Mid',
                line=dict(color='rgba(99,102,241,0.3)',width=1),mode='lines'),row=1,col=1)
            fig.add_trace(go.Scatter(x=df.index,y=bb[bbl_c],fill='tonexty',
                fillcolor='rgba(99,102,241,0.04)',name='BB Lower',
                line=dict(color='rgba(99,102,241,0.5)',width=1,dash='dot'),mode='lines'),row=1,col=1)
        if has_sma50:
            fig.add_trace(go.Scatter(x=df.index,y=sma50,name='50-SMA',
                line=dict(color='rgba(148,163,184,0.45)',width=1.2,dash='dot'),mode='lines'),row=1,col=1)
        if is_etf_ticker:
            sma200_chart = ta.sma(close_series, length=min(200,len(close_series)-1))
            if sma200_chart is not None and not sma200_chart.empty:
                fig.add_trace(go.Scatter(x=df.index,y=sma200_chart,name='200-SMA',
                    line=dict(color='rgba(251,191,36,0.55)',width=1.5,dash='dot'),mode='lines'),row=1,col=1)

        if show_forecast and not forecast_df.empty:
            fig.add_trace(go.Scatter(x=forecast_df['ds'],y=forecast_df['yhat_upper'],
                line=dict(width=0),mode='lines',showlegend=False),row=1,col=1)
            fig.add_trace(go.Scatter(x=forecast_df['ds'],y=forecast_df['yhat_lower'],
                line=dict(width=0),fill='tonexty',fillcolor='rgba(255,120,73,0.12)',
                name='80% CI',showlegend=False),row=1,col=1)
            fig.add_trace(go.Scatter(x=forecast_df['ds'],y=forecast_df['yhat'],
                name="14d Forecast",line=dict(color='#ff7849',width=2.2,dash='dash'),mode='lines'),row=1,col=1)

        vol_colors = [up_c if c>=o else dn_c for o,c in zip(df['Open'],df['Close'])]
        fig.add_trace(go.Bar(x=df.index,y=df['Volume'],name="Volume",
                             marker_color=vol_colors,opacity=0.85),row=2,col=1)
        if show_rsi:
            fig.add_trace(go.Scatter(x=df.index,y=df['RSI'],name=f"RSI ({rsi_period})",
                line=dict(color='#a855f7',width=1.8),mode='lines'),row=3,col=1)
            for lvl,clr in [(70,'#ef4444'),(30,'#10b981')]:
                fig.add_shape(type="line",x0=df.index[0],x1=df.index[-1],y0=lvl,y1=lvl,
                              line=dict(color=clr,width=1.5,dash='dash'),row=3,col=1)

        fig.update_layout(
            plot_bgcolor='rgba(15,23,42,0.7)',paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0',family='Outfit'),
            margin=dict(t=15,b=15,l=15,r=15),
            xaxis=dict(showgrid=True,gridcolor='rgba(255,255,255,0.05)',zeroline=False,
                       rangeslider=dict(visible=False)),
            yaxis=dict(title="Price ($)",showgrid=True,gridcolor='rgba(255,255,255,0.05)',
                       zeroline=False,tickprefix="$"),
            xaxis2=dict(showgrid=True,gridcolor='rgba(255,255,255,0.05)',zeroline=False),
            yaxis2=dict(title="Volume",showgrid=True,gridcolor='rgba(255,255,255,0.05)',zeroline=False),
            showlegend=True,
            legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1,
                        font=dict(size=11,color='#e2e8f0'),
                        bgcolor='rgba(15,23,42,0.5)',
                        bordercolor='rgba(255,255,255,0.05)',borderwidth=1),
            height=750 if show_rsi else 620,
        )
        if show_rsi:
            fig.update_layout(
                xaxis3=dict(showgrid=True,gridcolor='rgba(255,255,255,0.05)',zeroline=False,title="Date"),
                yaxis3=dict(title="RSI",showgrid=True,gridcolor='rgba(255,255,255,0.05)',
                            zeroline=False,range=[10,90],tickvals=[30,50,70]))
        else:
            fig.update_layout(xaxis2=dict(title="Date"))

        st.plotly_chart(fig, use_container_width=True)

        # ── Forecast Insights Card ────────────────────────────────────────────
        if show_forecast and not forecast_df.empty:
            last_row  = forecast_df.iloc[-1]
            fin_price = float(last_row['yhat'])
            fin_upper = float(last_row['yhat_upper'])
            fin_lower = float(last_row['yhat_lower'])
            fin_date  = last_row['ds'].strftime('%Y-%m-%d')
            exp_chg   = (fin_price-latest_close)/latest_close*100
            g_col     = "#10b981" if exp_chg>=0 else "#ef4444"
            dsym      = "▲" if exp_chg>=0 else "▼"
            st.markdown(
                f'<div style="background:linear-gradient(135deg,rgba(30,41,59,0.7),rgba(15,23,42,0.8));'
                f'border:1px solid rgba(255,120,73,0.35);border-radius:16px;padding:24px;margin-top:20px;">'
                f'<h3 style="color:#ff7849;margin-bottom:15px;font-weight:600;">🔮 Forecasting Insights (14-Day Horizon)</h3>'
                f'<p style="color:#94a3b8;font-size:0.93rem;margin-bottom:20px;">'
                f'Exponential Smoothing (additive trend) on {len(close_series)} observations. '
                f'Confidence intervals: 1.28σ × √h ≈ 80% coverage.</p>'
                f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:18px;">'
                f'<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);padding:15px;border-radius:12px;">'
                f'<span style="color:#94a3b8;font-size:.82rem;text-transform:uppercase;letter-spacing:.05em;">Projected Target</span><br/>'
                f'<strong style="color:#f8fafc;font-size:1.8rem;">${fin_price:,.2f}</strong><br/>'
                f'<span style="color:#64748b;font-size:.78rem;">Target Date: {fin_date}</span></div>'
                f'<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);padding:15px;border-radius:12px;">'
                f'<span style="color:#94a3b8;font-size:.82rem;text-transform:uppercase;letter-spacing:.05em;">Expected Trend</span><br/>'
                f'<strong style="color:{g_col};font-size:1.8rem;">{dsym} {exp_chg:+.2f}%</strong><br/>'
                f'<span style="color:#64748b;font-size:.78rem;">From close ${latest_close:,.2f}</span></div>'
                f'<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);padding:15px;border-radius:12px;">'
                f'<span style="color:#94a3b8;font-size:.82rem;text-transform:uppercase;letter-spacing:.05em;">80% Confidence Range</span><br/>'
                f'<strong style="color:#f8fafc;font-size:1.35rem;line-height:2.2;">${fin_lower:,.2f} – ${fin_upper:,.2f}</strong><br/>'
                f'<span style="color:#64748b;font-size:.78rem;">Statistical lower & upper bounds</span></div>'
                f'</div></div>',
                unsafe_allow_html=True)

        # ── Fundamentals ──────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🏢 Fundamentals Research Panel")
        info, fund_err = fetch_company_info(ticker)
        if fund_err:
            st.warning(f"⚠️ {fund_err}")
        elif info:
            mc   = info.get('marketCap')
            pe   = info.get('trailingPE') or info.get('forwardPE')
            pb   = info.get('priceToBook')
            dy   = info.get('dividendYield')
            summ = info.get('longBusinessSummary','No summary available.')
            name = info.get('longName', ticker)
            sect = info.get('sector','N/A')
            ind  = info.get('industry','N/A')
            h52  = info.get('fiftyTwoWeekHigh')
            l52  = info.get('fiftyTwoWeekLow')

            mc_s = (f"${mc/1e12:.2f}T" if mc and mc>=1e12 else
                    f"${mc/1e9:.2f}B"  if mc and mc>=1e9  else
                    f"${mc/1e6:.2f}M"  if mc              else "N/A")
            pe_s = f"{pe:.2f}"      if pe else "N/A"
            pb_s = f"{pb:.2f}"      if pb else "N/A"
            dy_s = f"{dy*100:.2f}%" if dy else "N/A"

            sentences  = re.split(r'(?<=[.!?]) +', summ)
            short_summ = " ".join(sentences[:2])

            range_bar = ""
            if h52 and l52 and h52 > l52:
                pp = max(0.0, min(100.0, (current_price-l52)/(h52-l52)*100))
                bc = "#10b981" if pp<40 else ("#fbbf24" if pp<70 else "#ef4444")
                range_bar = (
                    f'<div style="margin-top:12px;">'
                    f'<div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#64748b;margin-bottom:5px;">'
                    f'<span>52W Low ${l52:,.2f}</span><span>52W High ${h52:,.2f}</span></div>'
                    f'<div style="background:rgba(255,255,255,0.07);border-radius:999px;height:6px;overflow:hidden;">'
                    f'<div style="width:{pp:.0f}%;background:{bc};height:100%;border-radius:999px;"></div></div>'
                    f'<div style="font-size:0.75rem;color:#94a3b8;margin-top:4px;">'
                    f'Currently at <strong>{pp:.0f}%</strong> of 52-week range</div></div>'
                )

            def _fund_cell(label, val):
                return (f'<div style="background:rgba(255,255,255,0.02);padding:12px;border-radius:8px;">'
                        f'<span style="color:#64748b;font-size:.78rem;text-transform:uppercase;">{label}</span><br/>'
                        f'<strong style="color:#e2e8f0;font-size:1.15rem;">{val}</strong></div>')

            st.markdown(
                f'<div style="background:linear-gradient(135deg,rgba(30,41,59,0.7),rgba(15,23,42,0.8));'
                f'border:1px solid rgba(99,102,241,0.2);border-radius:16px;padding:24px;margin-top:10px;">'
                f'<h4 style="color:#f8fafc;margin-bottom:12px;font-weight:600;">{name} ({ticker})</h4>'
                f'<p style="color:#94a3b8;font-size:0.93rem;line-height:1.65;margin-bottom:4px;">'
                f'<strong>Business:</strong> {short_summ}</p>'
                f'{range_bar}'
                f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:12px;margin-top:16px;">'
                f'{_fund_cell("Market Cap", mc_s)}'
                f'{_fund_cell("P/E Ratio", pe_s)}'
                f'{_fund_cell("P/B Ratio", pb_s)}'
                f'{_fund_cell("Dividend Yield", dy_s)}'
                f'{_fund_cell("Sector", sect)}'
                f'{_fund_cell("Industry", ind)}'
                f'</div></div>',
                unsafe_allow_html=True)
