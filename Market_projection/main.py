import datetime
import re
import pandas as pd
import streamlit as st
import yfinance as yf
import pandas_ta as ta
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Research & Projection App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Outfit', sans-serif; }

    .status-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 30px rgba(0,0,0,0.2);
        backdrop-filter: blur(8px);
        margin-top: 20px;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        background: rgba(16, 185, 129, 0.1);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.2);
        padding: 6px 12px;
        border-radius: 9999px;
        font-weight: 500;
        font-size: 0.875rem;
        gap: 8px;
    }

    .status-pulse {
        width: 8px; height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 0 0 rgba(16,185,129,0.7);
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%   { transform: scale(0.95); box-shadow: 0 0 0 0   rgba(16,185,129,0.7); }
        70%  { transform: scale(1);    box-shadow: 0 0 0 6px rgba(16,185,129,0);   }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0   rgba(16,185,129,0);   }
    }

    .hero-title {
        background: linear-gradient(to right, #6366f1, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.75rem;
        margin-bottom: 0.25rem;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.125rem;
        margin-bottom: 1.5rem;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #f8fafc !important;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(30,41,59,0.4), rgba(15,23,42,0.5)) !important;
        border: 1px solid rgba(99,102,241,0.15) !important;
        border-radius: 12px !important;
        padding: 12px 18px !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1) !important;
    }

    /* News items in sidebar */
    .news-item {
        padding: 8px 0;
        border-bottom: 1px solid rgba(255,255,255,0.07);
        font-size: 0.82rem;
        line-height: 1.45;
    }
    .news-item:last-child { border-bottom: none; }
    .news-item a {
        color: #a5b4fc;
        text-decoration: none;
    }
    .news-item a:hover { color: #e0e7ff; text-decoration: underline; }
    .news-age { color: #475569; font-size: 0.75rem; margin-top: 2px; }

    /* Earnings countdown card */
    .earnings-card {
        background: linear-gradient(135deg, rgba(245,158,11,0.1), rgba(234,179,8,0.07));
        border: 1px solid rgba(245,158,11,0.35);
        border-radius: 12px;
        padding: 14px 16px;
        margin-top: 12px;
        text-align: center;
    }
    .earnings-days {
        font-size: 2rem;
        font-weight: 700;
        color: #fbbf24;
        line-height: 1.1;
    }
    .earnings-label { color: #94a3b8; font-size: 0.8rem; margin-top: 2px; }
    .earnings-warning { color: #fcd34d; font-size: 0.78rem; margin-top: 6px; font-style: italic; }

    /* Strategy bullet points */
    .strategy-bullets li {
        color: #cbd5e1;
        font-size: 0.95rem;
        line-height: 1.7;
        margin-bottom: 4px;
    }
    .strategy-bullets li::marker { color: #6366f1; }
    .strategy-bullets strong { color: #f1f5f9; }
    .strategy-highlight {
        display: inline-block;
        background: rgba(99,102,241,0.15);
        border: 1px solid rgba(99,102,241,0.25);
        padding: 1px 7px;
        border-radius: 6px;
        font-weight: 600;
        color: #a5b4fc;
        font-size: 0.92em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Helper Utilities ──────────────────────────────────────────────────────────

def _fmt_age(pub_ts):
    """Return a human-readable age string from a Unix timestamp or datetime."""
    try:
        if isinstance(pub_ts, (int, float)):
            pub_dt = datetime.datetime.utcfromtimestamp(pub_ts)
        else:
            pub_dt = pub_ts
        delta = datetime.datetime.utcnow() - pub_dt
        hours = int(delta.total_seconds() // 3600)
        if hours < 1:
            return "just now"
        if hours < 24:
            return f"{hours}h ago"
        days = hours // 24
        return f"{days}d ago"
    except Exception:
        return ""


def _days_until(target_date):
    """Days from today until target_date (datetime.date)."""
    today = datetime.date.today()
    return (target_date - today).days


# ─── Cached Data Loaders ───────────────────────────────────────────────────────

@st.cache_data(show_spinner="Fetching market data…")
def load_stock_data(ticker_symbol: str, start: datetime.date, end: datetime.date):
    try:
        df = yf.download(ticker_symbol, start=start, end=end, progress=False)
        if df.empty:
            return pd.DataFrame(), f"No historical data returned for '{ticker_symbol}'."
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col not in df.columns:
                return pd.DataFrame(), f"Missing required column: '{col}'."
        return df.sort_index(), None
    except Exception as e:
        return pd.DataFrame(), f"Failed to download data: {e}"


@st.cache_data(show_spinner="Fetching company profile…")
def fetch_company_info(ticker_symbol: str):
    try:
        info = yf.Ticker(ticker_symbol).info
        if not info or not isinstance(info, dict):
            return None, f"No profile for '{ticker_symbol}'."
        return info, None
    except Exception as e:
        return None, f"Profile fetch error: {e}"


@st.cache_data(show_spinner="Fetching news headlines…", ttl=900)
def fetch_news(ticker_symbol: str):
    """Return up to 5 recent news items: list of dicts with title, link, providerPublishTime."""
    try:
        ticker_obj = yf.Ticker(ticker_symbol)
        news = ticker_obj.news or []
        results = []
        for item in news[:5]:
            title = item.get('title', '')
            link  = item.get('link', '#')
            ts    = item.get('providerPublishTime', None)
            if title:
                results.append({'title': title, 'link': link, 'ts': ts})
        return results, None
    except Exception as e:
        return [], f"News fetch error: {e}"


@st.cache_data(show_spinner="Looking up earnings date…", ttl=3600)
def fetch_earnings_date(ticker_symbol: str):
    """Return next earnings date as datetime.date or None."""
    try:
        ticker_obj = yf.Ticker(ticker_symbol)
        cal = ticker_obj.calendar
        if cal is None:
            return None, None

        # calendar can be a dict or a DataFrame depending on yfinance version
        if isinstance(cal, dict):
            raw = cal.get('Earnings Date') or cal.get('earningsDate')
            if raw:
                if isinstance(raw, (list, tuple)) and len(raw) > 0:
                    raw = raw[0]
                if hasattr(raw, 'date'):
                    return raw.date(), None
                return pd.Timestamp(raw).date(), None

        elif isinstance(cal, pd.DataFrame):
            if 'Earnings Date' in cal.index:
                raw = cal.loc['Earnings Date'].values[0]
                return pd.Timestamp(raw).date(), None

        # Fallback: earnings_dates table
        ed = ticker_obj.earnings_dates
        if ed is not None and not ed.empty:
            future = ed[ed.index > pd.Timestamp.now()]
            if not future.empty:
                return future.index[-1].date(), None

        return None, None
    except Exception as e:
        return None, f"Earnings date error: {e}"


@st.cache_data(show_spinner="Running forecast…")
def generate_statistical_forecast(series: pd.Series, horizon: int = 14):
    try:
        s = series.squeeze().astype(float)
        if len(s) < 10:
            return pd.DataFrame(), "Need at least 10 trading days."
        model = ExponentialSmoothing(s, trend="add", seasonal=None, initialization_method="estimated")
        fitted = model.fit()
        fcast  = fitted.forecast(steps=horizon)
        last_date    = series.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq='D')
        std_resid = fitted.resid.std() if not fitted.resid.empty else 0.0
        lower, upper = [], []
        for h in range(1, horizon + 1):
            margin = 1.28 * std_resid * np.sqrt(h)
            lower.append(fcast.values[h - 1] - margin)
            upper.append(fcast.values[h - 1] + margin)
        return pd.DataFrame({'ds': future_dates, 'yhat': fcast.values,
                             'yhat_lower': lower, 'yhat_upper': upper}), None
    except Exception as e:
        return pd.DataFrame(), f"Forecast failed: {e}"


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluent/96/000000/line-chart.png", width=64)
    st.markdown("### **Navigation & Inputs**")

    ticker = st.text_input("Stock Ticker Symbol", value="SCHG",
                           help="e.g. AAPL, MSFT, SCHG").upper().strip()

    today        = datetime.date.today()
    one_year_ago = today - datetime.timedelta(days=365)
    start_date   = st.date_input("Start Date", value=one_year_ago, max_value=today)
    end_date     = st.date_input("End Date",   value=today,        max_value=today)

    st.markdown("---")
    st.markdown("### 📊 **Indicators**")

    show_ema  = st.checkbox("Short-term EMA",  value=False)
    ema_period = st.number_input("EMA Period",  min_value=2, max_value=200, value=20, step=1, disabled=not show_ema)

    show_sma  = st.checkbox("Long-term SMA",   value=False)
    sma_period = st.number_input("SMA Period",  min_value=2, max_value=500, value=50, step=1, disabled=not show_sma)

    show_rsi_indicator = st.checkbox("RSI", value=False)
    rsi_period = st.number_input("RSI Period",  min_value=2, max_value=100, value=14, step=1, disabled=not show_rsi_indicator)

    st.markdown("---")
    st.markdown("### 🔮 **Forecast**")
    show_forecast    = st.checkbox("Enable 14-Day Forecast", value=False)
    forecast_horizon = 14

    # ── Earnings Countdown ────────────────────────────────────────────────────
    if ticker:
        st.markdown("---")
        st.markdown("### 📅 **Earnings Countdown**")
        earnings_date, earn_err = fetch_earnings_date(ticker)
        if earn_err:
            st.caption(f"⚠️ {earn_err}")
        elif earnings_date:
            days_away = _days_until(earnings_date)
            if days_away >= 0:
                urgency_color = "#ef4444" if days_away <= 7 else ("#fbbf24" if days_away <= 21 else "#34d399")
                warning_text  = ""
                if days_away <= 14:
                    warning_text = "⚠️ Earnings are near — expect volatility. Consider waiting until after the report to buy."
                elif days_away <= 30:
                    warning_text = "📌 Earnings approaching. Watch for pre-report price swings."
                st.markdown(
                    f"""
                    <div class="earnings-card">
                        <div style="color:#94a3b8; font-size:0.78rem; text-transform:uppercase; letter-spacing:.06em;">Next Earnings Report</div>
                        <div class="earnings-days" style="color:{urgency_color};">
                            {"Today" if days_away == 0 else f"In {days_away} Day{'s' if days_away != 1 else ''}"}
                        </div>
                        <div class="earnings-label">{earnings_date.strftime('%B %d, %Y')}</div>
                        {f'<div class="earnings-warning">{warning_text}</div>' if warning_text else ''}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.caption(f"Last reported: {earnings_date.strftime('%b %d, %Y')}")
        else:
            st.caption("No upcoming earnings date found.")

    # ── Live News ─────────────────────────────────────────────────────────────
    if ticker:
        st.markdown("---")
        st.markdown("### 📰 **Latest News**")
        news_items, news_err = fetch_news(ticker)
        if news_err:
            st.caption(f"⚠️ {news_err}")
        elif not news_items:
            st.caption("No recent headlines found.")
        else:
            news_html = ""
            for item in news_items:
                age = _fmt_age(item['ts']) if item['ts'] else ""
                news_html += f"""
                <div class="news-item">
                    <a href="{item['link']}" target="_blank">{item['title']}</a>
                    {"<div class='news-age'>" + age + "</div>" if age else ""}
                </div>
                """
            st.markdown(news_html, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("💡 *Powered by Python · Streamlit · yFinance · Pandas*")


# ─── Main Header ──────────────────────────────────────────────────────────────
h1, h2 = st.columns([3, 1])
with h1:
    st.markdown('<h1 class="hero-title">📈 Stock Research & Projection App</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Interactive stock market analysis, historical forecasting, and accumulation strategy insights.</p>', unsafe_allow_html=True)
with h2:
    st.markdown(
        '<div style="text-align:right;padding-top:15px;">'
        '<div class="status-badge"><span class="status-pulse"></span> Engine Active</div>'
        '</div>',
        unsafe_allow_html=True,
    )

# ─── Main Execution ───────────────────────────────────────────────────────────
if not ticker:
    st.warning("⚠️ Please enter a stock ticker symbol in the sidebar.")
else:
    df, data_error = load_stock_data(ticker, start_date, end_date)

    if data_error:
        st.error(f"❌ {data_error}")
    elif df.empty:
        st.error(f"❌ No data for '{ticker}' in the selected range.")
    else:
        close_series = df['Close'].squeeze()

        # ── Indicator Calculations ────────────────────────────────────────────
        active_periods = []
        if show_ema: active_periods.append(ema_period)
        if show_sma: active_periods.append(sma_period)
        if show_rsi_indicator: active_periods.append(rsi_period)

        max_period = max(active_periods) if active_periods else 0
        if max_period > len(df):
            st.warning(f"⚠️ Only {len(df)} trading days available; indicator period {max_period} may show NaN.")

        if show_ema: df['EMA'] = ta.ema(close_series, length=ema_period)
        if show_sma: df['SMA'] = ta.sma(close_series, length=sma_period)
        if show_rsi_indicator: df['RSI'] = ta.rsi(close_series, length=rsi_period)

        # ── Forecast ──────────────────────────────────────────────────────────
        forecast_df = pd.DataFrame()
        if show_forecast:
            forecast_raw, forecast_err = generate_statistical_forecast(close_series, forecast_horizon)
            if forecast_err:
                st.error(f"❌ Forecasting Error: {forecast_err}")
            else:
                forecast_df = forecast_raw

        # ── Accumulation Logic ────────────────────────────────────────────────
        bb      = ta.bbands(close_series, length=20, std=2)
        sma_50  = ta.sma(close_series, length=50)

        current_price  = float(close_series.iloc[-1])
        has_bb         = bb is not None and not bb.empty
        has_sma50      = sma_50 is not None and not sma_50.empty

        # Detect which signal fired
        signal_source  = None   # "bb" | "sma50" | None
        current_bbl    = None
        current_sma50  = None
        pct_below_sma  = None

        if has_bb:
            bbl_col     = [c for c in bb.columns if c.startswith('BBL')][0]
            bbu_col     = [c for c in bb.columns if c.startswith('BBU')][0]
            current_bbl = float(bb[bbl_col].iloc[-1])
            current_bbu = float(bb[bbu_col].iloc[-1])
            if current_price <= current_bbl * 1.015:
                signal_source = "bb"

        if signal_source is None and has_sma50:
            current_sma50 = float(sma_50.iloc[-1])
            if current_price <= current_sma50:
                signal_source = "sma50"
                pct_below_sma = ((current_sma50 - current_price) / current_sma50) * 100

        status_type = "buy" if signal_source else "hold"

        # ── Forecast Direction ────────────────────────────────────────────────
        forecast_is_bearish = False
        if show_forecast and not forecast_df.empty:
            if forecast_df['yhat'].iloc[-1] < forecast_df['yhat'].iloc[0]:
                forecast_is_bearish = True

        # ── Build Strategy Explainer HTML ─────────────────────────────────────
        def _highlight(val):
            return f'<span class="strategy-highlight">{val}</span>'

        if status_type == "buy" and signal_source == "bb":
            card_style    = "background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(5,150,105,0.22)); border: 1px solid rgba(16,185,129,0.45);"
            signal_header = '<strong style="color:#10b981;font-size:1.25rem;">🛒 BUY SIGNAL — Lower Bollinger Band Breach</strong>'
            bullets_html  = f"""
            <ul class="strategy-bullets" style="margin:14px 0 0 20px;padding:0;">
                <li>The current price of {_highlight("$" + f"{current_price:,.2f}")} has touched or dropped through the
                    <strong>Lower Bollinger Band</strong> ({_highlight("$" + f"{current_bbl:,.2f}")}), which acts as
                    a statistically-derived <em>price floor</em> — calculated as 2 standard deviations below the 20-day average.</li>
                <li>Think of this band like a <strong>rubber band being stretched</strong>: the further the price deviates
                    downward from the mean, the stronger the mathematical tendency to snap back toward fair value.</li>
                <li>Historically, a breach of the lower Bollinger Band signals that the stock is
                    <strong>temporarily oversold</strong> — sellers have pushed the price to an extreme and
                    buying pressure typically builds from here.</li>
                <li>This is a <strong>high-probability value zone</strong>. If your thesis on the company is intact,
                    this is exactly the kind of discount a disciplined accumulator should be targeting.</li>
                <li><strong>Recommended action:</strong> Begin accumulating a partial position. You don't have to go
                    all-in — scaling in across 2–3 sessions reduces timing risk.</li>
            </ul>
            """

        elif status_type == "buy" and signal_source == "sma50":
            current_sma50_fmt = f"{current_sma50:,.2f}" if current_sma50 else "N/A"
            pct_str           = f"{pct_below_sma:.1f}" if pct_below_sma else "?"
            card_style    = "background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(5,150,105,0.22)); border: 1px solid rgba(16,185,129,0.45);"
            signal_header = '<strong style="color:#10b981;font-size:1.25rem;">🛒 BUY SIGNAL — Below 50-Day Moving Average</strong>'
            bullets_html  = f"""
            <ul class="strategy-bullets" style="margin:14px 0 0 20px;padding:0;">
                <li>The stock is currently trading at {_highlight("$" + f"{current_price:,.2f}")} — that is
                    {_highlight(pct_str + "% below")} its 50-day Simple Moving Average of
                    {_highlight("$" + current_sma50_fmt)}.</li>
                <li>The <strong>50-day SMA</strong> represents the market's consensus on fair value over the medium term.
                    When a fundamentally healthy company dips below it, you're effectively
                    <strong>cutting out the short-term hype premium</strong> and buying at a structural discount.</li>
                <li>This type of pullback is <em>normal and expected</em> in healthy bull trends — it shakes out
                    weak hands and reloads the stock for its next leg upward.</li>
                <li>The further a quality stock falls below its 50-day average, the <strong>wider your margin of safety</strong>
                    — and at {_highlight(pct_str + "%")} below, that margin is meaningful.</li>
                <li><strong>Recommended action:</strong> This is a discounted entry. Consider building a position here,
                    with a plan to add more if the price extends further below the average.</li>
            </ul>
            """

        else:  # HOLD
            card_style    = "background: linear-gradient(135deg, rgba(71,85,105,0.12), rgba(51,65,85,0.22)); border: 1px solid rgba(148,163,184,0.3);"
            signal_header = '<strong style="color:#94a3b8;font-size:1.25rem;">⏳ HOLD — Stock Trading at a Premium</strong>'
            upper_ref     = _highlight("$" + f"{current_bbu:,.2f}") if has_bb and current_bbu else ""
            sma_ref       = _highlight("$" + f"{float(sma_50.iloc[-1]):,.2f}") if has_sma50 else ""
            bullets_html  = f"""
            <ul class="strategy-bullets" style="margin:14px 0 0 20px;padding:0;">
                <li>At the current price of {_highlight("$" + f"{current_price:,.2f}")}, the stock is trading
                    <strong>above both its 50-day moving average</strong>{(" (" + sma_ref + ")") if sma_ref else ""}
                    and near the <strong>upper Bollinger Band</strong>{(" (" + upper_ref + ")") if upper_ref else ""}.
                    You are in <em>premium territory</em>.</li>
                <li>Buying here means paying a price that already reflects optimism and recent momentum —
                    <strong>you're not getting a deal; you're paying full price at peak excitement.</strong></li>
                <li>Markets breathe. Even the strongest stocks pull back 5–15% before continuing higher.
                    Your goal as a disciplined accumulator is to <strong>buy those dips, not chase the peaks.</strong></li>
                <li>There is no urgency. Cash is a position. <strong>Patience is the edge</strong> that separates
                    long-term wealth builders from emotional traders who buy high and sell low.</li>
                <li><strong>Recommended action:</strong> Set a price alert at the 50-day SMA
                    {("(" + sma_ref + ") ") if sma_ref else ""}or lower Bollinger Band
                    {("(" + _highlight("$" + f"{current_bbl:,.2f}") + ") ") if has_bb and current_bbl else ""}
                    and wait for the market to come to you.</li>
            </ul>
            """

        # Forecast dip warning block
        forecast_explainer_html = ""
        if forecast_is_bearish:
            f_start = float(forecast_df['yhat'].iloc[0])
            f_end   = float(forecast_df['yhat'].iloc[-1])
            drop_pct = ((f_start - f_end) / f_start) * 100
            forecast_explainer_html = f"""
            <div style="margin-top:18px; padding:14px 18px;
                        background:rgba(245,158,11,0.08); border:1px dashed rgba(245,158,11,0.4);
                        border-radius:10px;">
                <p style="color:#fbbf24; font-size:1rem; font-weight:600; margin:0 0 8px 0;">
                    🔮 Why the Dip Warning?
                </p>
                <p style="color:#cbd5e1; font-size:0.93rem; line-height:1.65; margin:0;">
                    The mathematical momentum embedded in the last several weeks of closing prices shows a
                    <strong style="color:#fcd34d;">short-term downward trajectory</strong> — a cooling-off period
                    that the Exponential Smoothing model projects will continue over the next 14 days
                    (an estimated drop of ~{_highlight(f"{drop_pct:.1f}%")} from the current level).
                    <br><br>
                    This is not a crash signal — it's a <strong>reversion to the mean in motion</strong>.
                    Think of it as the stock exhaling after a run. If you're sitting on cash and watching for
                    a buy entry, <strong>exercise patience</strong>: an even better price window may open up
                    over the next fortnight. Let the dip come to you.
                </p>
            </div>
            """

        # Render the full Accumulation Strategy Guide card
        st.markdown(
            f"""
            <div class="status-card" style="margin-top:5px; margin-bottom:25px; {card_style}">
                <h3 style="color:#f8fafc; font-size:1.4rem; font-weight:600; margin-top:0; margin-bottom:10px;">
                    🎯 Accumulation Strategy Guide
                </h3>
                {signal_header}
                {bullets_html}
                {forecast_explainer_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Key Metrics ───────────────────────────────────────────────────────
        latest_close = float(close_series.iloc[-1])
        if len(df) > 1:
            prev_close = float(close_series.iloc[-2])
            delta_val  = latest_close - prev_close
            delta_pct  = (delta_val / prev_close) * 100
            delta_str  = f"${delta_val:+,.2f} ({delta_pct:+.2f}%)"
        else:
            delta_str = "N/A"

        period_high = float(df['High'].max())
        period_low  = float(df['Low'].min())

        # Additional metrics: avg volume and annualized volatility
        avg_vol   = int(df['Volume'].mean())
        daily_ret = close_series.pct_change().dropna()
        ann_vol   = float(daily_ret.std() * np.sqrt(252) * 100) if len(daily_ret) > 1 else 0.0

        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric(label=f"Latest Close ({df.index[-1].strftime('%b %d')})",
                      value=f"${latest_close:,.2f}", delta=delta_str)
        with m2:
            st.metric(label="Period High", value=f"${period_high:,.2f}")
        with m3:
            st.metric(label="Period Low",  value=f"${period_low:,.2f}")
        with m4:
            st.metric(label="Avg Daily Volume",
                      value=f"{avg_vol / 1_000_000:.1f}M" if avg_vol >= 1_000_000 else f"{avg_vol / 1_000:.0f}K")
        with m5:
            st.metric(label="Ann. Volatility", value=f"{ann_vol:.1f}%")

        # ── Chart ─────────────────────────────────────────────────────────────
        show_rsi = show_rsi_indicator and 'RSI' in df.columns

        num_rows    = 3 if show_rsi else 2
        row_heights = [0.55, 0.15, 0.30] if show_rsi else [0.70, 0.30]

        fig = make_subplots(rows=num_rows, cols=1, shared_xaxes=True,
                            vertical_spacing=0.04, row_heights=row_heights)

        up_color, dn_color = '#10b981', '#ef4444'

        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'], name="Price",
            increasing_line_color=up_color, decreasing_line_color=dn_color,
            increasing_fillcolor=up_color, decreasing_fillcolor=dn_color
        ), row=1, col=1)

        if show_ema and 'EMA' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA'],
                name=f"EMA ({ema_period})", line=dict(color='#facc15', width=1.5), mode='lines'), row=1, col=1)

        if show_sma and 'SMA' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA'],
                name=f"SMA ({sma_period})", line=dict(color='#22d3ee', width=1.5), mode='lines'), row=1, col=1)

        # Always overlay Bollinger Bands (thin, subtle) for context
        if has_bb:
            bbl_col = [c for c in bb.columns if c.startswith('BBL')][0]
            bbm_col = [c for c in bb.columns if c.startswith('BBM')][0]
            bbu_col = [c for c in bb.columns if c.startswith('BBU')][0]
            fig.add_trace(go.Scatter(x=df.index, y=bb[bbu_col],
                name='BB Upper', line=dict(color='rgba(99,102,241,0.5)', width=1, dash='dot'), mode='lines'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=bb[bbm_col],
                name='BB Mid', line=dict(color='rgba(99,102,241,0.3)', width=1), mode='lines'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=bb[bbl_col], fill='tonexty',
                fillcolor='rgba(99,102,241,0.04)',
                name='BB Lower', line=dict(color='rgba(99,102,241,0.5)', width=1, dash='dot'), mode='lines'), row=1, col=1)

        # Always overlay 50-SMA (thin grey) for reference
        if has_sma50:
            fig.add_trace(go.Scatter(x=df.index, y=sma_50,
                name='50-SMA (ref)', line=dict(color='rgba(148,163,184,0.45)', width=1.2, dash='dot'), mode='lines'), row=1, col=1)

        if show_forecast and not forecast_df.empty:
            fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_upper'],
                line=dict(width=0), mode='lines', showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_lower'],
                line=dict(width=0), fill='tonexty', fillcolor='rgba(255,120,73,0.12)',
                name='80% CI', showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'],
                name="14d Forecast", line=dict(color='#ff7849', width=2.2, dash='dash'), mode='lines'), row=1, col=1)

        vol_colors = [up_color if c >= o else dn_color for o, c in zip(df['Open'], df['Close'])]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume",
                             marker_color=vol_colors, opacity=0.85), row=2, col=1)

        if show_rsi:
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'],
                name=f"RSI ({rsi_period})", line=dict(color='#a855f7', width=1.8), mode='lines'), row=3, col=1)
            for level, color in [(70, '#ef4444'), (30, '#10b981')]:
                fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1],
                              y0=level, y1=level, line=dict(color=color, width=1.5, dash='dash'), row=3, col=1)

        base_layout = dict(
            plot_bgcolor='rgba(15,23,42,0.7)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0', family='Outfit'),
            margin=dict(t=15, b=15, l=15, r=15),
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)',
                       zeroline=False, rangeslider=dict(visible=False)),
            yaxis=dict(title="Price ($)", showgrid=True,
                       gridcolor='rgba(255,255,255,0.05)', zeroline=False, tickprefix="$"),
            xaxis2=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False),
            yaxis2=dict(title="Volume", showgrid=True,
                        gridcolor='rgba(255,255,255,0.05)', zeroline=False),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        font=dict(size=11, color='#e2e8f0'),
                        bgcolor='rgba(15,23,42,0.5)',
                        bordercolor='rgba(255,255,255,0.05)', borderwidth=1),
            height=750 if show_rsi else 620,
        )
        fig.update_layout(**base_layout)

        if show_rsi:
            fig.update_layout(
                xaxis3=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)',
                            zeroline=False, title="Date"),
                yaxis3=dict(title="RSI", showgrid=True, gridcolor='rgba(255,255,255,0.05)',
                            zeroline=False, range=[10, 90], tickvals=[30, 50, 70])
            )
        else:
            fig.update_layout(xaxis2=dict(title="Date"))

        st.plotly_chart(fig, use_container_width=True)

        # ── Forecast Insights Card ────────────────────────────────────────────
        if show_forecast and not forecast_df.empty:
            last_row     = forecast_df.iloc[-1]
            final_date   = last_row['ds'].strftime('%Y-%m-%d')
            final_price  = float(last_row['yhat'])
            final_upper  = float(last_row['yhat_upper'])
            final_lower  = float(last_row['yhat_lower'])
            exp_change   = ((final_price - latest_close) / latest_close) * 100
            grow_color   = "#10b981" if exp_change >= 0 else "#ef4444"
            dir_sym      = "▲" if exp_change >= 0 else "▼"

            st.markdown(
                f"""
                <div class="status-card" style="margin-top:25px; border:1px solid rgba(255,120,73,0.35);">
                    <h3 style="color:#ff7849; margin-bottom:15px; font-weight:600;">
                        🔮 Statistical Forecasting Insights (14-Day Horizon)
                    </h3>
                    <p style="color:#94a3b8; font-size:0.95rem; margin-bottom:20px;">
                        Exponential Smoothing (additive trend) fitted on {len(close_series)} closing-price observations.
                        Confidence intervals reflect 1.28σ × √h propagation of residual variance (≈80% coverage).
                    </p>
                    <div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:20px;">
                        <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.05); padding:15px; border-radius:12px;">
                            <span style="color:#94a3b8; font-size:.85rem; text-transform:uppercase; letter-spacing:.05em;">Projected Target</span><br/>
                            <strong style="color:#f8fafc; font-size:1.8rem;">${final_price:,.2f}</strong><br/>
                            <span style="color:#64748b; font-size:.8rem;">Target Date: {final_date}</span>
                        </div>
                        <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.05); padding:15px; border-radius:12px;">
                            <span style="color:#94a3b8; font-size:.85rem; text-transform:uppercase; letter-spacing:.05em;">Expected Trend</span><br/>
                            <strong style="color:{grow_color}; font-size:1.8rem;">{dir_sym} {exp_change:+.2f}%</strong><br/>
                            <span style="color:#64748b; font-size:.8rem;">From current close ${latest_close:,.2f}</span>
                        </div>
                        <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.05); padding:15px; border-radius:12px;">
                            <span style="color:#94a3b8; font-size:.85rem; text-transform:uppercase; letter-spacing:.05em;">80% Confidence Range</span><br/>
                            <strong style="color:#f8fafc; font-size:1.4rem; line-height:2.2;">${final_lower:,.2f} – ${final_upper:,.2f}</strong><br/>
                            <span style="color:#64748b; font-size:.8rem;">Statistical lower & upper bounds</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Fundamentals Panel ────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🏢 **Fundamentals Research Panel**")

        info, fund_err = fetch_company_info(ticker)
        if fund_err:
            st.warning(f"⚠️ {fund_err}")
        elif info:
            market_cap   = info.get('marketCap')
            pe_ratio     = info.get('trailingPE') or info.get('forwardPE')
            pb_ratio     = info.get('priceToBook')
            div_yield    = info.get('dividendYield')
            summary      = info.get('longBusinessSummary', 'No business summary available.')
            company_name = info.get('longName', ticker)
            sector       = info.get('sector', 'N/A')
            industry     = info.get('industry', 'N/A')
            fifty2_high  = info.get('fiftyTwoWeekHigh')
            fifty2_low   = info.get('fiftyTwoWeekLow')

            if market_cap:
                if market_cap >= 1e12:
                    mc_str = f"${market_cap / 1e12:.2f}T"
                elif market_cap >= 1e9:
                    mc_str = f"${market_cap / 1e9:.2f}B"
                else:
                    mc_str = f"${market_cap / 1e6:.2f}M"
            else:
                mc_str = "N/A"

            pe_str  = f"{pe_ratio:.2f}"  if pe_ratio  else "N/A"
            pb_str  = f"{pb_ratio:.2f}"  if pb_ratio  else "N/A"
            div_str = f"{div_yield*100:.2f}%" if div_yield else "N/A"
            h52_str = f"${fifty2_high:,.2f}" if fifty2_high else "N/A"
            l52_str = f"${fifty2_low:,.2f}"  if fifty2_low  else "N/A"

            sentences       = re.split(r'(?<=[.!?]) +', summary)
            short_summary   = " ".join(sentences[:2])

            # 52-week position bar
            if fifty2_high and fifty2_low and fifty2_high > fifty2_low:
                pos_pct = ((current_price - fifty2_low) / (fifty2_high - fifty2_low)) * 100
                pos_pct = max(0.0, min(100.0, pos_pct))
                bar_color = "#10b981" if pos_pct < 40 else ("#fbbf24" if pos_pct < 70 else "#ef4444")
                range_bar = f"""
                <div style="margin-top:6px;">
                    <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#64748b; margin-bottom:4px;">
                        <span>52W Low {l52_str}</span><span>52W High {h52_str}</span>
                    </div>
                    <div style="background:rgba(255,255,255,0.07); border-radius:999px; height:6px; overflow:hidden;">
                        <div style="width:{pos_pct:.0f}%; background:{bar_color}; height:100%; border-radius:999px;"></div>
                    </div>
                    <div style="font-size:0.75rem; color:#94a3b8; margin-top:4px;">
                        Currently at <strong>{pos_pct:.0f}%</strong> of 52-week range
                    </div>
                </div>
                """
            else:
                range_bar = ""

            fundamentals_grid = f"""
            <div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(175px,1fr)); gap:14px; margin-top:16px;">
                <div style="background:rgba(255,255,255,0.02); padding:12px; border-radius:8px;">
                    <span style="color:#64748b; font-size:.78rem; text-transform:uppercase;">Market Cap</span><br/>
                    <strong style="color:#e2e8f0; font-size:1.15rem;">{mc_str}</strong>
                </div>
                <div style="background:rgba(255,255,255,0.02); padding:12px; border-radius:8px;">
                    <span style="color:#64748b; font-size:.78rem; text-transform:uppercase;">P/E Ratio</span><br/>
                    <strong style="color:#e2e8f0; font-size:1.15rem;">{pe_str}</strong>
                </div>
                <div style="background:rgba(255,255,255,0.02); padding:12px; border-radius:8px;">
                    <span style="color:#64748b; font-size:.78rem; text-transform:uppercase;">P/B Ratio</span><br/>
                    <strong style="color:#e2e8f0; font-size:1.15rem;">{pb_str}</strong>
                </div>
                <div style="background:rgba(255,255,255,0.02); padding:12px; border-radius:8px;">
                    <span style="color:#64748b; font-size:.78rem; text-transform:uppercase;">Dividend Yield</span><br/>
                    <strong style="color:#e2e8f0; font-size:1.15rem;">{div_str}</strong>
                </div>
                <div style="background:rgba(255,255,255,0.02); padding:12px; border-radius:8px;">
                    <span style="color:#64748b; font-size:.78rem; text-transform:uppercase;">Sector</span><br/>
                    <strong style="color:#e2e8f0; font-size:1.15rem;">{sector}</strong>
                </div>
                <div style="background:rgba(255,255,255,0.02); padding:12px; border-radius:8px;">
                    <span style="color:#64748b; font-size:.78rem; text-transform:uppercase;">Industry</span><br/>
                    <strong style="color:#e2e8f0; font-size:1.15rem;">{industry}</strong>
                </div>
            </div>
            """

            st.markdown(
                f"""
                <div class="status-card" style="margin-top:10px;">
                    <h4 style="color:#f8fafc; margin-bottom:12px; font-weight:600;">{company_name} ({ticker})</h4>
                    <p style="color:#94a3b8; font-size:0.93rem; line-height:1.65; margin-bottom:8px;">
                        <strong>Business:</strong> {short_summary}
                    </p>
                    {range_bar}
                    {fundamentals_grid}
                </div>
                """,
                unsafe_allow_html=True,
            )