import datetime
import pandas as pd
import streamlit as st
import yfinance as yf
import pandas_ta as ta
from prophet import Prophet
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Set up page configurations with wide layout
st.set_page_config(
    page_title="Stock Research & Projection App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for gorgeous styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Apply modern font */
    * {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Glowing card or live badge styles */
    .status-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
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
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        }
        70% {
            transform: scale(1);
            box-shadow: 0 0 0 6px rgba(16, 185, 129, 0);
        }
        100% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
        }
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
    
    /* Premium style override for Streamlit metrics */
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
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.4), rgba(15, 23, 42, 0.5)) !important;
        border: 1px solid rgba(99, 102, 241, 0.15) !important;
        border-radius: 12px !important;
        padding: 12px 18px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar setup
with st.sidebar:
    st.image("https://img.icons8.com/fluent/96/000000/line-chart.png", width=64)
    st.markdown("### **Navigation & Inputs**")
    
    # Stock Ticker input
    ticker = st.text_input("Stock Ticker Symbol", value="SCHG", help="Enter a valid exchange ticker symbol e.g., SCHG, AAPL, MSFT").upper().strip()
    
    # Date Pickers
    today = datetime.date(2026, 5, 26)  # Default current date based on system state
    one_year_ago = today - datetime.timedelta(days=365)
    
    start_date = st.date_input("Start Date", value=one_year_ago, max_value=today)
    end_date = st.date_input("End Date", value=today, max_value=today)
    
    # Technical Indicators section
    st.markdown("---")
    st.markdown("### 📊 **Indicators Configuration**")
    
    # EMA toggle & companion period input
    show_ema = st.checkbox("Short-term EMA", value=False, help="Exponential Moving Average calculated on closing prices.")
    ema_period = st.number_input("EMA Period", min_value=2, max_value=200, value=20, step=1, disabled=not show_ema)
    
    # SMA toggle & companion period input
    show_sma = st.checkbox("Long-term SMA", value=False, help="Simple Moving Average calculated on closing prices.")
    sma_period = st.number_input("SMA Period", min_value=2, max_value=500, value=50, step=1, disabled=not show_sma)
    
    # RSI toggle & companion period input
    show_rsi_indicator = st.checkbox("Relative Strength Index (RSI)", value=False, help="Momentum oscillator measuring velocity/magnitude of price movements.")
    rsi_period = st.number_input("RSI Period", min_value=2, max_value=100, value=14, step=1, disabled=not show_rsi_indicator)
    
    # Predictive Forecasting Engine section
    st.markdown("---")
    st.markdown("### 🔮 **Predictive Forecasting Engine**")
    
    show_forecast = st.checkbox("Enable Machine Learning Forecast", value=False, help="Utilize a local Prophet time-series model to project future price trends.")
    forecast_horizon = st.slider("Forecast Horizon (Days)", min_value=7, max_value=90, value=30, step=1, disabled=not show_forecast)
    
    st.markdown("---")
    st.markdown("💡 *Market forecasting & analytics engine powered by Python, Streamlit, yFinance, and Pandas.*")

# Cached stock data loader with validation & error handling
@st.cache_data(show_spinner="Fetching data from Yahoo Finance...")
def load_stock_data(ticker_symbol: str, start: datetime.date, end: datetime.date):
    try:
        # Fetch using yfinance download
        df = yf.download(ticker_symbol, start=start, end=end, progress=False)
        if df.empty:
            return pd.DataFrame()
        
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Ensure standard columns are present
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                return pd.DataFrame()
                
        # Sort index
        df = df.sort_index()
        return df
    except Exception as e:
        return pd.DataFrame()

# Cached Facebook Prophet forecasting logic
@st.cache_data(show_spinner="Training predictive Prophet model...")
def generate_forecast(df: pd.DataFrame, horizon: int):
    try:
        # Prophet expects 'ds' (datetimes) and 'y' (values)
        train_df = df[['Close']].reset_index()
        train_df.columns = ['ds', 'y']
        
        # Ensure ds is timezone-naive
        train_df['ds'] = pd.to_datetime(train_df['ds']).dt.tz_localize(None)
        
        # Fit Prophet model
        model = Prophet(
            yearly_seasonality='auto',
            weekly_seasonality='auto',
            daily_seasonality=False
        )
        model.fit(train_df)
        
        # Generate future dataframe
        future = model.make_future_dataframe(periods=horizon, freq='D')
        
        # Predict
        forecast = model.predict(future)
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    except Exception as e:
        return pd.DataFrame()

# Main Header Layout with Live Connection Badge aligned next to Title
header_col1, header_col2 = st.columns([3, 1])

with header_col1:
    st.markdown('<h1 class="hero-title">📈 Stock Research & Projection App</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Interactive stock market analysis, historical forecasting, and portfolio projections.</p>', unsafe_allow_html=True)

with header_col2:
    st.markdown(
        """
        <div style="text-align: right; padding-top: 15px;">
            <div class="status-badge">
                <span class="status-pulse"></span> Engine Active
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Execution Flow
if not ticker:
    st.warning("⚠️ Please enter a stock ticker symbol in the sidebar.")
else:
    # Load data
    df = load_stock_data(ticker, start_date, end_date)
    
    if df.empty:
        st.error(f"❌ Ticker '{ticker}' not found or no historical data available for the period {start_date} to {end_date}. Please check the symbol and dates.")
    else:
        # Step 1: Technical Indicators Calculations
        active_periods = []
        if show_ema:
            active_periods.append(ema_period)
        if show_sma:
            active_periods.append(sma_period)
        if show_rsi_indicator:
            active_periods.append(rsi_period)
            
        max_requested_period = max(active_periods) if active_periods else 0
        
        # Check if length is sufficient for calculation
        if max_requested_period > len(df):
            st.warning(
                f"⚠️ The selected date range contains only {len(df)} trading days, which is fewer than the "
                f"maximum configured indicator period of {max_requested_period} days. Some indicator trendlines may not fully display or will contain NaN values."
            )
            
        # Calculate EMA
        if show_ema:
            df['EMA'] = ta.ema(df['Close'], length=ema_period)
            
        # Calculate SMA
        if show_sma:
            df['SMA'] = ta.sma(df['Close'], length=sma_period)
            
        # Calculate RSI
        if show_rsi_indicator:
            df['RSI'] = ta.rsi(df['Close'], length=rsi_period)
            
        # Step 2: Run Prophet Machine Learning Forecast if active
        forecast_df = pd.DataFrame()
        if show_forecast:
            forecast_raw = generate_forecast(df, forecast_horizon)
            if not forecast_raw.empty:
                # Slice to future-only starting from the final historical close date to prevent overlapping
                last_hist_date = pd.to_datetime(df.index[-1]).tz_localize(None)
                forecast_df = forecast_raw[forecast_raw['ds'] >= last_hist_date]
            
        # Step 3: Calculate dashboard header metrics
        latest_close = float(df['Close'].iloc[-1])
        
        if len(df) > 1:
            prev_close = float(df['Close'].iloc[-2])
            delta_val = latest_close - prev_close
            delta_pct = (delta_val / prev_close) * 100
            delta_str = f"${delta_val:+,.2f} ({delta_pct:+.2f}%)"
        else:
            delta_str = "N/A"
            
        period_high = float(df['High'].max())
        period_low = float(df['Low'].min())
        
        # Display Key Metrics
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        with metrics_col1:
            st.metric(
                label=f"Latest Close ({df.index[-1].strftime('%Y-%m-%d')})", 
                value=f"${latest_close:,.2f}", 
                delta=delta_str
            )
        with metrics_col2:
            st.metric(
                label="Period High", 
                value=f"${period_high:,.2f}"
            )
        with metrics_col3:
            st.metric(
                label="Period Low", 
                value=f"${period_low:,.2f}"
            )
            
        # Step 4: Build Candlestick, Volume, and RSI subplots dynamically
        show_rsi = show_rsi_indicator and 'RSI' in df.columns
        
        if show_rsi:
            num_rows = 3
            row_heights = [0.55, 0.15, 0.30]
        else:
            num_rows = 2
            row_heights = [0.70, 0.30]
            
        fig = make_subplots(
            rows=num_rows, 
            cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.04, 
            row_heights=row_heights
        )
        
        # Colors matching professional dashboard styles
        up_color = '#10b981'    # Emerald green
        down_color = '#ef4444'  # Crimson red
        
        # Add Candlestick Chart to Row 1
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name="Candlestick",
                increasing_line_color=up_color,
                decreasing_line_color=down_color,
                increasing_fillcolor=up_color,
                decreasing_fillcolor=down_color
            ),
            row=1, col=1
        )
        
        # Overlay EMA if toggled and valid
        if show_ema and 'EMA' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['EMA'],
                    name=f"EMA ({ema_period})",
                    line=dict(color='#facc15', width=1.5),
                    mode='lines'
                ),
                row=1, col=1
            )
            
        # Overlay SMA if toggled and valid
        if show_sma and 'SMA' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['SMA'],
                    name=f"SMA ({sma_period})",
                    line=dict(color='#22d3ee', width=1.5),
                    mode='lines'
                ),
                row=1, col=1
            )
            
        # Overlay Forecast curves in Row 1 if active and valid
        if show_forecast and not forecast_df.empty:
            # 1. Shaded Uncertainty boundary fill (Upper & Lower)
            fig.add_trace(
                go.Scatter(
                    x=forecast_df['ds'],
                    y=forecast_df['yhat_upper'],
                    line=dict(width=0),
                    mode='lines',
                    name='Forecast Upper Limit',
                    showlegend=False
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=forecast_df['ds'],
                    y=forecast_df['yhat_lower'],
                    line=dict(width=0),
                    mode='lines',
                    fill='tonexty',
                    fillcolor='rgba(255, 120, 73, 0.15)',
                    name='Uncertainty Interval (80%)',
                    showlegend=False
                ),
                row=1, col=1
            )
            # 2. Main predicted line
            fig.add_trace(
                go.Scatter(
                    x=forecast_df['ds'],
                    y=forecast_df['yhat'],
                    name="Prophet Forecast",
                    line=dict(color='#ff7849', width=2, dash='dash'),
                    mode='lines'
                ),
                row=1, col=1
            )
        
        # Conditional colors for volume bars
        volume_colors = [up_color if c >= o else down_color for o, c in zip(df['Open'], df['Close'])]
        
        # Add Volume Chart to Row 2
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Volume'],
                name="Volume",
                marker_color=volume_colors,
                opacity=0.85
            ),
            row=2, col=1
        )
        
        # Add RSI Chart to Row 3 if active
        if show_rsi:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['RSI'],
                    name=f"RSI ({rsi_period})",
                    line=dict(color='#a855f7', width=1.8),
                    mode='lines'
                ),
                row=3, col=1
            )
            
            # Draw RSI overbought/oversold dashed bounds at 70 and 30
            fig.add_shape(
                type="line",
                x0=df.index[0],
                x1=df.index[-1],
                y0=70,
                y1=70,
                line=dict(color='#ef4444', width=1.5, dash='dash'),
                row=3, col=1
            )
            fig.add_shape(
                type="line",
                x0=df.index[0],
                x1=df.index[-1],
                y0=30,
                y1=30,
                line=dict(color='#10b981', width=1.5, dash='dash'),
                row=3, col=1
            )
        
        # layout configurations targeting dark-glass aesthetics
        fig.update_layout(
            plot_bgcolor='rgba(15, 23, 42, 0.7)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            font=dict(color='#e2e8f0', family='Outfit'),
            margin=dict(t=15, b=15, l=15, r=15),
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.05)',
                zeroline=False,
                rangeslider=dict(visible=False)
            ),
            yaxis=dict(
                title="Price ($)",
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.05)',
                zeroline=False,
                tickprefix="$"
            ),
            xaxis2=dict(
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.05)',
                zeroline=False,
            ),
            yaxis2=dict(
                title="Volume",
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.05)',
                zeroline=False
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=11, color='#e2e8f0'),
                bgcolor='rgba(15, 23, 42, 0.5)',
                bordercolor='rgba(255, 255, 255, 0.05)',
                borderwidth=1
            ),
            height=750 if show_rsi else 600
        )
        
        # Configure Row 3 axes if active, or set date title on bottom axis
        if show_rsi:
            fig.update_layout(
                xaxis3=dict(
                    showgrid=True,
                    gridcolor='rgba(255, 255, 255, 0.05)',
                    zeroline=False,
                    title="Date"
                ),
                yaxis3=dict(
                    title="RSI Oscillator",
                    showgrid=True,
                    gridcolor='rgba(255, 255, 255, 0.05)',
                    zeroline=False,
                    range=[10, 90],
                    tickvals=[30, 50, 70]
                )
            )
        else:
            fig.update_layout(
                xaxis2=dict(
                    title="Date"
                )
            )
            
        # Render Plotly Chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Step 5: Render Forecasting Summary Cards below the chart
        if show_forecast and not forecast_df.empty:
            final_forecast_row = forecast_df.iloc[-1]
            final_date = final_forecast_row['ds'].strftime('%Y-%m-%d')
            final_price = float(final_forecast_row['yhat'])
            final_upper = float(final_forecast_row['yhat_upper'])
            final_lower = float(final_forecast_row['yhat_lower'])

            # Expected change relative to latest close
            expected_change = ((final_price - latest_close) / latest_close) * 100
            
            # Trend layout styles
            growth_color = "#10b981" if expected_change >= 0 else "#ef4444"
            direction_symbol = "▲" if expected_change >= 0 else "▼"

            st.markdown(
                f"""
                <div class="status-card" style="margin-top: 25px; border: 1px solid rgba(255, 120, 73, 0.35);">
                    <h3 style="color: #ff7849; margin-bottom: 15px; font-weight: 600; display: flex; align-items: center; gap: 8px;">
                        🔮 Prophet Forecasting Insights ({forecast_horizon} Days Horizon)
                    </h3>
                    <p style="color: #94a3b8; font-size: 0.95rem; margin-bottom: 20px;">
                        Forecasting model successfully trained on {len(df)} days of historical data. Here are the projected insights:
                    </p>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 20px;">
                        <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 12px;">
                            <span style="color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em;">Projected Target Price</span><br/>
                            <strong style="color: #f8fafc; font-size: 1.8rem;">${final_price:,.2f}</strong><br/>
                            <span style="color: #64748b; font-size: 0.8rem;">Target Date: {final_date}</span>
                        </div>
                        <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 12px;">
                            <span style="color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em;">Expected Trend</span><br/>
                            <strong style="color: {growth_color}; font-size: 1.8rem;">{direction_symbol} {expected_change:+.2f}%</strong><br/>
                            <span style="color: #64748b; font-size: 0.8rem;">From current close of ${latest_close:,.2f}</span>
                        </div>
                        <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 12px;">
                            <span style="color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em;">Confidence Range (80%)</span><br/>
                            <strong style="color: #f8fafc; font-size: 1.4rem; line-height: 2.2;">${final_lower:,.2f} - ${final_upper:,.2f}</strong><br/>
                            <span style="color: #64748b; font-size: 0.8rem;">Statistical lower & upper limits</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
