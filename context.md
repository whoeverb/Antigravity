# Personal Stock Research & Projection App - System Blueprint

## App Overview & Context
This is a 100% local, desktop-based personal stock analytics dashboard 
running on a laptop (Windows/Mac) with a live internet connection. It is 
built using Python (Streamlit framework), utilizing native data science 
and machine learning packages. The goal is zero-friction, automated data 
analysis, and advanced time-series forecasting for personal asset 
management.

## Core Architecture & Technical Stack
- **UI & Presentation Layout:** Streamlit framework for rapid, reactive 
dashboard rendering.
- **Data Sourcing:** `yfinance` to dynamically pull historical equity 
pricing, metrics, and fundamental data.
- **Data Manipulation:** `pandas` and `numpy` for localized dataframes, 
time-series transformations, and calculations.
- **Technical Indicators:** Modern algorithmic indicator math (moving 
averages, RSI, MACD, Bollinger Bands).
- **Visualization Engine:** Interactive, high-fidelity financial graphing 
via `plotly.graph_objects` (candlesticks, line charts, volumes).
- **Forecasting / Projection Engine:** Time-series estimation models 
utilizing machine learning or quantitative forecasting (e.g., Meta's 
`prophet` or `scikit-learn` regressions).

## Features Implemented & Target Workflows
1. **Watchlist & Ticker Hub:** Persistent lookup for ticker metrics and 
saved holding states.
2. **Interactive Charting Block:** Detailed financial candlestick displays 
with technical analysis overlays.
3. **Local Investment Tracking Input:** Input fields where the user 
registers personal owned shares and purchase cost basis. Data 
auto-saves/loads natively.
4. **Local Prediction Module:** Quantitative logic that projects future 
price action trends directly from local memory variables onto the 
timeline.

## Guardrails & Styling Directives
- **Zero-Broke Layout Rules:** Use strict, defensive CSS layouts. Never 
utilize hardcoded fixed heights (`height: 40px`) or absolute positions 
that cause user input elements to collide or overlap when auto-populated. 
Prioritize flexible flexbox formatting (`display: flex; flex-direction: 
column; height: auto; gap: 12px;`).
- **Financial Dashboard Theme:** Deep, rich dark-mode or high-contrast 
modern visual palette. Text hierarchy must use thick font weights and 
crystal-clear colors (e.g., clean white text on dark cards, slate-grays 
for subtitles). 
- **Indicator Color Standardization:** Positive financial metrics (gains, 
bullish projections) must use clear, vibrant green text/lines. Negative 
metrics (losses, bearish curves) must use clear, distinct red text/lines. 
Avoid muddy, low-contrast UI colors.
