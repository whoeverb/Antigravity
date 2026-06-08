"""
email_renderer.py
Renders signals.json into a styled HTML email body.
Mobile-responsive via media queries (works in iOS Mail and Gmail app).
"""

import json
from datetime import datetime


SIGNAL_STYLE = {
    "BUY":  {"bg": "#d4edda", "color": "#155724", "border": "#28a745"},
    "DCA":  {"bg": "#cce5ff", "color": "#004085", "border": "#007bff"},
    "HOLD": {"bg": "#fff3cd", "color": "#856404", "border": "#ffc107"},
    "WAIT": {"bg": "#e2e3e5", "color": "#383d41", "border": "#6c757d"},
    "SELL": {"bg": "#f8d7da", "color": "#721c24", "border": "#dc3545"},
}

REGIME_EMOJI = {
    "Overheated":           "🔥",
    "Strong Uptrend":       "📈",
    "Bullish":              "🟢",
    "Neutral":              "⚪",
    "Pullback":             "🔻",
    "Oversold Opportunity": "💎",
    "Risk-Off":             "⚠️",
}

CONFIDENCE_BADGE = {
    "High":   ("🟢", "#28a745"),
    "Medium": ("🟡", "#ffc107"),
    "Low":    ("🔴", "#dc3545"),
}


def _signal_badge(signal: str) -> str:
    s = SIGNAL_STYLE.get(signal, SIGNAL_STYLE["HOLD"])
    return (
        f'<span style="display:inline-block;padding:3px 10px;border-radius:4px;'
        f'font-weight:700;font-size:13px;letter-spacing:.5px;'
        f'background:{s["bg"]};color:{s["color"]};border:1px solid {s["border"]};">'
        f'{signal}</span>'
    )


def _conf_badge(conf: str) -> str:
    emoji, _ = CONFIDENCE_BADGE.get(conf, ("⚪", "#6c757d"))
    return f'{emoji} {conf}'


def _price_change(chg: float) -> str:
    color = "#28a745" if chg >= 0 else "#dc3545"
    arrow = "▲" if chg >= 0 else "▼"
    return f'<span style="color:{color};font-weight:600;">{arrow} {abs(chg):.2f}%</span>'


def _mobile_card(ticker: str, data: dict) -> str:
    """Renders a single ticker as a stacked card for mobile."""
    signal  = data.get("signal", "HOLD")
    price   = data.get("price", 0.0)
    chg     = data.get("change_pct", 0.0)
    regime  = data.get("regime", "Neutral")
    conf    = data.get("confidence", "Low")
    reasons = data.get("reasons", "—")
    forecast = data.get("forecast_30d")
    s = SIGNAL_STYLE.get(signal, SIGNAL_STYLE["HOLD"])
    regime_icon = REGIME_EMOJI.get(regime, "")

    forecast_html = ""
    if forecast and price:
        diff_pct = (forecast - price) / price * 100
        fc_color = "#28a745" if diff_pct > 0 else "#dc3545"
        fc_arrow = "▲" if diff_pct > 0 else "▼"
        forecast_html = (
            f'<div style="margin-top:6px;font-size:12px;color:#6c757d;">'
            f'30d Forecast: <span style="color:{fc_color};font-weight:600;">'
            f'{fc_arrow} ${forecast:,.2f} ({diff_pct:+.1f}%)</span></div>'
        )

    return f"""
<div class="mobile-card" style="display:none;background:#fff;border-radius:10px;
     margin:8px 0;padding:14px 16px;border-left:4px solid {s["border"]};
     box-shadow:0 1px 4px rgba(0,0,0,.07);">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
    <span style="font-size:18px;font-weight:800;color:#0d1117;">{ticker}</span>
    {_signal_badge(signal)}
  </div>
  <div style="display:flex;gap:16px;margin-bottom:8px;">
    <div>
      <div style="font-size:15px;font-weight:700;">${price:,.2f}</div>
      <div style="font-size:12px;">{_price_change(chg)}</div>
    </div>
    <div>
      <div style="font-size:12px;color:#6c757d;">Regime</div>
      <div style="font-size:13px;font-weight:600;">{regime_icon} {regime}</div>
    </div>
    <div>
      <div style="font-size:12px;color:#6c757d;">Confidence</div>
      <div style="font-size:13px;">{_conf_badge(conf)}</div>
    </div>
  </div>
  <div style="font-size:12px;color:#495057;line-height:1.5;">{reasons}</div>
  {forecast_html}
</div>"""


def _desktop_section_header(title: str) -> str:
    return (
        f'<tr><td colspan="7" style="padding:18px 0 6px 0;">'
        f'<div style="font-size:13px;font-weight:700;letter-spacing:1px;'
        f'text-transform:uppercase;color:#6c757d;border-bottom:2px solid #e9ecef;'
        f'padding-bottom:6px;">{title}</div></td></tr>'
    )


def _desktop_row(ticker: str, data: dict, zebra: bool) -> str:
    bg      = "#f8f9fa" if zebra else "#ffffff"
    signal  = data.get("signal", "HOLD")
    price   = data.get("price", 0.0)
    chg     = data.get("change_pct", 0.0)
    regime  = data.get("regime", "Neutral")
    conf    = data.get("confidence", "Low")
    reasons = data.get("reasons", "—")
    forecast = data.get("forecast_30d")
    forecast_str = "—"
    if forecast and price:
        diff_pct = (forecast - price) / price * 100
        fc_color = "#28a745" if diff_pct > 0 else "#dc3545"
        fc_arrow = "▲" if diff_pct > 0 else "▼"
        forecast_str = (
            f'<span style="color:{fc_color};">{fc_arrow} ${forecast:,.2f} '
            f'({diff_pct:+.1f}%)</span>'
        )
    regime_icon = REGIME_EMOJI.get(regime, "")
    cell = 'style="padding:10px 8px;vertical-align:middle;font-size:13px;border-bottom:1px solid #e9ecef;"'
    return f"""
    <tr style="background:{bg};">
      <td {cell}><strong style="font-size:14px;">{ticker}</strong></td>
      <td {cell}>{_signal_badge(signal)}</td>
      <td {cell}><strong>${price:,.2f}</strong><br>{_price_change(chg)}</td>
      <td {cell}>{regime_icon} {regime}</td>
      <td {cell}>{_conf_badge(conf)}</td>
      <td {cell} style="max-width:220px;color:#495057;">{reasons}</td>
      <td {cell}>{forecast_str}</td>
    </tr>"""


def render_html(signals_path: str = "signals.json") -> str:
    with open(signals_path) as f:
        data = json.load(f)

    generated_at    = data.get("generated_at", str(datetime.now()))
    signals         = data.get("signals", {})
    top_candidates  = data.get("top_candidates", [])

    etfs   = {k: v for k, v in signals.items() if v.get("type") == "ETF"}
    stocks = {k: v for k, v in signals.items() if v.get("type") == "STOCK"}

    counts = {"BUY": 0, "DCA": 0, "HOLD": 0, "WAIT": 0, "SELL": 0}
    for v in signals.values():
        sig = v.get("signal", "HOLD")
        if sig in counts:
            counts[sig] += 1

    # ── Top candidates ─────────────────────────────────────────────────────────
    top_html = ""
    for c in top_candidates:
        s = SIGNAL_STYLE.get(c["signal"], SIGNAL_STYLE["HOLD"])
        top_html += (
            f'<div style="display:inline-block;margin:4px 6px;padding:8px 16px;'
            f'border-radius:6px;background:{s["bg"]};color:{s["color"]};'
            f'border:1px solid {s["border"]};font-weight:700;font-size:14px;">'
            f'{c["ticker"]} &nbsp;<span style="font-weight:400;font-size:12px;">'
            f'({c["signal"]})</span></div>'
        )

    # ── Summary pills ──────────────────────────────────────────────────────────
    summary_pills = ""
    for sig, count in counts.items():
        if count == 0:
            continue
        s = SIGNAL_STYLE[sig]
        summary_pills += (
            f'<span style="display:inline-block;margin:3px 4px;padding:5px 12px;'
            f'border-radius:20px;font-size:13px;font-weight:600;'
            f'background:{s["bg"]};color:{s["color"]};border:1px solid {s["border"]};">'
            f'{sig} &nbsp;<strong>{count}</strong></span>'
        )

    # ── Desktop table rows ─────────────────────────────────────────────────────
    desktop_rows = ""
    desktop_rows += _desktop_section_header("ETFs")
    for i, (ticker, info) in enumerate(etfs.items()):
        desktop_rows += _desktop_row(ticker, info, i % 2 == 1)
    desktop_rows += _desktop_section_header("Stocks")
    for i, (ticker, info) in enumerate(stocks.items()):
        desktop_rows += _desktop_row(ticker, info, i % 2 == 1)

    # ── Mobile cards ───────────────────────────────────────────────────────────
    mobile_etf_cards = "".join(_mobile_card(t, v) for t, v in etfs.items())
    mobile_stock_cards = "".join(_mobile_card(t, v) for t, v in stocks.items())

    th = ('style="padding:10px 8px;text-align:left;font-size:12px;font-weight:700;'
          'letter-spacing:.5px;text-transform:uppercase;color:#6c757d;'
          'border-bottom:2px solid #dee2e6;white-space:nowrap;"')

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Antigravity Signal Report</title>
<style>
  /* ── Mobile: show cards, hide table ── */
  @media only screen and (max-width: 620px) {{
    .desktop-table {{ display: none !important; }}
    .mobile-card   {{ display: block !important; }}
    .mobile-section-label {{ display: block !important; }}
    .email-wrapper {{ padding: 8px 4px !important; }}
    .header-pad    {{ padding: 20px 16px !important; }}
    .section-pad   {{ padding: 12px 16px !important; }}
    .header-title  {{ font-size: 18px !important; }}
  }}
  .mobile-section-label {{
    display: none;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #6c757d;
    padding: 12px 0 4px 0;
    border-bottom: 2px solid #e9ecef;
    margin-bottom: 4px;
  }}
</style>
</head>
<body style="margin:0;padding:0;background:#f0f2f5;
     font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">

<table width="100%" cellpadding="0" cellspacing="0" style="background:#f0f2f5;">
<tr><td align="center" class="email-wrapper" style="padding:24px 12px;">

  <table width="700" cellpadding="0" cellspacing="0"
         style="max-width:700px;width:100%;background:#ffffff;border-radius:12px;
                box-shadow:0 2px 12px rgba(0,0,0,.08);overflow:hidden;">

    <!-- Header -->
    <tr>
      <td class="header-pad"
          style="background:linear-gradient(135deg,#0d1117 0%,#1a2332 100%);padding:28px 32px;">
        <div class="header-title"
             style="color:#ffffff;font-size:22px;font-weight:800;letter-spacing:-0.5px;">
          🚀 Antigravity Signal Report
        </div>
        <div style="color:#8b949e;font-size:13px;margin-top:4px;">
          Generated {generated_at} &nbsp;·&nbsp; {len(signals)} positions tracked
        </div>
      </td>
    </tr>

    <!-- Summary bar -->
    <tr>
      <td class="section-pad"
          style="padding:16px 32px;background:#f8f9fa;border-bottom:1px solid #e9ecef;">
        <div style="font-size:12px;font-weight:700;text-transform:uppercase;
                    letter-spacing:.8px;color:#6c757d;margin-bottom:8px;">
          Signal Summary
        </div>
        {summary_pills}
      </td>
    </tr>

    <!-- Top candidates -->
    <tr>
      <td class="section-pad"
          style="padding:20px 32px;border-bottom:1px solid #e9ecef;">
        <div style="font-size:12px;font-weight:700;text-transform:uppercase;
                    letter-spacing:.8px;color:#6c757d;margin-bottom:10px;">
          ⭐ Top Candidates
        </div>
        {top_html if top_html else '<span style="color:#aaa;font-size:13px;">None this session</span>'}
      </td>
    </tr>

    <!-- Desktop table (hidden on mobile) -->
    <tr class="desktop-table">
      <td class="section-pad" style="padding:8px 32px 24px 32px;">
        <table width="100%" cellpadding="0" cellspacing="0"
               style="border-collapse:collapse;min-width:580px;">
          <thead>
            <tr>
              <th {th}>Ticker</th>
              <th {th}>Signal</th>
              <th {th}>Price</th>
              <th {th}>Regime</th>
              <th {th}>Confidence</th>
              <th {th}>Reasons</th>
              <th {th}>30d Forecast</th>
            </tr>
          </thead>
          <tbody>{desktop_rows}</tbody>
        </table>
      </td>
    </tr>

    <!-- Mobile cards (hidden on desktop) -->
    <tr>
      <td class="section-pad" style="padding:8px 16px 16px 16px;">
        <div class="mobile-section-label">ETFs</div>
        {mobile_etf_cards}
        <div class="mobile-section-label" style="margin-top:8px;">Stocks</div>
        {mobile_stock_cards}
      </td>
    </tr>

    <!-- Footer -->
    <tr>
      <td class="section-pad"
          style="padding:16px 32px;background:#f8f9fa;border-top:1px solid #e9ecef;
                 border-radius:0 0 12px 12px;">
        <div style="font-size:11px;color:#adb5bd;text-align:center;line-height:1.6;">
          ⚠️ This report is for informational purposes only and does not constitute
          financial advice. Always do your own research before making investment decisions.
          <br>Antigravity · Auto-generated by GitHub Actions
        </div>
      </td>
    </tr>

  </table>
</td></tr>
</table>

</body>
</html>"""

    return html


if __name__ == "__main__":
    html = render_html("signals.json")
    with open("signal_email_preview.html", "w") as f:
        f.write(html)
    print("Preview written to signal_email_preview.html")
