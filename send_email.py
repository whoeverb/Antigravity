"""
send_email.py
Sends the Antigravity HTML signal report via Gmail SMTP.

Required environment variables (set as GitHub Actions secrets):
  GMAIL_USER      — your Gmail address, e.g. you@gmail.com
  GMAIL_APP_PASS  — 16-char Gmail App Password (not your main password)
  RECIPIENT_EMAIL — where to send the report (can be same as GMAIL_USER)
"""

import os
import sys
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime

from email_renderer import render_html


def send_signal_email(signals_path: str = "signals.json") -> None:
    # ── Read config from environment ──────────────────────────────────────────
    gmail_user  = os.environ.get("GMAIL_USER")
    app_pass    = os.environ.get("GMAIL_APP_PASS")
    recipient   = os.environ.get("RECIPIENT_EMAIL", gmail_user)

    if not gmail_user or not app_pass:
        print("❌  Missing GMAIL_USER or GMAIL_APP_PASS environment variables.")
        sys.exit(1)

    # ── Build the message ─────────────────────────────────────────────────────
    today = datetime.now().strftime("%b %d, %Y")
    subject = f"🚀 Signal Report — {today}"

    html_body = render_html(signals_path)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = f"Antigravity Signals <{gmail_user}>"
    msg["To"]      = recipient

    # Plain-text fallback
    plain = (
        f"Signal Report — {today}\n\n"
        "Open this email in an HTML-capable client to view the full report.\n"
        f"Signals file: {signals_path}"
    )
    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    # ── Send via Gmail SMTP (TLS) ─────────────────────────────────────────────
    print(f"📧  Sending signal report to {recipient} ...")
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.ehlo()
            server.starttls()
            server.login(gmail_user, app_pass)
            server.sendmail(gmail_user, recipient, msg.as_string())
        print("✅  Email sent successfully.")
    except smtplib.SMTPAuthenticationError:
        print("❌  SMTP authentication failed. Check your App Password.")
        sys.exit(1)
    except Exception as e:
        print(f"❌  Failed to send email: {e}")
        sys.exit(1)


if __name__ == "__main__":
    signals_file = sys.argv[1] if len(sys.argv) > 1 else "signals.json"
    send_signal_email(signals_file)
