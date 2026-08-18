import os
import smtplib
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ==========================================
# SMTP CONFIGURATION (set these as environment variables — never hardcode)
# ==========================================
SENDER_NAME = "SafeVision AI"


def _get_smtp_config():
    """
    Read credentials at CALL TIME (not import time). This makes the app more
    robust if environment variables were set after Python started, and keeps
    behavior consistent no matter when this module was first imported.
    """
    return {
        "server": os.environ.get("SAFEVISION_SMTP_SERVER", "smtp.gmail.com"),
        "port": int(os.environ.get("SAFEVISION_SMTP_PORT", 587)),
        "username": os.environ.get("SAFEVISION_SMTP_USERNAME"),
        "password": os.environ.get("SAFEVISION_SMTP_PASSWORD"),
    }


def _send_email_blocking(to_email, subject, html_body):
    """Low-level helper: sends one HTML email. Returns True/False, never raises.
    This is the SLOW part (network + SMTP handshake) — always called from a
    background thread via _send_email(), never directly from a Flask route."""
    cfg = _get_smtp_config()

    if not cfg["username"] or not cfg["password"]:
        print("⚠️ Email not sent — SAFEVISION_SMTP_USERNAME / SAFEVISION_SMTP_PASSWORD not configured.")
        return False
    if not to_email:
        print("⚠️ Email not sent — recipient has no email on file.")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"{SENDER_NAME} <{cfg['username']}>"
    msg["To"] = to_email
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(cfg["server"], cfg["port"], timeout=15) as server:
            server.starttls()
            server.login(cfg["username"], cfg["password"])
            server.sendmail(cfg["username"], [to_email], msg.as_string())
        print(f"✅ Email sent to {to_email}: {subject}")
        return True
    except smtplib.SMTPAuthenticationError:
        print("❌ Email failed: Gmail rejected the username/App Password. "
              "Check SAFEVISION_SMTP_PASSWORD is the 16-digit App Password, not your normal Gmail password.")
        return False
    except Exception as e:
        print(f"⚠️ Email sending failed: {e}")
        return False


def _send_email(to_email, subject, html_body):
    """
    Fire-and-forget wrapper: runs _send_email_blocking() in a background
    daemon thread so the calling Flask route (login, /detect, /stop_recording)
    returns immediately instead of waiting on Gmail's SMTP round-trip.
    """
    thread = threading.Thread(
        target=_send_email_blocking, args=(to_email, subject, html_body), daemon=True
    )
    thread.start()


# ==========================================
# 1) LOGIN NOTIFICATION
# ==========================================
def send_login_notification(to_email, full_name, ip_address, login_time):
    subject = "🔐 SafeVision AI — New Login to Your Account"
    html_body = f"""
    <div style="font-family:Arial,sans-serif;max-width:480px;margin:auto;border:1px solid #e2e8f0;border-radius:8px;overflow:hidden">
        <div style="background:#0B3D91;color:#fff;padding:16px 20px">
            <h2 style="margin:0">🛡️ SafeVision AI</h2>
        </div>
        <div style="padding:20px;color:#1f2937">
            <p>Hi <strong>{full_name}</strong>,</p>
            <p>Your SafeVision AI account was just signed in to.</p>
            <table style="width:100%;border-collapse:collapse;margin:15px 0">
                <tr><td style="padding:6px 0;color:#6b7280">IP Address:</td><td style="padding:6px 0"><strong>{ip_address}</strong></td></tr>
                <tr><td style="padding:6px 0;color:#6b7280">Time:</td><td style="padding:6px 0"><strong>{login_time}</strong></td></tr>
            </table>
            <p style="color:#6b7280;font-size:.85rem">If this wasn't you, please change your password immediately.</p>
        </div>
    </div>
    """
    _send_email(to_email, subject, html_body)


# ==========================================
# 2) SAFETY VIOLATION NOTIFICATION
# ==========================================
def send_violation_alert(to_email, full_name, source_type, status, file_url=None):
    missing_items = [
        item.capitalize() for item in ["helmet", "vest", "gloves", "shoes", "glasses"]
        if status.get(item, {}).get("is_missing")
    ]
    missing_list_html = "".join(f"<li>{item}</li>" for item in missing_items) or "<li>None</li>"

    subject = "⚠️ SafeVision AI — Safety Violation Detected"
    html_body = f"""
    <div style="font-family:Arial,sans-serif;max-width:480px;margin:auto;border:1px solid #e2e8f0;border-radius:8px;overflow:hidden">
        <div style="background:#B00020;color:#fff;padding:16px 20px">
            <h2 style="margin:0">⚠️ Safety Violation Detected</h2>
        </div>
        <div style="padding:20px;color:#1f2937">
            <p>Hi <strong>{full_name}</strong>,</p>
            <p>A PPE compliance scan from <strong>{source_type}</strong> flagged a mandatory safety violation.</p>
            <p style="color:#6b7280">Missing PPE items:</p>
            <ul>{missing_list_html}</ul>
            {f'<p><a href="{file_url}" style="color:#0284c7">View the flagged file</a></p>' if file_url else ''}
            <p style="color:#6b7280;font-size:.85rem">This is an automated alert from SafeVision AI.</p>
        </div>
    </div>
    """
    _send_email(to_email, subject, html_body)


# ==========================================
# 3) SIGNUP / WELCOME NOTIFICATION
# ==========================================
def send_signup_notification(to_email, full_name, username):
    """
    Sent once, right after a new account is created (separate from the
    login-notification email that fires on every subsequent sign-in).
    Referenced optionally from app.py's /register route.
    """
    subject = "✅ Welcome to SafeVision AI"
    html_body = f"""
    <div style="font-family:Arial,sans-serif;max-width:480px;margin:auto;border:1px solid #e2e8f0;border-radius:8px;overflow:hidden">
        <div style="background:#16a34a;color:#fff;padding:16px 20px">
            <h2 style="margin:0">🛡️ SafeVision AI</h2>
        </div>
        <div style="padding:20px;color:#1f2937">
            <p>Hi <strong>{full_name}</strong>,</p>
            <p>Your SafeVision AI account has been created successfully.</p>
            <table style="width:100%;border-collapse:collapse;margin:15px 0">
                <tr><td style="padding:6px 0;color:#6b7280">Username:</td><td style="padding:6px 0"><strong>{username}</strong></td></tr>
            </table>
            <p>You can now sign in and start monitoring PPE compliance.</p>
            <p style="color:#6b7280;font-size:.85rem">If you didn't create this account, please contact your system administrator.</p>
        </div>
    </div>
    """
    _send_email(to_email, subject, html_body)


# ==========================================
# 4) PASSWORD RESET
# ==========================================
def send_password_reset_email(to_email, full_name, reset_url):
    """
    Sent when a user requests a password reset via /forgot_password.
    reset_url already points at /reset_password/<token> — the link is
    valid for 30 minutes (see utils/security.py: RESET_TOKEN_VALID_MINUTES).
    """
    subject = "🔑 SafeVision AI — Reset Your Password"
    html_body = f"""
    <div style="font-family:Arial,sans-serif;max-width:480px;margin:auto;border:1px solid #e2e8f0;border-radius:8px;overflow:hidden">
        <div style="background:#0B3D91;color:#fff;padding:16px 20px">
            <h2 style="margin:0">🛡️ SafeVision AI</h2>
        </div>
        <div style="padding:20px;color:#1f2937">
            <p>Hi <strong>{full_name}</strong>,</p>
            <p>We received a request to reset your SafeVision AI password. Click the button below to choose a new one — this link expires in 30 minutes.</p>
            <p style="text-align:center;margin:25px 0">
                <a href="{reset_url}" style="background:#0284c7;color:#fff;text-decoration:none;padding:12px 24px;border-radius:6px;font-weight:bold;display:inline-block">Reset Password</a>
            </p>
            <p style="color:#6b7280;font-size:.85rem">If the button doesn't work, copy and paste this link into your browser:<br>{reset_url}</p>
            <p style="color:#6b7280;font-size:.85rem">If you didn't request this, you can safely ignore this email — your password won't be changed.</p>
        </div>
    </div>
    """
    _send_email(to_email, subject, html_body)