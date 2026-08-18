"""
utils/security.py

Security hardening helpers for SafeVision AI:
  - Flask-Limiter setup (rate limiting on login/register/forgot-password)
  - Flask-WTF CSRFProtect setup
  - Session idle-timeout enforcement
  - Password reset token generation/verification
  - Account lockout after repeated failed logins
  - admin_required decorator (role-based access control)

Install the two new dependencies before running:
    pip install flask-limiter flask-wtf

Nothing in this file talks to the DB directly except through the
get_db_connection() helper you already have, so it drops into your
existing project without a new dependency on app.py's internals.
"""

import secrets
from datetime import datetime, timedelta
from functools import wraps

from flask import session, jsonify, redirect, url_for, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf import CSRFProtect

from database.db_helper import get_db_connection
from config import Config

# ==========================================
# RATE LIMITING
# ==========================================
# In-memory storage is fine for a single-process dev/small deployment.
# For multi-worker production (gunicorn with >1 worker), point storage_uri
# at Redis instead, e.g. storage_uri="redis://localhost:6379".
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[],  # no global default; we set limits per-route
    storage_uri="memory://",
)

# ==========================================
# CSRF PROTECTION
# ==========================================
csrf = CSRFProtect()

# ==========================================
# SESSION IDLE TIMEOUT
# ==========================================
# Sourced from config.py (which reads SESSION_IDLE_TIMEOUT_MINUTES from
# .env), so this number always matches app.py's PERMANENT_SESSION_LIFETIME
# instead of being a second hardcoded value that could drift out of sync.
SESSION_IDLE_TIMEOUT_MINUTES = Config.SESSION_IDLE_TIMEOUT_MINUTES


def enforce_session_timeout():
    """
    Call this from a Flask before_request hook. Logs the user out (clears
    session) if they've been idle longer than SESSION_IDLE_TIMEOUT_MINUTES.
    Otherwise refreshes the "last activity" stamp.

    IMPORTANT: session.permanent is set unconditionally, before the
    user_id check. Flask-WTF stores the CSRF token in the session cookie
    for EVERY request, including anonymous ones (e.g. viewing the login
    page). If session.permanent stayed False until after login, that
    anonymous session cookie had no real expiry (tied to browser-session
    lifetime instead of PERMANENT_SESSION_LIFETIME), which some browsers
    and redirect flows (e.g. ngrok's interstitial) don't persist/send
    reliably -- causing "CSRF session token is missing" on submit.
    """
    session.permanent = True

    if "user_id" not in session:
        return

    last_active_raw = session.get("last_active")
    now = datetime.utcnow()

    if last_active_raw:
        last_active = datetime.fromisoformat(last_active_raw)
        if now - last_active > timedelta(minutes=SESSION_IDLE_TIMEOUT_MINUTES):
            session.clear()
            return

    session["last_active"] = now.isoformat()


# ==========================================
# ACCOUNT LOCKOUT (brute-force protection layered on top of rate limiting)
# ==========================================
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_MINUTES = 15


def is_account_locked(user_row):
    locked_until = user_row["locked_until"] if "locked_until" in user_row.keys() else None
    if not locked_until:
        return False
    return datetime.utcnow() < datetime.fromisoformat(locked_until)


def register_failed_login(username):
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    if user is None:
        conn.close()
        return

    attempts = (user["failed_login_attempts"] or 0) + 1
    locked_until = None
    if attempts >= MAX_FAILED_ATTEMPTS:
        locked_until = (datetime.utcnow() + timedelta(minutes=LOCKOUT_MINUTES)).isoformat()

    conn.execute(
        "UPDATE users SET failed_login_attempts = ?, locked_until = ? WHERE id = ?",
        (attempts, locked_until, user["id"]),
    )
    conn.commit()
    conn.close()


def clear_failed_logins(user_id):
    conn = get_db_connection()
    conn.execute(
        "UPDATE users SET failed_login_attempts = 0, locked_until = NULL WHERE id = ?",
        (user_id,),
    )
    conn.commit()
    conn.close()


# ==========================================
# PASSWORD RESET TOKENS
# ==========================================
RESET_TOKEN_VALID_MINUTES = 30


def create_password_reset_token(email):
    """
    Generates and stores a one-time reset token for the account matching
    `email`. Returns the raw token string, or None if no account has that
    email (caller should show a generic "if that email exists..." message
    either way, to avoid leaking which emails are registered).
    """
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    if user is None:
        conn.close()
        return None

    token = secrets.token_urlsafe(32)
    expiry = (datetime.utcnow() + timedelta(minutes=RESET_TOKEN_VALID_MINUTES)).isoformat()

    conn.execute(
        "UPDATE users SET reset_token = ?, reset_token_expiry = ? WHERE id = ?",
        (token, expiry, user["id"]),
    )
    conn.commit()
    conn.close()
    return token


def verify_password_reset_token(token):
    """Returns the matching user row if the token is valid and unexpired, else None."""
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE reset_token = ?", (token,)).fetchone()
    conn.close()

    if user is None or not user["reset_token_expiry"]:
        return None
    if datetime.utcnow() > datetime.fromisoformat(user["reset_token_expiry"]):
        return None
    return user


def consume_password_reset_token(user_id, new_password_hash):
    """Sets the new password hash and invalidates the reset token."""
    conn = get_db_connection()
    conn.execute(
        """UPDATE users
           SET password_hash = ?, reset_token = NULL, reset_token_expiry = NULL,
               failed_login_attempts = 0, locked_until = NULL
           WHERE id = ?""",
        (new_password_hash, user_id),
    )
    conn.commit()
    conn.close()


# ==========================================
# ADMIN / ROLE-BASED ACCESS CONTROL
# ==========================================
def admin_required(f):
    """
    Use in addition to @login_required (put @login_required first/outermost).
    Blocks any non-admin user from hitting the route.
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            if request.path.startswith("/api/"):
                return jsonify({"status": "error", "message": "Please log in first."}), 401
            return redirect(url_for("index"))

        conn = get_db_connection()
        user = conn.execute("SELECT is_admin FROM users WHERE id = ?", (session["user_id"],)).fetchone()
        conn.close()

        if not user or not user["is_admin"]:
            return jsonify({"status": "error", "message": "Admin access required."}), 403
        return f(*args, **kwargs)
    return wrapper


def log_admin_action(admin_username, action, target=None):
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO admin_audit_log (admin_username, action, target) VALUES (?, ?, ?)",
        (admin_username, action, target),
    )
    conn.commit()
    conn.close()