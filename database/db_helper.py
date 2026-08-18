"""
database/db_helper.py

Handles the SQLite connection and schema (create + idempotent migration).
DATABASE_PATH now comes from config.py (Config.DATABASE_PATH) so there's a
single source of truth for where the .db file lives — no more duplicating
the BASE_DIR computation in every file that needs it.
"""

import sqlite3

from config import Config

DATABASE_PATH = Config.DATABASE_PATH


def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _column_exists(conn, table, column):
    cols = [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    return column in cols


def _table_exists(conn, table):
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def migrate_schema(conn):
    """
    Idempotent migration: adds security/admin columns and tables if they
    are missing. Safe to call every time the app starts.
    """
    # --- users: role/admin + account lockout + password reset + profile photo ---
    user_columns = {
        "is_admin": "INTEGER NOT NULL DEFAULT 0",
        "failed_login_attempts": "INTEGER NOT NULL DEFAULT 0",
        "locked_until": "TEXT",
        "reset_token": "TEXT",
        "reset_token_expiry": "TEXT",
        "created_at": "TEXT",
        # Web-relative path to the user's uploaded profile photo (e.g.
        # "/profile_photos/user_3.jpg"), used both as the profile-page
        # avatar and as the source image for face-recognition training.
        # NULL until the user uploads one from the Profile tab.
        "profile_photo": "TEXT",
    }
    for col, ddl in user_columns.items():
        if not _column_exists(conn, "users", col):
            conn.execute(f"ALTER TABLE users ADD COLUMN {col} {ddl}")

    # --- detection_logs: track which user ran each scan, so logs can be
    # filtered to "my own" logs for non-admin users ---
    if not _column_exists(conn, "detection_logs", "user_id"):
        conn.execute("ALTER TABLE detection_logs ADD COLUMN user_id INTEGER")

    # --- audit table for admin actions (role changes, deletions, target edits) ---
    if not _table_exists(conn, "admin_audit_log"):
        conn.execute("""
            CREATE TABLE admin_audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                admin_username TEXT NOT NULL,
                action TEXT NOT NULL,
                target TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

    conn.commit()

    # --- bootstrap: if no admin exists yet, promote the earliest registered user ---
    admin_row = conn.execute("SELECT id FROM users WHERE is_admin = 1 LIMIT 1").fetchone()
    if admin_row is None:
        first_user = conn.execute("SELECT id FROM users ORDER BY id ASC LIMIT 1").fetchone()
        if first_user is not None:
            conn.execute(
                "UPDATE users SET is_admin = 1, role = 'Admin' WHERE id = ?",
                (first_user["id"],),
            )
            conn.commit()


def init_db():
    conn = get_db_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT NOT NULL,
            email TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'Safety Inspector'
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS detection_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_type TEXT NOT NULL,
            status TEXT NOT NULL,
            file_path TEXT,
            report_path TEXT,
            timestamp TEXT NOT NULL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS login_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            username TEXT,
            ip_address TEXT,
            login_time TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS recording_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_path TEXT,
            report_path TEXT,
            start_time TEXT,
            end_time TEXT,
            total_frames INTEGER,
            violation_frames INTEGER,
            recorded_by TEXT
        )
    """)

    conn.commit()

    # apply security/admin/profile-photo migration
    migrate_schema(conn)

    conn.close()