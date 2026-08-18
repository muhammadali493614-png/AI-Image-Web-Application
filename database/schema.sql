-- ==========================================
-- SafeVision AI — Reference Schema
-- ==========================================
-- NOTE: this file is for reference only. The actual database is created
-- and kept up to date automatically by database/db_helper.py
-- (init_db() + migrate_schema()) every time the app starts — you never
-- need to run this .sql file manually. It's here so the full column set
-- is visible in one place without reading the Python migration code.

-- ==========================================
-- USERS TABLE
-- ==========================================
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    full_name TEXT NOT NULL,
    email TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'Safety Inspector',

    -- added by utils/security.py's admin RBAC + lockout + reset-token logic
    is_admin INTEGER NOT NULL DEFAULT 0,
    failed_login_attempts INTEGER NOT NULL DEFAULT 0,
    locked_until TEXT,
    reset_token TEXT,
    reset_token_expiry TEXT,
    created_at TEXT
);

-- ==========================================
-- DETECTION LOGS TABLE (image/video/live results)
-- ==========================================
CREATE TABLE IF NOT EXISTS detection_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type TEXT NOT NULL,
    status TEXT NOT NULL,
    file_path TEXT,
    report_path TEXT,
    timestamp TEXT NOT NULL,

    -- added so each user's Detection History can be filtered to their own
    -- scans only (admins see everyone's) — nullable because older rows
    -- created before this column existed won't have a user attached
    user_id INTEGER
);

-- ==========================================
-- LOGIN LOGS TABLE
-- ==========================================
CREATE TABLE IF NOT EXISTS login_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    username TEXT,
    ip_address TEXT,
    login_time TEXT DEFAULT (datetime('now', 'localtime')),
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- ==========================================
-- LIVE RECORDING SESSIONS TABLE
-- ==========================================
CREATE TABLE IF NOT EXISTS recording_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_path TEXT,
    report_path TEXT,
    start_time TEXT,
    end_time TEXT,
    total_frames INTEGER,
    violation_frames INTEGER,
    recorded_by TEXT
);

-- ==========================================
-- ADMIN AUDIT LOG (role changes, deletions, target-rule edits)
-- ==========================================
CREATE TABLE IF NOT EXISTS admin_audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    admin_username TEXT NOT NULL,
    action TEXT NOT NULL,
    target TEXT,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
);