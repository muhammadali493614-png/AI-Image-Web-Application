import os
import json
import logging
import platform
import sqlite3
import threading
import time
from logging.handlers import RotatingFileHandler
import cv2
import numpy as np
from functools import wraps
from datetime import datetime, timedelta
from flask import (Flask, render_template, request, Response, redirect, url_for,
                    send_from_directory, jsonify, send_file, session)
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
from ultralytics import YOLO

from config import Config
from database.db_helper import get_db_connection, init_db
from utils.report_generator import (generate_detection_report,
                                     generate_live_session_report,
                                     generate_live_snapshot_report)
from utils.email_notifier import send_login_notification, send_violation_alert
from utils.face_recognition_helper import register_face, recognize_faces_in_frame, remove_user_face

# --- optional email hooks: only used if defined in utils/email_notifier.py ---
try:
    from utils.email_notifier import send_signup_notification
except ImportError:
    send_signup_notification = None

try:
    from utils.email_notifier import send_password_reset_email
except ImportError:
    send_password_reset_email = None
    # Add this to utils/email_notifier.py to enable "forgot password" emails:
    #   def send_password_reset_email(to_email, full_name, reset_url): ...

# ==========================================
# SECURITY: rate limiting, CSRF, sessions, lockout, admin RBAC
# ==========================================
from utils.security import (
    limiter, csrf, enforce_session_timeout,
    is_account_locked, register_failed_login, clear_failed_logins,
    create_password_reset_token, verify_password_reset_token, consume_password_reset_token,
    admin_required, log_admin_action,
)
from flask_wtf.csrf import CSRFError

# Silences OpenCV's internal C++ warnings (e.g. the DSHOW "can't be used to
# capture by index" message) from flooding the terminal — _open_camera()
# below already handles that failure case explicitly and logs its own,
# clearer message, so the raw OpenCV warning is just noise once that's in
# place. Must be set before any cv2.VideoCapture() call.
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass  # older OpenCV builds may not expose this — safe to skip

app = Flask(__name__)

# --- all settings now come from config.py (which reads .env) ---
app.config.from_object(Config)
Config.ensure_folders_exist()

if app.config["SECRET_KEY"] == "change-this-in-production":
    print("⚠️  WARNING: SAFEVISION_SECRET_KEY is not set — using the default dev key. "
          "Set a real secret via the environment variable before deploying.")

# --- rate limiting + CSRF ---
limiter.init_app(app)
csrf.init_app(app)


# ==========================================
# PRODUCTION LOGGING (file-based, only when debug is off)
# ==========================================
if not app.config["DEBUG"]:
    file_handler = RotatingFileHandler(
        os.path.join(Config.LOG_FOLDER, "app.log"), maxBytes=5 * 1024 * 1024, backupCount=3
    )
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s in %(module)s: %(message)s"
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info("SafeVision AI startup (debug=False)")


@app.errorhandler(CSRFError)
def handle_csrf_error(e):
    # Without this, a CSRF failure returns Flask-WTF's default HTML error
    # page. The frontend always expects JSON, so res.json() would throw and
    # the UI would show a generic "Unable to reach server" message that
    # hides the real cause. This makes the real reason visible instead.
    print(f"⚠️ CSRF validation failed: {e.description}")
    return jsonify({
        "status": "error",
        "message": f"Security check failed: {e.description}. Please refresh the page (Ctrl+Shift+R) and try again."
    }), 400


@app.errorhandler(429)
def handle_rate_limit_error(e):
    # Same reasoning as the CSRF handler above — Flask-Limiter's default
    # 429 response isn't JSON, which would otherwise also show up in the
    # UI as a generic "Unable to reach server" message.
    return jsonify({
        "status": "error",
        "message": "Too many attempts. Please wait a bit and try again."
    }), 429


@app.errorhandler(413)
def handle_file_too_large(e):
    # Fires automatically once Config.MAX_CONTENT_LENGTH is exceeded (Flask
    # rejects the request before it even reaches /detect). Without this
    # handler the browser would just get a bare "Request Entity Too Large"
    # HTML page instead of the JSON the frontend expects.
    max_mb = Config.MAX_CONTENT_LENGTH // (1024 * 1024)
    return jsonify({
        "status": "error",
        "message": f"File is too large. Maximum allowed size is {max_mb}MB."
    }), 413


@app.errorhandler(500)
def handle_internal_error(e):
    # Last-resort safety net: ANY route that raises an uncaught exception
    # (not just update_profile) used to fall through to Flask's default
    # HTML error page. Since every frontend call in main.js does
    # `await res.json()`, that HTML would throw inside the try block and
    # get swallowed by a generic "Unable to reach server" / "Something
    # went wrong" alert — hiding the real cause. This guarantees the
    # frontend always gets parseable JSON back, even for bugs we didn't
    # anticipate. The real traceback is still printed to the server
    # console/log by Flask as usual.
    return jsonify({
        "status": "error",
        "message": "Something went wrong on our end. Please try again in a moment."
    }), 500

# ==========================================
# CONFIGURATION & PATH SETUP (all from config.py now)
# ==========================================
BASE_DIR = Config.BASE_DIR
UPLOAD_FOLDER_IMG = Config.UPLOAD_FOLDER_IMG
UPLOAD_FOLDER_VID = Config.UPLOAD_FOLDER_VID
RESULT_FOLDER_IMG = Config.RESULT_FOLDER_IMG
RESULT_FOLDER_VID = Config.RESULT_FOLDER_VID
REPORT_FOLDER = Config.REPORT_FOLDER
DATABASE_PATH = Config.DATABASE_PATH
MODEL_PATH = Config.MODEL_PATH
PROFILE_PHOTOS_FOLDER = Config.PROFILE_PHOTOS_FOLDER

# Which photo extensions we'll accept for a profile photo / face sample.
ALLOWED_PHOTO_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

# Where utils/model_validation.py writes the model's static validation
# accuracy (mAP50) after you run it against a labeled validation set.
# See load_model_accuracy() below for how this is consumed.
MODEL_ACCURACY_PATH = os.path.join(BASE_DIR, 'model_accuracy.json')

app.config['UPLOAD_FOLDER_IMG'] = UPLOAD_FOLDER_IMG
app.config['UPLOAD_FOLDER_VID'] = UPLOAD_FOLDER_VID

# ==========================================
# GLOBAL TARGET MANDATES
# ==========================================
target_mandates = {"helmet": True, "vest": True, "gloves": False, "shoes": True, "glasses": False}

# ==========================================
# GLOBAL REAL-TIME LIVE PPE STATUS TRACKER
# ==========================================
live_ppe_status = {
    "total_persons": 0,
    "helmet": {"status": "❌ Missing", "is_missing": True},
    "vest": {"status": "❌ Missing", "is_missing": True},
    "gloves": {"status": "❌ Missing", "is_missing": True},
    "shoes": {"status": "❌ Missing", "is_missing": True},
    "glasses": {"status": "❌ Missing", "is_missing": True},
    "alert": True,
    "alert_message": "System Ready",
    # Names of any registered users recognized in the current frame (via
    # their uploaded profile photo). "Unknown" faces are omitted from this
    # list — see recognize_faces_in_frame() in utils/face_recognition_helper.py.
    "person_names": [],
}

# ==========================================
# GLOBAL LIVE RECORDING STATE
# ==========================================
recording_state = {
    "active": False, "writer": None, "video_path": None, "start_time": None,
    "total_frames": 0, "violation_frames": 0, "frame_size": None, "aggregate_status": None,
}

# ==========================================
# FACE-RECOGNITION THROTTLING
# ==========================================
# recognize_faces_in_frame() is comparatively expensive (embedding lookup
# against every registered profile photo). Running it on EVERY frame, on
# top of the YOLO PPE inference that already runs every frame, was the
# main source of the live-feed lag reported by the user. Faces don't
# change position fast enough to need re-recognition 20+ times a second,
# so we only run it every FACE_RECOGNITION_INTERVAL frames and reuse the
# last known result (names + boxes) on the frames in between. This keeps
# the on-screen name labels and boxes stable instead of flickering off
# between recognition runs.
FACE_RECOGNITION_INTERVAL = 5


# ==========================================
# LOAD YOLOv8 PPE MODEL
# ==========================================
model = None
if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
    try:
        print(f"✅ Loading custom PPE model from: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"⚠️ Custom model load failed ({e}). Loading fallback model...")
        model = YOLO('yolov8n.pt')
else:
    print("⚠️ Custom PPE model not found at models/yolov8_ppe.pt!")
    model = YOLO('yolov8n.pt')

CONF_THRESHOLD = Config.CONF_THRESHOLD
INFERENCE_IMGSZ = Config.INFERENCE_IMGSZ

# ==========================================
# LIVE-STREAM INFERENCE SIZE
# ==========================================
# Uploaded images/videos (the /detect route) keep using the full
# INFERENCE_IMGSZ from Config for the most accurate result, since that's a
# one-shot operation. The LIVE webcam feed runs YOLO on every single frame
# in a tight loop, so it needs to be fast above all else — a smaller
# inference resolution cuts CPU inference time roughly quadratically (e.g.
# 640->384 is close to a 3x speedup) at a small accuracy cost that's
# acceptable for a live monitoring view. Capped at 384, or lower if
# INFERENCE_IMGSZ was already smaller than that.
LIVE_INFERENCE_IMGSZ = min(INFERENCE_IMGSZ, 384)


# ==========================================
# MODEL ACCURACY (static — read from validation report, NOT recomputed live)
# ==========================================
def load_model_accuracy():
    """
    Returns the model's validation accuracy (mAP50, as a percentage) or
    None if it hasn't been generated yet.

    IMPORTANT: this is intentionally NOT calculated from live detection
    activity (compliant/total scans). A YOLO model's accuracy is a
    property of the *model itself*, measured once against a labeled
    validation set — it doesn't change based on how many people logged in
    or how many of today's scans were compliant. To (re)generate this
    file, run:

        python utils/model_validation.py --data path/to/data.yaml

    That script writes model_accuracy.json to the project root, which
    this function reads on every /api/dashboard_stats call.
    """
    if not os.path.exists(MODEL_ACCURACY_PATH):
        return None
    try:
        with open(MODEL_ACCURACY_PATH, 'r') as f:
            data = json.load(f)
        map50 = data.get('map50')
        if map50 is None:
            return None
        return round(float(map50) * 100, 1)
    except Exception as e:
        print(f"⚠️ Could not read model_accuracy.json: {e}")
        return None


# ==========================================
# SHARED PPE ANALYSIS LOGIC (used by both live cam and uploads)
# ==========================================
def analyze_ppe_labels(detected_labels, mandates):
    labels = [str(l).lower().strip().replace('_', ' ').replace('-', ' ') for l in detected_labels]

    person_detected = any(k in lbl for lbl in labels for k in ['person', 'worker', 'man', 'human', 'people'])
    total_persons = 1 if person_detected or len(labels) > 0 else 0

    found = {"helmet": False, "vest": False, "gloves": False, "shoes": False, "glasses": False}

    positive_keywords = {
        "helmet": ['helmet', 'hardhat', 'safety hat', 'head protection', 'hat'],
        "vest": ['vest', 'safety vest', 'jacket', 'hi vis', 'reflective vest', 'safety jacket'],
        "gloves": ['glove', 'gloves', 'safety glove', 'hand protection', 'hands'],
        "shoes": ['shoe', 'shoes', 'boot', 'boots', 'safety boot', 'footwear', 'safety shoes'],
        "glasses": ['glass', 'glasses', 'goggle', 'goggles', 'eyewear', 'spectacles', 'eye protection', 'face shield'],
    }
    NEGATIVE_PREFIXES = ['no ', 'without ', 'non ']

    for lbl in labels:
        is_negative = any(lbl.startswith(pfx) for pfx in NEGATIVE_PREFIXES)

        stripped = lbl
        for pfx in NEGATIVE_PREFIXES:
            if lbl.startswith(pfx):
                stripped = lbl[len(pfx):]
                break

        for item, positive_kw in positive_keywords.items():
            if any(k in stripped for k in positive_kw):
                if not is_negative:
                    found[item] = True

    def item_status(is_present):
        return ("✓ Present", False) if is_present else ("❌ Missing", True)

    status = {"total_persons": total_persons}
    for item in found:
        st, miss = item_status(found[item])
        status[item] = {"status": st, "is_missing": miss}

    has_alert = any(status[item]["is_missing"] and mandates.get(item, True) for item in found)
    status["alert"] = has_alert

    if total_persons == 0:
        status["alert_message"] = "No Person Detected"
    elif has_alert:
        status["alert_message"] = "⚠️ Safety Violation: Missing Mandatory PPE Detected!"
    else:
        status["alert_message"] = "✓ All Required PPE Compliant"

    return status


def merge_status_into_aggregate(aggregate, frame_status):
    if aggregate is None:
        return frame_status
    for item in ["helmet", "vest", "gloves", "shoes", "glasses"]:
        if not frame_status[item]["is_missing"]:
            aggregate[item] = frame_status[item]
    aggregate["total_persons"] = max(aggregate.get("total_persons", 0), frame_status["total_persons"])
    return aggregate


# ==========================================
# DATABASE HELPERS (detection + login logs)
# ==========================================
def log_detection(source_type, status, file_path, report_path=None, user_id=None):
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO detection_logs (source_type, status, file_path, report_path, timestamp, user_id) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (source_type, status, file_path, report_path,
         datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_id)
    )
    conn.commit()
    conn.close()


def fetch_logs(user_id=None, is_admin=False, limit=20):
    """
    By default (is_admin=False) returns only the given user's own
    detection logs. Admins get every user's logs (user_id is ignored).
    Rows logged before the user_id column existed have user_id = NULL and
    won't show up for any non-admin user — that's expected for old data.
    """
    conn = get_db_connection()
    if is_admin:
        logs = conn.execute(
            "SELECT * FROM detection_logs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    else:
        logs = conn.execute(
            "SELECT * FROM detection_logs WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit)
        ).fetchall()
    conn.close()
    return logs


def log_recording_session(video_path, report_path, start_time, end_time, total_frames, violation_frames, recorded_by):
    conn = get_db_connection()
    conn.execute(
        """INSERT INTO recording_sessions
           (video_path, report_path, start_time, end_time, total_frames, violation_frames, recorded_by)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (video_path, report_path, start_time, end_time, total_frames, violation_frames, recorded_by)
    )
    conn.commit()
    conn.close()


def log_login(user_id, username, ip_address):
    conn = get_db_connection()
    conn.execute("INSERT INTO login_logs (user_id, username, ip_address) VALUES (?, ?, ?)",
                 (user_id, username, ip_address))
    conn.commit()
    conn.close()


def fetch_login_logs(user_id=None, is_admin=False, limit=20):
    """
    By default (is_admin=False) returns only the given user's own login
    history. Admins get everyone's login history.
    """
    conn = get_db_connection()
    if is_admin:
        logs = conn.execute(
            "SELECT * FROM login_logs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    else:
        logs = conn.execute(
            "SELECT * FROM login_logs WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit)
        ).fetchall()
    conn.close()
    return logs


def get_current_user():
    if "user_id" not in session:
        return None
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (session["user_id"],)).fetchone()
    conn.close()
    return user


def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            # Only routes actually called via fetch()/apiFetch() (the SPA's
            # /api/* endpoints, plus the record button's start/stop calls)
            # should get a raw JSON 401 back — the JS there is written to
            # parse JSON and show its "message" field.
            #
            # /detect (a real <form method="POST"> submit) and
            # /download_pdf_report (a plain <a href> click) are FULL BROWSER
            # NAVIGATIONS, not fetch calls. Returning JSON for those makes
            # the browser render the raw JSON as a page (exactly the
            # "Please log in first." text-only page bug) instead of taking
            # the person back into the app. Redirect those back to the
            # homepage instead, flagged so the frontend can pop the login
            # modal there.
            if request.path.startswith("/api/") or request.path in ("/start_recording", "/stop_recording"):
                return jsonify({"status": "error", "message": "Please log in first."}), 401
            return redirect(url_for('index', auth_required=1))
        return f(*args, **kwargs)
    return wrapper


init_db()

# ==========================================
# GUEST ACCOUNT (frictionless demo access, no registration needed)
# ==========================================
# A single shared, non-admin account so visitors can explore the app
# without creating a real account first. It intentionally has NO admin
# rights (can't touch the Admin panel or Target PPE rules), so opening
# access up this way doesn't expose anything sensitive. Credentials now
# come from config.py / .env (SAFEVISION_GUEST_USERNAME / _PASSWORD) —
# change them there before a public demo if you want them less guessable.
GUEST_USERNAME = Config.GUEST_USERNAME
GUEST_PASSWORD = Config.GUEST_PASSWORD


def ensure_guest_account():
    conn = get_db_connection()
    existing = conn.execute("SELECT id FROM users WHERE username = ?", (GUEST_USERNAME,)).fetchone()
    if not existing:
        conn.execute(
            """INSERT INTO users (username, password_hash, full_name, email, role, is_admin, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (GUEST_USERNAME, generate_password_hash(GUEST_PASSWORD), "Guest Visitor",
             "guest@safevision.local", "Guest", 0, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        conn.commit()
        print(f"👤 Guest account created (username: '{GUEST_USERNAME}', password: '{GUEST_PASSWORD}')")
    conn.close()


ensure_guest_account()


def _reset_session_preserving_csrf():
    """
    session.clear() wipes EVERY key in the session, including Flask-WTF's
    csrf_token. But this app is a single-page app — login/logout/guest-login
    happen over AJAX without reloading the page, so the CURRENTLY RENDERED
    HTML (the <meta name="csrf-token"> tag, and the hidden csrf_token field
    inside the Upload form) still embeds the OLD token value. Once
    session.clear() removes that token from the session, the next plain
    form POST (like the Upload tab's "Run AI Detection" button, which is a
    real multipart form submit, not a fetch call) has nothing to validate
    against anymore -> "CSRF session token is missing".

    Preserving the token across the clear() keeps every already-rendered
    form/meta-tag on the page valid without forcing a full page reload
    after logging in, logging out, or using the guest account.
    """
    csrf_token_value = session.get('csrf_token')
    session.clear()
    if csrf_token_value:
        session['csrf_token'] = csrf_token_value


@app.before_request
def _session_timeout_check():
    # Clears session + forces re-login after SESSION_IDLE_TIMEOUT_MINUTES of
    # inactivity. Also marks every session (including anonymous, pre-login
    # ones) as permanent, so the CSRF-token-bearing cookie always gets a
    # real expiry from PERMANENT_SESSION_LIFETIME instead of being a
    # browser-session-only cookie.
    enforce_session_timeout()


# ==========================================
# CAMERA-UNAVAILABLE PLACEHOLDER FRAME
# ==========================================
def _build_camera_unavailable_frame():
    """
    Generates a single static JPEG frame with a "Camera Not Available"
    message. Used by generate_webcam_frames() when cv2.VideoCapture(0)
    can't open a device — e.g. on a cloud/server deployment with no
    physical webcam attached. Without this, /video_feed would just stop
    streaming with no explanation, which looks like a crash in the UI.
    """
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (40, 40, 40)
    text = "Camera Not Available"
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    x = (640 - text_size[0]) // 2
    y = (480 + text_size[1]) // 2
    cv2.putText(frame, text, (x, y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    ret, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes() if ret else None


# ==========================================
# CAMERA OPENING (tries multiple backends/indices, verifies with a real frame)
# ==========================================
def _try_open_camera(index, backend=None, backend_name="default"):
    """
    Opens a single (index, backend) combination and verifies it actually
    delivers a frame — not just that isOpened() reports True.

    This distinction matters: on some Windows setups, cv2.CAP_DSHOW can
    report isOpened() == True while every read() call still fails (the
    "backend is generally available but can't be used to capture by
    index" warning). Relying on isOpened() alone silently accepts a dead
    camera handle, which is what caused the Live Camera tab to sit on the
    "Camera Not Available" placeholder even though OpenCV never raised an
    explicit error.
    """
    try:
        cam = cv2.VideoCapture(index, backend) if backend is not None else cv2.VideoCapture(index)
    except Exception as e:
        print(f"⚠️ Camera open raised for index {index} ({backend_name}): {e}")
        return None

    if not cam.isOpened():
        cam.release()
        return None

    ok, _ = cam.read()
    if not ok:
        print(f"⚠️ Camera index {index} ({backend_name}) opened but produced no frame — trying next option.")
        cam.release()
        return None

    print(f"✅ Camera opened successfully: index {index}, backend {backend_name}.")
    return cam


def _open_camera():
    """
    Tries a short list of (backend, index) combinations in order of speed/
    reliability and returns the first one that actually delivers a frame,
    or None if nothing worked.

    Order rationale:
      - Windows: DSHOW is normally fastest to open, but can silently fail
        to capture by index on some drivers/systems — MSMF is the more
        reliable fallback there, then the OS default.
      - Non-Windows: the default backend (V4L2 on Linux, AVFoundation on
        macOS) is tried directly; CAP_DSHOW is Windows-only and a no-op
        elsewhere.
      - Index 1 is tried after index 0 for every backend, since some
        laptops expose a virtual camera (e.g. an OBS/Iriun driver) at
        index 0 and the real webcam at index 1.
    """
    if platform.system() == "Windows":
        candidates = [
            (0, cv2.CAP_DSHOW, "DSHOW"),
            (0, cv2.CAP_MSMF, "MSMF"),
            (0, None, "default"),
            (1, cv2.CAP_DSHOW, "DSHOW"),
            (1, cv2.CAP_MSMF, "MSMF"),
        ]
    else:
        candidates = [
            (0, None, "default"),
            (1, None, "default"),
        ]

    for index, backend, name in candidates:
        cam = _try_open_camera(index, backend, name)
        if cam is not None:
            return cam

    return None


# ==========================================
# LATEST-FRAME CAMERA READER (prevents growing lag/delay over time)
# ==========================================
class _LatestFrameReader:
    """
    Wraps a cv2.VideoCapture and continuously grabs frames on a dedicated
    background thread, keeping only the SINGLE MOST RECENT frame.

    THE PROBLEM THIS SOLVES: cv2.VideoCapture keeps an internal frame
    buffer. YOLO inference + face recognition together take noticeably
    longer than the interval between camera frames. If the main loop just
    calls camera.read() directly, unread frames pile up in that buffer
    faster than they're consumed, so read() keeps handing back older and
    older backlog frames — the live feed falls further and further behind
    real time the longer it stays open. That's the "delay that keeps
    getting worse" symptom.

    The fix: a background thread does nothing but grab frames as fast as
    the camera can produce them and overwrite a single shared slot each
    time (discarding whatever frame was sitting there before, unread or
    not). The processing loop below always reads that slot, so it always
    works on the newest available frame — inference still takes the same
    amount of time per frame, but the delay no longer compounds over the
    life of the stream.
    """
    def __init__(self, capture):
        self._capture = capture
        self._lock = threading.Lock()
        self._latest_frame = None
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _reader_loop(self):
        while self._running:
            ok, frame = self._capture.read()
            if not ok:
                break
            with self._lock:
                self._latest_frame = frame

    def read(self):
        with self._lock:
            frame = self._latest_frame
        if frame is None:
            return False, None
        return True, frame

    def release(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self._capture.release()


# ==========================================
# WEBCAM STREAMING GENERATOR (with optional recording + face name tagging)
# ==========================================
def generate_webcam_frames():
    raw_camera = _open_camera()

    if raw_camera is None:
        print("❌ Error: Unable to access camera on any backend/index. Serving placeholder frame instead.")
        placeholder = _build_camera_unavailable_frame()
        if placeholder is None:
            return
        # Keep the MJPEG stream alive with a static placeholder instead of
        # closing the connection immediately — the frontend's <img> tag
        # then shows a clear message instead of a broken-image icon.
        while True:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n')
        return

    fps = raw_camera.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 20.0

    # Wrap the raw capture so we always process the newest frame instead
    # of an ever-growing backlog — see _LatestFrameReader above.
    camera = _LatestFrameReader(raw_camera)

    # --- face-recognition throttling state ---
    # frame_counter drives the every-Nth-frame check below.
    # cached_face_matches / cached_recognized_names hold the last actual
    # recognition result so frames in between re-use it instead of running
    # recognize_faces_in_frame() (the expensive call) every single frame.
    frame_counter = 0
    cached_face_matches = []
    cached_recognized_names = []

    try:
        while True:
            success, frame = camera.read()
            if not success:
                # The reader thread hasn't produced its first frame yet
                # (brief startup window) — wait a beat instead of tearing
                # the whole stream down, since the camera itself is fine.
                time.sleep(0.01)
                continue

            results = model(frame, conf=CONF_THRESHOLD, imgsz=LIVE_INFERENCE_IMGSZ)
            detected_classes = [model.names[int(c)] for c in results[0].boxes.cls] if results[0].boxes is not None else []

            frame_status = analyze_ppe_labels(detected_classes, target_mandates)

            # --- face recognition: who is this? (based on their uploaded
            # profile photo) — runs alongside the PPE model, independent of
            # it, so it works even before any PPE model retraining. ---
            # THROTTLED: this is the expensive call, so it only actually
            # runs every FACE_RECOGNITION_INTERVAL frames. On the frames in
            # between, we reuse the last computed result (both the names
            # for the sidebar/report AND the boxes drawn below) so the
            # on-screen labels stay put instead of flickering, while the
            # PPE detection above still runs on every single frame.
            frame_counter += 1
            if frame_counter % FACE_RECOGNITION_INTERVAL == 0:
                cached_face_matches = recognize_faces_in_frame(frame)
                cached_recognized_names = sorted(
                    {m["name"] for m in cached_face_matches if m["name"] != "Unknown"}
                )

            face_matches = cached_face_matches
            frame_status["person_names"] = cached_recognized_names

            live_ppe_status.update(frame_status)

            annotated_frame = results[0].plot()

            # Draw each recognized (or "Unknown") name above its face box
            # directly on the annotated frame, so it shows up both in the
            # live stream image and in the recording, not just the sidebar.
            for match in face_matches:
                x, y, w, h = match["box"]
                label = match["name"]
                color = (34, 197, 94) if label != "Unknown" else (156, 163, 175)  # BGR: green / gray
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
                text_y = max(y - 10, 20)
                cv2.putText(annotated_frame, label, (x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

            if recording_state["active"]:
                if recording_state["writer"] is None:
                    h, w = annotated_frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    recording_state["writer"] = cv2.VideoWriter(recording_state["video_path"], fourcc, fps, (w, h))
                    recording_state["frame_size"] = (w, h)

                recording_state["writer"].write(annotated_frame)
                recording_state["total_frames"] += 1
                if frame_status["alert"]:
                    recording_state["violation_frames"] += 1
                recording_state["aggregate_status"] = merge_status_into_aggregate(
                    recording_state.get("aggregate_status"), frame_status
                )

            # Slightly reduced JPEG quality (default is 95) trims encode
            # time and payload size per frame with no visible difference
            # in a browser <img> stream — another small but real
            # contributor to end-to-end frame latency.
            ret, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret:
                continue

            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        # Always release the camera, whether the loop ended from a read
        # failure, an exception, or the client disconnecting mid-stream —
        # otherwise the device stays locked and the next /video_feed
        # request can't open it (a common cause of "camera busy" bugs).
        camera.release()


# ==========================================
# AUTH ROUTES
# ==========================================
@app.route('/register', methods=['POST'])
@limiter.limit("5 per hour")
def register():
    data = request.get_json() or {}
    username = data.get('username', '').strip()
    password = data.get('password', '')
    full_name = data.get('full_name', '').strip()
    email = data.get('email', '').strip()

    if not username or not password or not full_name or not email:
        return jsonify({"status": "error", "message": "All fields are required."}), 400

    if len(password) < 6:
        return jsonify({"status": "error", "message": "Password must be at least 6 characters."}), 400

    conn = get_db_connection()
    existing = conn.execute("SELECT id FROM users WHERE username = ? COLLATE NOCASE", (username,)).fetchone()
    if existing:
        conn.close()
        return jsonify({"status": "error", "message": "This username is already taken."}), 409

    # First user ever registered becomes Admin automatically (bootstrap).
    # Everyone after that registers as a regular Safety Inspector; an
    # existing Admin can promote other users later from the Admin panel.
    is_first_user = conn.execute("SELECT COUNT(*) as cnt FROM users").fetchone()["cnt"] == 0
    role = "Admin" if is_first_user else "Safety Inspector"

    try:
        conn.execute(
            """INSERT INTO users (username, password_hash, full_name, email, role, is_admin, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (username, generate_password_hash(password), full_name, email, role,
             1 if is_first_user else 0, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        conn.commit()
    except sqlite3.IntegrityError:
        # Covers a UNIQUE constraint on email (username was already checked
        # above, but a race or a UNIQUE(email) constraint could still hit
        # here) — return a clear message instead of a raw 500.
        conn.close()
        return jsonify({"status": "error", "message": "That email address is already registered."}), 409
    conn.close()

    signup_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        if send_signup_notification:
            send_signup_notification(to_email=email, full_name=full_name, username=username)
        else:
            send_login_notification(
                to_email=email,
                full_name=full_name,
                ip_address=request.remote_addr,
                login_time=signup_time
            )
    except Exception as e:
        print(f"⚠️ Signup notification email failed for {email}: {e}")

    return jsonify({"status": "success", "message": "Account created successfully. You can now log in."})


@app.route('/login', methods=['POST'])
@limiter.limit("5 per minute")
def login():
    data = request.get_json() or {}
    username = data.get('username', '').strip()
    password = data.get('password', '')

    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE username = ? COLLATE NOCASE", (username,)).fetchone()
    conn.close()

    if user is not None and is_account_locked(user):
        return jsonify({
            "status": "error",
            "message": "This account is temporarily locked due to repeated failed login attempts. Try again later."
        }), 423

    if user is None or not check_password_hash(user['password_hash'], password):
        if user is not None:
            register_failed_login(username)
        return jsonify({"status": "error", "message": "Invalid username or password."}), 401

    clear_failed_logins(user['id'])

    _reset_session_preserving_csrf()
    session['user_id'] = user['id']
    session['username'] = user['username']
    session['full_name'] = user['full_name']
    # session.permanent is set globally by enforce_session_timeout() in the
    # before_request hook (see utils/security.py) — the reset above only
    # wipes session data (preserving csrf_token), not the cookie's
    # permanent/non-permanent flag.

    login_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_login(user['id'], user['username'], request.remote_addr)

    send_login_notification(
        to_email=user['email'],
        full_name=user['full_name'],
        ip_address=request.remote_addr,
        login_time=login_time
    )

    return jsonify({
        "status": "success", "full_name": user['full_name'], "email": user['email'],
        "role": user['role'], "is_admin": bool(user['is_admin']),
        "profile_photo": user['profile_photo']
    })


@app.route('/guest_login', methods=['POST'])
@limiter.limit("30 per hour")
def guest_login():
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE username = ?", (GUEST_USERNAME,)).fetchone()
    conn.close()

    if user is None:
        return jsonify({"status": "error", "message": "Guest access is not available right now."}), 500

    _reset_session_preserving_csrf()
    session['user_id'] = user['id']
    session['username'] = user['username']
    session['full_name'] = user['full_name']
    # see note in login() above — session.permanent is already handled
    # globally by enforce_session_timeout()

    log_login(user['id'], user['username'], request.remote_addr)

    return jsonify({
        "status": "success", "full_name": user['full_name'], "email": user['email'],
        "role": user['role'], "is_admin": bool(user['is_admin']),
        "profile_photo": user['profile_photo']
    })


@app.route('/logout', methods=['POST'])
def logout():
    _reset_session_preserving_csrf()
    return jsonify({"status": "success"})


@app.route('/api/current_user')
def api_current_user():
    user = get_current_user()
    if not user:
        return jsonify({"logged_in": False})
    return jsonify({"logged_in": True, "full_name": user['full_name'], "email": user['email'],
                     "role": user['role'], "username": user['username'], "is_admin": bool(user['is_admin']),
                     "profile_photo": user['profile_photo']})


# ==========================================
# FORGOT / RESET PASSWORD
# ==========================================
@app.route('/forgot_password', methods=['POST'])
@limiter.limit("3 per hour")
def forgot_password():
    data = request.get_json() or {}
    email = data.get('email', '').strip()

    generic_message = ("If an account with that email exists, we've sent a password "
                        "reset link to it.")

    if not email:
        return jsonify({"status": "error", "message": "Email is required."}), 400

    token = create_password_reset_token(email)

    if token:
        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        conn.close()

        reset_url = f"{request.host_url.rstrip('/')}/reset_password/{token}"
        try:
            if send_password_reset_email:
                send_password_reset_email(to_email=email, full_name=user['full_name'], reset_url=reset_url)
            else:
                print(f"⚠️ send_password_reset_email() not implemented in email_notifier.py. "
                      f"Reset link for {email}: {reset_url}")
        except Exception as e:
            print(f"⚠️ Password reset email failed for {email}: {e}")

    # Always return the same generic message — never reveal whether the email exists.
    return jsonify({"status": "success", "message": generic_message})


@app.route('/reset_password/<token>', methods=['GET'])
def reset_password_page(token):
    user = verify_password_reset_token(token)
    return render_template('reset_password.html', token=token, valid=user is not None)


@app.route('/api/reset_password', methods=['POST'])
@limiter.limit("5 per hour")
def api_reset_password():
    data = request.get_json() or {}
    token = data.get('token', '')
    new_password = data.get('password', '')

    if len(new_password) < 6:
        return jsonify({"status": "error", "message": "Password must be at least 6 characters."}), 400

    user = verify_password_reset_token(token)
    if user is None:
        return jsonify({"status": "error", "message": "This reset link is invalid or has expired."}), 400

    consume_password_reset_token(user['id'], generate_password_hash(new_password))
    return jsonify({"status": "success", "message": "Password updated. You can now log in."})


# ==========================================
# ADMIN PANEL (role management, target rules restricted to Admin)
# ==========================================
@app.route('/api/admin/users', methods=['GET'])
@login_required
@admin_required
def admin_list_users():
    conn = get_db_connection()
    users = conn.execute(
        "SELECT id, username, full_name, email, role, is_admin, created_at FROM users ORDER BY id ASC"
    ).fetchall()
    conn.close()
    return jsonify([dict(u) for u in users])


@app.route('/api/admin/users/<int:user_id>/role', methods=['POST'])
@login_required
@admin_required
def admin_set_user_role(user_id):
    data = request.get_json() or {}
    make_admin = bool(data.get('is_admin'))

    if user_id == session.get('user_id') and not make_admin:
        return jsonify({"status": "error", "message": "You can't remove your own admin access."}), 400

    conn = get_db_connection()
    conn.execute(
        "UPDATE users SET is_admin = ?, role = ? WHERE id = ?",
        (1 if make_admin else 0, "Admin" if make_admin else "Safety Inspector", user_id)
    )
    conn.commit()
    conn.close()

    log_admin_action(session.get('username', 'unknown'),
                      "grant_admin" if make_admin else "revoke_admin", target=f"user_id={user_id}")

    return jsonify({"status": "success", "message": "Role updated."})


@app.route('/api/admin/users/<int:user_id>', methods=['DELETE'])
@login_required
@admin_required
def admin_delete_user(user_id):
    if user_id == session.get('user_id'):
        return jsonify({"status": "error", "message": "You can't delete your own account."}), 400

    conn = get_db_connection()
    conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()

    # Also drop this user's face sample and retrain the recognizer without
    # it, so a deleted account doesn't keep getting identified on the live
    # feed (or keep occupying a label slot) after their account is gone.
    remove_user_face(user_id)

    log_admin_action(session.get('username', 'unknown'), "delete_user", target=f"user_id={user_id}")
    return jsonify({"status": "success", "message": "User deleted."})


# ==========================================
# CORE ROUTES
# ==========================================
@app.route('/')
def index():
    user = get_current_user()
    user_id = user['id'] if user else None
    is_admin = bool(user['is_admin']) if user else False
    return render_template(
        'index.html',
        logs=fetch_logs(user_id=user_id, is_admin=is_admin),
        login_logs=fetch_login_logs(user_id=user_id, is_admin=is_admin),
        is_admin=is_admin,
        # Set by login_required() when /detect or /download_pdf_report
        # redirected a logged-out visitor back here — tells the frontend to
        # pop the login modal instead of leaving them guessing why nothing
        # happened.
        auth_required=request.args.get('auth_required')
    )


@app.route('/api/live_status')
def live_status():
    return jsonify(live_ppe_status)


@app.route('/api/dashboard_stats')
def dashboard_stats():
    conn = get_db_connection()
    total_scanned = conn.execute("SELECT COUNT(*) as cnt FROM detection_logs").fetchone()["cnt"]
    violations = conn.execute("SELECT COUNT(*) as cnt FROM detection_logs WHERE status LIKE '%Violation%'").fetchone()["cnt"]
    conn.close()

    compliant = max(total_scanned - violations, 0)

    # model_accuracy is a static property of the trained model (mAP50 from
    # a validation run), NOT derived from today's compliance ratio. It only
    # changes when you re-run utils/model_validation.py after retraining.
    model_accuracy = load_model_accuracy()

    return jsonify({
        "total_scanned": total_scanned,
        "compliant_workers": compliant,
        "safety_violations": violations,
        "model_accuracy": model_accuracy if model_accuracy is not None else "N/A",
    })


@app.route('/api/analytics')
@login_required
def api_analytics():
    conn = get_db_connection()

    trend_rows = conn.execute(
        """SELECT strftime('%Y-%m-%d', timestamp) as day,
                  COUNT(*) as total,
                  SUM(CASE WHEN status LIKE '%Violation%' THEN 1 ELSE 0 END) as violations
           FROM detection_logs
           WHERE date(timestamp) >= date('now', '-6 days')
           GROUP BY day
           ORDER BY day ASC"""
    ).fetchall()
    trend_map = {row['day']: {'total': row['total'], 'violations': row['violations']} for row in trend_rows}

    trend = []
    for i in range(6, -1, -1):
        day = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
        entry = trend_map.get(day, {'total': 0, 'violations': 0})
        trend.append({"date": day, "total": entry['total'], "violations": entry['violations']})

    source_rows = conn.execute(
        "SELECT source_type, COUNT(*) as cnt FROM detection_logs GROUP BY source_type"
    ).fetchall()
    source_breakdown = {row['source_type']: row['cnt'] for row in source_rows}

    total_scanned = conn.execute("SELECT COUNT(*) as cnt FROM detection_logs").fetchone()['cnt']
    violations = conn.execute(
        "SELECT COUNT(*) as cnt FROM detection_logs WHERE status LIKE '%Violation%'"
    ).fetchone()['cnt']
    conn.close()

    compliant = max(total_scanned - violations, 0)

    return jsonify({
        "trend": trend,
        "source_breakdown": source_breakdown,
        "compliance_ratio": {"compliant": compliant, "violations": violations}
    })


@app.route('/api/detection_logs')
@login_required
def api_detection_logs():
    user = get_current_user()
    logs = fetch_logs(user_id=user['id'], is_admin=bool(user['is_admin']))
    return jsonify([dict(row) for row in logs])


@app.route('/api/login_logs')
@login_required
@admin_required
def api_login_logs():
    # Admin-only: normal users should not be able to pull everyone's
    # login history directly from the API even if the Login History table
    # is hidden in their UI. @admin_required returns a 403 JSON response
    # for anyone who isn't an admin — see utils/security.py.
    user = get_current_user()
    logs = fetch_login_logs(user_id=user['id'], is_admin=bool(user['is_admin']))
    return jsonify([dict(row) for row in logs])


@app.route('/api/targets', methods=['POST'])
@login_required
@admin_required
def save_targets():
    data = request.get_json() or {}
    for key in ['helmet', 'vest', 'gloves', 'shoes', 'glasses']:
        if key in data:
            target_mandates[key] = bool(data.get(key))

    log_admin_action(session.get('username', 'unknown'), "update_targets", target=str(target_mandates))
    return jsonify({"status": "success", "message": "Target PPE configuration saved successfully."})


@app.route('/api/update_profile', methods=['POST'])
@login_required
def update_profile():
    full_name = request.form.get('full_name')
    email = request.form.get('email')
    photo_file = request.files.get('photo')

    conn = get_db_connection()

    # --- update name/email, with a clear error if the email is already
    # taken by another account, instead of an uncaught IntegrityError
    # crashing this route into a non-JSON 500 response (which is what
    # produced the generic "Could not update profile" alert in the UI). ---
    try:
        conn.execute(
            "UPDATE users SET full_name = COALESCE(?, full_name), email = COALESCE(?, email) WHERE id = ?",
            (full_name, email, session['user_id'])
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({
            "status": "error",
            "message": "That email address is already in use by another account."
        }), 409
    except Exception as e:
        conn.close()
        app.logger.error(f"Profile name/email update failed for user {session['user_id']}: {e}")
        return jsonify({
            "status": "error",
            "message": "Could not update your profile details. Please try again."
        }), 500

    photo_url = None
    face_message = None

    # --- optional profile photo upload: also registers/updates this
    # user's face for live-camera name recognition ---
    if photo_file and photo_file.filename:
        ext = photo_file.filename.rsplit('.', 1)[-1].lower() if '.' in photo_file.filename else ''
        if ext not in ALLOWED_PHOTO_EXTENSIONS:
            conn.close()
            return jsonify({"status": "error",
                             "message": "Please upload a JPG, PNG, or WEBP photo."}), 400

        try:
            # Safety net in case PROFILE_PHOTOS_FOLDER was never created by
            # Config.ensure_folders_exist() (e.g. after a manual folder
            # deletion, or a config change that missed it).
            os.makedirs(PROFILE_PHOTOS_FOLDER, exist_ok=True)

            safe_name = secure_filename(f"user_{session['user_id']}.{ext}")
            save_path = os.path.join(PROFILE_PHOTOS_FOLDER, safe_name)
            photo_file.save(save_path)

            photo_url = f"/profile_photos/{safe_name}"
            conn.execute("UPDATE users SET profile_photo = ? WHERE id = ?", (photo_url, session['user_id']))
            conn.commit()
        except Exception as e:
            conn.close()
            app.logger.error(f"Profile photo save failed for user {session['user_id']}: {e}")
            return jsonify({
                "status": "error",
                "message": "Your name/email were saved, but the photo upload failed. Please try uploading it again."
            }), 500

        # Face registration failing should NOT fail the whole request —
        # the name/email and photo are already saved successfully by now.
        try:
            face_ok, face_message = register_face(session['user_id'], save_path)
            if not face_ok:
                app.logger.info(f"Face registration skipped for user {session['user_id']}: {face_message}")
        except Exception as e:
            app.logger.error(f"register_face() raised for user {session['user_id']}: {e}")
            face_message = "Photo saved, but face recognition setup failed. You can try re-uploading later."

    conn.close()

    if full_name:
        session['full_name'] = full_name

    response = {"status": "success", "full_name": full_name, "email": email}
    if photo_url:
        response["profile_photo"] = photo_url
    if face_message:
        response["face_message"] = face_message

    return jsonify(response)


@app.route('/profile_photos/<path:filename>')
def serve_profile_photo(filename):
    return send_from_directory(PROFILE_PHOTOS_FOLDER, filename)


@app.route('/download_pdf_report')
@login_required
def download_pdf_report():
    user = get_current_user()
    buffer = generate_live_snapshot_report(
        live_ppe_status, target_mandates,
        inspector_name=user['full_name'] if user else 'N/A',
        inspector_email=user['email'] if user else 'N/A'
    )
    return send_file(buffer, as_attachment=True, download_name="Live_PPE_Safety_Report.pdf", mimetype='application/pdf')


@app.route('/detect', methods=['POST'])
@login_required
def detect():
    if 'file' not in request.files or request.files['file'].filename == '':
        return redirect(url_for('index'))

    user = get_current_user()
    file = request.files['file']
    filename = file.filename
    ext = filename.rsplit('.', 1)[-1].lower()
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    result_image = result_video = report_url = None

    if ext in ['jpg', 'jpeg', 'png', 'webp']:
        save_path = os.path.join(app.config['UPLOAD_FOLDER_IMG'], f"{timestamp_str}_{filename}")
        file.save(save_path)

        results = model.predict(source=save_path, conf=CONF_THRESHOLD, imgsz=INFERENCE_IMGSZ)
        annotated_img = results[0].plot()
        detected_classes = [model.names[int(c)] for c in results[0].boxes.cls] if results[0].boxes is not None else []
        status = analyze_ppe_labels(detected_classes, target_mandates)

        out_filename = f"annotated_{timestamp_str}_{filename}"
        cv2.imwrite(os.path.join(RESULT_FOLDER_IMG, out_filename), annotated_img)
        result_image = f"/static_results/annotated_images/{out_filename}"

        report_buffer = generate_detection_report(
            "Image Upload", filename, status, target_mandates,
            inspector_name=user['full_name'] if user else 'N/A',
            inspector_email=user['email'] if user else 'N/A'
        )
        report_filename = f"report_{timestamp_str}.pdf"
        with open(os.path.join(REPORT_FOLDER, report_filename), 'wb') as f:
            f.write(report_buffer.getvalue())
        report_url = f"/static_results/reports/{report_filename}"

        log_detection("Image Upload", status["alert_message"], result_image, report_url,
                      user_id=user['id'] if user else None)

        if status.get("alert") and user:
            send_violation_alert(
                to_email=user['email'],
                full_name=user['full_name'],
                source_type="Image Upload",
                status=status,
                file_url=request.host_url.rstrip('/') + result_image
            )

    elif ext in ['mp4', 'avi', 'mov', 'mkv']:
        save_path = os.path.join(app.config['UPLOAD_FOLDER_VID'], f"{timestamp_str}_{filename}")
        file.save(save_path)

        cap = cv2.VideoCapture(save_path)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        out_filename = f"annotated_{timestamp_str}.mp4"
        out = cv2.VideoWriter(os.path.join(RESULT_FOLDER_VID, out_filename),
                               cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        aggregate_status = None
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame, conf=CONF_THRESHOLD, imgsz=INFERENCE_IMGSZ)
            out.write(results[0].plot())

            detected_classes = [model.names[int(c)] for c in results[0].boxes.cls] if results[0].boxes is not None else []
            frame_status = analyze_ppe_labels(detected_classes, target_mandates)
            aggregate_status = merge_status_into_aggregate(aggregate_status, frame_status)

        cap.release()
        out.release()

        aggregate_status = aggregate_status or analyze_ppe_labels([], target_mandates)
        result_video = f"/static_results/annotated_videos/{out_filename}"

        report_buffer = generate_detection_report(
            "Video Upload", filename, aggregate_status, target_mandates,
            inspector_name=user['full_name'] if user else 'N/A',
            inspector_email=user['email'] if user else 'N/A'
        )
        report_filename = f"report_{timestamp_str}.pdf"
        with open(os.path.join(REPORT_FOLDER, report_filename), 'wb') as f:
            f.write(report_buffer.getvalue())
        report_url = f"/static_results/reports/{report_filename}"

        log_detection("Video Upload", aggregate_status["alert_message"], result_video, report_url,
                      user_id=user['id'] if user else None)

        if aggregate_status.get("alert") and user:
            send_violation_alert(
                to_email=user['email'],
                full_name=user['full_name'],
                source_type="Video Upload",
                status=aggregate_status,
                file_url=request.host_url.rstrip('/') + result_video
            )

    return render_template(
        'index.html', result_image=result_image, result_video=result_video, report_url=report_url,
        logs=fetch_logs(user_id=user['id'] if user else None, is_admin=bool(user['is_admin']) if user else False),
        login_logs=fetch_login_logs(user_id=user['id'] if user else None, is_admin=bool(user['is_admin']) if user else False),
        is_admin=bool(user['is_admin']) if user else False
    )


# ==========================================
# LIVE RECORDING CONTROL
# ==========================================
@app.route('/start_recording', methods=['POST'])
@login_required
def start_recording():
    if recording_state["active"]:
        return jsonify({"status": "error", "message": "Recording already in progress."}), 400

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(RESULT_FOLDER_VID, f"live_recording_{timestamp_str}.mp4")

    recording_state.update({
        "active": True, "writer": None, "video_path": video_path, "start_time": datetime.now(),
        "total_frames": 0, "violation_frames": 0, "aggregate_status": None,
    })
    return jsonify({"status": "success", "message": "Recording started."})


@app.route('/stop_recording', methods=['POST'])
@login_required
def stop_recording():
    if not recording_state["active"]:
        return jsonify({"status": "error", "message": "No active recording."}), 400

    user = get_current_user()
    recording_state["active"] = False

    if recording_state["writer"] is not None:
        recording_state["writer"].release()
        recording_state["writer"] = None

    end_time = datetime.now()
    start_time = recording_state["start_time"]
    duration = (end_time - start_time).total_seconds() if start_time else 0
    aggregate_status = recording_state.get("aggregate_status") or analyze_ppe_labels([], target_mandates)

    session_data = {
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S") if start_time else "N/A",
        "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": duration,
        "video_path": os.path.basename(recording_state["video_path"]),
        "total_frames": recording_state["total_frames"],
        "violation_frames": recording_state["violation_frames"],
        "aggregate_status": aggregate_status,
    }

    report_buffer = generate_live_session_report(
        session_data, target_mandates,
        inspector_name=user['full_name'] if user else 'N/A',
        inspector_email=user['email'] if user else 'N/A'
    )
    report_filename = f"session_report_{start_time.strftime('%Y%m%d_%H%M%S') if start_time else 'unknown'}.pdf"
    with open(os.path.join(REPORT_FOLDER, report_filename), 'wb') as f:
        f.write(report_buffer.getvalue())

    video_url = f"/static_results/annotated_videos/{os.path.basename(recording_state['video_path'])}"
    report_url = f"/static_results/reports/{report_filename}"

    log_recording_session(video_url, report_url, session_data["start_time"], session_data["end_time"],
                           session_data["total_frames"], session_data["violation_frames"],
                           user['username'] if user else 'unknown')
    log_detection("Live Recording", aggregate_status["alert_message"], video_url, report_url,
                  user_id=user['id'] if user else None)

    if aggregate_status.get("alert") and user:
        send_violation_alert(
            to_email=user['email'],
            full_name=user['full_name'],
            source_type="Live Recording Session",
            status=aggregate_status,
            file_url=request.host_url.rstrip('/') + video_url
        )

    return jsonify({"status": "success", "video_url": video_url, "report_url": report_url,
                     "total_frames": session_data["total_frames"],
                     "violation_frames": session_data["violation_frames"], "duration_seconds": duration})


@app.route('/video_feed')
def video_feed():
    return Response(generate_webcam_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/static_results/<path:filename>')
def serve_results(filename):
    return send_from_directory(os.path.join(BASE_DIR, 'results'), filename)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting SafeVision AI Server on http://0.0.0.0:{port} ...")
    app.run(host='0.0.0.0', port=port, debug=app.config["DEBUG"], use_reloader=False)