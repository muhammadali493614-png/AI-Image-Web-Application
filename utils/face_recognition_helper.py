"""
utils/face_recognition_helper.py

Turns a user's uploaded profile photo into a face the live camera stream
can recognize, using OpenCV's built-in LBPH face recognizer (opencv-contrib
-> cv2.face). No external face-recognition service or heavy dlib install
needed — this stays inside your existing opencv-contrib-python dependency.

How it fits together:
    1. User uploads a profile photo -> register_face() finds the largest
       face in it, saves a normalized grayscale crop under
       models/face_samples/user_<id>.png, and retrains the shared
       recognizer over every registered user's sample.
    2. The live camera loop (app.py: generate_webcam_frames) calls
       recognize_faces_in_frame() on every frame and draws each person's
       name next to their PPE detection box.
    3. admin_delete_user() calls remove_user_face() so a deleted account's
       face sample doesn't linger and keep getting retrained forever.

Auto-generated files (do NOT create these by hand — they're written here):
    models/face_recognizer.yml   — the trained LBPH model
    models/face_labels.json      — {label_id: {user_id, full_name}}
    models/face_samples/*.png    — one normalized face crop per user

NOTE ON EMPTY/CORRUPT MODEL FILES:
    If face_recognizer.yml or face_labels.json ever end up on disk as
    0-byte (or otherwise unparsable) files — e.g. a previous training run
    crashed mid-write, or they were manually touched into existence —
    cv2.face.LBPHFaceRecognizer.read() / cv2.FileStorage will raise a
    persistence.cpp "buf" assertion. _ensure_loaded() below guards against
    this by checking file size before attempting a read, and by deleting
    any file that still fails to parse so it doesn't keep crashing every
    single frame afterwards.
"""

import os
import json
import threading

import cv2
import numpy as np

from config import Config

# Fixed size every face crop is normalized to before training/prediction —
# LBPH requires all training images to be the same dimensions.
FACE_SIZE = (200, 200)

_LOCK = threading.Lock()
_recognizer = None      # lazily loaded cv2.face_LBPHFaceRecognizer
_labels = {}             # {"0": {"user_id": 3, "full_name": "Ali Khan"}, ...}
_cascade = None          # lazily loaded Haar cascade face detector
_load_attempted = False  # prevents re-trying a known-corrupt file on every frame
_clahe = None            # lazily created CLAHE normalizer (see _normalize_face below)

# Prints "<name-or-Unknown> distance=NN.N (threshold=NN)" to the console for
# every recognized face on every frame. Turn this off once you've picked a
# good FACE_RECOGNITION_CONFIDENCE_THRESHOLD value — it's noisy on a live
# stream, but invaluable while tuning: it tells you exactly how far the
# real match distance is from your current threshold instead of guessing.
DEBUG_LOG_CONFIDENCE = True


def _get_cascade():
    global _cascade
    if _cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _cascade = cv2.CascadeClassifier(cascade_path)
    return _cascade


def _get_clahe():
    """CLAHE (Contrast Limited Adaptive Histogram Equalization) instead of
    plain equalizeHist. Plain equalizeHist normalizes contrast globally
    across the whole face, which is why a well-lit static profile photo and
    a dim/uneven webcam frame of the SAME person can still land far apart
    in LBPH's distance space — global brightness differences dominate the
    comparison. CLAHE normalizes contrast in small local tiles instead, so
    it's much more robust to the lighting mismatch between an uploaded
    profile photo and a live camera feed, without needing multiple training
    photos per user."""
    global _clahe
    if _clahe is None:
        _clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return _clahe


def _normalize_face(gray_roi):
    """Resizes + CLAHE-normalizes a grayscale face crop. Used identically
    at both training time (register_face) and prediction time
    (recognize_faces_in_frame) — the two MUST stay in sync, since LBPH
    compares raw pixel patterns, not learned features. If you change this
    function, every user must re-upload their profile photo afterward so
    their stored sample gets rebuilt with the new preprocessing."""
    roi = cv2.resize(gray_roi, FACE_SIZE)
    roi = _get_clahe().apply(roi)
    return roi


def _detect_largest_face(gray_image):
    """Returns a normalized (FACE_SIZE, CLAHE-equalized) crop of the largest
    face found in a grayscale image, or None if no face was detected."""
    cascade = _get_cascade()
    faces = cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])  # pick the largest by area
    roi = gray_image[y:y + h, x:x + w]
    return _normalize_face(roi)


def register_face(user_id, image_path):
    """
    Detects the largest face in the uploaded profile photo, saves it as
    this user's canonical face sample, then retrains the recognizer over
    every registered user. Call this right after saving a user's uploaded
    profile photo to disk.

    Returns (success: bool, message: str).
    """
    img = cv2.imread(image_path)
    if img is None:
        return False, "Could not read the uploaded image."

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_roi = _detect_largest_face(gray)
    if face_roi is None:
        return False, ("No face detected in the uploaded photo. Please upload a clear, "
                        "front-facing photo with good lighting.")

    os.makedirs(Config.FACE_SAMPLES_DIR, exist_ok=True)
    sample_path = os.path.join(Config.FACE_SAMPLES_DIR, f"user_{user_id}.png")
    cv2.imwrite(sample_path, face_roi)

    _retrain()
    return True, "Face registered — this person will now be identified by name on the live camera feed."


def remove_user_face(user_id):
    """
    Deletes a user's saved face sample and retrains the recognizer without
    it. Call this from wherever a user account gets deleted (e.g.
    admin_delete_user in app.py) so their face doesn't keep getting
    trained into the model after the account is gone.
    """
    sample_path = os.path.join(Config.FACE_SAMPLES_DIR, f"user_{user_id}.png")
    if os.path.exists(sample_path):
        os.remove(sample_path)
    _retrain()


def _clear_model_files():
    """Removes any on-disk recognizer/labels files. Used both when there
    are zero samples to train on, and when a corrupt file is detected."""
    for path in (Config.FACE_RECOGNIZER_PATH, Config.FACE_LABELS_PATH):
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError as e:
                print(f"⚠️ Could not remove stale face model file {path}: {e}")


def _retrain():
    """
    Rebuilds the LBPH recognizer from every face_samples/user_<id>.png
    file on disk, pulling each user's current full_name from the database
    so labels stay in sync with profile edits. Safe to call with zero
    samples (just clears the model). Imports db_helper lazily to avoid a
    circular import at module load time (db_helper doesn't import this
    module, but app.py imports both).
    """
    from database.db_helper import get_db_connection

    global _recognizer, _labels, _load_attempted

    os.makedirs(Config.FACE_SAMPLES_DIR, exist_ok=True)
    sample_files = sorted(
        f for f in os.listdir(Config.FACE_SAMPLES_DIR)
        if f.startswith("user_") and f.endswith(".png")
    )

    if not sample_files:
        with _LOCK:
            _recognizer = None
            _labels = {}
            _load_attempted = True  # nothing to load — don't let _ensure_loaded try again
            _clear_model_files()
        return

    conn = get_db_connection()
    faces, label_ids, labels_map = [], [], {}
    next_label = 0

    for fname in sample_files:
        try:
            uid = int(fname[len("user_"):-len(".png")])
        except ValueError:
            continue

        user_row = conn.execute("SELECT id, full_name FROM users WHERE id = ?", (uid,)).fetchone()
        if user_row is None:
            continue  # stale sample for a since-deleted user — ignored, cleaned up on next register/delete

        img = cv2.imread(os.path.join(Config.FACE_SAMPLES_DIR, fname), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        faces.append(img)
        label_ids.append(next_label)
        labels_map[str(next_label)] = {"user_id": uid, "full_name": user_row["full_name"]}
        next_label += 1

    conn.close()

    if not faces:
        with _LOCK:
            _recognizer = None
            _labels = {}
            _load_attempted = True
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(label_ids))

    # Hold the lock across the disk write + in-memory swap so a concurrent
    # recognize_faces_in_frame() call can't read a half-written .yml file
    # (write() isn't atomic) or a labels dict that's out of sync with it.
    with _LOCK:
        os.makedirs(Config.FACE_MODEL_DIR, exist_ok=True)
        recognizer.write(Config.FACE_RECOGNIZER_PATH)
        with open(Config.FACE_LABELS_PATH, "w") as f:
            json.dump(labels_map, f)

        _recognizer = recognizer
        _labels = labels_map
        _load_attempted = True


def _is_usable_file(path):
    """A file only counts as loadable if it exists AND has content — a
    0-byte file (interrupted write, manually created stub, etc.) is what
    triggers OpenCV's persistence.cpp 'buf' assertion if you try to read
    it, so we treat it the same as 'missing'."""
    return os.path.exists(path) and os.path.getsize(path) > 0


def _ensure_loaded():
    """Lazily loads the on-disk recognizer/labels the first time they're
    needed (e.g. right after app startup, before any register_face() call
    has happened in this process).

    Only attempts the load once per process per known-good state: if the
    files are missing/empty/corrupt, it marks itself as attempted so it
    doesn't re-try (and re-fail) on every single camera frame. A
    successful register_face() -> _retrain() call resets this via the
    _LOCK-protected globals, so recognition picks up automatically once a
    real model exists.
    """
    global _recognizer, _labels, _load_attempted

    if _load_attempted:
        return
    with _LOCK:
        if _load_attempted:
            return

        if not (_is_usable_file(Config.FACE_RECOGNIZER_PATH) and _is_usable_file(Config.FACE_LABELS_PATH)):
            # Nothing valid to load yet (or one/both files are empty stubs).
            # Clean up any empty leftovers so a future check doesn't have to
            # reason about partial state, then wait for the first successful
            # register_face() to populate things for real.
            _clear_model_files()
            _load_attempted = True
            return

        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(Config.FACE_RECOGNIZER_PATH)
            with open(Config.FACE_LABELS_PATH, "r") as f:
                labels_map = json.load(f)
            _recognizer = recognizer
            _labels = labels_map
        except Exception as e:
            # Corrupt file that nonetheless has nonzero size (e.g. write
            # was interrupted partway) — remove it instead of leaving it to
            # fail this same read on every future frame/request.
            print(f"⚠️ Could not load face recognizer, removing corrupt file(s): {e}")
            _clear_model_files()
            _recognizer = None
            _labels = {}
        finally:
            _load_attempted = True


def recognize_faces_in_frame(frame_bgr):
    """
    Detects every face in a BGR frame and, for each, returns the best
    matching registered user if the recognizer is confident enough.

    Returns a list of dicts, one per detected face:
        [{"name": "Ali", "confidence": 42.1, "box": (x, y, w, h)}, ...]

    "name" is "Unknown" when nobody registered matches closely enough.
    LBPH's predict() confidence is actually a *distance* — lower means a
    better match — compared against Config.FACE_RECOGNITION_CONFIDENCE_THRESHOLD.
    """
    _ensure_loaded()

    cascade = _get_cascade()
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    results = []
    for (x, y, w, h) in faces:
        name = "Unknown"
        confidence = None

        if _recognizer is not None and _labels:
            roi = cv2.resize(gray[y:y + h, x:x + w], FACE_SIZE)
            roi = cv2.equalizeHist(roi)
            label_id, distance = _recognizer.predict(roi)
            confidence = round(float(distance), 1)
            if distance <= Config.FACE_RECOGNITION_CONFIDENCE_THRESHOLD:
                entry = _labels.get(str(label_id))
                if entry:
                    name = entry["full_name"]

        results.append({"name": name, "confidence": confidence, "box": (int(x), int(y), int(w), int(h))})

    return results