"""
utils/face_recognition_helper.py

Turns a user's uploaded profile photo into a face the live camera stream
can recognize, using OpenCV's built-in LBPH face recognizer (opencv-contrib
-> cv2.face). No external face-recognition service or heavy dlib install
needed — this stays inside your existing opencv-contrib-python dependency.

How it fits together:
    1. User uploads a profile photo -> register_face() finds the largest
       face in it, generates several augmented variants of it (different
       small rotations / brightness / a horizontal flip — see
       _augment_face), saves each under
       models/face_samples/user_<id>_<n>.png, and retrains the shared
       recognizer over every registered user's samples.
    2. The live camera loop (app.py: generate_webcam_frames) calls
       recognize_faces_in_frame() on every Nth frame and draws each
       person's name next to their PPE detection box.
    3. admin_delete_user() calls remove_user_face() so a deleted account's
       face samples don't linger and keep getting retrained forever.

FACE DETECTION (2026-08-18 upgrade):
    Detection now prefers OpenCV's SSD-based DNN face detector (res10
    300x300) over the old Haar cascade whenever its two model files are
    present in models/ (see _get_dnn_net() docstring for the one-time
    download commands). The DNN detector handles off-angle faces and
    small/distant faces far better than Haar, which was frontal-only and
    needed a face to fill a good chunk of the frame. If the model files
    aren't downloaded yet, everything automatically falls back to the
    original Haar cascade — nothing breaks, it's just shorter-range.

WHY AUGMENTATION HELPS AT A DISTANCE/ANGLE:
    Users only upload ONE profile photo, almost always a straight-on,
    well-lit shot. LBPH compares raw pixel patterns, so a live camera view
    of the same person turned slightly to the side, or farther away (lower
    resolution), can land surprisingly far from that single stored sample
    in LBPH's distance space — even though it's obviously the same person
    to a human. register_face() now generates several synthetic variants
    of the single uploaded photo (small rotations, brighter/darker,
    mirrored) and trains on all of them, so the recognizer has a wider net
    to match against without asking users for multiple photos.

Auto-generated files (do NOT create these by hand — they're written here):
    models/face_recognizer.yml     — the trained LBPH model
    models/face_labels.json        — {label_id: {user_id, full_name}}
    models/face_samples/*.png      — augmented face crops, several per user

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
import glob
import json
import threading

import cv2
import numpy as np

from config import Config

# Fixed size every face crop is normalized to before training/prediction —
# LBPH requires all training images to be the same dimensions.
FACE_SIZE = (200, 200)

_LOCK = threading.Lock()
_recognizer = None          # lazily loaded cv2.face_LBPHFaceRecognizer
_labels = {}                 # {"0": {"user_id": 3, "full_name": "Ali Khan"}, ...}
_cascade = None              # lazily loaded Haar cascade (fallback detector)
_dnn_net = None               # lazily loaded DNN face detector (preferred)
_dnn_load_attempted = False
_load_attempted = False      # prevents re-trying a known-corrupt recognizer file every frame
_clahe = None                 # lazily created CLAHE normalizer (see _normalize_face below)

# Prints "<name-or-Unknown> distance=NN.N (threshold=NN)" to the console for
# every recognized face on every frame this runs. Invaluable while tuning
# FACE_RECOGNITION_CONFIDENCE_THRESHOLD — turn off once you're happy with it.
DEBUG_LOG_CONFIDENCE = True


# ==========================================
# FACE DETECTION — DNN (preferred) with Haar cascade fallback
# ==========================================
def _get_cascade():
    global _cascade
    if _cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _cascade = cv2.CascadeClassifier(cascade_path)
    return _cascade


def _get_dnn_net():
    """
    Lazily loads OpenCV's SSD-based DNN face detector (res10 300x300).
    Far more robust than the Haar cascade to off-angle faces and small/
    distant faces, at a modest extra cost per detection pass (~15-40ms on
    CPU). Since detection is already throttled to run every
    FACE_RECOGNITION_INTERVAL frames (see app.py), this extra cost does
    not add up over a live stream.

    Returns None (caller falls back to the Haar cascade) if the model
    files haven't been downloaded yet.

    ONE-TIME SETUP — run in PowerShell from the project root:
        Invoke-WebRequest -Uri "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt" -OutFile "models\\deploy.prototxt"
        Invoke-WebRequest -Uri "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel" -OutFile "models\\res10_300x300_ssd_iter_140000_fp16.caffemodel"
    """
    global _dnn_net, _dnn_load_attempted
    if _dnn_load_attempted:
        return _dnn_net
    _dnn_load_attempted = True

    proto = Config.FACE_DETECTOR_PROTOTXT_PATH
    weights = Config.FACE_DETECTOR_MODEL_PATH
    if os.path.exists(proto) and os.path.exists(weights) and os.path.getsize(weights) > 0:
        try:
            _dnn_net = cv2.dnn.readNetFromCaffe(proto, weights)
            print("✅ DNN face detector loaded — off-angle and distant faces will be detected "
                  "far more reliably than with the Haar cascade.")
        except Exception as e:
            print(f"⚠️ Failed to load DNN face detector, falling back to Haar cascade: {e}")
            _dnn_net = None
    else:
        print("ℹ️ DNN face detector model files not found in models/ — using the Haar cascade "
              "fallback (shorter range, frontal faces only). See _get_dnn_net() in "
              "utils/face_recognition_helper.py for the one-time download commands.")
        _dnn_net = None
    return _dnn_net


def _detect_faces_dnn(frame_bgr, conf_threshold=None):
    """Returns a list of (x, y, w, h) boxes, or None if the DNN detector
    isn't available (caller should fall back to Haar in that case)."""
    net = _get_dnn_net()
    if net is None:
        return None

    if conf_threshold is None:
        conf_threshold = Config.FACE_DETECTION_CONFIDENCE

    h, w = frame_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame_bgr, (300, 300)), 1.0, (300, 300),
                                  (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < conf_threshold:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w), min(y2, h)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes


def _detect_faces_haar(gray_image):
    cascade = _get_cascade()
    # minSize lowered from the original (60, 60) to (30, 30) so smaller/
    # farther faces are still picked up when running in Haar fallback mode.
    faces = cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    return [tuple(int(v) for v in f) for f in faces]


def _detect_faces(frame_bgr):
    """
    Unified multi-face detector: tries the DNN detector first (handles
    angle + distance far better), falls back to the Haar cascade if the
    DNN model files aren't present. Returns a list of (x, y, w, h) boxes
    in the original frame's coordinates.
    """
    dnn_boxes = _detect_faces_dnn(frame_bgr)
    if dnn_boxes is not None:
        return dnn_boxes
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return _detect_faces_haar(gray)


def _detect_largest_face_bgr(bgr_image):
    """Returns the (raw, un-normalized) grayscale crop of the largest face
    found in a BGR image, or None if no face was detected."""
    boxes = _detect_faces(bgr_image)
    if not boxes:
        return None
    x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])  # pick the largest by area
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    return gray[y:y + h, x:x + w]


# ==========================================
# NORMALIZATION + AUGMENTATION
# ==========================================
def _get_clahe():
    """CLAHE (Contrast Limited Adaptive Histogram Equalization) instead of
    plain equalizeHist. Plain equalizeHist normalizes contrast globally
    across the whole face, which is why a well-lit static profile photo and
    a dim/uneven webcam frame of the SAME person can still land far apart
    in LBPH's distance space — global brightness differences dominate the
    comparison. CLAHE normalizes contrast in small local tiles instead, so
    it's much more robust to the lighting mismatch between an uploaded
    profile photo and a live camera feed."""
    global _clahe
    if _clahe is None:
        _clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return _clahe


def _normalize_face(gray_roi):
    """
    Resizes + CLAHE-normalizes a grayscale face crop. Used identically at
    both training time (register_face, via _augment_face) and prediction
    time (recognize_faces_in_frame) — the two MUST stay in sync, since
    LBPH compares raw pixel patterns, not learned features.

    Small/far ROIs are upscaled with cubic interpolation (smoother, keeps
    more usable detail) rather than the default resize, since a distant
    face on the live camera is naturally much lower-resolution than the
    uploaded profile photo.
    """
    h, w = gray_roi.shape[:2]
    if h < FACE_SIZE[1] or w < FACE_SIZE[0]:
        roi = cv2.resize(gray_roi, FACE_SIZE, interpolation=cv2.INTER_CUBIC)
    else:
        roi = cv2.resize(gray_roi, FACE_SIZE, interpolation=cv2.INTER_AREA)
    roi = _get_clahe().apply(roi)
    return roi


def _augment_face(gray_face_roi):
    """
    Generates several normalized variants of a single registered face crop
    so LBPH has more to match against than one exact frontal pose/lighting.
    This is what lets the recognizer generalize to a slightly turned head,
    a closer/farther distance, or different lighting on the live camera —
    without asking users to upload multiple photos.

    Returns a list of FACE_SIZE, CLAHE-normalized grayscale images:
    the original, 4 small rotations (±8°, ±15° — mimics a turned head),
    a brighter and a darker version, and a horizontal mirror.
    """
    variants = [_normalize_face(gray_face_roi)]

    h, w = gray_face_roi.shape[:2]
    center = (w // 2, h // 2)
    for angle in (-15, -8, 8, 15):
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(gray_face_roi, rot_matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
        variants.append(_normalize_face(rotated))

    for alpha, beta in ((1.25, 15), (0.75, -15)):  # brighter / darker
        adjusted = cv2.convertScaleAbs(gray_face_roi, alpha=alpha, beta=beta)
        variants.append(_normalize_face(adjusted))

    variants.append(_normalize_face(cv2.flip(gray_face_roi, 1)))

    return variants


# ==========================================
# REGISTER / REMOVE
# ==========================================
def register_face(user_id, image_path):
    """
    Detects the largest face in the uploaded profile photo, generates
    several augmented variants of it (see _augment_face), saves all of
    them as this user's samples, then retrains the recognizer over every
    registered user. Call this right after saving a user's uploaded
    profile photo to disk.

    Returns (success: bool, message: str).
    """
    img = cv2.imread(image_path)
    if img is None:
        return False, "Could not read the uploaded image."

    face_roi = _detect_largest_face_bgr(img)
    if face_roi is None:
        return False, ("No face detected in the uploaded photo. Please upload a clear, "
                        "front-facing photo with good lighting.")

    os.makedirs(Config.FACE_SAMPLES_DIR, exist_ok=True)

    # Clear this user's previous samples first (re-uploading a new photo
    # shouldn't leave old variants lying around diluting the new ones).
    for old in glob.glob(os.path.join(Config.FACE_SAMPLES_DIR, f"user_{user_id}_*.png")):
        os.remove(old)
    legacy_path = os.path.join(Config.FACE_SAMPLES_DIR, f"user_{user_id}.png")
    if os.path.exists(legacy_path):
        os.remove(legacy_path)

    variants = _augment_face(face_roi)
    for i, variant in enumerate(variants):
        sample_path = os.path.join(Config.FACE_SAMPLES_DIR, f"user_{user_id}_{i}.png")
        cv2.imwrite(sample_path, variant)

    _retrain()
    return True, (f"Face registered ({len(variants)} training variants generated from your "
                   f"photo) — this person will now be identified by name on the live camera "
                   f"feed, including at a distance or a slight angle.")


def remove_user_face(user_id):
    """
    Deletes all of a user's saved face sample variants and retrains the
    recognizer without them. Call this from wherever a user account gets
    deleted (e.g. admin_delete_user in app.py) so their face doesn't keep
    getting trained into the model after the account is gone.
    """
    for path in glob.glob(os.path.join(Config.FACE_SAMPLES_DIR, f"user_{user_id}_*.png")):
        os.remove(path)
    legacy_path = os.path.join(Config.FACE_SAMPLES_DIR, f"user_{user_id}.png")
    if os.path.exists(legacy_path):
        os.remove(legacy_path)
    _retrain()


# ==========================================
# TRAINING / LOADING
# ==========================================
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
    Rebuilds the LBPH recognizer from every face_samples/user_<id>_<n>.png
    file on disk (multiple augmented variants per user — see
    _augment_face), pulling each user's current full_name from the
    database so labels stay in sync with profile edits. Also accepts
    legacy user_<id>.png files (pre-augmentation) for backward
    compatibility. Safe to call with zero samples (just clears the model).
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
    user_to_label = {}
    next_label = 0

    for fname in sample_files:
        stem = fname[len("user_"):-len(".png")]
        uid_part = stem.split("_")[0]  # "user_3_0.png" -> "3"; legacy "user_3.png" -> "3"
        try:
            uid = int(uid_part)
        except ValueError:
            continue

        user_row = conn.execute("SELECT id, full_name FROM users WHERE id = ?", (uid,)).fetchone()
        if user_row is None:
            continue  # stale sample for a since-deleted user

        img = cv2.imread(os.path.join(Config.FACE_SAMPLES_DIR, fname), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        if uid not in user_to_label:
            user_to_label[uid] = next_label
            labels_map[str(next_label)] = {"user_id": uid, "full_name": user_row["full_name"]}
            next_label += 1

        faces.append(img)
        label_ids.append(user_to_label[uid])

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
    needed. Only attempts the load once per process per known-good state —
    see original docstring reasoning preserved from the earlier version."""
    global _recognizer, _labels, _load_attempted

    if _load_attempted:
        return
    with _LOCK:
        if _load_attempted:
            return

        if not (_is_usable_file(Config.FACE_RECOGNIZER_PATH) and _is_usable_file(Config.FACE_LABELS_PATH)):
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
            print(f"⚠️ Could not load face recognizer, removing corrupt file(s): {e}")
            _clear_model_files()
            _recognizer = None
            _labels = {}
        finally:
            _load_attempted = True


# ==========================================
# LIVE RECOGNITION
# ==========================================
def recognize_faces_in_frame(frame_bgr):
    """
    Detects every face in a BGR frame (via the DNN detector when its model
    files are present, the Haar cascade otherwise — see _detect_faces) and,
    for each, returns the best matching registered user if the recognizer
    is confident enough.

    Returns a list of dicts, one per detected face:
        [{"name": "Ali", "confidence": 42.1, "box": (x, y, w, h)}, ...]

    "name" is "Unknown" when nobody registered matches closely enough.
    LBPH's predict() confidence is actually a *distance* — lower means a
    better match — compared against Config.FACE_RECOGNITION_CONFIDENCE_THRESHOLD.
    """
    _ensure_loaded()

    boxes = _detect_faces(frame_bgr)
    if not boxes:
        return []

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    results = []

    for (x, y, w, h) in boxes:
        name = "Unknown"
        confidence = None

        if _recognizer is not None and _labels:
            roi = _normalize_face(gray[y:y + h, x:x + w])
            label_id, distance = _recognizer.predict(roi)
            confidence = round(float(distance), 1)

            if DEBUG_LOG_CONFIDENCE:
                entry = _labels.get(str(label_id))
                candidate_name = entry["full_name"] if entry else "?"
                print(f"👤 face match candidate={candidate_name} distance={confidence} "
                      f"(threshold={Config.FACE_RECOGNITION_CONFIDENCE_THRESHOLD})")

            if distance <= Config.FACE_RECOGNITION_CONFIDENCE_THRESHOLD:
                entry = _labels.get(str(label_id))
                if entry:
                    name = entry["full_name"]

        results.append({"name": name, "confidence": confidence, "box": (int(x), int(y), int(w), int(h))})

    return results