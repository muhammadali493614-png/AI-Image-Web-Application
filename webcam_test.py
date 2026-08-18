"""
webcam_test.py
---------------
Standalone camera diagnostic + live-test script for SafeVision AI.

Purpose:
  1. Tell you EXACTLY which camera index/backend combo works on this
     machine (or which one silently fails after isOpened() == True).
  2. Run a live YOLO-annotated preview window, using the SAME robust
     camera-opening logic that app.py's _open_camera() uses — so if this
     script works, app.py's /video_feed will work too, and if this
     script fails, you know the problem is NOT in the Flask code.

IMPORTANT: Close app.py (and any other app using the webcam — Zoom,
Teams, Windows Camera app, browser tabs with camera permission, OBS,
Iriun, etc.) before running this. Only ONE process can hold a camera
device at a time on Windows; if app.py's server is still running, this
script will "steal" the camera from it (or fail to open it at all) and
you'll see the exact same "Camera Not Available" / DSHOW warning.
"""

import platform
import sys
import time

import cv2
from ultralytics import YOLO


def _try_open_camera(index, backend=None, backend_name="default"):
    """
    Opens a single (index, backend) combo and verifies it actually
    delivers a real frame — not just that isOpened() reports True.
    This is the key fix: on Windows, cv2.CAP_DSHOW can report
    isOpened() == True while every read() still fails (the
    "backend is generally available but can't be used to capture by
    index" warning). Checking isOpened() alone is not enough.
    """
    try:
        cam = cv2.VideoCapture(index, backend) if backend is not None else cv2.VideoCapture(index)
    except Exception as e:
        print(f"   ⚠️  index {index} ({backend_name}): open() raised -> {e}")
        return None

    if not cam.isOpened():
        cam.release()
        print(f"   ❌ index {index} ({backend_name}): isOpened() == False")
        return None

    # Give the driver a brief moment to warm up before the first read —
    # some USB webcams return a bad frame on the very first read()
    # immediately after opening.
    time.sleep(0.3)
    ok, frame = cam.read()
    if not ok or frame is None:
        print(f"   ⚠️  index {index} ({backend_name}): opened but produced NO frame")
        cam.release()
        return None

    print(f"   ✅ index {index} ({backend_name}): OK — frame size {frame.shape[1]}x{frame.shape[0]}")
    return cam


def diagnose_all_cameras(max_index=3):
    """
    Scans every (index, backend) combination and prints a report, so you
    can see at a glance which ones actually work on this machine.
    """
    print("\n🔍 Scanning for available cameras...\n")

    if platform.system() == "Windows":
        backends = [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"), (None, "default")]
    else:
        backends = [(None, "default")]

    working = []
    for index in range(max_index):
        print(f"Index {index}:")
        for backend, name in backends:
            cam = _try_open_camera(index, backend, name)
            if cam is not None:
                working.append((index, name))
                cam.release()

    print("\n" + "=" * 50)
    if working:
        print("✅ WORKING CAMERA(S) FOUND:")
        for index, name in working:
            print(f"   -> index={index}, backend={name}")
    else:
        print("❌ NO CAMERA COULD BE OPENED ON ANY INDEX/BACKEND.")
        print("   This means it's NOT a Python/OpenCV code issue. Check:")
        print("   1. Is another app using the camera right now? (Zoom, Teams,")
        print("      Windows Camera app, a browser tab, OBS, Iriun, app.py's")
        print("      Flask server still running in another terminal, etc.)")
        print("      -> Close ALL of them, then re-run this script.")
        print("   2. Windows Settings > Privacy & Security > Camera ->")
        print("      make sure 'Camera access' and 'Let desktop apps access")
        print("      your camera' are turned ON.")
        print("   3. Device Manager -> Cameras -> check for a yellow warning")
        print("      icon (driver problem) -> update/reinstall the driver.")
        print("   4. Unplug/replug the camera (if external/USB) and retry.")
        print("   5. Restart your PC — a crashed previous Python process can")
        print("      leave the camera device locked at the OS level.")
    print("=" * 50 + "\n")

    return working


def open_best_camera():
    """Same fallback order as app.py's _open_camera()."""
    if platform.system() == "Windows":
        candidates = [
            (0, cv2.CAP_DSHOW, "DSHOW"),
            (0, cv2.CAP_MSMF, "MSMF"),
            (0, None, "default"),
            (1, cv2.CAP_DSHOW, "DSHOW"),
            (1, cv2.CAP_MSMF, "MSMF"),
        ]
    else:
        candidates = [(0, None, "default"), (1, None, "default")]

    for index, backend, name in candidates:
        cam = _try_open_camera(index, backend, name)
        if cam is not None:
            return cam
    return None


def run_live_preview():
    print("✅ Loading YOLO model (yolov8n.pt)...")
    model = YOLO('yolov8n.pt')

    cam = open_best_camera()
    if cam is None:
        print("\n❌ Could not open the camera on any backend/index. Run this script "
              "again with `python webcam_test.py --diagnose` for a full report.")
        sys.exit(1)

    print("\n🚀 Live Camera Stream running... Press 'q' to quit.\n")
    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                print("❌ Camera se feed nahi mil rahi (mid-stream read failure).")
                break

            results = model(frame, conf=0.35)
            annotated_frame = results[0].plot()
            cv2.imshow("SafeVision AI - Live Webcam Feed (Test)", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if "--diagnose" in sys.argv:
        diagnose_all_cameras()
    else:
        # Run a quick diagnostic first so failures are self-explanatory,
        # then fall into the live preview if a camera was found.
        found = diagnose_all_cameras()
        if found:
            run_live_preview()