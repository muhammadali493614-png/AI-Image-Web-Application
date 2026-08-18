import sys
import os
from ultralytics import YOLO

# Project Base Directory Setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUSTOM_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'yolov8_ppe.pt')
FALLBACK_MODEL_PATH = os.path.join(BASE_DIR, 'yolov8n.pt')

def run_prediction(source_path):
    """
    Run YOLOv8 prediction on an image or video file.
    Automatically handles missing or corrupted model files.
    """
    if not os.path.exists(source_path):
        print(f"❌ Error: File '{source_path}' does not exist.")
        return

    # Model file selection with fallback check
    if os.path.exists(CUSTOM_MODEL_PATH) and os.path.getsize(CUSTOM_MODEL_PATH) > 0:
        model_file = CUSTOM_MODEL_PATH
        print(f"📦 Loading custom PPE model: {model_file}")
    else:
        model_file = FALLBACK_MODEL_PATH if os.path.exists(FALLBACK_MODEL_PATH) else "yolov8n.pt"
        print(f"⚠️ Custom PPE model not found or empty. Using default model: {model_file}")

    # Load Model & Run Inference
    try:
        model = YOLO(model_file)
        print(f"🚀 Running AI Detection on: {source_path}")
        
        # Save output in 'runs/detect/predict' folder
        results = model.predict(source=source_path, conf=0.35, save=True, show=False)
        print("✅ Detection completed! Output saved in 'runs/detect/' directory.")
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        run_prediction(input_file)
    else:
        print("ℹ️ Usage Example:")
        print("   python models/predict.py uploads/images/test.jpg")