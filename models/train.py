import os
from ultralytics import YOLO

# Project Directory Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_CONFIG = os.path.join(BASE_DIR, 'dataset', 'data.yaml') # Path to dataset config if training custom PPE dataset

def train_ppe_model():
    """
    Script to train or fine-tune YOLOv8 on custom PPE dataset.
    """
    print("🚀 Initializing YOLOv8 Model for Training...")
    
    # Load base pretrained model
    model = YOLO('yolov8n.pt')

    if not os.path.exists(DATASET_CONFIG):
        print(f"⚠️ Dataset configuration file not found at: {DATASET_CONFIG}")
        print("ℹ️ To train a custom model, place your 'data.yaml' inside a 'dataset/' directory.")
        return

    # Start Training
    try:
        print("⚡ Starting Training Process...")
        model.train(
            data=DATASET_CONFIG,
            epochs=50,
            imgsz=640,
            batch=16,
            name='yolov8_ppe_custom',
            project=os.path.join(BASE_DIR, 'runs', 'train')
        )
        print("✅ Training completed! Trained weights saved in 'runs/train/yolov8_ppe_custom/weights/best.pt'")
    except Exception as e:
        print(f"❌ Error during training: {e}")

if __name__ == '__main__':
    train_ppe_model()