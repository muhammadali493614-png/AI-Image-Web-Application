"""
utils/model_validation.py

Standalone script (NOT a Flask route) to evaluate the accuracy of your
trained yolov8_ppe.pt model against a labeled validation set, using
Ultralytics' built-in validation pipeline.

WHAT YOU NEED BEFORE RUNNING THIS:
  1. A validation dataset in YOLO format:
       ppe_val_dataset/
         images/val/*.jpg
         labels/val/*.txt        <- YOLO-format bounding box labels
  2. A data.yaml describing it, e.g.:

        path: /absolute/path/to/ppe_val_dataset
        train: images/train      # can point anywhere, unused for val-only runs
        val: images/val
        names:
          0: helmet
          1: no-helmet
          2: vest
          3: no-vest
          ... etc, matching the class list your model was trained on

  If you don't have a held-out labeled validation set yet, this is the
  actual blocker for "model accuracy & validation" — the metrics below
  can't be computed without ground-truth labels to compare against. If
  your training data already has a val split (most YOLOv8 training runs
  do, from the same data.yaml used for training), point --data at that
  same file and it'll just work.

USAGE:
    python utils/model_validation.py --data path/to/data.yaml
    python utils/model_validation.py --data path/to/data.yaml --conf 0.25 --iou 0.6

OUTPUT:
  - Precision, Recall, mAP50, mAP50-95 per class and overall (printed + saved)
  - Confusion matrix PNG (saved to runs/detect/val*/confusion_matrix.png)
  - F1/PR curves (saved alongside it)
  - A plain-text summary written to model_validation_report.txt in the
    current directory, so you can attach it to a report/FYP submission.
  - model_accuracy.json written to the project root (one level above this
    utils/ folder) — this is what SafeVision AI's dashboard reads to show
    "Model Accuracy". Re-run this script any time you retrain the model;
    the dashboard value only updates when this file is regenerated.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Validate SafeVision AI PPE model accuracy.")
    parser.add_argument("--model", default="models/yolov8_ppe.pt", help="Path to trained model weights.")
    parser.add_argument("--data", required=True, help="Path to data.yaml describing the validation set.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (match app.py's CONF_THRESHOLD).")
    parser.add_argument("--iou", type=float, default=0.6, help="IoU threshold for NMS during validation.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size (match app.py).")
    parser.add_argument(
        "--output",
        default=None,
        help="Where to write model_accuracy.json for the dashboard. "
             "Defaults to <project_root>/model_accuracy.json (one level above utils/)."
    )
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"❌ Model not found at {args.model}. Point --model at your yolov8_ppe.pt file.")
        sys.exit(1)
    if not Path(args.data).exists():
        print(f"❌ data.yaml not found at {args.data}. See the docstring at the top of this file.")
        sys.exit(1)

    print(f"📦 Loading model: {args.model}")
    model = YOLO(args.model)

    print("🔎 Running validation...")
    metrics = model.val(data=args.data, conf=args.conf, iou=args.iou, imgsz=args.imgsz, plots=True)

    # metrics.box holds the detection metrics (Ultralytics DetMetrics object)
    box = metrics.box
    class_names = model.names

    lines = []
    lines.append("SafeVision AI — Model Validation Report")
    lines.append("=" * 45)
    lines.append(f"Model: {args.model}")
    lines.append(f"Data:  {args.data}")
    lines.append(f"Conf threshold: {args.conf} | IoU threshold: {args.iou} | imgsz: {args.imgsz}")
    lines.append("")
    lines.append(f"Overall mAP50:     {box.map50:.4f}")
    lines.append(f"Overall mAP50-95:  {box.map:.4f}")
    lines.append(f"Overall Precision: {box.mp:.4f}")
    lines.append(f"Overall Recall:    {box.mr:.4f}")
    lines.append("")
    lines.append("Per-class breakdown:")
    lines.append(f"{'Class':<25}{'Precision':>12}{'Recall':>12}{'mAP50':>12}")

    try:
        p, r, ap50, ap = box.p, box.r, box.ap50, box.ap
        for i, cls_idx in enumerate(box.ap_class_index):
            name = class_names.get(int(cls_idx), str(cls_idx))
            lines.append(f"{name:<25}{p[i]:>12.4f}{r[i]:>12.4f}{ap50[i]:>12.4f}")
    except Exception as e:
        lines.append(f"(Per-class arrays unavailable in this Ultralytics version: {e})")

    lines.append("")
    lines.append(f"Confusion matrix + PR/F1 curves saved under: {metrics.save_dir}")

    report_text = "\n".join(lines)
    print("\n" + report_text)

    out_path = Path("model_validation_report.txt")
    out_path.write_text(report_text, encoding="utf-8")
    print(f"\n📄 Report written to {out_path.resolve()}")

    # --- Write model_accuracy.json for the SafeVision AI dashboard ---
    # This is the file app.py's load_model_accuracy() reads from. It is a
    # static snapshot — the dashboard does NOT recompute this per request.
    accuracy_output_path = Path(args.output) if args.output else Path(__file__).resolve().parent.parent / "model_accuracy.json"
    accuracy_payload = {
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "model": args.model,
        "data": args.data,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    accuracy_output_path.write_text(json.dumps(accuracy_payload, indent=2), encoding="utf-8")
    print(f"📊 Dashboard accuracy file written to {accuracy_output_path.resolve()} "
          f"(mAP50 = {box.map50 * 100:.1f}%)")


if __name__ == "__main__":
    main()