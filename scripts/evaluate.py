import argparse
import logging
from pathlib import Path
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO model on custom eval set.")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path or name of YOLO model")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--data", type=str, default="data/eval/data.yaml", help="Path to data.yaml")
    args = parser.parse_args()

    logger.info(f"Loading model {args.model} for evaluation on CPU")
    model = YOLO(args.model)

    logger.info(f"Running validation on {args.data} with imgsz={args.imgsz}, filtering class 0 (person)")
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        device="cpu",
        classes=[0], # Filter pre-trained COCO model to only class 0 ('person')
        verbose=True
    )

    map50 = metrics.box.map50
    precision = metrics.box.mp
    recall = metrics.box.mr
    
    # speed dict has inference time in ms per image
    inference_time_ms = metrics.speed.get("inference", 0.0)

    print(f"mAP@0.5: {map50:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"ms/image: {inference_time_ms:.2f}")

if __name__ == "__main__":
    main()
