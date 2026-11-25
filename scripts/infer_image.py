import argparse
import logging
from pathlib import Path
import cv2
import yaml

from players_det import PlayerDetector, draw_detections

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Loads configuration from YAML file."""
    path = Path(config_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def main():
    parser = argparse.ArgumentParser(description="Run player detection on an image.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="outputs/output_image.jpg", help="Path to output image")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--model", type=str, default=None, help="Path or name of YOLO model")
    parser.add_argument("--imgsz", type=int, default=None, help="Inference image size")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold")
    args = parser.parse_args()

    config = load_config(args.config)
    model_path = args.model or config.get("model_path", "yolov8n.pt")
    imgsz = args.imgsz if args.imgsz is not None else config.get("imgsz", 640)
    conf = args.conf if args.conf is not None else config.get("conf_thres", 0.25)
    device = config.get("device", "cpu")

    image_path = Path(args.image)
    if not image_path.exists():
        logger.error(f"Input image not found: {image_path}")
        return

    logger.info(f"Reading image {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"Failed to read image {image_path}")
        return

    detector = PlayerDetector(model_path=model_path, device=device, conf=conf)
    logger.info(f"Running detection with imgsz={imgsz}")
    detections = detector.detect(image, imgsz=imgsz)
    logger.info(f"Detected {len(detections)} players")

    viz_image = draw_detections(image, detections)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), viz_image)
    logger.info(f"Saved result to {output_path}")


if __name__ == "__main__":
    main()
