import logging
from pathlib import Path
from typing import List, Tuple, Union
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class PlayerDetector:
    """YOLO-based player detector filtering the person class."""

    def __init__(self, model_path: Union[str, Path] = "yolov8n.pt", device: str = "cpu", conf: float = 0.25):
        """Initializes the YOLO model for CPU inference."""
        self.model_path = Path(model_path)
        self.device = device
        self.conf = conf
        logger.info(f"Loading YOLO model from {self.model_path} on device {self.device}")
        self.model = YOLO(str(self.model_path))

    def detect(self, image: np.ndarray, imgsz: int = 640) -> List[Tuple[float, float, float, float, float]]:
        """Detects players in an image, returning list of (x1, y1, x2, y2, score)."""
        results = self.model(image, device=self.device, conf=self.conf, imgsz=imgsz, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0].item())
                # COCO class 0 is 'person'
                if cls_id == 0:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    score = float(box.conf[0].item())
                    detections.append((x1, y1, x2, y2, score))

        return detections
