import logging
from typing import List, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


def draw_detections(image: np.ndarray, detections: List[Tuple[float, float, float, float, float]]) -> np.ndarray:
    """Draws player bounding boxes and confidence scores on the image."""
    img_viz = image.copy()
    for x1, y1, x2, y2, score in detections:
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2))
        # Green color for bounding boxes, thickness 2
        cv2.rectangle(img_viz, pt1, pt2, (0, 255, 0), 2)
        label = f"player {score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw background rectangle for text
        text_y1 = max(int(y1) - text_height - baseline - 4, 0)
        cv2.rectangle(
            img_viz,
            (int(x1), text_y1),
            (int(x1) + text_width + 4, int(y1)),
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            img_viz,
            label,
            (int(x1) + 2, int(y1) - baseline - 2),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )
    return img_viz
