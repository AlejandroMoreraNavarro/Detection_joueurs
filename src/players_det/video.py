import logging
from pathlib import Path
from typing import Generator, Tuple, Union
import cv2
import numpy as np

logger = logging.getLogger(__name__)


def get_video_info(video_path: Union[str, Path]) -> Tuple[int, int, float, int]:
    """Returns width, height, fps, and total frame count of a video."""
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return width, height, fps, frame_count


def read_video(video_path: Union[str, Path], vid_stride: int = 1) -> Generator[Tuple[int, np.ndarray], None, None]:
    """Yields frame index and frame array from video with given stride."""
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {video_path}")
    
    frame_idx = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % vid_stride == 0:
                yield frame_idx, frame
            frame_idx += 1
    finally:
        cap.release()


def get_video_writer(output_path: Union[str, Path], fps: float, width: int, height: int) -> cv2.VideoWriter:
    """Returns an OpenCV VideoWriter configured with avc1 codec."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        # Fallback to mp4v if avc1 is not available on this platform
        logger.warning("avc1 codec failed, falling back to mp4v")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise ValueError(f"Could not create video writer for {output_path}")
    return writer
