import argparse
import logging
from pathlib import Path
import yaml

from players_det import PlayerDetector, draw_detections
from players_det.video import get_video_info, get_video_writer, read_video

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
    parser = argparse.ArgumentParser(description="Run player detection on a video.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, default="outputs/output_video.mp4", help="Path to output video")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--model", type=str, default=None, help="Path or name of YOLO model")
    parser.add_argument("--imgsz", type=int, default=None, help="Inference image size")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold")
    parser.add_argument("--stride", type=int, default=None, help="Video stride (frame skipping)")
    args = parser.parse_args()

    config = load_config(args.config)
    model_path = args.model or config.get("model_path", "yolov8n.pt")
    imgsz = args.imgsz if args.imgsz is not None else config.get("imgsz", 640)
    conf = args.conf if args.conf is not None else config.get("conf_thres", 0.25)
    stride = args.stride if args.stride is not None else config.get("vid_stride", 1)
    device = config.get("device", "cpu")

    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"Input video not found: {video_path}")
        return

    width, height, fps, frame_count = get_video_info(video_path)
    logger.info(f"Video info: {width}x{height}, {fps} fps, {frame_count} frames, stride={stride}")

    detector = PlayerDetector(model_path=model_path, device=device, conf=conf)
    writer = get_video_writer(args.output, fps, width, height)

    logger.info(f"Processing video frames with imgsz={imgsz}")
    processed_frames = 0
    for frame_idx, frame in read_video(video_path, vid_stride=stride):
        detections = detector.detect(frame, imgsz=imgsz)
        viz_frame = draw_detections(frame, detections)
        writer.write(viz_frame)
        processed_frames += 1
        if processed_frames % 30 == 0:
            logger.info(f"Processed {processed_frames} frames...")

    writer.release()
    logger.info(f"Saved processed video to {args.output}")


if __name__ == "__main__":
    main()
