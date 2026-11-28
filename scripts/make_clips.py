import os
from pathlib import Path
import cv2
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def make_clips():
    eval_img_dir = Path("data/eval/images")
    output_dir = Path("data/clips_foot")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not eval_img_dir.exists():
        logger.error(f"Directory not found: {eval_img_dir}")
        return

    # Group images by prefix
    sequences = {}
    for img_path in sorted(eval_img_dir.glob("*.jpg")):
        name = img_path.name
        if name.startswith("0"):
            prefix = "v_gQNyhv8y0QY_c013"
        else:
            parts = name.split("_")
            prefix = "_".join(parts[:3]) if len(parts) >= 3 else "other"
        sequences.setdefault(prefix, []).append(img_path)

    logger.info(f"Found {len(sequences)} sequence prefixes in {eval_img_dir}")

    target_duration = 15.0  # seconds
    target_fps = 15.0
    target_frames = int(target_duration * target_fps)

    for prefix, img_paths in sorted(sequences.items()):
        img_paths = sorted(img_paths)
        if not img_paths:
            continue

        sample_img = cv2.imread(str(img_paths[0]))
        if sample_img is None:
            logger.warning(f"Could not read sample image {img_paths[0]}")
            needed_w, needed_h = 1280, 720
        else:
            needed_h, needed_w = sample_img.shape[:2]

        frame_list = []
        if len(img_paths) >= target_frames:
            step = len(img_paths) / target_frames
            frame_list = [img_paths[int(i * step)] for i in range(target_frames)]
        else:
            while len(frame_list) < target_frames:
                for p in img_paths:
                    frame_list.append(p)
                    if len(frame_list) >= target_frames:
                        break

        output_path = output_dir / f"{prefix}.mp4"
        
        # Use avc1 codec as requested, with fallback to mp4v
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(output_path), fourcc, target_fps, (needed_w, needed_h))
        if not writer.isOpened():
            logger.warning("avc1 codec failed, falling back to mp4v")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, target_fps, (needed_w, needed_h))
        
        if not writer.isOpened():
            logger.error(f"Failed to create video writer for {output_path}")
            continue

        for img_p in frame_list:
            frame = cv2.imread(str(img_p))
            if frame is not None:
                writer.write(frame)

        writer.release()
        duration = len(frame_list) / target_fps
        logger.info(f"Created clip (avc1): {output_path.name} ({len(frame_list)} frames, {duration:.1f}s, {target_fps} fps)")

    logger.info(f"All clips successfully regenerated in {output_dir}")

if __name__ == "__main__":
    make_clips()
