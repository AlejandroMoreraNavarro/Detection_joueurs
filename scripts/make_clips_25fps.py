from pathlib import Path
import cv2
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def make_clips_25fps():
    eval_img_dir = Path("data/eval/images")
    output_dir = Path("data/clips_foot")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group images by sequence name prefix
    sequences = {}
    for img_path in sorted(eval_img_dir.glob("*.jpg")):
        name = img_path.name
        # Format is {seq_name}_{frame}.jpg
        # Find sequence name by matching known prefixes or splitting before last underscore
        parts = name.split("_")
        # seq_name can have multiple parts connected by '_'
        # Let's find which sequence name it belongs to
        for seq_prefix in [
            'v_1yHWGw8DH4A_c029', 'v_1yHWGw8DH4A_c047', 'v_1yHWGw8DH4A_c077',
            'v_1yHWGw8DH4A_c601', 'v_1yHWGw8DH4A_c609', 'v_1yHWGw8DH4A_c610',
            'v_HdiyOtliFiw_c003', 'v_HdiyOtliFiw_c004', 'v_HdiyOtliFiw_c008',
            'v_HdiyOtliFiw_c010', 'v_HdiyOtliFiw_c602', 'v_gQNyhv8y0QY_c003',
            'v_gQNyhv8y0QY_c012', 'v_gQNyhv8y0QY_c013', 'v_iIxMOsCGH58_c013'
        ]:
            if name.startswith(seq_prefix):
                sequences.setdefault(seq_prefix, []).append(img_path)
                break

    target_fps = 25.0

    for seq_name, img_paths in sorted(sequences.items()):
        img_paths = sorted(img_paths)
        if not img_paths:
            continue

        sample_img = cv2.imread(str(img_paths[0]))
        if sample_img is None:
            continue
        h, w = sample_img.shape[:2]

        output_path = output_dir / f"{seq_name}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(output_path), fourcc, target_fps, (w, h))
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, target_fps, (w, h))

        if not writer.isOpened():
            continue

        for img_p in img_paths:
            frame = cv2.imread(str(img_p))
            if frame is not None:
                writer.write(frame)

        writer.release()
        duration = len(img_paths) / target_fps
        logger.info(f"Clip {output_path.name}: {len(img_paths)} frames, {duration:.1f}s at {target_fps} fps")

if __name__ == "__main__":
    make_clips_25fps()
