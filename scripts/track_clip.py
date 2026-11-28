from pathlib import Path
import time
import cv2
from ultralytics import YOLO

def track_clip():
    clip_path = next(Path('data/clips_foot').glob('*.mp4'))
    print(f'Input clip: {clip_path.name}')

    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(str(clip_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Input info: {width}x{height}, {frame_count} frames')

    output_path = Path('outputs/tracked_video.mp4')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        print('Error: Could not open VideoWriter.')
        return

    unique_ids = set()
    frames_processed = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames_processed += 1

        results = model.track(
            frame,
            tracker="bytetrack.yaml",
            classes=[0],
            imgsz=640,
            device="cpu",
            persist=True,
            verbose=False
        )[0]

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy().astype(int)
            confidences = results.boxes.conf.cpu().numpy()

            for box, track_id, conf in zip(boxes, track_ids, confidences):
                unique_ids.add(track_id)
                x1, y1, x2, y2 = map(int, box)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = f'ID {track_id} ({conf:.2f})'
                cv2.putText(frame, label, (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        writer.write(frame)

    elapsed_time = time.time() - start_time
    total_ms_per_frame = (elapsed_time * 1000.0) / max(frames_processed, 1)

    cap.release()
    writer.release()

    print(f"Number of frames: {frames_processed}")
    print(f"Number of unique track IDs created: {len(unique_ids)}")
    print(f"Total ms/frame: {total_ms_per_frame:.2f} ms")
    print(f"Tracked video saved to {output_path}")

if __name__ == '__main__':
    track_clip()
