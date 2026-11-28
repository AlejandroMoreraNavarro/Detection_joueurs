from pathlib import Path
import cv2
from ultralytics import YOLO

def annotate_clip():
    clip_path = next(Path('data/clips_foot').glob('*.mp4'))
    print(f'Input clip: {clip_path.name}')

    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(str(clip_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Force 25 FPS as requested
    fps = 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Input info: {width}x{height}, {frame_count} frames')

    output_path = Path('outputs/annotated_ball.mp4')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        print('Error: Could not open VideoWriter.')
        return

    conf_thres = 0.25
    frames_processed = 0
    ball_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames_processed += 1
        
        results = model(frame, imgsz=640, conf=conf_thres, classes=[0, 32], device='cpu', verbose=False)[0]
        
        has_ball = False
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            if cls_id == 32: # sports ball
                has_ball = True
                color = (0, 0, 255) # Red for ball
                label = f'ball {conf:.2f}'
            else: # person
                color = (0, 255, 0) # Green for person
                label = f'player {conf:.2f}'
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        if has_ball:
            ball_frames += 1
            
        writer.write(frame)

    cap.release()
    writer.release()

    print(f'Processed {frames_processed} frames.')
    print(f'Frames with sports ball: {ball_frames}')
    print(f'Output video saved to {output_path} at {fps} FPS.')

if __name__ == '__main__':
    annotate_clip()
