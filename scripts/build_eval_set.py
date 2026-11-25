import shutil
from pathlib import Path

def build_eval():
    src_seq = Path("data/sportsmot/dataset/train/v_gQNyhv8y0QY_c013")
    img_src = src_seq / "img1"
    gt_path = src_seq / "gt" / "gt.txt"
    
    eval_dir = Path("data/eval")
    eval_img_dir = eval_dir / "images"
    eval_lbl_dir = eval_dir / "labels"
    
    eval_img_dir.mkdir(parents=True, exist_ok=True)
    eval_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    gt_data = {}
    if gt_path.exists():
        with open(gt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = [float(x.strip()) for x in line.split(",")]
                if len(parts) >= 9:
                    frame_id = int(parts[0])
                    conf = int(parts[6])
                    if conf == 0:
                        continue
                    x, y, w, h = parts[2], parts[3], parts[4], parts[5]
                    gt_data.setdefault(frame_id, []).append((x, y, w, h))
                    
    images = sorted(list(img_src.glob("*.jpg")))
    img_w, img_h = 1280, 720
    selected_images = images[::10]
    
    total_boxes = 0
    sample_label_lines = []
    
    for img_p in selected_images:
        try:
            frame_id = int(img_p.stem)
        except ValueError:
            continue
            
        dest_img = eval_img_dir / img_p.name
        shutil.copy(img_p, dest_img)
        
        boxes = gt_data.get(frame_id, [])
        label_p = eval_lbl_dir / f"{img_p.stem}.txt"
        file_lines = []
        with open(label_p, "w", encoding="utf-8") as lf:
            for (x, y, w, h) in boxes:
                x1 = max(0.0, min(x, img_w))
                y1 = max(0.0, min(y, img_h))
                x2 = max(0.0, min(x + w, img_w))
                y2 = max(0.0, min(y + h, img_h))
                
                clamped_w = x2 - x1
                clamped_h = y2 - y1
                if clamped_w <= 0 or clamped_h <= 0:
                    continue
                    
                xc = (x1 + clamped_w / 2.0) / img_w
                yc = (y1 + clamped_h / 2.0) / img_h
                nw = clamped_w / img_w
                nh = clamped_h / img_h
                
                line_str = f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}"
                lf.write(line_str + "\n")
                file_lines.append(line_str)
                total_boxes += 1
                
        if len(sample_label_lines) < 3 and file_lines:
            sample_label_lines.extend(file_lines[:3 - len(sample_label_lines)])
            
    print(f"IMAGES:{len(selected_images)}")
    print(f"BOXES:{total_boxes}")
    print("SAMPLE_LINES:")
    for l in sample_label_lines[:3]:
        print(l)

if __name__ == "__main__":
    build_eval()
