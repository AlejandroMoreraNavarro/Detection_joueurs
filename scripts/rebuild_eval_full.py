from pathlib import Path
import shutil

def rebuild():
    eval_dir = Path("data/eval")
    eval_img_dir = eval_dir / "images"
    eval_lbl_dir = eval_dir / "labels"
    
    if eval_img_dir.exists():
        shutil.rmtree(eval_img_dir)
    if eval_lbl_dir.exists():
        shutil.rmtree(eval_lbl_dir)
        
    eval_img_dir.mkdir(parents=True, exist_ok=True)
    eval_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    # Get 15 football prefixes from existing eval images or hardcoded list
    train_dir = Path("data/sportsmot/dataset/train_extracted/train")
    
    # Identify football prefixes from existing eval/images
    old_eval_imgs = list(Path("data/eval/images").glob("*.jpg")) if Path("data/eval/images").exists() else []
    prefixes = set()
    for p in Path("data/eval/images").glob("*.jpg"):
        name = p.name
        if name.startswith('0'):
            prefixes.add('v_gQNyhv8y0QY_c013')
        else:
            parts = name.split('_')
            if len(parts) >= 3:
                prefixes.add('_'.join(parts[:3]))
                
    if not prefixes:
        # fallback list of the 15 football sequences
        prefixes = {
            'v_1yHWGw8DH4A_c029', 'v_1yHWGw8DH4A_c047', 'v_1yHWGw8DH4A_c077',
            'v_1yHWGw8DH4A_c601', 'v_1yHWGw8DH4A_c609', 'v_1yHWGw8DH4A_c610',
            'v_HdiyOtliFiw_c003', 'v_HdiyOtliFiw_c004', 'v_HdiyOtliFiw_c008',
            'v_HdiyOtliFiw_c010', 'v_HdiyOtliFiw_c602', 'v_gQNyhv8y0QY_c003',
            'v_gQNyhv8y0QY_c012', 'v_gQNyhv8y0QY_c013', 'v_iIxMOsCGH58_c013'
        }

    sequences = [d for d in train_dir.iterdir() if d.is_dir() and d.name in prefixes]
    
    img_w, img_h = 1280, 720
    total_images = 0
    total_labels = 0
    
    for seq_p in sequences:
        img_src_dir = seq_p / "img1"
        gt_path = seq_p / "gt" / "gt.txt"
        
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
                        
        if img_src_dir.exists():
            for img_p in sorted(img_src_dir.glob("*.jpg")):
                try:
                    frame_id = int(img_p.stem)
                except ValueError:
                    continue
                    
                new_img_name = f"{seq_p.name}_{img_p.name}"
                dest_img = eval_img_dir / new_img_name
                shutil.copy(img_p, dest_img)
                total_images += 1
                
                boxes = gt_data.get(frame_id, [])
                label_p = eval_lbl_dir / f"{Path(new_img_name).stem}.txt"
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
                        
                        lf.write(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")
                total_labels += 1

    print(f"SEQUENCES:{len(sequences)}")
    print(f"IMAGES:{total_images}")
    print(f"LABELS:{total_labels}")

if __name__ == "__main__":
    rebuild()
