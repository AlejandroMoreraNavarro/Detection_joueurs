import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def compute_iou(box1, box2):
    # box: [x1, y1, x2, y2]
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
        
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return interArea / float(box1Area + box2Area - interArea)

def match_boxes(gt_boxes, pred_boxes, iou_thresh=0.5):
    # gt_boxes: list of [x1, y1, x2, y2, height]
    # pred_boxes: list of [x1, y1, x2, y2]
    matched_gt = set()
    matched_pred = set()
    
    matches = []
    for g_idx, g in enumerate(gt_boxes):
        for p_idx, p in enumerate(pred_boxes):
            if p_idx in matched_pred:
                continue
            iou = compute_iou(g[:4], p)
            if iou >= iou_thresh:
                matches.append((g_idx, p_idx, iou))
                
    # sort by iou descending
    matches.sort(key=lambda x: x[2], reverse=True)
    
    for g_idx, p_idx, _ in matches:
        if g_idx not in matched_gt and p_idx not in matched_pred:
            matched_gt.add(g_idx)
            matched_pred.add(p_idx)
            
    fn_indices = [i for i in range(len(gt_boxes)) if i not in matched_gt]
    fp_indices = [i for i in range(len(pred_boxes)) if i not in matched_pred]
    return fn_indices, fp_indices

def main():
    model = YOLO("yolov8n.pt")
    img_dir = Path("data/eval/images")
    lbl_dir = Path("data/eval/labels")
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    img_w, img_h = 1280, 720
    
    size_buckets = {"< 40 px": {"total": 0, "fn": 0},
                    "40 to 80 px": {"total": 0, "fn": 0},
                    "> 80 px": {"total": 0, "fn": 0}}
    
    image_stats = []
    
    for img_path in sorted(img_dir.glob("*.jpg")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        gt_boxes = []
        if lbl_path.exists():
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = [float(x) for x in line.split()]
                    if len(parts) == 5:
                        _, xc, yc, nw, nh = parts
                        w_abs = nw * img_w
                        h_abs = nh * img_h
                        x1 = (xc - nw / 2.0) * img_w
                        y1 = (yc - nh / 2.0) * img_h
                        x2 = x1 + w_abs
                        y2 = y1 + h_abs
                        gt_boxes.append([x1, y1, x2, y2, h_abs])
                        
        results = model(img, conf=0.25, classes=[0], verbose=False)[0]
        pred_boxes = []
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                pred_boxes.append([x1, y1, x2, y2])
                
        fn_indices, fp_indices = match_boxes(gt_boxes, pred_boxes, iou_thresh=0.5)
        
        # update size buckets
        for i, g in enumerate(gt_boxes):
            h_val = g[4]
            if h_val < 40:
                bucket = "< 40 px"
            elif h_val <= 80:
                bucket = "40 to 80 px"
            else:
                bucket = "> 80 px"
                
            size_buckets[bucket]["total"] += 1
            if i in fn_indices:
                size_buckets[bucket]["fn"] += 1
                
        image_stats.append({
            "path": img_path,
            "img": img,
            "gt": gt_boxes,
            "pred": pred_boxes,
            "fn_count": len(fn_indices),
            "fp_count": len(fp_indices)
        })
        
    # Print table
    print("--- Tableau des Faux Négatifs par Taille ---")
    print(f"{'Taille (hauteur)':<15} | {'Annotés':<10} | {'Manqués (FN)':<12} | {'% Manqué':<10}")
    print("-" * 55)
    for bucket, data in size_buckets.items():
        tot = data["total"]
        fn = data["fn"]
        pct = (fn / tot * 100) if tot > 0 else 0.0
        print(f"{bucket:<15} | {tot:<10} | {fn:<12} | {pct:.1f}%")
        
    # Save top 3 FN and top 3 FP images
    image_stats.sort(key=lambda x: x["fn_count"], reverse=True)
    top_fn = image_stats[:3]
    
    image_stats.sort(key=lambda x: x["fp_count"], reverse=True)
    top_fp = image_stats[:3]
    
    for rank, item in enumerate(top_fn):
        canvas = item["img"].copy()
        for g in item["gt"]:
            cv2.rectangle(canvas, (int(g[0]), int(g[1])), (int(g[2]), int(g[3])), (0, 255, 0), 2)
        for p in item["pred"]:
            cv2.rectangle(canvas, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), (0, 0, 255), 2)
        cv2.imwrite(str(out_dir / f"top_fn_{rank+1}_{item['path'].name}"), canvas)
        
    for rank, item in enumerate(top_fp):
        canvas = item["img"].copy()
        for g in item["gt"]:
            cv2.rectangle(canvas, (int(g[0]), int(g[1])), (int(g[2]), int(g[3])), (0, 255, 0), 2)
        for p in item["pred"]:
            cv2.rectangle(canvas, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), (0, 0, 255), 2)
        cv2.imwrite(str(out_dir / f"top_fp_{rank+1}_{item['path'].name}"), canvas)

if __name__ == "__main__":
    main()
