import cv2, glob, os, numpy as np
seqs = sorted({os.path.basename(p).rsplit('_', 1)[0] for p in glob.glob('data/eval/images/v_*.jpg')})
tiles = []
for s in seqs:
    img = cv2.imread(f'data/eval/images/{s}_000001.jpg')
    if img is None:
        continue
    img = cv2.resize(img, (320, 180))
    cv2.putText(img, s[-14:], (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    tiles.append(img)
while len(tiles) % 6:
    tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]
os.makedirs('outputs', exist_ok=True)
cv2.imwrite('outputs/contact_sheet.jpg', np.vstack(rows))