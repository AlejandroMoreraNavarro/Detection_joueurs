# Détection de joueurs (Football / Rugby)

Détection monoclasse `player` sur plans larges (bounding boxes + score de confiance) optimisée pour l'inférence locale sur CPU.

## Installation (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

## Utilisation

### Inférence sur une image
```powershell
$env:PYTHONPATH="src"
python scripts/infer_image.py --image data/test_image.jpg --model yolov8n.pt --output outputs/output_image.jpg
```

### Inférence sur une vidéo
```powershell
$env:PYTHONPATH="src"
python scripts/infer_video.py --video data/test_video.mp4 --model yolov8n.pt --output outputs/output_video.mp4
```

---

## Bilan de la Phase 1 : Setup et Baseline

- **Modèles testés :** 
  - **YOLOv8n** (Nano) et **YOLOv8s** (Small).
  - **OpenVINO** (exporté depuis YOLOv8n pour accélération CPU Intel).
- **Entraînement :** Poids pré-entraînés sur le dataset **COCO**, filtrage de la classe `person` (id 0) en tant que `player`.
- **Performances (CPU Intel Core i7, 1280x720) :**
  - **YOLOv8n :** ~70.1 ms/image (~14.2 FPS)
  - **YOLOv8s :** ~131.5 ms/image (~7.6 FPS)
  - **YOLOv8n (OpenVINO) :** ~81.2 ms/image (~12.3 FPS)
- **Observations :** Excellente détection des joueurs sur plans larges. Les rares faux positifs proviennent de personnes hors terrain (staff, public) détectées comme `person`.
