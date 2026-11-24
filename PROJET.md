# PROJET : Détection de joueurs sur plans larges (football / rugby)

## 1. Vision et objectifs

Détecter et localiser les **joueurs** sur des images/vidéos de matchs (plans larges) sous forme de bounding boxes, avec une **seule classe : `player`**.

| Inclus | Hors périmètre (pour l'instant) |
|---|---|
| Détection monoclasse `player` | Distinction des équipes |
| Inférence locale sur CPU (image et vidéo) | Suivi / Re-ID / OCR de maillots |
| Baseline zéro-entraînement, puis amélioration mesurée | Ballon, arbitres (classes dédiées), pose, segmentation |
| Fine-tuning éventuel sur Kaggle (GPU) | Homographie terrain, interface web, API |
| Export optimisé CPU (ONNX / OpenVINO) | Déploiement cloud |

**Règle d'arrêt :** le projet reste le plus simple possible. Si un modèle pré-entraîné (éventuellement exporté en OpenVINO) donne des résultats satisfaisants à la fin de la Phase 1 ou 2, on le garde, on documente et on s'arrête. Les Phases 3 et 4 ne sont lancées que si une limite mesurée le justifie.

**Mesures de succès** (seuils chiffrés à fixer à la fin de la Phase 1, une fois la baseline mesurée) :
- Qualité : mAP@0.5 et rappel sur un petit jeu d'évaluation annoté à la main (voir Phase 2).
- Vitesse : ms/image et FPS sur ta machine CPU.

## 2. Choix techniques de départ

| Option | Rôle | Remarque |
|---|---|---|
| YOLO (Ultralytics) `n` / `s` | Baseline locale | Poids COCO : la classe `person` (id 0) sert de `player` en Phase 1. Licence **AGPL-3.0** (à vérifier) : à garder en tête si le code est distribué. |
| RF-DETR Nano / Small | Alternative | Licence Apache 2.0 pour ces tailles. Surtout intéressant côté GPU ; sur CPU, YOLO est a priori plus simple. |
| SAHI / tiling | Joueurs très petits | Coût CPU multiplié par le nombre de tuiles : à n'utiliser que si la mesure le justifie. |
| ONNX Runtime / OpenVINO (INT8) | Accélération CPU | Souvent le plus gros gain sur CPU, devant le choix du modèle. OpenVINO surtout pertinent sur CPU Intel. |

## 3. Architecture cible

```
.
├── GEMINI.md
├── PROJET.md
├── README.md
├── requirements.txt          # inférence locale uniquement
├── .gitignore
├── configs/
│   └── default.yaml          # modèle, imgsz, seuils, tiling
├── src/
│   └── players_det/
│       ├── __init__.py
│       ├── detector.py       # chargement modèle + predict (sortie x1,y1,x2,y2,score)
│       ├── tiling.py         # tiling / SAHI (Phase 2+)
│       ├── video.py          # lecture / écriture vidéo
│       ├── viz.py            # dessin des boîtes
│       └── export.py         # export ONNX / OpenVINO
├── scripts/
│   ├── infer_image.py
│   ├── infer_video.py
│   ├── evaluate.py           # mAP / rappel sur le jeu d'éval
│   └── export_model.py
├── kaggle/                   # autonome, jamais importé par src/
│   ├── README.md
│   └── train_finetune.ipynb
├── tests/
├── data/                     # ignoré par Git (raw/, processed/, eval/)
├── models/                   # ignoré par Git (poids)
└── outputs/                  # ignoré par Git (vidéos/images annotées)
```

Extrait de `.gitignore` minimal :

```gitignore
.venv/
__pycache__/
runs/
outputs/
data/
models/
*.pt
*.pth
*.onnx
*.engine
*.bin
*.mp4
*.avi
*.mov
*.mkv
kaggle.json
.env
```

## 4. Feuille de route

### Phase 1 : Setup et baseline zéro-entraînement (local, CPU)
- [ ] Créer le dépôt, `.venv`, `.gitignore`, `requirements.txt` léger.
- [ ] Installer PyTorch CPU + Ultralytics + OpenCV.
- [ ] Écrire `detector.py` (chargement, `predict`, filtrage classe `person` → `player`).
- [ ] Écrire `infer_image.py` et `infer_video.py`.
- [ ] Tester `yolo*n` et `yolo*s` sur 1 image puis 1 vidéo courte (≤ 15 s) de plan large.
- [ ] Mesurer ms/image et FPS ; noter les résultats dans ce fichier.
- [ ] Essayer l'export OpenVINO (CPU Intel i7), puis ONNX en repli ; comparer vitesse et sorties.
- **Sortie de phase :** baseline qui tourne en local + chiffres de vitesse. Fixer ici les seuils cibles.

### Phase 2 : Évaluation des limites et choix d'amélioration
- [ ] Constituer un **jeu d'évaluation** : ~50 à 100 images de plans larges (foot et rugby), annotées à la main en `player` (nombre indicatif, à ajuster). Hors Git.
- [ ] Écrire `evaluate.py` : mAP@0.5, rappel, découpage par taille de joueur (petit / moyen / grand).
- [ ] Lister les erreurs : joueurs manqués (loin, occlusions, mêlées au rugby), faux positifs (public, arbitres, staff).
- [ ] Tester `imgsz` plus grand, puis tiling/SAHI : gain de rappel vs coût CPU.
- [ ] Décision documentée : on s'arrête à la baseline, on ajoute du tiling, ou on fine-tune (Phase 3).
- **Sortie de phase :** tableau comparatif (config → mAP, rappel, FPS) + décision.

### Phase 3 : Données et fine-tuning sur Kaggle (si décidé)
- [ ] Choisir des sources de données (SoccerNet, datasets Roboflow Universe, autres) ; **vérifier licences et conditions d'accès**.
- [ ] Convertir tout en une classe `player`, format YOLO ; séparer train/val/test **par match** (pas par image) pour éviter les fuites.
- [ ] Notebook `kaggle/train_finetune.ipynb` : partir de poids pré-entraînés `n` ou `s`, `imgsz` élevé si joueurs petits.
- [ ] Évaluer sur le jeu d'éval local (jamais utilisé pour l'entraînement).
- [ ] Exporter les poids optimisés pour CPU (ONNX / OpenVINO), les télécharger dans `models/`.
- **Phase conditionnelle** (seulement si la décision de la Phase 2 est « fine-tuning »).
- **Sortie de phase :** poids fine-tunés + mesures comparées à la baseline.

### Phase 4 : Validation finale et optimisation de l'inférence locale
*(Allégée si on s'est arrêté après la Phase 2 : seule la validation et le README restent utiles.)*

- [ ] Rejouer `evaluate.py` sur le jeu d'éval avec le modèle final, version exportée.
- [ ] Optimiser l'inférence : quantification INT8 (vérifier la perte de mAP), `vid_stride`, taille d'entrée, tiling ciblé sur la zone de terrain.
- [ ] Valider sur plusieurs vidéos jamais vues (foot et rugby).
- [ ] Rédiger le `README.md` : installation PowerShell, commandes, résultats.
- [ ] Nettoyer : vérifier qu'aucun fichier lourd n'est suivi par Git.
- **Sortie de phase :** pipeline reproductible, chiffres finaux, README.

## 5. Journal de mesures (à remplir)

| Date | Modèle / config | imgsz | Runtime | ms/image | FPS | mAP@0.5 | Rappel | Notes |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
