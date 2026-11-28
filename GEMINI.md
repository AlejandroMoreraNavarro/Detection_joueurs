# GEMINI.md : règles permanentes du dépôt

## 1. Rôle et périmètre

Tu assistes le développement d'un projet de **détection de joueurs** (football / rugby) sur plans larges, par bounding boxes.

- **Tâche principale :** détection monoclasse `player` (boîtes englobantes + score de confiance).
- **Extension en cours (Phase 2b) :** suivi **anonyme** des joueurs (identifiants persistants d'une image à l'autre), par suivi par détection avec un tracker heuristique (ByteTrack / BoT-SORT d'Ultralytics ou bibliothèque `trackers`). Pas d'identité réelle (nom, numéro de maillot).
- **Hors périmètre (ne pas implémenter, même "pour anticiper") :** distinction des équipes, OCR de maillots, détection du ballon, modèles de suivi appris ou de bout en bout (CAMELTrack, MOTR...), Re-ID visuel ajouté par nous (seul un Re-ID intégré à un tracker, mesuré, est permis), segmentation, estimation de pose, homographie/calibration du terrain, interface web, API, déploiement cloud.
- **Principe directeur : simplicité d'abord.** Si un modèle pré-entraîné remplit déjà la tâche de façon satisfaisante, on l'utilise tel quel et on s'arrête. Ne propose jamais fine-tuning, tiling/SAHI, Kaggle ou changement de modèle tant qu'une limite **mesurée** (Phases 1-2 de `PROJET.md`) ne le justifie.
- **Pas de sur-ingénierie :** solution la plus simple qui fonctionne. Pas de classes abstraites, de systèmes de plugins, de frameworks de config ou d'options non demandés.
- Si une demande sort du périmètre ou si elle est ambiguë, **pose une question avant de coder** (une seule, la plus utile).
- Ne présente jamais une supposition comme un fait. N'invente pas d'API, d'argument ou de résultat de benchmark : en cas de doute, dis-le et vérifie la documentation.
- Réponds en français, de façon concise.

## 2. Contraintes techniques

### Exécution locale = CPU uniquement
- Aucune dépendance à CUDA. Pas de `.cuda()`, `device=0` ou `device="cuda"` en dur : `device="cpu"` par défaut.
- Machine : CPU **Intel Core i7** → l'export **OpenVINO** est le premier choix d'accélération, ONNX Runtime en repli.
- Privilégier les modèles `nano`/`small`, l'export ONNX ou OpenVINO, un `imgsz` raisonnable et le saut d'images (`vid_stride`) pour les vidéos.
- Le tiling/SAHI multiplie le coût d'inférence par le nombre de tuiles : ne l'activer que si une mesure montre que c'est nécessaire.
- Toujours tester sur une image ou une vidéo courte (≤ 15 s) avant toute exécution longue.

### Environnement Windows / PowerShell
- Environnement virtuel : `.venv`. Activation : `.\.venv\Scripts\Activate.ps1`.
- Écris les commandes pour **PowerShell** : pas de `&&` (utiliser `;`), pas de `rm`, `export`, `source`, `cat`/`grep` style bash. Continuation de ligne avec le backtick `` ` ``.
- Installer avec `python -m pip install ...`.
- Chemins en Python : `pathlib.Path`, jamais de séparateurs en dur. Fichiers texte en UTF-8.
- PyTorch CPU : `python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.

### Séparation local / Kaggle
- `src/` et `scripts/` : inférence, évaluation et export, **exécutés en local sur CPU**.
- `kaggle/` : notebooks/scripts d'entraînement (GPU), **autonomes** (installation des dépendances, chemins `/kaggle/input` et `/kaggle/working` dans le notebook).
- `src/` n'importe **jamais** rien de `kaggle/`, et inversement. L'inférence ne dépend pas du code d'entraînement.
- Les poids entraînés sont téléchargés manuellement depuis Kaggle vers `models/` (dossier ignoré par Git).

### Style de code
- Python 3.10+, type hints, fonctions courtes, docstrings d'une ligne.
- Pas de chemins ou de seuils en dur : arguments `argparse` ou `configs/default.yaml`.
- Format de sortie unique pour les détections : `(x1, y1, x2, y2, score)`, classe implicite `player`.
- Logs simples (`logging`), pas de `print` dispersés dans `src/`.

## 3. Bonnes pratiques Git

- **Ne jamais commiter :** vidéos (`.mp4 .avi .mov .mkv`), images de données, datasets, poids et modèles (`.pt .pth .onnx .engine .bin`, dossiers OpenVINO), `.venv/`, `runs/`, `outputs/`, `data/`, `models/`, secrets (`kaggle.json`, `.env`).
- Vérifier que `.gitignore` couvre ces éléments **avant** tout `git add`. Préférer `git add <fichiers précis>` à `git add .`, et lire `git status` avant de commiter.
- Ne jamais lancer `git commit` ni `git push` sans demande explicite. Messages de commit courts, à l'impératif.
- Aucun fichier > 5 Mo dans le dépôt.
- `requirements.txt` **léger** : uniquement ce qui sert à l'inférence locale. Toute nouvelle dépendance doit être justifiée. Les dépendances d'entraînement vivent dans le notebook Kaggle.
- Code modulaire : un module = une responsabilité (chargement du modèle, inférence, tiling, E/S vidéo, visualisation, évaluation).

## 4. Règles pour le suivi (Phase 2b)

- **Mesurer avec la vérité terrain**, jamais avec le nombre brut d'IDs : utiliser `gt/gt.txt` de SportsMOT (colonne 2 = identifiant) et rapporter IDF1, HOTA et nombre de changements d'ID. Le nombre d'IDs uniques et leur durée de vie restent un indicateur complémentaire.
- **Un seul changement de configuration à la fois**, mesuré avant le suivant. Configs de tracker dans `configs/` (fichiers YAML), pas de valeurs en dur.
- Mesurer aussi le coût CPU **total** par image (détection + suivi).
- Toute bibliothèque d'évaluation ou de suivi supplémentaire s'installe dans un venv séparé (`.venv-eval`), n'est **pas** ajoutée à `requirements.txt`, et se justifie en 2 lignes avant installation.
- Ne pas inventer de paramètres : vérifier les noms d'arguments et de clés YAML dans la documentation.

## 5. Workflow

- Avant de coder, lire `PROJET.md` et indiquer la phase en cours.
- Quand une étape de `PROJET.md` est terminée, cocher la case correspondante.
- Mesurer avant d'optimiser : temps par image (ms) et FPS sur CPU, pas d'optimisation "au feeling".

### Garde-fous : ne pas tourner indéfiniment
- **Même erreur après 3 tentatives de correction** (ou 3 approches différentes sans progrès sur un même problème) : **stop**. Résume ce qui a été essayé, l'hypothèse la plus probable, puis demande à l'utilisateur s'il faut continuer ou s'il a un conseil.
- **Commande ou entraînement qui dépasse ~5 minutes** sans résultat attendu : ne pas relancer ni attendre en boucle. Interromps, explique pourquoi c'est long, propose une version plus légère (image plus petite, vidéo plus courte, `vid_stride`, modèle plus petit) et demande confirmation.
- **Pas de relance automatique en boucle** (retry, boucle `while`, essais successifs de modèles ou de paramètres) sans accord explicite.
- Si la tâche s'étend au-delà de la demande initiale (nouvelle dépendance, refonte, nouveau modèle), **stop et demande** avant de continuer.
- Ces seuils (3 tentatives, ~5 min) sont des valeurs par défaut : l'utilisateur peut les modifier.
