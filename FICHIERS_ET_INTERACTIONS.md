# 📚 Guide Complet des Fichiers et leurs Interactions

## 🎯 Vue d'Ensemble

Ce document explique **chaque fichier** du projet, son rôle, et **comment il interagit** avec les autres fichiers dans l'architecture globale.

---

## 📁 Structure par Catégories

### **1. PRÉPARATION DES DONNÉES**

#### `prepare_data.py` ⭐ CORE
**Rôle :** Script principal de préparation du dataset IAM

**Fonctions principales :**
- `check_dataset_exists()` : Vérifie la présence du dataset IAM
- `get_stroke_sequence(filename)` : Extrait et normalise les strokes depuis XML
- `get_ascii_sequences(filename)` : Extrait les transcriptions ASCII
- `collect_data()` : Collecte tous les fichiers et crée les correspondances

**Interactions :**
```
prepare_data.py
    │
    ├─→ import drawing
    │   └─ Utilise drawing.align(), drawing.denoise(), 
    │      drawing.coords_to_offsets(), drawing.normalize()
    │      drawing.encode_ascii(), drawing.MAX_STROKE_LEN, 
    │      drawing.MAX_CHAR_LEN, drawing.alphabet
    │
    ├─→ Lit: data/raw/ascii/*.txt
    ├─→ Lit: data/raw/lineStrokes/*.xml
    ├─→ Lit: data/raw/original-xml/*.xml
    │
    └─→ Écrit: data/processed/
        ├─ x.npy      (strokes normalisés)
        ├─ x_len.npy  (longueurs)
        ├─ c.npy      (transcriptions)
        ├─ c_len.npy  (longueurs textes)
        └─ w_id.npy   (IDs écrivains)
```

**Utilisé par :**
- `rnn.py` (DataReader charge ces fichiers)
- `check_data.py` (vérifie leur existence)
- `check_data_rendering.py` (charge x.npy et x_len.npy)
- `prepare_evaluation_data.py` (utilise collect_data())

---

#### `check_data.py`
**Rôle :** Vérification des fichiers de données préprocessées

**Fonctions :**
- Vérifie l'existence de `data/processed/`
- Liste les fichiers `.npy` présents
- Vérifie les shapes et dtypes de chaque fichier

**Interactions :**
```
check_data.py
    │
    └─→ Lit: data/processed/*.npy
        ├─ x.npy
        ├─ x_len.npy
        ├─ c.npy
        └─ c_len.npy
```

**Utilisé par :** Script de diagnostic manuel

---

#### `check_data_rendering.py`
**Rôle :** Visualisation des strokes pour vérification

**Fonctions :**
- Charge `x.npy` et `x_len.npy`
- Convertit les strokes en images
- Sauvegarde des échantillons dans `debug_render/`

**Interactions :**
```
check_data_rendering.py
    │
    ├─→ import drawing
    │   └─ Utilise drawing.draw() pour convertir strokes → images
    │
    ├─→ Lit: data/processed/x.npy
    ├─→ Lit: data/processed/x_len.npy
    │
    └─→ Écrit: debug_render/sample_X.png
```

**Utilisé par :** Vérification visuelle après préparation

---

#### `diag_collect_stats.py`
**Rôle :** Collecte de statistiques sur le dataset

**Fonctions :**
- Parcourt le dataset IAM
- Compte les correspondances/non-correspondances
- Identifie les problèmes de structure

**Interactions :**
```
diag_collect_stats.py
    │
    ├─→ import prepare_data
    │   └─ Utilise prepare_data.RAW_BASE_DIR, 
    │      prepare_data.get_ascii_sequences()
    │
    └─→ Lit: data/raw/ascii/, data/raw/lineStrokes/
```

**Utilisé par :** Diagnostic du dataset

---

#### `diag_prepare.py`
**Rôle :** Diagnostic du processus de préparation

**Interactions :**
```
diag_prepare.py
    │
    └─→ Utilise prepare_data.py
```

---

### **2. MODÈLES RNN/LSTM** ⭐ PRINCIPAL

#### `rnn.py` ⭐ CORE
**Rôle :** Modèle RNN principal avec LSTM et attention

**Classes principales :**
- `LSTMAttentionCell` : Cellule LSTM avec mécanisme d'attention
- `DataReader` : Lecteur de données pour l'entraînement
- `RNN` : Modèle RNN complet

**Interactions :**
```
rnn.py
    │
    ├─→ import drawing
    │   └─ Utilise drawing.alphabet, drawing.MAX_STROKE_LEN, 
    │      drawing.MAX_CHAR_LEN
    │
    ├─→ import data_frame (DataFrame)
    │   └─ Utilise DataFrame pour gérer les batches
    │
    ├─→ Lit: data/processed/*.npy
    │   ├─ x.npy
    │   ├─ x_len.npy
    │   ├─ c.npy
    │   └─ c_len.npy
    │
    ├─→ Utilise: rnn_cell.py (LSTMAttentionCell)
    │   └─ Importe la cellule LSTM avec attention
    │
    └─→ Utilise: rnn_ops.py (optionnel)
        └─ Opérations RNN optimisées
```

**Utilisé par :** Script d'entraînement principal

**Fonctionnalités :**
- `DataReader` : Charge et prépare les données
- `RNN.forward()` : Forward pass avec GMM
- `RNN.nll()` : Calcul de la loss (Negative Log Likelihood)
- `RNN._rnn_free_run()` : Génération séquentielle

---

#### `rnn_cell.py` ⭐ CORE
**Rôle :** Implémentation de la cellule LSTM avec attention

**Classe principale :**
- `LSTMAttentionCell` : Cellule LSTM avec mécanisme d'attention

**Interactions :**
```
rnn_cell.py
    │
    ├─→ import drawing
    │   └─ Utilise drawing.MAX_CHAR_LEN
    │
    └─→ Utilisé par: rnn.py
        └─ Importé dans RNN.__init__()
```

**Fonctionnalités :**
- `forward()` : Forward pass avec attention
- `_compute_attention()` : Calcul des poids d'attention
- `zero_state()` : Initialisation de l'état

**Architecture :**
```
Input: [dx, dy, eos] (3-D) + attention context
    ↓
Attention Projection (Linear 2→3)
    ↓
Concaténation [dx, dy, eos, attn_proj] (6-D)
    ↓
LSTMCell (6 → lstm_size)
    ↓
Attention Computation (Mixture of Gaussians)
    ↓
Output: hidden state (lstm_size)
```

---

#### `rnn_ops.py`
**Rôle :** Opérations RNN optimisées et utilitaires

**Fonctions principales :**
- `raw_rnn()` : Boucle RNN générique
- `rnn_teacher_force()` : Entraînement avec ground truth
- `rnn_free_run()` : Génération autonome

**Interactions :**
```
rnn_ops.py
    │
    └─→ Utilisé par: rnn.py (optionnel)
        └─ Pour optimiser les opérations RNN
```

**Utilisé par :** `rnn.py` si besoin d'optimisations

---

#### `tf_base_model.py`
**Rôle :** Modèle de base TensorFlow (alternative)

**Interactions :**
```
tf_base_model.py
    │
    ├─→ Utilise: tf_utils.py
    │   └─ Utilitaires TensorFlow
    │
    └─→ Alternative à rnn.py (si TensorFlow préféré)
```

**Utilisé par :** Si on veut utiliser TensorFlow au lieu de PyTorch

---

#### `tf_utils.py`
**Rôle :** Utilitaires TensorFlow

**Fonctions :**
- `raw_rnn()` : Boucle RNN TensorFlow
- `rnn_teacher_force()` : Teacher forcing TensorFlow
- `rnn_free_run()` : Free run TensorFlow

**Interactions :**
```
tf_utils.py
    │
    └─→ Utilisé par: tf_base_model.py
```

---

### **3. RENDU ET VISUALISATION**

#### `drawing.py` ⭐ CORE
**Rôle :** Utilitaires pour la conversion et manipulation des strokes

**Fonctions principales :**
- `align()` : Correction de l'inclinaison globale
- `denoise()` : Lissage Savitzky-Golay
- `normalize()` : Normalisation des offsets
- `coords_to_offsets()` : Conversion coordonnées → offsets
- `offsets_to_coords()` : Conversion offsets → coordonnées
- `draw()` : Rendu strokes → image matplotlib
- `encode_ascii()` : Encodage texte → indices

**Constantes :**
- `alphabet` : Liste des caractères supportés
- `MAX_STROKE_LEN` : 1200
- `MAX_CHAR_LEN` : 75

**Interactions :**
```
drawing.py
    │
    ├─→ Utilisé par: prepare_data.py
    │   ├─ align(), denoise(), coords_to_offsets(), normalize()
    │   └─ encode_ascii(), MAX_STROKE_LEN, MAX_CHAR_LEN, alphabet
    │
    ├─→ Utilisé par: rnn.py
    │   └─ alphabet, MAX_STROKE_LEN, MAX_CHAR_LEN
    │
    ├─→ Utilisé par: rnn_cell.py
    │   └─ MAX_CHAR_LEN
    │
    ├─→ Utilisé par: check_data_rendering.py
    │   └─ draw() pour visualiser les strokes
    │
    └─→ Utilisé par: prepare_evaluation_data.py
        └─ draw() pour générer images réelles
```

**C'est le fichier central** utilisé par presque tous les autres modules !

---

#### `handwriting_renderer.py` ⭐ CORE
**Rôle :** Rendu stylisé d'écriture manuscrite avec polices

**Classes principales :**
- `PaperStyle` : Styles de papier (plain, ruled, grid)
- `RenderConfig` : Configuration de rendu
- `HandwritingRenderer` : Moteur de rendu principal

**Interactions :**
```
handwriting_renderer.py
    │
    ├─→ Utilise: PIL (Image, ImageDraw, ImageFont)
    ├─→ Utilise: matplotlib.font_manager
    │
    ├─→ Utilisé par: streamlit_app.py
    │   └─ HandwritingRenderer pour génération interactive
    │
    └─→ Utilisé par: prepare_evaluation_data.py
        └─ Génère images "générées" pour évaluation
```

**Fonctionnalités :**
- `render()` : Génère une image depuis texte
- `available_fonts()` : Liste les polices disponibles
- `to_bytes()` : Conversion image → bytes

---

### **4. GESTION DES DONNÉES**

#### `data_frame.py`
**Rôle :** Structure de données similaire à pandas DataFrame

**Classe principale :**
- `DataFrame` : Gestion de matrices NumPy avec batching

**Interactions :**
```
data_frame.py
    │
    └─→ Utilisé par: rnn.py
        └─ DataReader utilise DataFrame pour gérer les batches
```

**Fonctionnalités :**
- `batch_generator()` : Génération de batches
- `train_test_split()` : Division train/test
- `shuffle()` : Mélange des données

---

#### `data/dataset.py`
**Rôle :** Dataset personnalisé (si utilisé)

**Interactions :**
```
data/dataset.py
    │
    └─→ Peut être utilisé par les modèles
```

---

### **5. ÉVALUATION ET MÉTRIQUES**

#### `metrics.py` ⭐ CORE
**Rôle :** Implémentation de toutes les métriques d'évaluation

**Classes et fonctions :**
- `InceptionFeatureExtractor` : Extraction features pour FID/KID
- `calculate_fid()` : Fréchet Inception Distance
- `calculate_kid()` : Kernel Inception Distance
- `calculate_cer()` : Character Error Rate
- `calculate_wer()` : Word Error Rate
- `calculate_ssim()` : Structural Similarity Index
- `calculate_psnr()` : Peak Signal-to-Noise Ratio
- `calculate_lpips()` : Learned Perceptual Similarity
- `ocr_image()` : OCR avec Tesseract
- `calculate_ocr_accuracy()` : Précision OCR
- `evaluate_handwriting_metrics()` : Fonction principale

**Interactions :**
```
metrics.py
    │
    ├─→ Utilise: PyTorch (torch, torchvision)
    │   └─ Pour FID, KID, LPIPS
    │
    ├─→ Utilise: scikit-image
    │   └─ Pour SSIM
    │
    ├─→ Utilise: pytesseract
    │   └─ Pour OCR (CER, WER, OCR Accuracy)
    │
    ├─→ Utilise: lpips
    │   └─ Pour LPIPS
    │
    ├─→ Utilisé par: calculate_metrics.py
    │   └─ evaluate_handwriting_metrics()
    │
    ├─→ Utilisé par: quick_metrics.py
    │   └─ evaluate_handwriting_metrics()
    │
    ├─→ Utilisé par: evaluate_metrics.py
    │   └─ evaluate_handwriting_metrics()
    │
    ├─→ Utilisé par: streamlit_metrics.py
    │   └─ evaluate_handwriting_metrics()
    │
    └─→ Utilisé par: prepare_evaluation_data.py
        └─ evaluate_handwriting_metrics()
```

**C'est le fichier central** pour toutes les métriques !

---

#### `calculate_metrics.py`
**Rôle :** Script interactif guidé pour calculer les métriques

**Interactions :**
```
calculate_metrics.py
    │
    ├─→ import metrics
    │   └─ Utilise metrics.evaluate_handwriting_metrics()
    │
    ├─→ Lit: Images depuis dossiers spécifiés
    │
    └─→ Écrit: Résultats à l'écran
```

**Utilisé par :** Utilisateur pour calcul interactif

---

#### `quick_metrics.py`
**Rôle :** Script rapide avec chemins en dur ou variables d'environnement

**Interactions :**
```
quick_metrics.py
    │
    ├─→ import metrics
    │   └─ Utilise metrics.evaluate_handwriting_metrics()
    │
    └─→ Lit: Images depuis chemins configurés
```

**Utilisé par :** Automatisation et scripts batch

---

#### `evaluate_metrics.py`
**Rôle :** Script avancé avec options ligne de commande

**Interactions :**
```
evaluate_metrics.py
    │
    ├─→ import metrics
    │   └─ Utilise metrics.evaluate_handwriting_metrics()
    │
    ├─→ Arguments CLI:
    │   ├─ --real_dir
    │   ├─ --gen_dir
    │   ├─ --ground_truth_texts
    │   ├─ --use_ocr
    │   └─ --output
    │
    └─→ Écrit: JSON avec résultats
```

**Utilisé par :** Scripts automatisés et pipelines

---

#### `streamlit_metrics.py`
**Rôle :** Interface Streamlit pour les métriques

**Interactions :**
```
streamlit_metrics.py
    │
    ├─→ import metrics
    │   └─ Utilise metrics.evaluate_handwriting_metrics()
    │
    ├─→ Interface graphique:
    │   ├─ Sélection dossiers
    │   ├─ Aperçu images
    │   ├─ Calcul métriques
    │   └─ Export JSON
    │
    └─→ Utilisé par: streamlit run streamlit_metrics.py
```

**Utilisé par :** Interface utilisateur graphique

---

#### `prepare_evaluation_data.py`
**Rôle :** Préparation des données pour l'évaluation

**Interactions :**
```
prepare_evaluation_data.py
    │
    ├─→ import prepare_data
    │   └─ Utilise prepare_data.check_dataset_exists(),
    │      prepare_data.collect_data(),
    │      prepare_data.get_stroke_sequence()
    │
    ├─→ import drawing
    │   └─ Utilise drawing.draw(), drawing.alphabet
    │
    ├─→ import handwriting_renderer
    │   └─ Utilise HandwritingRenderer pour générer images
    │
    ├─→ import metrics
    │   └─ Utilise metrics.evaluate_handwriting_metrics()
    │
    ├─→ Lit: Dataset IAM
    │
    ├─→ Écrit: evaluation/real/*.png (images réelles)
    ├─→ Écrit: evaluation/gen/*.png (images générées)
    │
    └─→ Écrit: metrics_results_full.json
```

**Utilisé par :** Préparation avant évaluation complète

---

### **6. INTERFACES UTILISATEUR**

#### `streamlit_app.py` ⭐ CORE
**Rôle :** Interface principale Streamlit pour génération

**Interactions :**
```
streamlit_app.py
    │
    ├─→ import handwriting_renderer
    │   └─ Utilise HandwritingRenderer, PAPER_PRESETS, RenderConfig
    │
    └─→ Interface:
        ├─ Saisie texte
        ├─ Choix police
        ├─ Paramètres style
        └─ Génération + téléchargement
```

**Utilisé par :** `streamlit run streamlit_app.py`

---

#### `streamlit_metrics.py`
**Rôle :** Interface Streamlit pour métriques

**Interactions :**
```
streamlit_metrics.py
    │
    ├─→ import metrics
    │   └─ Utilise metrics.evaluate_handwriting_metrics()
    │
    └─→ Interface graphique pour métriques
```

**Utilisé par :** `streamlit run streamlit_metrics.py`

---

### **7. DOCUMENTATION**

#### `ARCHITECTURE_GUIDE.md`
**Rôle :** Guide complet de l'architecture

**Contenu :**
- Vue d'ensemble du projet
- Pipeline détaillé
- Architecture des modèles
- Workflow

---

#### `PIPELINE_DIAGRAM.md`
**Rôle :** Diagrammes visuels du pipeline

**Contenu :**
- Diagrammes ASCII
- Flux de données
- Comparaisons d'approches

---

#### `METRICS_GUIDE.md`
**Rôle :** Guide d'utilisation des métriques

**Contenu :**
- Explication des métriques
- Guide d'utilisation des scripts
- Interprétation des résultats

---

#### `FICHIERS_ET_INTERACTIONS.md` (ce fichier)
**Rôle :** Guide complet des fichiers et interactions

---

### **8. CONFIGURATION**

#### `requirements.txt`
**Rôle :** Dépendances Python du projet

**Contenu :**
- matplotlib, pandas, scikit-learn, scipy
- svgwrite, tensorflow, Pillow
- streamlit, torch, torchvision
- scikit-image, pytesseract, lpips

---

## 🔄 Graphique des Interactions Principales

```
┌─────────────────────────────────────────────────────────────────┐
│              GRAPHE DES DÉPENDANCES PRINCIPALES                 │
└─────────────────────────────────────────────────────────────────┘

                    drawing.py ⭐ CENTRAL
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
  prepare_data.py    rnn.py          check_data_rendering.py
        │                │                │
        │                │                │
        ▼                ▼                │
  data/processed/     rnn_cell.py         │
        │                │                │
        │                │                │
        └────────────────┼────────────────┘
                         │
                         ▼
                    data_frame.py
                         │
                         ▼
                    [Entraînement RNN]
                         │
                         ▼
                    [Génération Strokes]
                         │
                         ▼
                    drawing.draw()
                         │
                         ▼
                    [Image 128×128]


handwriting_renderer.py
        │
        ▼
streamlit_app.py ──→ [Interface Utilisateur]


prepare_evaluation_data.py
        │
        ├─→ prepare_data.py
        ├─→ drawing.py
        ├─→ handwriting_renderer.py
        └─→ metrics.py
            │
            ▼
    evaluation/real/ + evaluation/gen/
            │
            ▼
    metrics.py ──→ [Toutes les métriques]
            │
            ├─→ calculate_metrics.py
            ├─→ quick_metrics.py
            ├─→ evaluate_metrics.py
            └─→ streamlit_metrics.py
```

---

## 📊 Matrice des Dépendances

| Fichier | Utilise | Utilisé par |
|---------|---------|-------------|
| `drawing.py` | - | `prepare_data.py`, `rnn.py`, `rnn_cell.py`, `check_data_rendering.py`, `prepare_evaluation_data.py` |
| `prepare_data.py` | `drawing.py` | `rnn.py`, `check_data.py`, `prepare_evaluation_data.py`, `diag_collect_stats.py` |
| `rnn.py` | `drawing.py`, `data_frame.py`, `rnn_cell.py` | Script d'entraînement |
| `rnn_cell.py` | `drawing.py` | `rnn.py` |
| `data_frame.py` | - | `rnn.py` |
| `handwriting_renderer.py` | PIL, matplotlib | `streamlit_app.py`, `prepare_evaluation_data.py` |
| `metrics.py` | PyTorch, scikit-image, pytesseract, lpips | `calculate_metrics.py`, `quick_metrics.py`, `evaluate_metrics.py`, `streamlit_metrics.py`, `prepare_evaluation_data.py` |
| `check_data_rendering.py` | `drawing.py` | Diagnostic |
| `prepare_evaluation_data.py` | `prepare_data.py`, `drawing.py`, `handwriting_renderer.py`, `metrics.py` | Évaluation |
| `streamlit_app.py` | `handwriting_renderer.py` | Interface utilisateur |
| `streamlit_metrics.py` | `metrics.py` | Interface utilisateur |

---

## 🔗 Flux de Données Complet

### **Flux 1 : Préparation des Données**
```
Dataset IAM (XML, ASCII)
    ↓
prepare_data.py
    ├─→ Utilise drawing.py (align, denoise, normalize, etc.)
    ↓
data/processed/*.npy
    ├─→ x.npy (strokes)
    ├─→ x_len.npy (longueurs)
    ├─→ c.npy (transcriptions)
    ├─→ c_len.npy (longueurs textes)
    └─→ w_id.npy (IDs écrivains)
```

### **Flux 2 : Entraînement RNN**
```
data/processed/*.npy
    ↓
rnn.py (DataReader)
    ├─→ Utilise data_frame.py (DataFrame)
    ├─→ Utilise rnn_cell.py (LSTMAttentionCell)
    ├─→ Utilise drawing.py (alphabet, constants)
    ↓
RNN Model
    ├─→ Forward pass
    ├─→ Loss calculation
    └─→ Backpropagation
```

### **Flux 3 : Génération**
```
Texte utilisateur
    ↓
Option A: handwriting_renderer.py
    └─→ streamlit_app.py
        └─→ Image stylisée

Option B: RNN entraîné
    └─→ rnn.py (génération)
        └─→ drawing.draw()
            └─→ Image depuis strokes
```

### **Flux 4 : Évaluation**
```
prepare_evaluation_data.py
    ├─→ prepare_data.py (collect_data)
    ├─→ drawing.py (draw strokes → images réelles)
    ├─→ handwriting_renderer.py (génère images stylisées)
    ↓
evaluation/real/ + evaluation/gen/
    ↓
metrics.py (evaluate_handwriting_metrics)
    ├─→ calculate_metrics.py
    ├─→ quick_metrics.py
    ├─→ evaluate_metrics.py
    └─→ streamlit_metrics.py
    ↓
Résultats JSON
```

---

## 🎯 Points d'Entrée Principaux

### **1. Préparation des Données**
```bash
python prepare_data.py
```
- **Fichiers impliqués :** `prepare_data.py`, `drawing.py`
- **Sortie :** `data/processed/*.npy`

### **2. Vérification**
```bash
python check_data.py
python check_data_rendering.py
```
- **Fichiers impliqués :** `check_data.py`, `check_data_rendering.py`, `drawing.py`
- **Sortie :** Vérifications + `debug_render/*.png`

### **3. Entraînement**
```bash
python rnn.py
```
- **Fichiers impliqués :** `rnn.py`, `rnn_cell.py`, `rnn_ops.py`, `data_frame.py`, `drawing.py`
- **Sortie :** Modèle entraîné

### **4. Génération (Interface)**
```bash
streamlit run streamlit_app.py
```
- **Fichiers impliqués :** `streamlit_app.py`, `handwriting_renderer.py`
- **Sortie :** Images générées interactivement

### **5. Évaluation**
```bash
python prepare_evaluation_data.py
python calculate_metrics.py
# ou
streamlit run streamlit_metrics.py
```
- **Fichiers impliqués :** `prepare_evaluation_data.py`, `metrics.py`, `drawing.py`, `handwriting_renderer.py`
- **Sortie :** Métriques calculées

---

## 🔍 Fichiers Clés par Rôle

### **Fichiers Centraux (utilisés par beaucoup)**
1. **`drawing.py`** ⭐ - Utilisé par 5+ fichiers
2. **`metrics.py`** ⭐ - Utilisé par 4+ fichiers
3. **`prepare_data.py`** ⭐ - Point d'entrée principal

### **Fichiers Modèles**
1. **`rnn.py`** ⭐ - Modèle principal
2. **`rnn_cell.py`** ⭐ - Cellule LSTM
3. **`rnn_ops.py`** - Utilitaires RNN

### **Fichiers Interface**
1. **`streamlit_app.py`** ⭐ - Interface principale
2. **`streamlit_metrics.py`** - Interface métriques

### **Fichiers Utilitaires**
1. **`data_frame.py`** - Gestion données
2. **`handwriting_renderer.py`** ⭐ - Rendu stylisé
3. **`tf_utils.py`** - Utilitaires TensorFlow

### **Fichiers Diagnostic**
1. **`check_data.py`** - Vérification données
2. **`check_data_rendering.py`** - Vérification rendu
3. **`diag_collect_stats.py`** - Statistiques
4. **`diag_prepare.py`** - Diagnostic préparation

---

## 📝 Résumé des Interactions

### **Hiérarchie des Dépendances**

**Niveau 0 (Fondations) :**
- `drawing.py` - Utilitaires de base
- `data_frame.py` - Structure de données

**Niveau 1 (Préparation) :**
- `prepare_data.py` → utilise `drawing.py`
- `check_data.py` → lit `data/processed/`
- `check_data_rendering.py` → utilise `drawing.py`

**Niveau 2 (Modèles) :**
- `rnn_cell.py` → utilise `drawing.py`
- `rnn.py` → utilise `drawing.py`, `data_frame.py`, `rnn_cell.py`

**Niveau 3 (Rendu) :**
- `handwriting_renderer.py` - Indépendant
- `streamlit_app.py` → utilise `handwriting_renderer.py`

**Niveau 4 (Évaluation) :**
- `metrics.py` - Indépendant (utilise libs externes)
- `prepare_evaluation_data.py` → utilise `prepare_data.py`, `drawing.py`, `handwriting_renderer.py`, `metrics.py`
- Scripts métriques → utilisent `metrics.py`

---

---

## 🎨 Diagramme Visuel des Interactions

```
┌─────────────────────────────────────────────────────────────────┐
│              DIAGRAMME COMPLET DES INTERACTIONS                 │
└─────────────────────────────────────────────────────────────────┘

                    ┌──────────────┐
                    │ drawing.py │ ⭐ CENTRAL
                    └──────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│prepare_data  │   │   rnn.py     │   │check_data_   │
│    .py       │   │              │   │rendering.py  │
└──────────────┘   └──────────────┘   └──────────────┘
        │                  │                  │
        │                  │                  │
        ▼                  ▼                  ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│data/processed│   │rnn_cell.py   │   │debug_render/ │
│   /*.npy     │   │              │   │  *.png       │
└──────────────┘   └──────────────┘   └──────────────┘
        │                  │
        │                  │
        └──────────┬───────┘
                   │
                   ▼
            ┌──────────────┐
            │data_frame.py │
            └──────────────┘
                   │
                   ▼
            [ENTRAÎNEMENT]
                   │
                   ▼
            [GÉNÉRATION]
                   │
                   ▼
            ┌──────────────┐
            │drawing.draw()│
            └──────────────┘
                   │
                   ▼
            [IMAGE 128×128]


┌─────────────────────────────────────────────────────────────┐
│                    BRANCHE RENDU STYLISÉ                    │
└─────────────────────────────────────────────────────────────┘

            ┌──────────────┐
            │handwriting_  │
            │renderer.py   │
            └──────────────┘
                   │
                   ▼
            ┌──────────────┐
            │streamlit_    │
            │app.py        │
            └──────────────┘
                   │
                   ▼
            [INTERFACE WEB]


┌─────────────────────────────────────────────────────────────┐
│                    BRANCHE ÉVALUATION                       │
└─────────────────────────────────────────────────────────────┘

┌──────────────┐
│prepare_      │
│evaluation_   │
│data.py       │
└──────────────┘
        │
        ├─→ prepare_data.py
        ├─→ drawing.py
        ├─→ handwriting_renderer.py
        │
        ▼
┌──────────────┐
│evaluation/   │
│real/ + gen/  │
└──────────────┘
        │
        ▼
┌──────────────┐
│  metrics.py  │ ⭐ CENTRAL
└──────────────┘
        │
        ├─→ calculate_metrics.py
        ├─→ quick_metrics.py
        ├─→ evaluate_metrics.py
        └─→ streamlit_metrics.py
        │
        ▼
    [RÉSULTATS JSON]
```

---

## 🔄 Cycles de Vie des Données

### **Cycle 1 : Préparation → Entraînement → Génération**
```
1. Dataset IAM brut
   ↓
2. prepare_data.py + drawing.py
   ↓
3. data/processed/*.npy
   ↓
4. rnn.py + rnn_cell.py + data_frame.py
   ↓
5. Modèle entraîné
   ↓
6. Génération strokes
   ↓
7. drawing.draw()
   ↓
8. Image finale
```

### **Cycle 2 : Évaluation Complète**
```
1. prepare_evaluation_data.py
   ├─→ prepare_data.py (collect_data)
   ├─→ drawing.py (images réelles)
   └─→ handwriting_renderer.py (images générées)
   ↓
2. evaluation/real/ + evaluation/gen/
   ↓
3. metrics.py (calcul métriques)
   ↓
4. Scripts métriques (interface)
   ↓
5. Résultats JSON
```

---

## 📋 Checklist des Fichiers par Catégorie

### ✅ **Fichiers Core (Essentiels)**
- [x] `drawing.py` - Utilitaires strokes
- [x] `prepare_data.py` - Préparation données
- [x] `rnn.py` - Modèle principal
- [x] `rnn_cell.py` - Cellule LSTM
- [x] `handwriting_renderer.py` - Rendu stylisé
- [x] `metrics.py` - Métriques
- [x] `streamlit_app.py` - Interface principale

### ✅ **Fichiers Utilitaires**
- [x] `data_frame.py` - Gestion données
- [x] `rnn_ops.py` - Opérations RNN
- [x] `tf_utils.py` - Utilitaires TensorFlow
- [x] `tf_base_model.py` - Modèle TensorFlow

### ✅ **Fichiers Diagnostic**
- [x] `check_data.py` - Vérification données
- [x] `check_data_rendering.py` - Vérification rendu
- [x] `diag_collect_stats.py` - Statistiques
- [x] `diag_prepare.py` - Diagnostic

### ✅ **Fichiers Évaluation**
- [x] `prepare_evaluation_data.py` - Préparation évaluation
- [x] `calculate_metrics.py` - Script interactif
- [x] `quick_metrics.py` - Script rapide
- [x] `evaluate_metrics.py` - Script avancé
- [x] `streamlit_metrics.py` - Interface métriques

### ✅ **Fichiers Documentation**
- [x] `ARCHITECTURE_GUIDE.md` - Guide architecture
- [x] `PIPELINE_DIAGRAM.md` - Diagrammes pipeline
- [x] `METRICS_GUIDE.md` - Guide métriques
- [x] `FICHIERS_ET_INTERACTIONS.md` - Ce document

---

---

## 🎯 Intersections Critiques

### **Intersection 1 : drawing.py (Hub Central)**

`drawing.py` est le **fichier le plus utilisé** dans le projet :

```
                    drawing.py
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
prepare_data.py      rnn.py      check_data_rendering.py
        │                │                │
        │                │                │
        ▼                ▼                ▼
    [Data]          [Model]          [Debug]
```

**Pourquoi central ?**
- Définit `alphabet` (utilisé partout)
- Définit `MAX_STROKE_LEN`, `MAX_CHAR_LEN` (constantes globales)
- Fournit toutes les fonctions de transformation strokes
- Point unique de conversion strokes ↔ images

---

### **Intersection 2 : data/processed/ (Hub de Données)**

Tous les fichiers de données convergent vers `data/processed/` :

```
prepare_data.py ──→ data/processed/*.npy
                            │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
    rnn.py            check_data.py      check_data_rendering.py
    (entraînement)    (vérification)    (visualisation)
```

**Fichiers créés :**
- `x.npy` : Strokes normalisés
- `x_len.npy` : Longueurs réelles
- `c.npy` : Transcriptions encodées
- `c_len.npy` : Longueurs textes
- `w_id.npy` : IDs écrivains

---

### **Intersection 3 : metrics.py (Hub d'Évaluation)**

Tous les scripts d'évaluation utilisent `metrics.py` :

```
                    metrics.py
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
calculate_metrics.py  quick_metrics.py  evaluate_metrics.py
        │                │                │
        └────────────────┼────────────────┘
                         │
                         ▼
                streamlit_metrics.py
```

**Fonction centrale :**
- `evaluate_handwriting_metrics()` : Calcule toutes les métriques

---

### **Intersection 4 : RNN Pipeline**

Le pipeline RNN connecte plusieurs fichiers :

```
data/processed/*.npy
        │
        ▼
    rnn.py (DataReader)
        │
        ├─→ data_frame.py (DataFrame)
        ├─→ rnn_cell.py (LSTMAttentionCell)
        └─→ drawing.py (alphabet, constants)
        │
        ▼
    [Entraînement]
        │
        ▼
    [Génération Strokes]
        │
        ▼
    drawing.draw()
        │
        ▼
    [Image 128×128]
```

---

## 🔗 Chaînes de Dépendances

### **Chaîne 1 : Préparation → Entraînement**
```
prepare_data.py
    → utilise drawing.py
    → crée data/processed/*.npy
        ↓
rnn.py (DataReader)
    → lit data/processed/*.npy
    → utilise data_frame.py
    → utilise rnn_cell.py
    → utilise drawing.py
```

### **Chaîne 2 : Génération Utilisateur**
```
streamlit_app.py
    → utilise handwriting_renderer.py
    → génère image directement
```

### **Chaîne 3 : Évaluation Complète**
```
prepare_evaluation_data.py
    → utilise prepare_data.py
    → utilise drawing.py
    → utilise handwriting_renderer.py
    → crée evaluation/real/ + evaluation/gen/
        ↓
metrics.py
    → lit evaluation/real/ + evaluation/gen/
    → calcule métriques
        ↓
Scripts métriques
    → utilisent metrics.py
    → affichent résultats
```

---

## 📊 Tableau Récapitulatif des Fichiers

| Fichier | Catégorie | Rôle | Dépendances | Utilisé par |
|---------|-----------|------|-------------|-------------|
| `drawing.py` | Core | Utilitaires strokes | - | 5+ fichiers |
| `prepare_data.py` | Préparation | Préparation dataset | `drawing.py` | 4+ fichiers |
| `rnn.py` | Modèle | Modèle RNN principal | `drawing.py`, `data_frame.py`, `rnn_cell.py` | Entraînement |
| `rnn_cell.py` | Modèle | Cellule LSTM | `drawing.py` | `rnn.py` |
| `rnn_ops.py` | Modèle | Opérations RNN | - | `rnn.py` (optionnel) |
| `data_frame.py` | Utilitaires | Gestion données | - | `rnn.py` |
| `handwriting_renderer.py` | Rendu | Rendu stylisé | PIL, matplotlib | `streamlit_app.py`, `prepare_evaluation_data.py` |
| `metrics.py` | Évaluation | Métriques | PyTorch, scikit-image, pytesseract | 4+ scripts |
| `streamlit_app.py` | Interface | Interface principale | `handwriting_renderer.py` | Utilisateur |
| `streamlit_metrics.py` | Interface | Interface métriques | `metrics.py` | Utilisateur |
| `check_data.py` | Diagnostic | Vérification | - | Diagnostic |
| `check_data_rendering.py` | Diagnostic | Visualisation | `drawing.py` | Diagnostic |
| `prepare_evaluation_data.py` | Évaluation | Préparation éval | `prepare_data.py`, `drawing.py`, `handwriting_renderer.py`, `metrics.py` | Évaluation |
| `calculate_metrics.py` | Évaluation | Script interactif | `metrics.py` | Utilisateur |
| `quick_metrics.py` | Évaluation | Script rapide | `metrics.py` | Automatisation |
| `evaluate_metrics.py` | Évaluation | Script avancé | `metrics.py` | Scripts |
| `diag_collect_stats.py` | Diagnostic | Statistiques | `prepare_data.py` | Diagnostic |
| `diag_prepare.py` | Diagnostic | Diagnostic | `prepare_data.py` | Diagnostic |
| `tf_base_model.py` | Modèle | Modèle TensorFlow | `tf_utils.py` | Alternative |
| `tf_utils.py` | Utilitaires | Utils TensorFlow | - | `tf_base_model.py` |

---

## 🎓 Points Clés à Retenir

1. **`drawing.py` est le hub central** - Utilisé par presque tous les modules
2. **`data/processed/` est le hub de données** - Tous les fichiers de données y convergent
3. **`metrics.py` est le hub d'évaluation** - Tous les scripts métriques l'utilisent
4. **Le pipeline RNN** connecte : `prepare_data.py` → `rnn.py` → `rnn_cell.py` → `drawing.py`
5. **Deux approches de génération** :
   - RNN/LSTM : `rnn.py` → `drawing.draw()` → Image
   - Rendu stylisé : `handwriting_renderer.py` → Image directe

---

Ce document fournit une vue complète de tous les fichiers et de leurs interactions dans l'architecture du projet !

