# 🔍 Traçabilité Complète du Projet

## 📋 Vue d'Ensemble de la Traçabilité

Ce document fournit une **traçabilité complète et personnalisée** de tous les composants du projet, leurs relations, et le flux de données de bout en bout.

---

## 🗺️ Carte de Traçabilité Globale

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRACABILITÉ COMPLÈTE                         │
└─────────────────────────────────────────────────────────────────┘

SOURCE DE DONNÉES
    │
    ├─ Dataset IAM (data/raw/)
    │  ├─ ascii/*.txt
    │  ├─ lineStrokes/*.xml
    │  └─ original-xml/*.xml
    │
    └─ Images PNG (words/) [Source alternative]
    
         │
         ▼
    PRÉPARATION
    │
    ├─ prepare_data.py
    │  ├─ Utilise: drawing.py
    │  ├─ Lit: data/raw/
    │  └─ Écrit: data/processed/*.npy
    │
    ├─ check_data.py
    │  └─ Vérifie: data/processed/*.npy
    │
    └─ check_data_rendering.py
       ├─ Utilise: drawing.py
       ├─ Lit: data/processed/x.npy, x_len.npy
       └─ Écrit: debug_render/*.png
    
         │
         ▼
    DONNÉES PRÉPARÉES
    │
    └─ data/processed/
       ├─ x.npy (strokes)
       ├─ x_len.npy (longueurs)
       ├─ c.npy (transcriptions)
       ├─ c_len.npy (longueurs textes)
       └─ w_id.npy (IDs écrivains)
    
         │
         ▼
    MODÈLES
    │
    ├─ RNN/LSTM ⭐ PRINCIPAL
    │  │
    │  ├─ rnn.py
    │  │  ├─ Utilise: drawing.py, data_frame.py, rnn_cell.py
    │  │  ├─ Lit: data/processed/*.npy
    │  │  └─ Génère: Strokes [dx, dy, eos]
    │  │
    │  ├─ rnn_cell.py
    │  │  ├─ Utilise: drawing.py
    │  │  └─ Utilisé par: rnn.py
    │  │
    │  └─ rnn_ops.py
    │     └─ Utilisé par: rnn.py (optionnel)
    │
    └─ TensorFlow (Alternative)
       ├─ tf_base_model.py
       └─ tf_utils.py
    
         │
         ▼
    GÉNÉRATION
    │
    ├─ RNN/LSTM
    │  │
    │  ├─ rnn.py (génération)
    │  │  └─ Strokes générés
    │  │     │
    │  │     └─→ drawing.draw()
    │  │        └─→ Image 128×128
    │
    └─ Rendu Stylisé
       │
       ├─ handwriting_renderer.py
       │  └─ Image directe depuis texte
       │
       └─ streamlit_app.py
          ├─ Utilise: handwriting_renderer.py
          └─ Interface web interactive
    
         │
         ▼
    ÉVALUATION
    │
    ├─ prepare_evaluation_data.py
    │  ├─ Utilise: prepare_data.py, drawing.py, 
    │  │           handwriting_renderer.py, metrics.py
    │  ├─ Lit: Dataset IAM
    │  └─ Écrit: evaluation/real/ + evaluation/gen/
    │
    ├─ metrics.py ⭐ CENTRAL
    │  └─ Implémente toutes les métriques
    │
    └─ Scripts métriques
       ├─ calculate_metrics.py → Utilise metrics.py
       ├─ quick_metrics.py → Utilise metrics.py
       ├─ evaluate_metrics.py → Utilise metrics.py
       └─ streamlit_metrics.py → Utilise metrics.py
       │
       └─→ Résultats JSON
```

---

## 🔗 Matrice de Traçabilité Détaillée

### **Niveau 1 : Fichiers Sources**

| Fichier Source | Type | Utilisé par | Format |
|----------------|------|-------------|--------|
| `data/raw/ascii/*.txt` | Transcriptions | `prepare_data.py` | Texte ASCII |
| `data/raw/lineStrokes/*.xml` | Strokes | `prepare_data.py` | XML |
| `data/raw/original-xml/*.xml` | Métadonnées | `prepare_data.py` | XML |
| `words/*/` | Images PNG | Source alternative | PNG |

### **Niveau 2 : Fichiers Préprocessés**

| Fichier | Créé par | Utilisé par | Contenu |
|---------|---------|-------------|---------|
| `data/processed/x.npy` | `prepare_data.py` | `rnn.py`, `check_data_rendering.py` | Strokes (N, 1200, 3) |
| `data/processed/x_len.npy` | `prepare_data.py` | `rnn.py`, `check_data_rendering.py` | Longueurs (N,) |
| `data/processed/c.npy` | `prepare_data.py` | `rnn.py` | Transcriptions (N, 75) |
| `data/processed/c_len.npy` | `prepare_data.py` | `rnn.py` | Longueurs textes (N,) |
| `data/processed/w_id.npy` | `prepare_data.py` | (Optionnel) | IDs écrivains (N,) |

### **Niveau 3 : Fichiers de Modèles**

| Fichier | Rôle | Dépendances | Génère |
|---------|------|-------------|--------|
| `rnn.py` | Modèle principal | `drawing.py`, `data_frame.py`, `rnn_cell.py` | Strokes |
| `rnn_cell.py` | Cellule LSTM | `drawing.py` | Hidden states |
| `rnn_ops.py` | Opérations RNN | - | Utilitaires |
| `tf_base_model.py` | Modèle TF | `tf_utils.py` | Alternative |
| `tf_utils.py` | Utils TF | - | Utilitaires |

### **Niveau 4 : Fichiers de Rendu**

| Fichier | Rôle | Dépendances | Génère |
|---------|------|-------------|--------|
| `drawing.py` | Conversion strokes | - | Images matplotlib |
| `handwriting_renderer.py` | Rendu stylisé | PIL, matplotlib | Images PIL |

### **Niveau 5 : Fichiers d'Évaluation**

| Fichier | Rôle | Dépendances | Génère |
|---------|------|-------------|--------|
| `metrics.py` | Métriques | PyTorch, scikit-image, pytesseract, lpips | Scores |
| `prepare_evaluation_data.py` | Préparation | `prepare_data.py`, `drawing.py`, `handwriting_renderer.py`, `metrics.py` | Images évaluation |
| `calculate_metrics.py` | Script interactif | `metrics.py` | Résultats console |
| `quick_metrics.py` | Script rapide | `metrics.py` | Résultats console |
| `evaluate_metrics.py` | Script avancé | `metrics.py` | JSON |
| `streamlit_metrics.py` | Interface | `metrics.py` | Interface web |

### **Niveau 6 : Fichiers Interface**

| Fichier | Rôle | Dépendances | Utilisé par |
|---------|------|-------------|-------------|
| `streamlit_app.py` | Interface principale | `handwriting_renderer.py` | Utilisateur |
| `streamlit_metrics.py` | Interface métriques | `metrics.py` | Utilisateur |

---

## 📊 Flux de Traçabilité par Cas d'Usage

### **Cas 1 : Préparation Complète**

```
Utilisateur
    │
    └─→ python prepare_data.py
            │
            ├─→ check_dataset_exists()
            │   └─→ Vérifie data/raw/
            │
            ├─→ collect_data()
            │   ├─→ Parcourt data/raw/ascii/
            │   ├─→ Trouve lineStrokes correspondants
            │   └─→ Trouve original-xml correspondants
            │
            ├─→ get_stroke_sequence()
            │   ├─→ Lit XML
            │   ├─→ drawing.align()
            │   ├─→ drawing.denoise()
            │   ├─→ drawing.coords_to_offsets()
            │   └─→ drawing.normalize()
            │
            ├─→ get_ascii_sequences()
            │   └─→ drawing.encode_ascii()
            │
            └─→ Écrit data/processed/*.npy
                │
                └─→ [TRACÉ] Fichiers créés avec timestamps
```

**Traçabilité :**
- ✅ Source : `data/raw/`
- ✅ Processus : `prepare_data.py` + `drawing.py`
- ✅ Destination : `data/processed/*.npy`
- ✅ Vérification : `check_data.py`

---

### **Cas 2 : Entraînement RNN**

```
Utilisateur
    │
    └─→ python rnn.py
            │
            ├─→ DataReader('data/processed/')
            │   ├─→ Lit x.npy, x_len.npy, c.npy, c_len.npy
            │   ├─→ Crée DataFrame (data_frame.py)
            │   └─→ Train/test split
            │
            ├─→ RNN(lstm_size=400, ...)
            │   ├─→ Utilise rnn_cell.py (LSTMAttentionCell)
            │   └─→ Utilise drawing.py (alphabet, constants)
            │
            ├─→ Boucle d'entraînement
            │   ├─→ Forward: rnn.forward(x, c, c_len, x_len, y)
            │   ├─→ Loss: Negative Log Likelihood
            │   └─→ Backward: optimizer.step()
            │
            └─→ Sauvegarde modèle
                │
                └─→ [TRACÉ] Modèle sauvegardé avec hyperparamètres
```

**Traçabilité :**
- ✅ Données : `data/processed/*.npy`
- ✅ Modèle : `rnn.py` + `rnn_cell.py`
- ✅ Utilitaires : `data_frame.py`, `drawing.py`
- ✅ Sortie : Modèle entraîné

---

### **Cas 3 : Génération Utilisateur**

```
Utilisateur
    │
    └─→ streamlit run streamlit_app.py
            │
            ├─→ Interface web
            │   ├─→ Saisie texte
            │   ├─→ Paramètres style
            │   └─→ Bouton "Générer"
            │
            ├─→ handwriting_renderer.py
            │   ├─→ HandwritingRenderer.render()
            │   ├─→ Sélection police
            │   ├─→ Application effets (jitter, tilt, noise)
            │   └─→ Génération image
            │
            └─→ Affichage + Téléchargement
                │
                └─→ [TRACÉ] Image générée avec paramètres
```

**Traçabilité :**
- ✅ Interface : `streamlit_app.py`
- ✅ Moteur : `handwriting_renderer.py`
- ✅ Sortie : Image PNG
- ✅ Paramètres : Sauvegardés dans session

---

### **Cas 4 : Évaluation Complète**

```
Utilisateur
    │
    └─→ python prepare_evaluation_data.py
            │
            ├─→ prepare_data.collect_data()
            │   └─→ Charge dataset IAM
            │
            ├─→ Pour chaque échantillon:
            │   │
            │   ├─→ Images RÉELLES
            │   │   ├─→ prepare_data.get_stroke_sequence()
            │   │   └─→ drawing.draw()
            │   │       └─→ evaluation/real/sample_X.png
            │   │
            │   └─→ Images GÉNÉRÉES
            │       └─→ handwriting_renderer.render()
            │           └─→ evaluation/gen/sample_X.png
            │
            ├─→ metrics.evaluate_handwriting_metrics()
            │   ├─→ FID, KID (Inception v3)
            │   ├─→ CER, WER (Tesseract OCR)
            │   ├─→ SSIM, PSNR, LPIPS
            │   └─→ OCR Accuracy
            │
            └─→ Écrit metrics_results_full.json
                │
                └─→ [TRACÉ] Résultats avec timestamps
```

**Traçabilité :**
- ✅ Source : Dataset IAM
- ✅ Préparation : `prepare_evaluation_data.py`
- ✅ Images : `evaluation/real/` + `evaluation/gen/`
- ✅ Métriques : `metrics.py`
- ✅ Résultats : JSON avec toutes les métriques

---

## 🔍 Traçabilité des Dépendances

### **Dépendances Externes**

| Package | Utilisé par | Rôle |
|---------|-------------|------|
| `torch` | `rnn.py`, `rnn_cell.py`, `metrics.py` | Deep learning |
| `numpy` | Tous les fichiers | Calculs numériques |
| `PIL` | `handwriting_renderer.py`, `metrics.py` | Manipulation images |
| `matplotlib` | `drawing.py`, `handwriting_renderer.py` | Visualisation |
| `scikit-image` | `metrics.py` | SSIM |
| `pytesseract` | `metrics.py` | OCR |
| `lpips` | `metrics.py` | LPIPS |
| `streamlit` | `streamlit_app.py`, `streamlit_metrics.py` | Interfaces web |
| `pandas` | `data_frame.py` | Gestion données |

### **Dépendances Internes (Hiérarchie)**

```
Niveau 0 (Fondations)
    ├─ drawing.py (constantes, fonctions de base)
    └─ data_frame.py (structure données)

Niveau 1 (Préparation)
    ├─ prepare_data.py → drawing.py
    ├─ check_data.py → (lecture fichiers)
    └─ check_data_rendering.py → drawing.py

Niveau 2 (Modèles)
    ├─ rnn_cell.py → drawing.py
    ├─ rnn.py → drawing.py, data_frame.py, rnn_cell.py
    └─ tf_utils.py → (indépendant)

Niveau 3 (Rendu)
    ├─ handwriting_renderer.py → (indépendant)
    └─ streamlit_app.py → handwriting_renderer.py

Niveau 4 (Évaluation)
    ├─ metrics.py → (libs externes)
    ├─ prepare_evaluation_data.py → prepare_data.py, drawing.py, 
    │                                handwriting_renderer.py, metrics.py
    └─ Scripts métriques → metrics.py
```

---

## 📝 Journal de Traçabilité

### **Événements Traçables**

1. **Création données préprocessées**
   - Fichier : `data/processed/*.npy`
   - Créé par : `prepare_data.py`
   - Timestamp : Date/heure création
   - Vérifié par : `check_data.py`

2. **Entraînement modèle**
   - Modèle : RNN/LSTM
   - Fichiers : `rnn.py`
   - Hyperparamètres : lstm_size=400, output_mixture=20, etc.
   - Checkpoints : (si sauvegardés)

3. **Génération images**
   - Source : Texte utilisateur
   - Méthode : RNN ou Rendu stylisé
   - Paramètres : (selon méthode)
   - Sortie : Image PNG

4. **Calcul métriques**
   - Images : `evaluation/real/` + `evaluation/gen/`
   - Métriques calculées : FID, KID, CER, WER, SSIM, PSNR, LPIPS, OCR Accuracy
   - Résultats : `metrics_results.json` ou `metrics_results_full.json`
   - Timestamp : Date/heure calcul

---

## 🎯 Points de Contrôle de Traçabilité

### **Checkpoint 1 : Données Préparées**
- ✅ Fichiers `data/processed/*.npy` existent
- ✅ Shapes corrects
- ✅ Dtypes corrects
- ✅ Vérifié par : `check_data.py`

### **Checkpoint 2 : Modèle Entraîné**
- ✅ Modèle sauvegardé
- ✅ Hyperparamètres documentés
- ✅ Performance validée

### **Checkpoint 3 : Génération Fonctionnelle**
- ✅ RNN génère des strokes valides
- ✅ Rendu stylisé fonctionne
- ✅ Interface utilisateur opérationnelle

### **Checkpoint 4 : Évaluation Complète**
- ✅ Images réelles et générées créées
- ✅ Métriques calculées
- ✅ Résultats sauvegardés

---

## 🔄 Cycle de Vie des Données (Traçabilité)

```
1. DONNÉES BRUTES
   Source: Dataset IAM
   Format: XML, ASCII
   Localisation: data/raw/
   │
   └─→ [TRACÉ] Fichiers source identifiés

2. PRÉPARATION
   Script: prepare_data.py
   Utilise: drawing.py
   │
   └─→ [TRACÉ] Transformation appliquée

3. DONNÉES PRÉPROCESSÉES
   Format: NumPy arrays
   Localisation: data/processed/
   │
   └─→ [TRACÉ] Fichiers créés avec metadata

4. ENTRAÎNEMENT
   Script: rnn.py
   Utilise: data/processed/*.npy
   │
   └─→ [TRACÉ] Modèle entraîné avec hyperparamètres

5. GÉNÉRATION
   Source: Texte utilisateur
   Méthode: RNN ou Rendu stylisé
   │
   └─→ [TRACÉ] Image générée avec paramètres

6. ÉVALUATION
   Script: prepare_evaluation_data.py + metrics.py
   Utilise: Dataset + Images générées
   │
   └─→ [TRACÉ] Métriques calculées et sauvegardées
```

---

## 📋 Checklist de Traçabilité

### **Pour chaque composant :**

- [ ] **Source identifiée** : D'où viennent les données ?
- [ ] **Transformation documentée** : Quelles transformations sont appliquées ?
- [ ] **Destination claire** : Où vont les résultats ?
- [ ] **Dépendances listées** : Quels fichiers sont utilisés ?
- [ ] **Utilisateurs identifiés** : Qui utilise ce composant ?
- [ ] **Paramètres documentés** : Quels paramètres sont utilisés ?
- [ ] **Résultats traçables** : Comment vérifier les résultats ?

---

## 🎓 Utilisation de la Traçabilité

### **Pour Déboguer :**
1. Identifier le composant problématique
2. Remonter la chaîne de dépendances
3. Vérifier les données à chaque étape
4. Utiliser les scripts de diagnostic

### **Pour Améliorer :**
1. Identifier les goulots d'étranglement
2. Comprendre les dépendances
3. Optimiser les composants critiques
4. Ajouter de nouveaux composants de manière modulaire

### **Pour Documenter :**
1. Suivre le flux de données
2. Documenter les transformations
3. Maintenir la cohérence
4. Faciliter la maintenance

---

Ce document de traçabilité est **personnalisé pour ce projet spécifique** et reflète l'architecture réelle avec RNN/LSTM comme approche principale.

