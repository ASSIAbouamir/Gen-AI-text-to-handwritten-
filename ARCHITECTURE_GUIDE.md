# 📐 Architecture et Pipeline Complet du Projet de Génération d'Écriture Manuscrite

## 🎯 Vue d'ensemble du Projet

Ce projet implémente un **système de génération d'écriture manuscrite** à partir de texte, utilisant **trois approches complémentaires** :

1. **RNN/LSTM avec Attention** ⭐ **PRINCIPAL** : Génération séquentielle de strokes (traits) via un modèle RNN conditionné par le texte
2. **Rendu stylisé basé sur polices** : Génération rapide utilisant des polices manuscrites avec effets réalistes (jitter, inclinaison, texture)
3. **TensorFlow (Alternative)** : Implémentation alternative avec TensorFlow pour compatibilité

**Note :** Le GAN conditionnel est documenté mais **non implémenté** dans cette version du projet.

---

## 🏗️ Architecture Globale

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE COMPLET DU PROJET                   │
└─────────────────────────────────────────────────────────────────┘

1. PRÉPARATION DES DONNÉES
   │
   ├─ Dataset IAM (Images + Strokes + Transcriptions)
   │
   ├─ prepare_data.py
   │  ├─ Extraction des strokes (traits) depuis XML
   │  ├─ Extraction des transcriptions ASCII
   │  ├─ Normalisation et préprocessing
   │  └─ Sauvegarde en format NumPy (.npy)
   │
   └─ data/processed/
      ├─ x.npy      (strokes: offsets dx, dy, eos)
      ├─ x_len.npy  (longueurs réelles)
      ├─ c.npy      (transcriptions encodées)
      ├─ c_len.npy  (longueurs de texte)
      └─ w_id.npy   (IDs des écrivains)

2. ENTRÂINEMENT RNN/LSTM ⭐ PRINCIPAL
   │
   ├─ rnn.py
   │  ├─ DataReader : Chargement data/processed/*.npy
   │  ├─ RNN Model : Modèle principal avec GMM
   │  └─ Entraînement avec Negative Log Likelihood
   │
   ├─ rnn_cell.py
   │  └─ LSTMAttentionCell : Cellule LSTM avec mécanisme d'attention
   │
   ├─ rnn_ops.py
   │  └─ Opérations RNN optimisées (raw_rnn, teacher_force, free_run)
   │
   ├─ data_frame.py
   │  └─ DataFrame : Gestion batches et train/test split
   │
   └─ Modèle entraîné sauvegardé

3. GÉNÉRATION & INFÉRENCE
   │
   ├─ RNN/LSTM (rnn.py)
   │  ├─ Génération séquentielle de strokes [dx, dy, eos]
   │  └─ Conversion strokes → image via drawing.draw()
   │
   ├─ Rendu stylisé (handwriting_renderer.py)
   │  └─ Génération directe image depuis texte avec polices
   │
   └─ Interface utilisateur (streamlit_app.py)
      └─ Interface web interactive pour génération

4. ÉVALUATION
   │
   ├─ prepare_evaluation_data.py
   │  └─ Génère paires (réel, généré) pour métriques
   │
   ├─ metrics.py
   │  ├─ FID, KID (qualité visuelle)
   │  ├─ CER, WER (reconnaissance de texte)
   │  ├─ SSIM, PSNR, LPIPS (similarité)
   │  └─ OCR Accuracy
   │
   └─ evaluate_metrics.py
      └─ Script d'évaluation complète
```

---

## 📊 Pipeline Détaillé Étape par Étape

### **ÉTAPE 1 : Préparation des Données (`prepare_data.py`)**

#### 1.1 Vérification du Dataset IAM
```python
check_dataset_exists()
```
- Vérifie la présence des répertoires :
  - `data/ascii/` : Transcriptions textuelles
  - `data/lineStrokes/` : Fichiers de traits (strokes)
  - `data/original-xml/` : Métadonnées XML

#### 1.2 Collecte des Données
```python
collect_data()
```
**Processus :**
1. Parcourt récursivement `data/ascii/` pour trouver tous les fichiers `.txt`
2. Pour chaque fichier ASCII :
   - Extrait le texte (transcription)
   - Trouve le fichier XML correspondant dans `original-xml/`
   - Récupère l'ID de l'écrivain (`writerID`)
   - Trouve les fichiers de strokes correspondants dans `lineStrokes/`
3. Filtre les échantillons blacklistés (qualité faible)
4. Retourne : `(stroke_fnames, transcriptions, writer_ids)`

#### 1.3 Traitement des Strokes
```python
get_stroke_sequence(filename)
```
**Transformation :**
```
XML (coordonnées absolues)
  ↓
Coordonnées (x, y, eos)
  ↓ drawing.align()      → Correction de l'inclinaison
  ↓ drawing.denoise()     → Lissage Savitzky-Golay
  ↓ drawing.coords_to_offsets() → Conversion en déplacements
  ↓ drawing.normalize()   → Normalisation
  ↓
Offsets normalisés [dx, dy, eos] (MAX_STROKE_LEN=1200)
```

**Format des offsets :**
- `dx, dy` : Déplacements relatifs (normalisés)
- `eos` : End-of-stroke (1 = fin de trait, 0 = continuation)

#### 1.4 Traitement des Transcriptions
```python
get_ascii_sequences(filename)
```
**Processus :**
1. Lit le fichier ASCII
2. Extrait les lignes après `CSR:`
3. Encode chaque caractère en index dans `drawing.alphabet`
4. Tronque à `MAX_CHAR_LEN=75` caractères

#### 1.5 Sauvegarde
```python
# Tableaux NumPy créés
x = np.zeros([N, MAX_STROKE_LEN, 3])      # Strokes
x_len = np.zeros([N])                      # Longueurs réelles
c = np.zeros([N, MAX_CHAR_LEN])           # Transcriptions
c_len = np.zeros([N])                      # Longueurs de texte
w_id = np.zeros([N])                       # IDs écrivains

# Filtrage des échantillons valides
valid_mask = ~np.any(np.linalg.norm(x_i[:, :2], axis=1) > 60)

# Sauvegarde
np.save('data/processed/x.npy', x[valid_mask])
np.save('data/processed/x_len.npy', x_len[valid_mask])
np.save('data/processed/c.npy', c[valid_mask])
np.save('data/processed/c_len.npy', c_len[valid_mask])
np.save('data/processed/w_id.npy', w_id[valid_mask])
```

---

### **ÉTAPE 2 : Dataset PyTorch (`GAN/dataset.py`)**

#### 2.1 Chargement des Données
```python
IAMDataset(img_size=128, max_text_len=20)
```
- Charge les fichiers `.npy` depuis `data/processed/`
- Filtre les textes > `max_text_len` caractères

#### 2.2 Rendu Strokes → Image
```python
__getitem__(idx)
```

**Processus de conversion :**

1. **Récupération des strokes**
   ```python
   strokes = x[real_idx][:stroke_len]  # (L, 3) : [dx, dy, eos]
   ```

2. **Conversion offsets → coordonnées**
   ```python
   coords = np.cumsum(strokes[:, :2], axis=0)  # Accumulation des déplacements
   ```

3. **Normalisation et centrage**
   ```python
   # Calcul des min/max
   min_x, min_y = np.min(coords[:, 0]), np.min(coords[:, 1])
   max_x, max_y = np.max(coords[:, 0]), np.max(coords[:, 1])
   
   # Scaling pour tenir dans 128×128 avec padding
   scale = min(target_size / width, target_size / height)
   coords = (coords - [min_x, min_y]) * scale + padding
   ```

4. **Dessin avec PIL**
   ```python
   img = Image.new('L', (128, 128), color=255)  # Fond blanc
   draw = ImageDraw.Draw(img)
   
   # Dessine chaque trait (séparé par eos=1)
   for i in range(len(coords)):
       if coords[i, 2] == 1:  # End of stroke
           points = coords[start_idx:i+1, :2]
           draw.line(points, fill=0, width=2)  # Noir
   ```

5. **Transformation**
   ```python
   transform = transforms.Compose([
       transforms.ToTensor(),           # [0, 255] → [0, 1]
       transforms.Normalize((0.5,), (0.5,))  # [0, 1] → [-1, 1]
   ])
   ```

6. **Traitement du texte**
   ```python
   text = "".join([drawing.alphabet[i] for i in text_codes[:text_len]])
   text_indices = [char_to_idx.get(c, 0) for c in text]
   # Padding/truncation à max_text_len=20
   text_tensor = torch.tensor(text_indices, dtype=torch.long)
   ```

**Sortie :** `(img_tensor, text_tensor)`
- `img_tensor` : `(1, 128, 128)` dans `[-1, 1]`
- `text_tensor` : `(20,)` indices de caractères

---

### **ÉTAPE 3 : Architecture RNN/LSTM ⭐ PRINCIPALE (`rnn.py`, `rnn_cell.py`)**

#### 3.1 LSTMAttentionCell (`rnn_cell.py`)

**Rôle :** Cellule LSTM avec mécanisme d'attention pour conditionner la génération sur le texte

**Architecture :**

```
Input:
  - inputs: (B, 3)          # [dx, dy, eos] - stroke actuel
  - state: (lstm_state, prev_attention, prev_kappa)
  - attention_values: (B, seq_len, alphabet_size)  # One-hot encoding du texte

1. Projection d'Attention
   mean_attn = mean(prev_attention)  # [B]
   max_attn = max(prev_attention)    # [B]
   summary = [mean_attn, max_attn]  # [B, 2]
   attn_proj = tanh(Linear(2→3)(summary))  # [B, 3]

2. Concaténation Input
   lstm_input = concat([inputs[3], attn_proj[3]])  # [B, 6]

3. LSTM Cell
   new_h, new_c = LSTMCell(6 → lstm_size)(lstm_input, (h, c))

4. Calcul Attention (Mixture of Gaussians)
   attention_params = Linear(lstm_size → 3*K)(new_h)
   attention = compute_attention(attention_params, attention_values)
   new_kappa = update_kappa(prev_kappa, attention_params)

Output:
  - new_h: (B, lstm_size)  # Hidden state
  - new_state: (lstm_state, attention, new_kappa)
```

**Caractéristiques :**
- **Attention** : Permet au modèle de "regarder" différentes parties du texte pendant la génération
- **Mixture of Gaussians** : Modélise la distribution d'attention de manière probabiliste
- **État persistant** : Maintient kappa pour suivre la position dans le texte

#### 3.2 RNN Model (`rnn.py`)

**Rôle :** Modèle RNN complet avec GMM pour générer des strokes

**Architecture :**

```
Input:
  - x: (B, T, 3)           # Strokes d'entrée [dx, dy, eos]
  - c: (B, L)              # Transcriptions (indices caractères)
  - c_len: (B,)            # Longueurs textes
  - x_len: (B,)            # Longueurs strokes
  - y: (B, T, 3)           # Strokes cibles (pour entraînement)

1. Encodage Texte
   attention_values = one_hot(c, alphabet_size)  # [B, L, vocab_size]

2. Initialisation État
   state = cell.zero_state(B)

3. Free Run RNN
   Pour t = 0 à T-1:
     input_t = x[:, t, :]  # [B, 3]
     h_t, state = cell(input_t, state, attention_values, c_len)
     outputs.append(h_t)

4. GMM Layer
   params = Linear(lstm_size → K*6 + 1)(outputs)
   # K = output_mixture_components (typiquement 20)
   # 6 = 2 (mus) + 2 (sigmas) + 1 (rho) + 1 (pi)
   # +1 = end-of-stroke probability

5. Parse Parameters
   pis = softmax(params[:, :K])           # Mixing coefficients
   mus = params[:, K:3*K]                 # Means [mu_x, mu_y]
   sigmas = exp(clamp(params[:, 3*K:5*K])) # Std devs
   rhos = tanh(params[:, 5*K:6*K])        # Correlations
   es = sigmoid(params[:, 6*K])           # End-of-stroke prob

6. Loss (Negative Log Likelihood)
   nll = -log(GMM_likelihood + Bernoulli_likelihood)
   GMM_likelihood = Σ(pi * Gaussian(x, y | mu, sigma, rho))
   Bernoulli_likelihood = eos^eos * (1-eos)^(1-eos)

Output:
  - sequence_loss: Scalar  # Loss moyenne par séquence
  - element_loss: Scalar   # Loss moyenne par élément
```

**Hyperparamètres typiques :**
- `lstm_size`: 400
- `output_mixture_components`: 20
- `attention_mixture_components`: 10

---

### **ÉTAPE 4 : Entraînement RNN/LSTM (`rnn.py`)**

#### 4.1 Initialisation
```python
# DataReader charge les données
dr = DataReader(data_dir='data/processed/')
# Crée train/test split automatiquement

# Modèle RNN
model = RNN(
    lstm_size=400,
    output_mixture_components=20,
    attention_mixture_components=10
)

# Optimizer (à configurer)
optimizer = Adam(model.parameters(), lr=0.001)
```

#### 4.2 Boucle d'Entraînement

**Pour chaque batch :**

1. **Forward Pass**
   ```python
   # Batch depuis DataReader
   batch = next(dr.train_df.batch_generator(batch_size=32))
   
   # Forward
   loss = model.forward(
       x=batch['x'],        # Strokes d'entrée
       c=batch['c'],        # Transcriptions
       c_len=batch['c_len'], # Longueurs textes
       x_len=batch['x_len'], # Longueurs strokes
       y=batch['y']         # Strokes cibles
   )
   ```

2. **Backward Pass**
   ```python
   # Backpropagation
   loss.backward()
   
   # Gradient clipping (recommandé)
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   
   # Mise à jour
   optimizer.step()
   optimizer.zero_grad()
   ```

3. **Sauvegarde**
   - Périodiquement : sauvegarder le modèle
   - Validation : évaluer sur validation set

---

### **ÉTAPE 5 : Génération (Inference)**

#### 5.1 Avec RNN/LSTM (`rnn.py`)
```python
# 1. Charger le modèle entraîné
model.load_state_dict(torch.load('rnn_model.pth'))
model.eval()

# 2. Préparer le texte
text = "Hello World"
text_indices = [char_to_idx.get(c, 0) for c in text]
text_tensor = torch.tensor([text_indices], dtype=torch.long)

# 3. Génération séquentielle
state = model.cell.zero_state(1)
strokes = []

for t in range(max_length):
    # Calculer attention
    attention_values = one_hot(text_tensor, vocab_size)
    
    # LSTM step
    if t == 0:
        input_t = torch.zeros(1, 3)  # Point initial
    else:
        input_t = torch.tensor([[dx, dy, eos]], dtype=torch.float32)
    
    h_t, state = model.cell(input_t, state, attention_values, text_len)
    
    # GMM layer
    params = model.gmm_layer(h_t)
    pis, mus, sigmas, rhos, es = model.parse_parameters(params)
    
    # Échantillonner stroke
    dx, dy, eos = sample_from_gmm(pis, mus, sigmas, rhos, es)
    strokes.append([dx, dy, eos])
    
    if eos > 0.5:  # End of stroke
        break

# 4. Conversion strokes → image
drawing.draw(strokes, save_file='output.png')
```

#### 5.2 Avec le Rendu Stylisé (`handwriting_renderer.py`)
```python
renderer = HandwritingRenderer(RenderConfig())

image = renderer.render(
    text="Hello World",
    font_name="Segoe Script",
    font_size=64,
    ink_color=(32, 32, 32),
    paper_style="plain",
    jitter_px=1.4,      # Tremblement
    tilt_degrees=-3.0,   # Inclinaison
    noise_strength=0.08, # Texture papier
    line_spacing=1.35
)
```

**Processus de rendu :**
1. Création d'une image blanche
2. Dessin du texte avec la police sélectionnée
3. Application du jitter (déplacement aléatoire des caractères)
4. Application de l'inclinaison (transformation affine)
5. Ajout d'ombre (Gaussian blur)
6. Ajout de bruit (texture papier)

---

### **ÉTAPE 6 : Évaluation (`metrics.py`, `evaluate_metrics.py`)**

#### 6.1 Préparation des Données d'Évaluation
```python
prepare_evaluation_data(num_samples=50)
```

**Génère deux ensembles :**
- `evaluation/real/` : Images rendues depuis les strokes réels
- `evaluation/gen/` : Images générées (GAN ou rendu stylisé)

#### 6.2 Métriques Calculées

**1. FID (Fréchet Inception Distance)**
- Mesure la distance entre distributions d'images réelles et générées
- Utilise Inception v3 pour extraire des features
- Plus bas = meilleur (typiquement < 50)

**2. KID (Kernel Inception Distance)**
- Version non-biaisée du FID
- Utilise un kernel polynomial
- Plus bas = meilleur

**3. CER (Character Error Rate)**
- Taux d'erreur au niveau des caractères
- Utilise la distance de Levenshtein
- 0.0 = parfait, 1.0 = toutes erreurs

**4. WER (Word Error Rate)**
- Taux d'erreur au niveau des mots
- 0.0 = parfait, 1.0 = toutes erreurs

**5. SSIM (Structural Similarity Index)**
- Similarité structurelle entre images
- 1.0 = identique, 0.0 = complètement différent

**6. PSNR (Peak Signal-to-Noise Ratio)**
- Ratio signal/bruit
- Plus haut = meilleur (typiquement 20-50 dB)

**7. LPIPS (Learned Perceptual Image Patch Similarity)**
- Similarité perceptuelle apprise
- Plus bas = meilleur (0.0 = identique)

**8. OCR Accuracy**
- Pourcentage de caractères correctement reconnus par OCR
- 1.0 = 100% correct

---

## 🔄 Flux de Données Complet

```
┌─────────────────────────────────────────────────────────────┐
│                    FLUX DE DONNÉES                          │
└─────────────────────────────────────────────────────────────┘

1. DONNÉES BRUTES (IAM Dataset)
   │
   ├─ XML Files (strokes)
   │  └─ Coordonnées absolues (x, y, eos)
   │
   ├─ ASCII Files (transcriptions)
   │  └─ Texte brut
   │
   └─ Metadata (writer IDs)
      └─ Identifiants écrivains

2. PRÉTRAITEMENT (prepare_data.py)
   │
   ├─ Strokes
   │  └─ XML → Offsets normalisés [dx, dy, eos]
   │
   ├─ Textes
   │  └─ ASCII → Indices dans alphabet
   │
   └─ Sauvegarde
      └─ NumPy arrays (.npy)

3. DATASET PYTORCH (GAN/dataset.py)
   │
   ├─ Chargement .npy
   │
   ├─ Conversion strokes → images 128×128
   │  └─ PIL ImageDraw
   │
   └─ Transformation
      └─ Tensor + Normalisation [-1, 1]

4. ENTRAÎNEMENT (GAN/train.py)
   │
   ├─ Batch: (images, text_indices)
   │
   ├─ Generator
   │  └─ [noise + text] → image générée
   │
   ├─ Discriminator
   │  └─ [image + text] → score réel/faux
   │
   └─ Loss & Backprop
      └─ Mise à jour des poids

5. INFÉRENCE
   │
   ├─ GAN (GAN/app.py)
   │  └─ Texte → Image via modèle entraîné
   │
   └─ Rendu stylisé (handwriting_renderer.py)
      └─ Texte → Image via polices + effets

6. ÉVALUATION
   │
   ├─ Génération de paires (réel, généré)
   │
   ├─ Calcul métriques
   │  ├─ FID, KID (qualité visuelle)
   │  ├─ CER, WER (reconnaissance)
   │  └─ SSIM, PSNR, LPIPS (similarité)
   │
   └─ Rapport JSON
```

---

## 📁 Structure des Fichiers

```
GEN - Copie/
│
├── data/
│   ├── raw/                    # Dataset IAM brut
│   │   ├── ascii/              # Transcriptions ASCII
│   │   ├── lineStrokes/        # Fichiers strokes XML
│   │   └── original-xml/       # Métadonnées XML
│   │
│   ├── processed/              # Données préprocessées ⭐
│   │   ├── x.npy               # Strokes normalisés (N, 1200, 3)
│   │   ├── x_len.npy           # Longueurs réelles strokes
│   │   ├── c.npy               # Transcriptions encodées (N, 75)
│   │   ├── c_len.npy           # Longueurs textes
│   │   └── w_id.npy            # IDs écrivains
│   │
│   ├── dataset.py              # Dataset personnalisé (optionnel)
│   └── all_datasets.pickle     # Cache datasets
│
├── words/                      # Images PNG (source alternative)
│   └── [a01, a02, ..., r06]/  # Dossiers par écrivain
│
├── evaluation/                 # Données d'évaluation
│   ├── real/                   # Images réelles (150+ fichiers)
│   └── gen/                    # Images générées (100+ fichiers)
│
├── debug_render/               # Images de debug
│   └── sample_*.png
│
├── logs/                       # Fichiers de logs
│
├── PRÉPARATION DES DONNÉES ⭐
│   ├── prepare_data.py         # Script principal préparation
│   ├── check_data.py           # Vérification fichiers .npy
│   ├── check_data_rendering.py # Visualisation strokes
│   ├── diag_collect_stats.py   # Statistiques dataset
│   └── diag_prepare.py         # Diagnostic préparation
│
├── MODÈLES RNN/LSTM ⭐ PRINCIPAL
│   ├── rnn.py                  # Modèle RNN principal
│   ├── rnn_cell.py             # Cellule LSTM avec attention
│   ├── rnn_ops.py              # Opérations RNN optimisées
│   ├── tf_base_model.py        # Modèle TensorFlow (alternative)
│   └── tf_utils.py             # Utilitaires TensorFlow
│
├── RENDU ET VISUALISATION
│   ├── drawing.py              # Utilitaires conversion strokes ⭐
│   └── handwriting_renderer.py # Rendu stylisé avec polices
│
├── ÉVALUATION ET MÉTRIQUES
│   ├── metrics.py              # Implémentation métriques ⭐
│   ├── calculate_metrics.py   # Script interactif guidé
│   ├── quick_metrics.py        # Script rapide
│   ├── evaluate_metrics.py     # Script avancé CLI
│   ├── streamlit_metrics.py    # Interface Streamlit métriques
│   └── prepare_evaluation_data.py # Préparation données évaluation
│
├── INTERFACES UTILISATEUR
│   ├── streamlit_app.py        # Interface principale (rendu stylisé) ⭐
│   └── streamlit_metrics.py    # Interface métriques
│
├── UTILITAIRES
│   ├── data_frame.py           # Gestion données (DataFrame)
│   └── requirements.txt        # Dépendances Python
│
└── DOCUMENTATION
    ├── ARCHITECTURE_GUIDE.md   # Guide architecture (ce fichier)
    ├── PIPELINE_DIAGRAM.md     # Diagrammes visuels
    ├── METRICS_GUIDE.md        # Guide métriques
    ├── FICHIERS_ET_INTERACTIONS.md # Guide fichiers et interactions
    └── TRACABILITE_PROJET.md   # Traçabilité complète du projet
```

---

## 🎯 Points Clés de l'Architecture

### 1. **Représentation des Strokes** ⭐
- Format : Offsets normalisés `[dx, dy, eos]`
- Avantages :
  - Invariant à la translation
  - Normalisé pour stabilité
  - Compact (1200 points max)
- Utilisé par : `prepare_data.py`, `rnn.py`, `drawing.py`

### 2. **Modèle RNN/LSTM avec Attention** ⭐ PRINCIPAL
- **Architecture séquentielle** : Génération point par point
- **Mécanisme d'attention** : Le modèle "regarde" différentes parties du texte
- **GMM (Gaussian Mixture Model)** : Modélise la distribution des strokes de manière probabiliste
- **Conditionnement texte** : Le texte guide la génération via attention
- **Avantages** :
  - Génération naturelle et séquentielle
  - Modèle temporel qui capture la dynamique de l'écriture
  - Strokes réalistes grâce au GMM

### 3. **Rendu On-the-Fly**
- Les strokes sont convertis en images à la volée via `drawing.draw()`
- Évite de stocker des milliers d'images
- Permet des transformations dynamiques
- Utilisé dans : `check_data_rendering.py`, `prepare_evaluation_data.py`

### 4. **Trois Approches Complémentaires**
- **RNN/LSTM** ⭐ : Apprentissage profond, génération séquentielle, strokes naturels
- **Rendu stylisé** : Contrôle précis, rapide, pas d'entraînement, interface interactive
- **TensorFlow (Alternative)** : Compatibilité avec écosystème TensorFlow

### 5. **Évaluation Multi-Métriques**
- **Qualité visuelle** : FID, KID (via Inception v3)
- **Reconnaissance texte** : CER, WER, OCR Accuracy (via Tesseract)
- **Similarité** : SSIM, PSNR, LPIPS (similarité perceptuelle)
- Toutes implémentées dans `metrics.py` et accessibles via 4 interfaces différentes

### 6. **Hub Central : drawing.py**
- Fichier le plus utilisé dans le projet (5+ fichiers)
- Définit les constantes globales : `alphabet`, `MAX_STROKE_LEN`, `MAX_CHAR_LEN`
- Fournit toutes les fonctions de transformation strokes
- Point unique de conversion strokes ↔ images

### 7. **Pipeline Modulaire**
- Chaque étape est indépendante et réutilisable
- Préparation → Entraînement → Génération → Évaluation
- Scripts de diagnostic pour chaque étape

---

## 🚀 Workflow Typique du Projet

### **1. Préparation Initiale :**
```bash
# 1. Vérifier le dataset IAM
python prepare_data.py
# → Vérifie existence et structure du dataset

# 2. Préparer les données
python prepare_data.py
# → Génère data/processed/*.npy

# 3. Vérifier les données
python check_data.py
# → Vérifie que les fichiers .npy sont corrects

# 4. Visualiser des échantillons
python check_data_rendering.py
# → Génère debug_render/sample_*.png
```

### **2. Entraînement RNN/LSTM :**
```bash
# Entraîner le modèle RNN
python rnn.py
# → Entraîne le modèle avec les données préparées
# → Sauvegarde le modèle périodiquement
```

### **3. Génération :**
```bash
# Option A: Interface Streamlit (rendu stylisé)
streamlit run streamlit_app.py
# → Interface web interactive
# → Génération instantanée avec polices

# Option B: Utilisation programmatique
# → Utiliser handwriting_renderer.py directement
# → Ou utiliser rnn.py pour génération avec modèle entraîné
```

### **4. Évaluation :**
```bash
# 1. Préparer les données d'évaluation
python prepare_evaluation_data.py
# → Génère evaluation/real/ et evaluation/gen/

# 2. Calculer les métriques (4 options)

# Option A: Script interactif
python calculate_metrics.py

# Option B: Interface Streamlit
streamlit run streamlit_metrics.py

# Option C: Script rapide
python quick_metrics.py

# Option D: Script avancé
python evaluate_metrics.py \
    --real_dir evaluation/real \
    --gen_dir evaluation/gen \
    --output metrics_results.json
```

### **5. Diagnostic (si nécessaire) :**
```bash
# Collecter des statistiques
python diag_collect_stats.py

# Diagnostic préparation
python diag_prepare.py
```

---

## 📈 Hyperparamètres Principaux

### **RNN/LSTM ⭐**
| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `lstm_size` | 400 | Taille de la couche cachée LSTM |
| `output_mixture_components` | 20 | Nombre de composantes GMM pour sortie |
| `attention_mixture_components` | 10 | Nombre de composantes pour attention |
| `MAX_STROKE_LEN` | 1200 | Longueur max des séquences de strokes |
| `MAX_CHAR_LEN` | 75 | Longueur max des transcriptions |
| `vocab_size` | ~70 | Taille de l'alphabet (drawing.alphabet) |
| `learning_rate` | 0.001 | Taux d'apprentissage (à configurer) |
| `batch_size` | 32 | Taille des batches (à configurer) |

### **Rendu Stylisé**
| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `image_width` | 1400 | Largeur image générée |
| `image_height` | 900 | Hauteur image générée |
| `font_size` | 64 | Taille police par défaut |
| `jitter_px` | 1.4 | Amplitude tremblement |
| `tilt_degrees` | -3.0 | Inclinaison par défaut |
| `noise_strength` | 0.08 | Intensité texture papier |
| `line_spacing` | 1.35 | Interligne |

### **Données**
| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `image_size` | 128×128 | Taille images pour visualisation |
| `max_text_len` | 20 | Longueur max texte (pour certains modèles) |

---

## 🔍 Détails Techniques

### **Normalisation des Strokes**
- Les offsets sont normalisés par la médiane de leur norme
- Évite les problèmes d'échelle
- Rend l'entraînement plus stable

### **Padding et Truncation**
- Strokes : Padding avec `[0, 0, 0]` jusqu'à `MAX_STROKE_LEN`
- Textes : Padding avec `0` (caractère nul) jusqu'à `max_text_len`
- Les longueurs réelles sont stockées séparément

### **Loss Function (LSGAN)**
- Utilise MSE au lieu de BCE
- Plus stable pour l'entraînement
- Labels : `1` pour réel, `0` pour faux

### **Data Augmentation**
- Pas d'augmentation explicite dans le code actuel
- Possibilité d'ajouter : rotation, scaling, noise

---

---

## 🎯 Caractéristiques Spécifiques de ce Projet

### **Points Forts de l'Implémentation**

1. **Modèle RNN/LSTM Robuste**
   - Architecture avec attention pour conditionnement texte
   - GMM pour modélisation probabiliste des strokes
   - Génération séquentielle naturelle

2. **Pipeline Complet et Modulaire**
   - Préparation → Entraînement → Génération → Évaluation
   - Chaque étape est indépendante et testable
   - Scripts de diagnostic intégrés

3. **Double Approche de Génération**
   - RNN/LSTM pour qualité et réalisme
   - Rendu stylisé pour rapidité et contrôle

4. **Évaluation Complète**
   - 8 métriques différentes
   - 4 interfaces d'accès (interactif, CLI, rapide, Streamlit)
   - Support OCR pour évaluation texte

5. **Documentation Complète**
   - 4 guides détaillés
   - Diagrammes visuels
   - Explications étape par étape

### **Fichiers Clés par Fonctionnalité**

| Fonctionnalité | Fichiers Principaux |
|----------------|---------------------|
| **Préparation données** | `prepare_data.py`, `drawing.py` |
| **Modèle RNN** | `rnn.py`, `rnn_cell.py`, `rnn_ops.py` |
| **Gestion données** | `data_frame.py` |
| **Rendu** | `drawing.py`, `handwriting_renderer.py` |
| **Métriques** | `metrics.py` |
| **Interface** | `streamlit_app.py`, `streamlit_metrics.py` |
| **Diagnostic** | `check_data.py`, `check_data_rendering.py`, `diag_*.py` |

### **Flux de Travail Recommandé**

```
1. PREPARATION
   prepare_data.py → data/processed/*.npy
   check_data.py → Vérification
   check_data_rendering.py → Visualisation

2. ENTRAÎNEMENT
   rnn.py → Modèle entraîné

3. GÉNÉRATION
   streamlit_app.py → Interface interactive
   OU
   handwriting_renderer.py → Programmatique

4. ÉVALUATION
   prepare_evaluation_data.py → Données évaluation
   calculate_metrics.py → Métriques
   streamlit_metrics.py → Interface métriques
```

---

## 🎓 Conclusion

Ce projet implémente un **pipeline complet et modulaire** de génération d'écriture manuscrite, de la préparation des données à l'évaluation. L'approche principale utilise un **modèle RNN/LSTM avec attention** pour générer des strokes de manière séquentielle, complétée par un **rendu stylisé rapide** pour les applications interactives.

**Points distinctifs :**
- ✅ Modèle RNN/LSTM avec attention implémenté et fonctionnel
- ✅ Pipeline complet avec scripts de diagnostic
- ✅ Double approche (RNN + Rendu stylisé)
- ✅ Évaluation multi-métriques (8 métriques, 4 interfaces)
- ✅ Documentation complète et détaillée

**Architecture modulaire** permettant d'ajouter facilement de nouvelles fonctionnalités ou d'améliorer les composants existants.

---

## 📚 Documentation Complémentaire

Pour plus de détails, consultez :

- **`PIPELINE_DIAGRAM.md`** : Diagrammes visuels du pipeline complet
- **`FICHIERS_ET_INTERACTIONS.md`** : Guide détaillé de tous les fichiers et leurs interactions
- **`TRACABILITE_PROJET.md`** : Traçabilité complète et personnalisée du projet
- **`METRICS_GUIDE.md`** : Guide d'utilisation des métriques d'évaluation

