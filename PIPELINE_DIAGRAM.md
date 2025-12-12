# 🔄 Diagramme Visuel du Pipeline

## Vue d'Ensemble Simplifiée

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GÉNÉRATION D'ÉCRITURE MANUSCRITE               │
└─────────────────────────────────────────────────────────────────────┘

INPUT: Texte ("Hello World")
         │
         ├─────────────────────────────────┬──────────────────────────┐
         │                                   │                          │
         ▼                                   ▼                          ▼
    ┌─────────┐                        ┌─────────┐              ┌─────────┐
    │   RNN   │                        │ RENDU  │              │   GAN   │
    │ / LSTM  │ ⭐ PRINCIPAL           │STYLISÉ  │              │ (cGAN)  │
    │         │                        │        │              │ FUTUR   │
    └─────────┘                        └─────────┘              └─────────┘
         │                                   │                          │
         │                                   │                          │
         ▼                                   ▼                          ▼
    Strokes générés                    Image stylisée            Image 128×128
    (séquences [dx,dy,eos])            (polices + effets)        (non implémenté)
         │                                   │                          │
         │                                   │                          │
         ▼                                   │                          │
    drawing.draw()                          │                          │
    → Image 128×128                         │                          │
         │                                   │                          │
         └───────────────────────────────────┴──────────────────────────┘
                         │
                         ▼
                    OUTPUT: Image d'écriture manuscrite
```

---

## Pipeline Complet de Préparation des Données

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRÉPARATION DES DONNÉES                     │
└─────────────────────────────────────────────────────────────────┘

Dataset IAM (Brut)
│
├─ data/raw/ascii/*.txt
│  └─ Transcriptions: "Hello World"
│
├─ data/raw/lineStrokes/*.xml
│  └─ Strokes XML: <Stroke><Point x="100" y="200"/></Stroke>
│
└─ data/raw/original-xml/*.xml
   └─ Métadonnées: writerID, etc.

         │
         │ prepare_data.py
         ▼

┌─────────────────────────────────────────────────────────────┐
│  TRAITEMENT DES STROKES                                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  XML → Coordonnées (x, y, eos)                              │
│    │                                                         │
│    ├─ align()      → Correction inclinaison                 │
│    ├─ denoise()     → Lissage Savitzky-Golay                │
│    ├─ coords_to_offsets() → Conversion en déplacements      │
│    └─ normalize()  → Normalisation                         │
│                                                              │
│  Résultat: [dx, dy, eos] normalisés                          │
└─────────────────────────────────────────────────────────────┘
         │
         │
┌─────────────────────────────────────────────────────────────┐
│  TRAITEMENT DES TRANSCRIPTIONS                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ASCII → Texte brut                                          │
│    │                                                         │
│    └─ encode_ascii() → Indices dans alphabet                │
│                                                              │
│  Résultat: [char_idx1, char_idx2, ...]                      │
└─────────────────────────────────────────────────────────────┘
         │
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  SAUVEGARDE (data/processed/)                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  x.npy      → Strokes: (N, 1200, 3)                         │
│  x_len.npy  → Longueurs: (N,)                               │
│  c.npy      → Transcriptions: (N, 75)                        │
│  c_len.npy  → Longueurs textes: (N,)                        │
│  w_id.npy   → IDs écrivains: (N,)                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Architecture RNN/LSTM ⭐ PRINCIPALE

```
┌─────────────────────────────────────────────────────────────────┐
│                    MODÈLE RNN AVEC ATTENTION                    │
└─────────────────────────────────────────────────────────────────┘

Input:
  ┌─────────┐         ┌──────────┐
  │ Strokes │         │   Text   │
  │ [dx,dy, │         │ (chars)  │
  │  eos]   │         │          │
  └─────────┘         └──────────┘
       │                   │
       │                   ▼
       │            ┌──────────────┐
       │            │  One-Hot    │
       │            │  Encoding   │
       │            │  (alphabet) │
       │            └──────────────┘
       │                   │
       └───────┬───────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│  LSTMAttentionCell (rnn_cell.py)                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Projection d'Attention                                  │
│     prev_attention → [mean, max] → tanh(Linear) → [3-D]     │
│                                                              │
│  2. Concaténation Input                                     │
│     [dx, dy, eos] + attn_proj → [6-D]                      │
│                                                              │
│  3. LSTM Cell                                               │
│     LSTMCell(6 → lstm_size)                                 │
│     → (h, c)                                                │
│                                                              │
│  4. Calcul Attention                                         │
│     Mixture de Gaussians sur texte                          │
│     → attention weights                                     │
│                                                              │
│  Output: h (hidden state)                                    │
└─────────────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│  RNN Model (rnn.py)                                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Free Run RNN                                             │
│     Pour chaque timestep t:                                  │
│       input_t = strokes[t]                                   │
│       h_t, state = cell(input_t, state, attention)           │
│                                                              │
│  2. GMM Layer                                                │
│     Linear(lstm_size → output_units)                         │
│     output_units = K*6 + 1                                  │
│     (K = output_mixture_components)                          │
│                                                              │
│  3. Parse Parameters                                         │
│     → pis (mixing coeffs)                                   │
│     → mus (means: [mu_x, mu_y])                             │
│     → sigmas (std devs: [sigma_x, sigma_y])                 │
│     → rhos (correlations)                                    │
│     → es (end-of-stroke prob)                               │
│                                                              │
│  4. Loss (Negative Log Likelihood)                           │
│     GMM likelihood + Bernoulli likelihood                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
               │
               ▼
        ┌──────────────┐
        │   Output     │
        │ Strokes:     │
        │ [dx, dy, eos]│
        └──────────────┘
               │
               ▼
        ┌──────────────┐
        │ drawing.draw()│
        │ Strokes →    │
        │ Image 128×128 │
        └──────────────┘


┌─────────────────────────────────────────────────────────────────┐
│              DÉTAILS DE L'ARCHITECTURE                         │
└─────────────────────────────────────────────────────────────────┘

COMPOSANTS PRINCIPAUX:

1. LSTMAttentionCell (rnn_cell.py)
   ├─ LSTM Cell: 6 inputs → lstm_size hidden
   ├─ Attention Projection: [2] → [3]
   ├─ Attention Parameters: lstm_size → 3*K
   └─ Attention Computation: Mixture of Gaussians

2. RNN Model (rnn.py)
   ├─ Free Run: Génération séquentielle
   ├─ GMM Layer: lstm_size → (K*6 + 1)
   ├─ Parameter Parsing: Extraction params GMM
   └─ Loss Function: NLL (GMM + Bernoulli)

3. Opérations RNN (rnn_ops.py)
   ├─ raw_rnn: Boucle RNN générique
   ├─ rnn_teacher_force: Entraînement avec ground truth
   └─ rnn_free_run: Génération autonome

HYPERPARAMÈTRES TYPIQUES:
   ├─ lstm_size: 400
   ├─ output_mixture_components: 20
   ├─ attention_mixture_components: 10
   └─ alphabet_size: ~70 (drawing.alphabet)
```

---

## Processus d'Entraînement RNN/LSTM

```
┌─────────────────────────────────────────────────────────────────┐
│                    BOUCLE D'ENTRAÎNEMENT RNN                   │
└─────────────────────────────────────────────────────────────────┘

Pour chaque EPOCH:
    │
    └─ Pour chaque BATCH:
           │
           ├─ DataReader (rnn.py)
           │  │
           │  ├─ Charger data/processed/*.npy
           │  ├─ Créer batches avec padding
           │  └─ Préparer (x, y, c, x_len, c_len)
           │
           ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  FORWARD PASS                                               │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │  1. Encoder texte                                            │
    │     c → one_hot(c) → attention_values                       │
    │                                                              │
    │  2. Initialiser état                                        │
    │     state = cell.zero_state(batch_size)                     │
    │                                                              │
    │  3. Free Run RNN                                            │
    │     Pour t = 0 à T-1:                                       │
    │       input_t = x[:, t, :]  # [dx, dy, eos]                │
    │       h_t, state = cell(input_t, state, attention)          │
    │                                                              │
    │  4. GMM Layer                                               │
    │     params = gmm_layer(outputs)                              │
    │     → pis, mus, sigmas, rhos, es                           │
    │                                                              │
    │  5. Calculer Loss                                           │
    │     nll = negative_log_likelihood(y, params)                │
    │     → sequence_loss, element_loss                           │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  BACKWARD PASS                                               │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │  1. Backpropagation                                          │
    │     loss.backward()                                          │
    │                                                              │
    │  2. Mise à jour poids                                        │
    │     optimizer.step()                                         │
    │                                                              │
    │  3. Gradient clipping (optionnel)                            │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘
           │
           ▼
    Sauvegarde checkpoints
    (périodiquement)
```

---

## Conversion Strokes → Image (dans Dataset)

```
┌─────────────────────────────────────────────────────────────────┐
│          CONVERSION STROKES → IMAGE (IAMDataset)               │
└─────────────────────────────────────────────────────────────────┘

Input: Strokes (offsets)
  [dx₁, dy₁, 0]  →  Point 1
  [dx₂, dy₂, 0]  →  Point 2
  [dx₃, dy₃, 1]  →  Point 3 (fin de trait)
  [dx₄, dy₄, 0]  →  Point 4
  ...

         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  ÉTAPE 1: Conversion Offsets → Coordonnées                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  coords = cumsum(offsets[:, :2])                            │
│                                                              │
│  Résultat:                                                   │
│    [x₁, y₁, 0]                                               │
│    [x₂, y₂, 0]                                               │
│    [x₃, y₃, 1]  ← Fin de trait                             │
│    [x₄, y₄, 0]                                               │
│    ...                                                       │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  ÉTAPE 2: Normalisation et Centrage                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Calculer min/max:                                       │
│     min_x, min_y = min(coords)                              │
│     max_x, max_y = max(coords)                              │
│                                                              │
│  2. Calculer scale:                                         │
│     scale = min(target_size/width, target_size/height)      │
│                                                              │
│  3. Centrer et redimensionner:                               │
│     coords = (coords - [min_x, min_y]) * scale + padding   │
│                                                              │
│  Résultat: Coordonnées dans [0, 128]                        │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  ÉTAPE 3: Dessin avec PIL                                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  img = Image.new('L', (128, 128), color=255)  # Fond blanc    │
│  draw = ImageDraw.Draw(img)                                 │
│                                                              │
│  Pour chaque trait (séparé par eos=1):                       │
│    points = coords[start:end, :2]                           │
│    draw.line(points, fill=0, width=2)  # Noir              │
│                                                              │
│  Résultat: Image PIL 128×128 (niveaux de gris)              │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  ÉTAPE 4: Transformation PyTorch                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  transform = Compose([                                      │
│      ToTensor(),        # [0,255] → [0,1]                   │
│      Normalize(0.5, 0.5)  # [0,1] → [-1,1]                 │
│  ])                                                          │
│                                                              │
│  Résultat: Tensor (1, 128, 128) dans [-1, 1]                │
└─────────────────────────────────────────────────────────────┘
```

---

## Pipeline d'Évaluation (Section détaillée)

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE D'ÉVALUATION                       │
└─────────────────────────────────────────────────────────────────┘

1. PRÉPARATION
   │
   ├─ prepare_evaluation_data.py
   │  │
   │  ├─ Charger données IAM
   │  │
   │  ├─ Pour chaque échantillon:
   │  │  │
   │  │  ├─ Générer image RÉELLE
   │  │  │  └─ drawing.draw(strokes) → evaluation/real/
   │  │  │
   │  │  └─ Générer image GÉNÉRÉE
   │  │     └─ renderer.render(text) → evaluation/gen/
   │  │
   │  └─ Sauvegarder paires d'images
   │
   ▼

2. CALCUL DES MÉTRIQUES (4 méthodes disponibles)
   │
   ├─ Méthode 1: calculate_metrics.py (Interactif) ⭐
   │  │
   │  └─ Guide pas à pas interactif
   │
   ├─ Méthode 2: streamlit_metrics.py (Interface graphique) 🎨
   │  │
   │  └─ Interface Streamlit avec visualisation
   │
   ├─ Méthode 3: quick_metrics.py (Rapide) ⚡
   │  │
   │  └─ Chemins en dur ou variables d'environnement
   │
   └─ Méthode 4: evaluate_metrics.py (Avancé) 🔧
      │
      └─ Options ligne de commande complètes

3. MÉTRIQUES CALCULÉES
   │
   ├─ Métriques visuelles:
   │  ├─ FID (Fréchet Inception Distance)
   │  │  └─ Inception v3 → Features → Distance Fréchet
   │  │
   │  └─ KID (Kernel Inception Distance)
   │     └─ Inception v3 → Features → Kernel polynomial
   │
   ├─ Métriques de texte:
   │  ├─ CER (Character Error Rate)
   │  │  └─ OCR → Texte → Distance Levenshtein
   │  │
   │  ├─ WER (Word Error Rate)
   │  │  └─ OCR → Mots → Distance Levenshtein
   │  │
   │  └─ OCR Accuracy
   │     └─ Pourcentage caractères corrects
   │
   └─ Métriques de similarité:
      ├─ SSIM (Structural Similarity)
      ├─ PSNR (Peak Signal-to-Noise Ratio)
      └─ LPIPS (Learned Perceptual Similarity)

4. RAPPORT
   │
   └─ Sauvegarde JSON
      ├─ metrics_results.json
      └─ metrics_results_full.json
```

---

## Comparaison des Deux Approches

```
┌─────────────────────────────────────────────────────────────────┐
│              GAN vs RENDU STYLISÉ                              │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────────┐    ┌──────────────────────┐
│   GAN (cGAN)         │    │  RENDU STYLISÉ       │
├──────────────────────┤    ├──────────────────────┤
│                      │    │                      │
│ ✓ Style variable     │    │ ✓ Contrôle précis    │
│ ✓ Apprentissage      │    │ ✓ Rapide             │
│ ✓ Réaliste           │    │ ✓ Pas d'entraînement│
│                      │    │ ✓ Personnalisable   │
│ ✗ Nécessite          │    │                      │
│   entraînement       │    │ ✗ Style limité      │
│ ✗ Lent (inference)   │    │ ✗ Moins réaliste     │
│ ✗ Moins contrôlable │    │    (selon police)    │
│                      │    │                      │
│ Utilisation:         │    │ Utilisation:         │
│ - Génération         │    │ - Prototypage       │
│   créative           │    │ - Applications       │
│ - Style unique       │    │   production         │
│                      │    │ - Personnalisation  │
└──────────────────────┘    └──────────────────────┘
```

---

## Format des Données à Chaque Étape

```
┌─────────────────────────────────────────────────────────────────┐
│                    FORMAT DES DONNÉES                           │
└─────────────────────────────────────────────────────────────────┘

1. DONNÉES BRUTES (IAM)
   │
   ├─ XML Strokes:
   │  <Point x="100" y="200"/>
   │  → Coordonnées absolues
   │
   └─ ASCII:
      "Hello World"
      → Texte brut

2. APRÈS PRÉTRAITEMENT
   │
   ├─ x.npy: (N, 1200, 3)
   │  [[dx, dy, eos], ...]
   │  → Offsets normalisés
   │
   └─ c.npy: (N, 75)
      [char_idx1, char_idx2, ...]
      → Indices de caractères

3. DANS LE DATASET
   │
   ├─ Image: (1, 128, 128)
   │  Tensor dans [-1, 1]
   │  → Image PIL convertie
   │
   └─ Text: (20,)
      [idx1, idx2, ..., 0, 0, 0]
      → Indices padding à 20

4. ENTRÉE DU GAN
   │
   ├─ Noise: (B, 100)
   │  → Vecteur aléatoire
   │
   └─ Text: (B, 20)
      → Indices de caractères

5. SORTIE DU GAN
   │
   └─ Image: (B, 1, 128, 128)
      → Tensor dans [-1, 1]
```

---

## Workflow Utilisateur

```
┌─────────────────────────────────────────────────────────────────┐
│                    WORKFLOW UTILISATEUR                         │
└─────────────────────────────────────────────────────────────────┘

UTILISATEUR
    │
    ├─ Option 1: Interface Streamlit (Rendu stylisé)
    │  │
    │  └─ streamlit_app.py
    │     │
    │     ├─ Saisie texte
    │     ├─ Choix police
    │     ├─ Paramètres style
    │     └─ Génération instantanée
    │
    ├─ Option 2: Interface Métriques Streamlit
    │  │
    │  └─ streamlit_metrics.py
    │     │
    │     ├─ Chargement images
    │     ├─ Calcul métriques
    │     └─ Visualisation résultats
    │
    ├─ Option 3: Scripts Python Interactifs
    │  │
    │  ├─ calculate_metrics.py
    │  │  └─ Script interactif guidé
    │  │
    │  ├─ quick_metrics.py
    │  │  └─ Script rapide (chemins en dur)
    │  │
    │  └─ evaluate_metrics.py
    │     └─ Script avancé (ligne de commande)
    │
    └─ Option 4: Scripts de Préparation
       │
       ├─ prepare_data.py
       │  └─ Préparation dataset IAM
       │
       ├─ check_data.py
       │  └─ Vérification données
       │
       ├─ check_data_rendering.py
       │  └─ Vérification rendu strokes
       │
       ├─ diag_collect_stats.py
       │  └─ Statistiques dataset
       │
       └─ diag_prepare.py
          └─ Diagnostic préparation
```

---

## Architecture Complète du Projet

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMPOSANTS DU PROJET                         │
└─────────────────────────────────────────────────────────────────┘

1. PRÉPARATION DES DONNÉES
   │
   ├─ prepare_data.py
   │  └─ Extraction et normalisation dataset IAM
   │
   ├─ check_data.py
   │  └─ Vérification fichiers .npy
   │
   ├─ check_data_rendering.py
   │  └─ Visualisation strokes → images
   │
   ├─ diag_collect_stats.py
   │  └─ Collecte statistiques dataset
   │
   └─ diag_prepare.py
      └─ Diagnostic processus préparation

2. MODÈLES DE GÉNÉRATION
   │
   ├─ GAN (Conditionnel)
   │  ├─ model.py (Generator + Discriminator)
   │  ├─ dataset.py (IAMDataset PyTorch)
   │  ├─ train.py (Entraînement)
   │  └─ app.py (Interface Streamlit)
   │
   ├─ RNN/LSTM
   │  ├─ rnn.py (Modèle RNN principal)
   │  ├─ rnn_cell.py (Cellule LSTM avec attention)
   │  └─ rnn_ops.py (Opérations RNN)
   │
   └─ TensorFlow (Alternative)
      ├─ tf_base_model.py
      └─ tf_utils.py

3. RENDU ET VISUALISATION
   │
   ├─ drawing.py
   │  └─ Utilitaires conversion strokes → images
   │
   └─ handwriting_renderer.py
      └─ Rendu stylisé avec polices

4. ÉVALUATION ET MÉTRIQUES
   │
   ├─ metrics.py
   │  └─ Implémentation toutes les métriques
   │
   ├─ calculate_metrics.py
   │  └─ Script interactif guidé
   │
   ├─ quick_metrics.py
   │  └─ Script rapide
   │
   ├─ evaluate_metrics.py
   │  └─ Script avancé CLI
   │
   ├─ streamlit_metrics.py
   │  └─ Interface Streamlit métriques
   │
   └─ prepare_evaluation_data.py
      └─ Génération paires (réel, généré)

5. INTERFACES UTILISATEUR
   │
   ├─ streamlit_app.py
   │  └─ Interface principale (rendu stylisé)
   │
   └─ streamlit_metrics.py
      └─ Interface métriques

6. UTILITAIRES
   │
   ├─ data_frame.py
   │  └─ Gestion données (analogue pandas)
   │
   └─ METRICS_GUIDE.md
      └─ Documentation métriques
```

---

## Pipeline d'Évaluation Complet

```
┌─────────────────────────────────────────────────────────────────┐
│              PIPELINE D'ÉVALUATION MULTI-MÉTHODES               │
└─────────────────────────────────────────────────────────────────┘

1. PRÉPARATION DES DONNÉES D'ÉVALUATION
   │
   └─ prepare_evaluation_data.py
      │
      ├─ Charger données IAM
      ├─ Générer images RÉELLES (strokes)
      ├─ Générer images GÉNÉRÉES (rendu stylisé)
      └─ Sauvegarder dans evaluation/real/ et evaluation/gen/

2. CALCUL DES MÉTRIQUES (4 méthodes disponibles)
   │
   ├─ Méthode 1: Script Interactif ⭐ RECOMMANDÉ
   │  │
   │  └─ calculate_metrics.py
   │     │
   │     ├─ Guide interactif
   │     ├─ Chargement images
   │     ├─ Calcul automatique
   │     └─ Affichage résultats
   │
   ├─ Méthode 2: Interface Streamlit 🎨
   │  │
   │  └─ streamlit_metrics.py
   │     │
   │     ├─ Interface graphique
   │     ├─ Aperçu images
   │     ├─ Calcul en un clic
   │     └─ Export JSON
   │
   ├─ Méthode 3: Script Rapide ⚡
   │  │
   │  └─ quick_metrics.py
   │     │
   │     ├─ Chemins en dur ou variables d'environnement
   │     ├─ Calcul rapide
   │     └─ Bon pour automatisation
   │
   └─ Méthode 4: Script Avancé 🔧
      │
      └─ evaluate_metrics.py
         │
         ├─ Options ligne de commande
         ├─ Toutes les options disponibles
         └─ Export JSON personnalisé

3. MÉTRIQUES CALCULÉES
   │
   ├─ Qualité Visuelle:
   │  ├─ FID (Fréchet Inception Distance)
   │  └─ KID (Kernel Inception Distance)
   │
   ├─ Reconnaissance Texte:
   │  ├─ CER (Character Error Rate)
   │  ├─ WER (Word Error Rate)
   │  └─ OCR Accuracy
   │
   └─ Similarité:
      ├─ SSIM (Structural Similarity)
      ├─ PSNR (Peak Signal-to-Noise Ratio)
      └─ LPIPS (Learned Perceptual Similarity)

4. RAPPORTS
   │
   └─ Sauvegarde JSON
      ├─ metrics_results.json
      └─ metrics_results_full.json
```

---

## Pipeline RNN/LSTM Complet ⭐ PRINCIPAL

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE RNN/LSTM COMPLET                  │
└─────────────────────────────────────────────────────────────────┘

ÉTAPE 1: PRÉPARATION DES DONNÉES
   │
   ├─ prepare_data.py
   │  └─ Dataset IAM → data/processed/*.npy
   │
   └─ DataReader (rnn.py)
      └─ Chargement et batching

ÉTAPE 2: ENCODAGE
   │
   ├─ Texte → One-Hot
   │  └─ F.one_hot(c, alphabet_size)
   │
   └─ Strokes → Normalisés
      └─ [dx, dy, eos] déjà normalisés

ÉTAPE 3: ENTRAÎNEMENT
   │
   ├─ Initialisation
   │  └─ RNN(lstm_size=400, output_mixture=20, attn_mixture=10)
   │
   ├─ Boucle d'entraînement
   │  │
   │  ├─ Forward Pass
   │  │  ├─ Free Run RNN
   │  │  ├─ GMM Layer
   │  │  └─ Loss (NLL)
   │  │
   │  └─ Backward Pass
   │     ├─ Backpropagation
   │     └─ Optimizer step
   │
   └─ Sauvegarde modèles

ÉTAPE 4: GÉNÉRATION (INFERENCE)
   │
   ├─ Input: Texte
   │
   ├─ Encoder texte
   │  └─ One-hot encoding
   │
   ├─ Initialiser état
   │  └─ state = cell.zero_state()
   │
   ├─ Génération séquentielle
   │  │
   │  ├─ Pour chaque timestep:
   │  │  ├─ Calculer attention
   │  │  ├─ LSTM step
   │  │  ├─ GMM layer
   │  │  ├─ Échantillonner stroke
   │  │  │  └─ [dx, dy, eos]
   │  │  └─ Mettre à jour état
   │  │
   │  └─ Jusqu'à eos=1 ou max_length
   │
   └─ Output: Séquences de strokes

ÉTAPE 5: RENDU IMAGE
   │
   └─ drawing.draw(strokes)
      └─ Conversion strokes → Image 128×128
```

---

## Architecture Détaillée RNN/LSTM

```
┌─────────────────────────────────────────────────────────────────┐
│              COMPOSANTS DU SYSTÈME RNN/LSTM                    │
└─────────────────────────────────────────────────────────────────┘

1. LSTMAttentionCell (rnn_cell.py)
   │
   ├─ Input: [dx, dy, eos] (3-D) + attention context
   │
   ├─ Attention Projection
   │  └─ Linear(2 → 3) avec tanh
   │
   ├─ LSTM Cell
   │  └─ LSTMCell(6 → lstm_size)
   │     Input: [dx, dy, eos, attn_proj[3]]
   │
   ├─ Attention Computation
   │  └─ Mixture of Gaussians sur texte
   │
   └─ Output: hidden state (lstm_size)

2. RNN Model (rnn.py)
   │
   ├─ Free Run
   │  └─ Boucle séquentielle sur strokes
   │
   ├─ GMM Layer
   │  └─ Linear(lstm_size → K*6 + 1)
   │     K = output_mixture_components
   │
   ├─ Parameter Parsing
   │  ├─ pis: Softmax (mixing coefficients)
   │  ├─ mus: Means [mu_x, mu_y]
   │  ├─ sigmas: Exp + clamp (std devs)
   │  ├─ rhos: Tanh (correlations)
   │  └─ es: Sigmoid (end-of-stroke prob)
   │
   └─ Loss Function
      └─ Negative Log Likelihood
         ├─ GMM likelihood (Gaussian mixture)
         └─ Bernoulli likelihood (eos)

3. Opérations RNN (rnn_ops.py)
   │
   ├─ raw_rnn: Boucle RNN générique
   ├─ rnn_teacher_force: Entraînement avec ground truth
   └─ rnn_free_run: Génération autonome

4. Utilitaires TensorFlow (tf_utils.py)
   │
   └─ Compatibilité TensorFlow (si nécessaire)
```

---

## Structure Complète des Fichiers

```
GEN - Copie/
│
├── data/
│   ├── raw/                    # Dataset IAM brut
│   │   ├── ascii/
│   │   ├── lineStrokes/
│   │   └── original-xml/
│   │
│   ├── processed/              # Données préprocessées
│   │   ├── x.npy               # Strokes
│   │   ├── x_len.npy           # Longueurs strokes
│   │   ├── c.npy               # Transcriptions
│   │   ├── c_len.npy           # Longueurs textes
│   │   └── w_id.npy            # IDs écrivains
│   │
│   ├── dataset.py              # Dataset personnalisé
│   └── all_datasets.pickle     # Cache datasets
│
├── words/                      # Images PNG (source alternative)
│   └── [a01, a02, ..., r06]/  # Dossiers par écrivain
│
├── evaluation/
│   ├── real/                  # Images réelles
│   └── gen/                   # Images générées
│
├── debug_render/              # Images de debug
│
├── logs/                      # Fichiers de logs
│
├── PRÉPARATION DES DONNÉES
│   ├── prepare_data.py        # Préparation principale
│   ├── check_data.py          # Vérification données
│   ├── check_data_rendering.py # Vérification rendu
│   ├── diag_collect_stats.py  # Statistiques
│   └── diag_prepare.py        # Diagnostic
│
├── MODÈLES
│   ├── RNN/LSTM ⭐ PRINCIPAL
│   │   ├── rnn.py               # Modèle RNN principal
│   │   ├── rnn_cell.py          # Cellule LSTM avec attention
│   │   └── rnn_ops.py           # Opérations RNN optimisées
│   │
│   ├── TensorFlow (Alternative)
│   │   ├── tf_base_model.py    # Modèle TensorFlow
│   │   └── tf_utils.py          # Utilitaires TensorFlow
│   │
│   └── GAN (FUTUR - Non implémenté)
│       └── [À implémenter]
│
├── RENDU
│   ├── drawing.py              # Utilitaires strokes
│   └── handwriting_renderer.py # Rendu stylisé
│
├── ÉVALUATION
│   ├── metrics.py              # Implémentation métriques
│   ├── calculate_metrics.py    # Script interactif
│   ├── quick_metrics.py         # Script rapide
│   ├── evaluate_metrics.py     # Script avancé
│   ├── prepare_evaluation_data.py # Préparation évaluation
│   └── streamlit_metrics.py        # Interface Streamlit
│
├── INTERFACES
│   ├── streamlit_app.py        # Interface principale
│   └── streamlit_metrics.py    # Interface métriques
│
├── UTILITAIRES
│   ├── data_frame.py            # Gestion données
│   └── requirements.txt         # Dépendances
│
└── DOCUMENTATION
    ├── ARCHITECTURE_GUIDE.md   # Guide architecture
    ├── PIPELINE_DIAGRAM.md      # Ce fichier
    └── METRICS_GUIDE.md         # Guide métriques
```

---

## Comparaison des Approches

```
┌─────────────────────────────────────────────────────────────────┐
│          RNN/LSTM vs RENDU STYLISÉ vs GAN (FUTUR)              │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│   RNN/LSTM ⭐        │  │  RENDU STYLISÉ       │  │   GAN (cGAN)         │
│   PRINCIPAL          │  │                      │  │   FUTUR              │
├──────────────────────┤  ├──────────────────────┤  ├──────────────────────┤
│                      │  │                      │  │                      │
│ ✓ Génération         │  │ ✓ Contrôle précis    │  │ ✓ Style variable     │
│   séquentielle       │  │ ✓ Rapide             │  │ ✓ Apprentissage      │
│ ✓ Modèle temporel    │  │ ✓ Pas d'entraînement│  │ ✓ Réaliste           │
│ ✓ Attention          │  │ ✓ Personnalisable   │  │ ✓ Images directes    │
│   mechanism          │  │ ✓ Facile à utiliser  │  │                      │
│ ✓ Strokes naturels   │  │                      │  │ ✗ Non implémenté    │
│ ✓ Modèle             │  │ ✗ Style limité       │  │ ✗ Nécessite          │
│   probabiliste       │  │ ✗ Moins réaliste     │  │   entraînement       │
│ ✓ Implémenté         │  │    (selon police)    │  │ ✗ Lent (inference)   │
│                      │  │                      │  │ ✗ Moins contrôlable │
│ ✗ Nécessite          │  │                      │  │                      │
│   entraînement      │  │                      │  │                      │
│ ✗ Plus complexe      │  │ Utilisation:         │  │ Utilisation:         │
│ ✗ Conversion requise │  │ - Prototypage       │  │ - Génération         │
│   (strokes→image)    │  │ - Applications       │  │   créative           │
│                      │  │   production         │  │ - Style unique       │
│                      │  │ - Personnalisation  │  │ - Images haute       │
│ Utilisation:         │  │ - Démonstrations    │  │   qualité            │
│ - Génération         │  │                      │  │                      │
│   séquentielle      │  │                      │  │                      │
│ - Modélisation       │  │                      │  │                      │
│   temporelle         │  │                      │  │                      │
│ - Recherche          │  │                      │  │                      │
│ - Production         │  │                      │  │                      │
└──────────────────────┘  └──────────────────────┘  └──────────────────────┘
```

---

## Pipeline de Diagnostic et Vérification

```
┌─────────────────────────────────────────────────────────────────┐
│              OUTILS DE DIAGNOSTIC ET VÉRIFICATION              │
└─────────────────────────────────────────────────────────────────┘

1. VÉRIFICATION DES DONNÉES
   │
   ├─ check_data.py
   │  │
   │  ├─ Vérifie existence data/processed/
   │  ├─ Liste fichiers .npy présents
   │  ├─ Vérifie shapes et dtypes
   │  └─ Affiche statut de chaque fichier
   │
   └─ check_data_rendering.py
      │
      ├─ Charge x.npy et x_len.npy
      ├─ Convertit strokes → images
      ├─ Sauvegarde échantillons dans debug_render/
      └─ Permet vérification visuelle

2. STATISTIQUES ET DIAGNOSTIC
   │
   ├─ diag_collect_stats.py
   │  │
   │  ├─ Parcourt dataset IAM
   │  ├─ Collecte statistiques:
   │  │  ├─ Nombre fichiers ASCII
   │  │  ├─ Correspondances strokes
   │  │  ├─ Fichiers XML originaux
   │  │  └─ Erreurs de correspondance
   │  └─ Affiche exemples d'erreurs
   │
   └─ diag_prepare.py
      │
      └─ Diagnostic processus prepare_data.py

3. UTILISATION TYPIQUE
   │
   ├─ Après prepare_data.py:
   │  └─ python check_data.py
   │     → Vérifier que les fichiers sont corrects
   │
   ├─ Pour visualiser les données:
   │  └─ python check_data_rendering.py
   │     → Génère images dans debug_render/
   │
   └─ Pour diagnostiquer problèmes:
      └─ python diag_collect_stats.py
         → Identifie problèmes de correspondance
```

---

## Workflow Complet Recommandé

```
┌─────────────────────────────────────────────────────────────────┐
│                    WORKFLOW COMPLET                             │
└─────────────────────────────────────────────────────────────────┘

ÉTAPE 1: PRÉPARATION INITIALE
   │
   ├─ 1.1 Vérifier dataset IAM
   │  └─ python prepare_data.py
   │     → Vérifie existence et structure
   │
   ├─ 1.2 Préparer les données
   │  └─ python prepare_data.py
   │     → Génère data/processed/*.npy
   │
   ├─ 1.3 Vérifier les données
   │  └─ python check_data.py
   │     → Confirme que tout est OK
   │
   └─ 1.4 Visualiser échantillons
      └─ python check_data_rendering.py
         → Vérifie visuellement le rendu

ÉTAPE 2: ENTRAÎNEMENT
   │
   ├─ 2.1 Entraîner RNN/LSTM ⭐
   │  └─ python rnn.py
   │     → Entraîne le modèle RNN avec attention
   │
   └─ 2.2 Alternative: TensorFlow
      └─ Utiliser tf_base_model.py si nécessaire

ÉTAPE 3: GÉNÉRATION
   │
   ├─ 3.1 Interface Streamlit (Rendu stylisé)
   │  └─ streamlit run streamlit_app.py
   │
   ├─ 3.2 Interface GAN (si entraîné)
   │  └─ streamlit run GAN/app.py
   │
   └─ 3.3 Scripts Python
      └─ Génération programmatique

ÉTAPE 4: ÉVALUATION
   │
   ├─ 4.1 Préparer données évaluation
   │  └─ python prepare_evaluation_data.py
   │
   ├─ 4.2 Calculer métriques
   │  │
   │  ├─ Option A: Interactif
   │  │  └─ python calculate_metrics.py
   │  │
   │  ├─ Option B: Interface graphique
   │  │  └─ streamlit run streamlit_metrics.py
   │  │
   │  ├─ Option C: Rapide
   │  │  └─ python quick_metrics.py
   │  │
   │  └─ Option D: Avancé
   │     └─ python evaluate_metrics.py --real_dir ... --gen_dir ...
   │
   └─ 4.3 Analyser résultats
      └─ Consulter metrics_results.json
```

---

Ce diagramme complète le guide d'architecture en fournissant des représentations visuelles du pipeline complet du projet, incluant tous les composants disponibles, les outils de diagnostic, et les workflows recommandés.

