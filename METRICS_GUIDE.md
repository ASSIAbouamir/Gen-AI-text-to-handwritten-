# 📊 Guide d'Utilisation des Métriques

Ce guide explique comment utiliser les différents scripts pour calculer les métriques d'évaluation (FID, KID, CER, WER, SSIM, PSNR, LPIPS, OCR Accuracy).

## 🚀 Solutions Disponibles

### 1. **Script Interactif** (`calculate_metrics.py`) ⭐ RECOMMANDÉ
Le plus simple pour commencer - vous guide étape par étape.

```bash
python calculate_metrics.py
```

Le script vous demandera :
- Le dossier des images réelles (optionnel)
- Le dossier des images générées
- Les textes de référence (optionnel, pour CER/WER)
- Si vous voulez utiliser OCR

**Avantages :** Simple, interactif, pas besoin de modifier le code

---

### 2. **Interface Streamlit** (`streamlit_metrics.py`) 🎨
Interface graphique moderne et intuitive.

```bash
streamlit run streamlit_metrics.py
```

**Fonctionnalités :**
- Interface visuelle avec aperçu des images
- Calcul en un clic
- Téléchargement des résultats en JSON
- Explications des métriques intégrées

**Avantages :** Interface moderne, visualisation des résultats, facile à utiliser

---

### 3. **Script Rapide** (`quick_metrics.py`) ⚡
Pour usage répété - modifiez les chemins dans le script.

```bash
# Option 1: Modifier les chemins dans le script
python quick_metrics.py

# Option 2: Utiliser des variables d'environnement
export REAL_IMAGES_DIR="path/to/real"
export GEN_IMAGES_DIR="path/to/generated"
python quick_metrics.py
```

**Avantages :** Rapide, scriptable, bon pour l'automatisation

---

### 4. **Script Avancé** (`evaluate_metrics.py`) 🔧
Pour usage avancé avec toutes les options en ligne de commande.

```bash
python evaluate_metrics.py \
    --real_dir path/to/real/images \
    --gen_dir path/to/generated/images \
    --ground_truth_texts "Hello" "World" \
    --use_ocr \
    --output results.json
```

**Avantages :** Flexible, toutes les options disponibles, export JSON

---

## 📋 Prérequis

### Installation des dépendances
```bash
pip install -r requirements.txt
```

### Dépendances optionnelles

**Pour FID/KID :**
- PyTorch et torchvision (déjà dans requirements.txt)
- Télécharge automatiquement Inception v3 au premier usage

**Pour SSIM :**
- scikit-image (déjà dans requirements.txt)

**Pour CER/WER/OCR Accuracy avec OCR :**
- Tesseract OCR installé sur votre système
- `pytesseract` (déjà dans requirements.txt)

**Pour LPIPS :**
- `lpips` (déjà dans requirements.txt)
- PyTorch (déjà dans requirements.txt)
- Télécharge automatiquement les modèles pré-entraînés au premier usage

**Installation de Tesseract :**
- Windows: Télécharger depuis https://github.com/UB-Mannheim/tesseract/wiki
- Linux: `sudo apt-get install tesseract-ocr tesseract-ocr-fra`
- Mac: `brew install tesseract`

---

## 📖 Exemples d'Utilisation

### Exemple 1: Calculer toutes les métriques avec images réelles

```bash
python calculate_metrics.py
# Suivez les instructions interactives
```

### Exemple 2: Calculer seulement FID et KID

```python
from metrics import calculate_fid, calculate_kid
from PIL import Image

real_images = [Image.open("real1.png"), Image.open("real2.png")]
gen_images = [Image.open("gen1.png"), Image.open("gen2.png")]

fid = calculate_fid(real_images, gen_images)
kid = calculate_kid(real_images, gen_images)

print(f"FID: {fid:.4f}")
print(f"KID: {kid:.6f}")
```

### Exemple 3: Calculer CER/WER avec OCR

```python
from metrics import evaluate_handwriting_metrics, ocr_image
from PIL import Image

gen_images = [Image.open("gen1.png"), Image.open("gen2.png")]

results = evaluate_handwriting_metrics(
    real_images=[],  # Pas nécessaire pour CER/WER
    generated_images=gen_images,
    use_ocr=True  # Utilise OCR automatiquement
)

print(f"CER: {results['cer']:.4f}")
print(f"WER: {results['wer']:.4f}")
```

### Exemple 4: Calculer SSIM, PSNR et LPIPS pour une paire d'images

```python
from metrics import calculate_ssim, calculate_psnr, calculate_lpips
from PIL import Image

img1 = Image.open("real.png")
img2 = Image.open("generated.png")

ssim = calculate_ssim(img1, img2)
psnr = calculate_psnr(img1, img2)
lpips_score = calculate_lpips(img1, img2)

print(f"SSIM: {ssim:.4f}")
print(f"PSNR: {psnr:.2f} dB")
print(f"LPIPS: {lpips_score:.4f}")
```

### Exemple 5: Calculer OCR Accuracy

```python
from metrics import evaluate_handwriting_metrics
from PIL import Image

gen_images = [Image.open("gen1.png"), Image.open("gen2.png")]
ground_truth_texts = ["Hello", "World"]

results = evaluate_handwriting_metrics(
    real_images=[],  # Pas nécessaire pour OCR Accuracy
    generated_images=gen_images,
    ground_truth_texts=ground_truth_texts
)

print(f"OCR Accuracy: {results['ocr_accuracy']:.4f} ({results['ocr_accuracy']*100:.2f}%)")
```

---

## 📊 Interprétation des Résultats

### FID (Fréchet Inception Distance)
- **< 20** : Excellent
- **20-50** : Bon
- **50-100** : Acceptable
- **> 100** : À améliorer

### KID (Kernel Inception Distance)
- **< 0.01** : Excellent
- **0.01-0.05** : Bon
- **> 0.05** : À améliorer

### CER (Character Error Rate)
- **0.0** : Parfait (aucune erreur)
- **< 0.05** : Excellent (< 5% d'erreurs)
- **0.05-0.15** : Bon
- **> 0.15** : À améliorer

### WER (Word Error Rate)
- **0.0** : Parfait
- **< 0.1** : Excellent (< 10% d'erreurs)
- **0.1-0.3** : Acceptable
- **> 0.3** : À améliorer

### SSIM (Structural Similarity Index)
- **1.0** : Images identiques
- **> 0.9** : Très similaire
- **0.7-0.9** : Similaire
- **< 0.7** : Différent

### PSNR (Peak Signal-to-Noise Ratio)
- **> 40 dB** : Excellent
- **30-40 dB** : Bon
- **20-30 dB** : Acceptable
- **< 20 dB** : Faible qualité

### LPIPS (Learned Perceptual Image Patch Similarity)
- **0.0** : Images identiques
- **< 0.1** : Très similaire perceptuellement
- **0.1-0.3** : Similaire
- **> 0.3** : Différent perceptuellement
- Plus bas = meilleur

### OCR Accuracy
- **1.0 (100%)** : Tous les caractères correctement reconnus
- **> 0.9 (90%)** : Excellent
- **0.7-0.9 (70-90%)** : Bon
- **< 0.7 (70%)** : À améliorer
- Plus haut = meilleur

---

## 🔧 Dépannage

### Erreur: "PyTorch non disponible"
```bash
pip install torch torchvision
```

### Erreur: "scikit-image non disponible"
```bash
pip install scikit-image
```

### Erreur: "Tesseract OCR non disponible"
- Installez Tesseract sur votre système (voir Prérequis)
- Vérifiez que `pytesseract` est installé: `pip install pytesseract`

### Erreur: "lpips non disponible"
```bash
pip install lpips
```

### Erreur: "CUDA out of memory"
- Utilisez `device='cpu'` dans les fonctions
- Réduisez le nombre d'images
- Utilisez `batch_size` plus petit

### Les métriques ne se calculent pas
- Vérifiez que vous avez des images dans les dossiers
- Vérifiez que les images sont au format RGB
- Pour FID/KID/LPIPS, assurez-vous d'avoir des images réelles ET générées
- Pour CER/WER/OCR Accuracy, fournissez des textes de référence ou activez OCR
- Pour LPIPS, assurez-vous que le package `lpips` est installé

---

## 💡 Conseils

1. **Pour une évaluation rapide** : Utilisez `quick_metrics.py`
2. **Pour une première utilisation** : Utilisez `calculate_metrics.py` (interactif)
3. **Pour une interface moderne** : Utilisez `streamlit_metrics.py`
4. **Pour l'automatisation** : Utilisez `evaluate_metrics.py` avec des scripts

---

## 📝 Notes

- Les métriques FID et KID nécessitent un modèle Inception v3 qui sera téléchargé automatiquement (~100 MB) au premier usage
- Le calcul de FID/KID peut prendre plusieurs minutes selon le nombre d'images
- LPIPS nécessite un modèle pré-entraîné qui sera téléchargé automatiquement au premier usage
- Le calcul de LPIPS peut prendre du temps selon le nombre d'images et le réseau utilisé (alex/vgg/squeeze)
- SSIM, PSNR et LPIPS nécessitent des paires d'images (réelles vs générées) de même taille
- CER, WER et OCR Accuracy nécessitent soit des textes de référence, soit OCR activé




