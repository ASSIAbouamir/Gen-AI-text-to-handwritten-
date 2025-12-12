"""
Module de métriques d'évaluation pour la génération d'écriture manuscrite.

Implémente:
- FID (Fréchet Inception Distance)
- KID (Kernel Inception Distance)
- CER (Character Error Rate)
- WER (Word Error Rate)
- SSIM (Structural Similarity Index)
- PSNR (Peak Signal-to-Noise Ratio)
- LPIPS (Learned Perceptual Image Patch Similarity)
- OCR Accuracy (pourcentage de caractères correctement reconnus)
"""

import os
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from scipy import linalg
from sklearn.metrics.pairwise import polynomial_kernel

try:
    import torch
    import torch.nn.functional as F
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch non disponible. FID et KID nécessitent PyTorch.")

try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    warnings.warn("scikit-image non disponible. SSIM nécessite scikit-image.")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    warnings.warn("Tesseract OCR non disponible. CER et WER nécessitent pytesseract.")

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    warnings.warn("lpips non disponible. LPIPS nécessite le package lpips.")


class InceptionFeatureExtractor:
    """Extracteur de features Inception v3 pour FID et KID."""
    
    def __init__(self, device: Optional[str] = None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch est requis pour FID/KID")
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.inception_v3(pretrained=True, transform_input=False)
        self.model.fc = torch.nn.Identity()  # Retirer la couche de classification
        self.model.eval()
        self.model.to(self.device)
        
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, images: List[Image.Image]) -> np.ndarray:
        """Extrait les features Inception pour une liste d'images."""
        features = []
        with torch.no_grad():
            for img in images:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                tensor = self.transform(img).unsqueeze(0).to(self.device)
                feat = self.model(tensor).cpu().numpy()
                features.append(feat.flatten())
        return np.array(features)


def calculate_fid(
    real_images: List[Image.Image],
    generated_images: List[Image.Image],
    device: Optional[str] = None,
    batch_size: int = 50
) -> float:
    """
    Calcule la Fréchet Inception Distance (FID).
    
    Args:
        real_images: Liste d'images réelles (ground truth)
        generated_images: Liste d'images générées
        device: Device PyTorch ('cuda' ou 'cpu')
        batch_size: Taille du batch pour l'extraction de features
        
    Returns:
        Score FID (plus bas = meilleur)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch est requis pour FID")
    
    if len(real_images) != len(generated_images):
        warnings.warn(f"Nombre d'images différent: {len(real_images)} vs {len(generated_images)}")
    
    extractor = InceptionFeatureExtractor(device=device)
    
    # Extraire les features
    real_features = extractor.extract_features(real_images)
    gen_features = extractor.extract_features(generated_images)
    
    # Calculer les statistiques
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)
    
    # Calculer la distance de Fréchet
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(fid)


def calculate_kid(
    real_images: List[Image.Image],
    generated_images: List[Image.Image],
    device: Optional[str] = None,
    subsample_size: Optional[int] = None
) -> float:
    """
    Calcule la Kernel Inception Distance (KID).
    
    Args:
        real_images: Liste d'images réelles
        generated_images: Liste d'images générées
        device: Device PyTorch
        subsample_size: Taille de sous-échantillonnage (None = utiliser toutes les images)
        
    Returns:
        Score KID (plus bas = meilleur)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch est requis pour KID")
    
    extractor = InceptionFeatureExtractor(device=device)
    
    # Extraire les features
    real_features = extractor.extract_features(real_images)
    gen_features = extractor.extract_features(generated_images)
    
    # Sous-échantillonnage si nécessaire
    if subsample_size:
        np.random.seed(42)
        real_idx = np.random.choice(len(real_features), min(subsample_size, len(real_features)), replace=False)
        gen_idx = np.random.choice(len(gen_features), min(subsample_size, len(gen_features)), replace=False)
        real_features = real_features[real_idx]
        gen_features = gen_features[gen_idx]
    
    # Calculer le kernel polynomial
    kernel_real_real = polynomial_kernel(real_features, real_features, degree=3, gamma=None, coef0=1)
    kernel_gen_gen = polynomial_kernel(gen_features, gen_features, degree=3, gamma=None, coef0=1)
    kernel_real_gen = polynomial_kernel(real_features, gen_features, degree=3, gamma=None, coef0=1)
    
    # Calculer KID
    kid = (
        kernel_real_real.mean() +
        kernel_gen_gen.mean() -
        2 * kernel_real_gen.mean()
    )
    
    return float(kid)


def calculate_cer(
    predicted_text: str,
    ground_truth_text: str
) -> float:
    """
    Calcule le Character Error Rate (CER).
    
    Args:
        predicted_text: Texte prédit/reconnu
        ground_truth_text: Texte de référence
        
    Returns:
        CER (0.0 = parfait, 1.0 = toutes les erreurs)
    """
    pred = predicted_text.lower().strip()
    gt = ground_truth_text.lower().strip()
    
    if len(gt) == 0:
        return 1.0 if len(pred) > 0 else 0.0
    
    # Calculer la distance de Levenshtein
    n, m = len(gt), len(pred)
    dp = np.zeros((n + 1, m + 1))
    
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if gt[i - 1] == pred[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    
    errors = int(dp[n][m])
    cer = errors / len(gt)
    return cer


def calculate_wer(
    predicted_text: str,
    ground_truth_text: str
) -> float:
    """
    Calcule le Word Error Rate (WER).
    
    Args:
        predicted_text: Texte prédit/reconnu
        ground_truth_text: Texte de référence
        
    Returns:
        WER (0.0 = parfait, 1.0 = toutes les erreurs)
    """
    pred_words = predicted_text.lower().strip().split()
    gt_words = ground_truth_text.lower().strip().split()
    
    if len(gt_words) == 0:
        return 1.0 if len(pred_words) > 0 else 0.0
    
    # Calculer la distance de Levenshtein au niveau des mots
    n, m = len(gt_words), len(pred_words)
    dp = np.zeros((n + 1, m + 1))
    
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if gt_words[i - 1] == pred_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    
    errors = int(dp[n][m])
    wer = errors / len(gt_words)
    return wer


def calculate_ssim(
    img1: Image.Image,
    img2: Image.Image,
    data_range: Optional[float] = None
) -> float:
    """
    Calcule le Structural Similarity Index (SSIM).
    
    Args:
        img1: Première image
        img2: Deuxième image
        data_range: Plage de valeurs des pixels (None = auto-détection)
        
    Returns:
        Score SSIM (1.0 = identique, 0.0 = complètement différent)
    """
    if not SKIMAGE_AVAILABLE:
        raise ImportError("scikit-image est requis pour SSIM")
    
    # Convertir en numpy arrays
    arr1 = np.array(img1.convert("L"))  # Grayscale
    arr2 = np.array(img2.convert("L"))
    
    # Redimensionner si nécessaire
    if arr1.shape != arr2.shape:
        min_h = min(arr1.shape[0], arr2.shape[0])
        min_w = min(arr1.shape[1], arr2.shape[1])
        arr1 = arr1[:min_h, :min_w]
        arr2 = arr2[:min_h, :min_w]
    
    if data_range is None:
        data_range = max(arr1.max() - arr1.min(), arr2.max() - arr2.min())
        if data_range == 0:
            data_range = 255.0
    
    score = ssim(arr1, arr2, data_range=data_range)
    return float(score)


def calculate_psnr(
    img1: Image.Image,
    img2: Image.Image,
    max_pixel_value: float = 255.0
) -> float:
    """
    Calcule le Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        img1: Première image
        img2: Deuxième image
        max_pixel_value: Valeur maximale des pixels (255 pour uint8)
        
    Returns:
        Score PSNR en dB (plus haut = meilleur, typiquement 20-50 dB)
    """
    # Convertir en numpy arrays
    arr1 = np.array(img1.convert("RGB"), dtype=np.float64)
    arr2 = np.array(img2.convert("RGB"), dtype=np.float64)
    
    # Redimensionner si nécessaire
    if arr1.shape != arr2.shape:
        min_h = min(arr1.shape[0], arr2.shape[0])
        min_w = min(arr1.shape[1], arr2.shape[1])
        arr1 = arr1[:min_h, :min_w, :]
        arr2 = arr2[:min_h, :min_w, :]
    
    # Calculer MSE
    mse = np.mean((arr1 - arr2) ** 2)
    
    if mse == 0:
        return float('inf')  # Images identiques
    
    # Calculer PSNR
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return float(psnr)


def ocr_image(image: Image.Image, lang: str = "fra+eng") -> str:
    """
    Effectue de l'OCR sur une image pour extraire le texte.
    
    Args:
        image: Image à analyser
        lang: Langues pour Tesseract (ex: "fra+eng")
        
    Returns:
        Texte extrait
    """
    if not TESSERACT_AVAILABLE:
        raise ImportError("pytesseract est requis pour l'OCR")
    
    return pytesseract.image_to_string(image, lang=lang).strip()


def calculate_lpips(
    img1: Image.Image,
    img2: Image.Image,
    net: str = "alex",
    device: Optional[str] = None
) -> float:
    """
    Calcule le Learned Perceptual Image Patch Similarity (LPIPS).
    
    Args:
        img1: Première image
        img2: Deuxième image
        net: Réseau à utiliser ('alex', 'vgg', 'squeeze')
        device: Device PyTorch ('cuda' ou 'cpu')
        
    Returns:
        Score LPIPS (plus bas = meilleur, 0.0 = identique)
    """
    if not LPIPS_AVAILABLE:
        raise ImportError("lpips est requis pour LPIPS. Installez avec: pip install lpips")
    
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch est requis pour LPIPS")
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialiser le modèle LPIPS
    loss_fn = lpips.LPIPS(net=net, verbose=False).to(device)
    
    # Convertir les images PIL en tenseurs
    def pil_to_tensor(img):
        if img.mode != "RGB":
            img = img.convert("RGB")
        # Redimensionner si nécessaire (LPIPS fonctionne mieux avec des images carrées)
        img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
        return img_tensor
    
    # Redimensionner les images à la même taille si nécessaire
    if img1.size != img2.size:
        min_size = min(img1.size[0], img1.size[1], img2.size[0], img2.size[1])
        img1 = img1.resize((min_size, min_size), Image.Resampling.LANCZOS)
        img2 = img2.resize((min_size, min_size), Image.Resampling.LANCZOS)
    
    tensor1 = pil_to_tensor(img1)
    tensor2 = pil_to_tensor(img2)
    
    # Normaliser pour LPIPS (les images doivent être dans [-1, 1])
    tensor1 = tensor1 * 2.0 - 1.0
    tensor2 = tensor2 * 2.0 - 1.0
    
    # Calculer LPIPS
    with torch.no_grad():
        distance = loss_fn(tensor1, tensor2)
    
    return float(distance.cpu().item())


def calculate_ocr_accuracy(
    generated_images: List[Image.Image],
    ground_truth_texts: List[str],
    lang: str = "fra+eng"
) -> float:
    """
    Calcule l'OCR Accuracy : pourcentage de caractères correctement reconnus.
    
    Args:
        generated_images: Liste d'images générées
        ground_truth_texts: Liste de textes de référence
        lang: Langues pour Tesseract (ex: "fra+eng")
        
    Returns:
        OCR Accuracy (0.0 à 1.0, 1.0 = 100% de caractères corrects)
    """
    if not TESSERACT_AVAILABLE:
        raise ImportError("pytesseract est requis pour OCR Accuracy")
    
    if len(generated_images) != len(ground_truth_texts):
        raise ValueError(f"Nombre d'images ({len(generated_images)}) != nombre de textes ({len(ground_truth_texts)})")
    
    total_chars = 0
    correct_chars = 0
    
    for img, gt_text in zip(generated_images, ground_truth_texts):
        # Extraire le texte avec OCR
        predicted_text = ocr_image(img, lang=lang)
        
        # Normaliser les textes (minuscules, supprimer espaces)
        pred_normalized = predicted_text.lower().replace(" ", "")
        gt_normalized = gt_text.lower().replace(" ", "")
        
        # Compter les caractères corrects
        min_len = min(len(pred_normalized), len(gt_normalized))
        for i in range(min_len):
            if pred_normalized[i] == gt_normalized[i]:
                correct_chars += 1
            total_chars += 1
        
        # Ajouter les caractères manquants ou en trop comme erreurs
        total_chars += abs(len(pred_normalized) - len(gt_normalized))
    
    if total_chars == 0:
        return 0.0
    
    accuracy = correct_chars / total_chars
    return float(accuracy)


def evaluate_handwriting_metrics(
    real_images: List[Image.Image],
    generated_images: List[Image.Image],
    ground_truth_texts: Optional[List[str]] = None,
    use_ocr: bool = False,
    device: Optional[str] = None
) -> dict:
    """
    Calcule toutes les métriques disponibles pour évaluer la génération.
    
    Args:
        real_images: Liste d'images réelles (ground truth)
        generated_images: Liste d'images générées
        ground_truth_texts: Liste de textes de référence (pour CER/WER)
        use_ocr: Si True, utilise OCR pour extraire le texte des images générées
        device: Device PyTorch pour FID/KID
        
    Returns:
        Dictionnaire avec toutes les métriques calculées
    """
    results = {}
    
    # Métriques d'images (FID, KID)
    if TORCH_AVAILABLE and len(real_images) > 0 and len(generated_images) > 0:
        try:
            results['fid'] = calculate_fid(real_images, generated_images, device=device)
        except Exception as e:
            results['fid'] = None
            warnings.warn(f"Erreur lors du calcul de FID: {e}")
        
        try:
            results['kid'] = calculate_kid(real_images, generated_images, device=device)
        except Exception as e:
            results['kid'] = None
            warnings.warn(f"Erreur lors du calcul de KID: {e}")
    
    # Métriques de texte (CER, WER)
    if ground_truth_texts:
        if use_ocr and TESSERACT_AVAILABLE:
            predicted_texts = [ocr_image(img) for img in generated_images]
        else:
            predicted_texts = None
        
        if predicted_texts and len(predicted_texts) == len(ground_truth_texts):
            cers = [calculate_cer(pred, gt) for pred, gt in zip(predicted_texts, ground_truth_texts)]
            wers = [calculate_wer(pred, gt) for pred, gt in zip(predicted_texts, ground_truth_texts)]
            results['cer'] = np.mean(cers)
            results['wer'] = np.mean(wers)
        elif not use_ocr:
            warnings.warn("ground_truth_texts fourni mais use_ocr=False. CER/WER non calculés.")
    
    # Métriques de similarité (SSIM, PSNR, LPIPS) - nécessitent des paires d'images
    if len(real_images) == len(generated_images) and len(real_images) > 0:
        if SKIMAGE_AVAILABLE:
            ssims = []
            for img1, img2 in zip(real_images, generated_images):
                try:
                    ssims.append(calculate_ssim(img1, img2))
                except Exception as e:
                    warnings.warn(f"Erreur SSIM: {e}")
            if ssims:
                results['ssim'] = np.mean(ssims)
        
        psnrs = []
        for img1, img2 in zip(real_images, generated_images):
            try:
                psnrs.append(calculate_psnr(img1, img2))
            except Exception as e:
                warnings.warn(f"Erreur PSNR: {e}")
        if psnrs:
            results['psnr'] = np.mean(psnrs)
        
        # LPIPS
        if LPIPS_AVAILABLE and TORCH_AVAILABLE:
            lpips_scores = []
            for img1, img2 in zip(real_images, generated_images):
                try:
                    lpips_scores.append(calculate_lpips(img1, img2, device=device))
                except Exception as e:
                    warnings.warn(f"Erreur LPIPS: {e}")
            if lpips_scores:
                results['lpips'] = np.mean(lpips_scores)
    
    # OCR Accuracy
    if ground_truth_texts and len(ground_truth_texts) == len(generated_images):
        if TESSERACT_AVAILABLE:
            try:
                results['ocr_accuracy'] = calculate_ocr_accuracy(generated_images, ground_truth_texts)
            except Exception as e:
                results['ocr_accuracy'] = None
                warnings.warn(f"Erreur lors du calcul de OCR Accuracy: {e}")
        else:
            warnings.warn("Tesseract OCR non disponible. OCR Accuracy non calculé.")
    
    return results


if __name__ == "__main__":
    # Exemple d'utilisation
    print("Module de métriques d'évaluation pour l'écriture manuscrite")
    print("\nMétriques disponibles:")
    print("  - FID (Fréchet Inception Distance)")
    print("  - KID (Kernel Inception Distance)")
    print("  - CER (Character Error Rate)")
    print("  - WER (Word Error Rate)")
    print("  - SSIM (Structural Similarity Index)")
    print("  - PSNR (Peak Signal-to-Noise Ratio)")
    print("  - LPIPS (Learned Perceptual Image Patch Similarity)")
    print("  - OCR Accuracy (pourcentage de caractères correctement reconnus)")
    print("\nUtilisez evaluate_handwriting_metrics() pour calculer toutes les métriques.")

