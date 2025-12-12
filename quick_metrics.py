"""
Script rapide pour calculer les métriques avec des chemins en dur ou variables d'environnement.

Usage:
    # Avec variables d'environnement
    export REAL_IMAGES_DIR="path/to/real"
    export GEN_IMAGES_DIR="path/to/generated"
    python quick_metrics.py

    # Ou modifier directement les chemins dans le script
"""

import os
from pathlib import Path
from PIL import Image
import metrics


def load_images_from_dir(directory: str):
    """Charge toutes les images d'un répertoire."""
    images = []
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}
    
    if not os.path.exists(directory):
        print(f"⚠ Dossier non trouvé: {directory}")
        return images
    
    for filepath in sorted(Path(directory).rglob('*')):
        if filepath.suffix.lower() in extensions:
            try:
                img = Image.open(filepath)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                images.append(img)
            except Exception:
                pass
    
    return images


def main():
    # ============================================
    # CONFIGURATION - MODIFIEZ ICI
    # ============================================
    
    # Option 1: Utiliser des variables d'environnement
    REAL_IMAGES_DIR = os.getenv('REAL_IMAGES_DIR', '')
    GEN_IMAGES_DIR = os.getenv('GEN_IMAGES_DIR', '')
    
    # Option 2: Définir directement les chemins ici
    if not REAL_IMAGES_DIR:
        REAL_IMAGES_DIR = r'data/real_images'  # ← MODIFIEZ ICI
    if not GEN_IMAGES_DIR:
        GEN_IMAGES_DIR = r'data/generated_images'  # ← MODIFIEZ ICI
    
    # Textes de référence (optionnel)
    GROUND_TRUTH_TEXTS = None  # Exemple: ["Hello", "World", "Test"]
    USE_OCR = False  # Mettre à True pour utiliser OCR automatique
    
    # ============================================
    # CALCUL DES MÉTRIQUES
    # ============================================
    
    print("="*70)
    print("CALCUL RAPIDE DES MÉTRIQUES")
    print("="*70)
    
    # Charger les images
    print(f"\n📁 Chargement des images réelles: {REAL_IMAGES_DIR}")
    real_images = load_images_from_dir(REAL_IMAGES_DIR)
    print(f"   ✓ {len(real_images)} images chargées")
    
    print(f"\n📁 Chargement des images générées: {GEN_IMAGES_DIR}")
    gen_images = load_images_from_dir(GEN_IMAGES_DIR)
    print(f"   ✓ {len(gen_images)} images chargées")
    
    if not gen_images:
        print("\n❌ Erreur: Aucune image générée trouvée!")
        return
    
    # Calculer les métriques
    print("\n" + "="*70)
    print("CALCUL EN COURS...")
    print("="*70)
    
    results = metrics.evaluate_handwriting_metrics(
        real_images=real_images,
        generated_images=gen_images,
        ground_truth_texts=GROUND_TRUTH_TEXTS,
        use_ocr=USE_OCR,
        device=None
    )
    
    # Afficher les résultats
    print("\n" + "="*70)
    print("RÉSULTATS")
    print("="*70)
    
    if 'fid' in results and results['fid'] is not None:
        print(f"\nFID: {results['fid']:.4f}")
    if 'kid' in results and results['kid'] is not None:
        print(f"KID: {results['kid']:.6f}")
    if 'cer' in results:
        print(f"CER: {results['cer']:.4f} ({results['cer']*100:.2f}%)")
    if 'wer' in results:
        print(f"WER: {results['wer']:.4f} ({results['wer']*100:.2f}%)")
    if 'ssim' in results:
        print(f"SSIM: {results['ssim']:.4f}")
    if 'psnr' in results:
        psnr = results['psnr']
        print(f"PSNR: {psnr:.2f} dB" if psnr != float('inf') else "PSNR: ∞")
    if 'lpips' in results and results['lpips'] is not None:
        print(f"LPIPS: {results['lpips']:.4f}")
    if 'ocr_accuracy' in results and results['ocr_accuracy'] is not None:
        print(f"OCR Accuracy: {results['ocr_accuracy']:.4f} ({results['ocr_accuracy']*100:.2f}%)")
    
    print("\n" + "="*70)
    
    # Sauvegarder en JSON
    import json
    output_file = 'metrics_results.json'
    json_results = {}
    for key, value in results.items():
        if value is None:
            json_results[key] = None
        elif isinstance(value, float) and (value == float('inf') or value == float('-inf')):
            json_results[key] = str(value)
        else:
            json_results[key] = value
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Résultats sauvegardés dans {output_file}")


if __name__ == "__main__":
    main()

