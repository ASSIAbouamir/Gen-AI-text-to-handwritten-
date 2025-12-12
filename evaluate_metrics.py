"""
Script d'exemple pour évaluer les métriques de génération d'écriture manuscrite.

Usage:
    python evaluate_metrics.py --real_dir path/to/real/images --gen_dir path/to/generated/images
"""

import argparse
import os
from pathlib import Path
from typing import List

from PIL import Image
import metrics


def load_images_from_dir(directory: str) -> List[Image.Image]:
    """Charge toutes les images d'un répertoire."""
    images = []
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    
    for filepath in Path(directory).rglob('*'):
        if filepath.suffix.lower() in extensions:
            try:
                img = Image.open(filepath)
                images.append(img)
            except Exception as e:
                print(f"Erreur lors du chargement de {filepath}: {e}")
    
    return images


def main():
    parser = argparse.ArgumentParser(
        description="Évalue les métriques de génération d'écriture manuscrite"
    )
    parser.add_argument(
        '--real_dir',
        type=str,
        help='Répertoire contenant les images réelles (ground truth)'
    )
    parser.add_argument(
        '--gen_dir',
        type=str,
        help='Répertoire contenant les images générées'
    )
    parser.add_argument(
        '--real_images',
        type=str,
        nargs='+',
        help='Liste de chemins vers les images réelles'
    )
    parser.add_argument(
        '--gen_images',
        type=str,
        nargs='+',
        help='Liste de chemins vers les images générées'
    )
    parser.add_argument(
        '--ground_truth_texts',
        type=str,
        nargs='+',
        help='Textes de référence pour CER/WER (doit correspondre aux images générées)'
    )
    parser.add_argument(
        '--use_ocr',
        action='store_true',
        help='Utiliser OCR pour extraire le texte des images générées (pour CER/WER)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device PyTorch pour FID/KID'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Fichier de sortie pour sauvegarder les résultats (JSON)'
    )
    
    args = parser.parse_args()
    
    # Charger les images
    real_images = []
    gen_images = []
    
    if args.real_dir:
        print(f"Chargement des images réelles depuis {args.real_dir}...")
        real_images = load_images_from_dir(args.real_dir)
        print(f"  {len(real_images)} images chargées")
    
    if args.gen_dir:
        print(f"Chargement des images générées depuis {args.gen_dir}...")
        gen_images = load_images_from_dir(args.gen_dir)
        print(f"  {len(gen_images)} images chargées")
    
    if args.real_images:
        real_images = [Image.open(path) for path in args.real_images]
        print(f"  {len(real_images)} images réelles chargées")
    
    if args.gen_images:
        gen_images = [Image.open(path) for path in args.gen_images]
        print(f"  {len(gen_images)} images générées chargées")
    
    if not real_images or not gen_images:
        print("Erreur: Aucune image chargée. Utilisez --real_dir/--gen_dir ou --real_images/--gen_images")
        return
    
    # Déterminer le device
    device = None if args.device == 'auto' else args.device
    
    # Calculer les métriques
    print("\n" + "="*60)
    print("CALCUL DES MÉTRIQUES")
    print("="*60)
    
    results = metrics.evaluate_handwriting_metrics(
        real_images=real_images,
        generated_images=gen_images,
        ground_truth_texts=args.ground_truth_texts,
        use_ocr=args.use_ocr,
        device=device
    )
    
    # Afficher les résultats
    print("\nRÉSULTATS:")
    print("-" * 60)
    
    if 'fid' in results and results['fid'] is not None:
        print(f"FID (Fréchet Inception Distance): {results['fid']:.4f}")
        print("  → Plus bas = meilleur (typiquement < 50 pour de bonnes générations)")
    
    if 'kid' in results and results['kid'] is not None:
        print(f"KID (Kernel Inception Distance): {results['kid']:.6f}")
        print("  → Plus bas = meilleur")
    
    if 'cer' in results:
        print(f"CER (Character Error Rate): {results['cer']:.4f} ({results['cer']*100:.2f}%)")
        print("  → Plus bas = meilleur (0.0 = parfait)")
    
    if 'wer' in results:
        print(f"WER (Word Error Rate): {results['wer']:.4f} ({results['wer']*100:.2f}%)")
        print("  → Plus bas = meilleur (0.0 = parfait)")
    
    if 'ssim' in results:
        print(f"SSIM (Structural Similarity Index): {results['ssim']:.4f}")
        print("  → Plus haut = meilleur (1.0 = identique)")
    
    if 'psnr' in results:
        psnr_val = results['psnr']
        if psnr_val == float('inf'):
            print(f"PSNR (Peak Signal-to-Noise Ratio): ∞ (images identiques)")
        else:
            print(f"PSNR (Peak Signal-to-Noise Ratio): {psnr_val:.2f} dB")
        print("  → Plus haut = meilleur (typiquement 20-50 dB)")
    
    if 'lpips' in results and results['lpips'] is not None:
        print(f"LPIPS (Learned Perceptual Image Patch Similarity): {results['lpips']:.4f}")
        print("  → Plus bas = meilleur (0.0 = identique, typiquement < 0.1)")
    
    if 'ocr_accuracy' in results and results['ocr_accuracy'] is not None:
        print(f"OCR Accuracy: {results['ocr_accuracy']:.4f} ({results['ocr_accuracy']*100:.2f}%)")
        print("  → Plus haut = meilleur (1.0 = 100% de caractères corrects)")
    
    print("-" * 60)
    
    # Sauvegarder si demandé
    if args.output:
        import json
        # Convertir les résultats en format JSON-serializable
        json_results = {}
        for key, value in results.items():
            if value is None:
                json_results[key] = None
            elif isinstance(value, float) and (value == float('inf') or value == float('-inf')):
                json_results[key] = str(value)
            else:
                json_results[key] = value
        
        with open(args.output, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"\nRésultats sauvegardés dans {args.output}")


if __name__ == "__main__":
    main()

