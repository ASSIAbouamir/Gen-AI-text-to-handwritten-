"""
Script simple et direct pour calculer les métriques d'évaluation.

Usage simple:
    python calculate_metrics.py

Le script vous guidera pour charger les images et calculer les métriques.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional

try:
    from PIL import Image
    import metrics
except ImportError as e:
    print(f"Erreur d'import: {e}")
    print("Assurez-vous d'avoir installé toutes les dépendances:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


def load_images_interactive() -> tuple[List[Image.Image], List[Image.Image], Optional[List[str]]]:
    """Charge les images de manière interactive."""
    print("\n" + "="*70)
    print("CHARGEMENT DES IMAGES")
    print("="*70)
    
    # Images réelles
    print("\n1. Images RÉELLES (ground truth):")
    real_path = input("   Chemin vers le dossier d'images réelles (ou 'skip' pour ignorer): ").strip()
    real_images = []
    if real_path.lower() != 'skip' and real_path:
        if os.path.isdir(real_path):
            real_images = load_images_from_dir(real_path)
            print(f"   ✓ {len(real_images)} images chargées")
        elif os.path.isfile(real_path):
            real_images = [Image.open(real_path)]
            print(f"   ✓ 1 image chargée")
        else:
            print(f"   ✗ Chemin invalide: {real_path}")
    
    # Images générées
    print("\n2. Images GÉNÉRÉES:")
    gen_path = input("   Chemin vers le dossier d'images générées: ").strip()
    gen_images = []
    if gen_path:
        if os.path.isdir(gen_path):
            gen_images = load_images_from_dir(gen_path)
            print(f"   ✓ {len(gen_images)} images chargées")
        elif os.path.isfile(gen_path):
            gen_images = [Image.open(gen_path)]
            print(f"   ✓ 1 image chargée")
        else:
            print(f"   ✗ Chemin invalide: {gen_path}")
            return [], [], None
    
    # Textes de référence (optionnel)
    print("\n3. Textes de référence (optionnel, pour CER/WER):")
    use_texts = input("   Voulez-vous fournir des textes de référence? (o/n): ").strip().lower()
    ground_truth_texts = None
    if use_texts == 'o':
        texts_input = input("   Entrez les textes séparés par '|' (ex: 'Hello|World'): ").strip()
        if texts_input:
            ground_truth_texts = [t.strip() for t in texts_input.split('|')]
            print(f"   ✓ {len(ground_truth_texts)} textes enregistrés")
    
    return real_images, gen_images, ground_truth_texts


def load_images_from_dir(directory: str) -> List[Image.Image]:
    """Charge toutes les images d'un répertoire."""
    images = []
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}
    
    for filepath in sorted(Path(directory).rglob('*')):
        if filepath.suffix.lower() in extensions:
            try:
                img = Image.open(filepath)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                images.append(img)
            except Exception as e:
                print(f"   ⚠ Erreur lors du chargement de {filepath.name}: {e}")
    
    return images


def print_results(results: dict):
    """Affiche les résultats de manière formatée."""
    print("\n" + "="*70)
    print("RÉSULTATS DES MÉTRIQUES")
    print("="*70)
    
    if not results:
        print("\n⚠ Aucune métrique calculée.")
        return
    
    # Métriques d'images
    if 'fid' in results:
        if results['fid'] is not None:
            print(f"\n📊 FID (Fréchet Inception Distance): {results['fid']:.4f}")
            print("   → Plus bas = meilleur (typiquement < 50 pour de bonnes générations)")
        else:
            print("\n❌ FID: Non calculé (erreur ou dépendances manquantes)")
    
    if 'kid' in results:
        if results['kid'] is not None:
            print(f"\n📊 KID (Kernel Inception Distance): {results['kid']:.6f}")
            print("   → Plus bas = meilleur")
        else:
            print("\n❌ KID: Non calculé (erreur ou dépendances manquantes)")
    
    # Métriques de texte
    if 'cer' in results:
        cer = results['cer']
        print(f"\n📝 CER (Character Error Rate): {cer:.4f} ({cer*100:.2f}%)")
        print("   → Plus bas = meilleur (0.0 = parfait)")
    
    if 'wer' in results:
        wer = results['wer']
        print(f"\n📝 WER (Word Error Rate): {wer:.4f} ({wer*100:.2f}%)")
        print("   → Plus bas = meilleur (0.0 = parfait)")
    
    # Métriques de similarité
    if 'ssim' in results:
        ssim_val = results['ssim']
        print(f"\n🖼️  SSIM (Structural Similarity Index): {ssim_val:.4f}")
        print("   → Plus haut = meilleur (1.0 = identique)")
    
    if 'psnr' in results:
        psnr_val = results['psnr']
        if psnr_val == float('inf'):
            print(f"\n🖼️  PSNR (Peak Signal-to-Noise Ratio): ∞ (images identiques)")
        else:
            print(f"\n🖼️  PSNR (Peak Signal-to-Noise Ratio): {psnr_val:.2f} dB")
        print("   → Plus haut = meilleur (typiquement 20-50 dB)")
    
    if 'lpips' in results and results['lpips'] is not None:
        print(f"\n🖼️  LPIPS (Learned Perceptual Image Patch Similarity): {results['lpips']:.4f}")
        print("   → Plus bas = meilleur (0.0 = identique, typiquement < 0.1)")
    
    if 'ocr_accuracy' in results and results['ocr_accuracy'] is not None:
        ocr_acc = results['ocr_accuracy']
        print(f"\n🔤 OCR Accuracy: {ocr_acc:.4f} ({ocr_acc*100:.2f}%)")
        print("   → Plus haut = meilleur (1.0 = 100% de caractères corrects)")
    
    print("\n" + "="*70)


def main():
    print("="*70)
    print("CALCULATEUR DE MÉTRIQUES D'ÉVALUATION")
    print("="*70)
    print("\nCe script calcule les métriques suivantes:")
    print("  • FID (Fréchet Inception Distance)")
    print("  • KID (Kernel Inception Distance)")
    print("  • CER (Character Error Rate)")
    print("  • WER (Word Error Rate)")
    print("  • SSIM (Structural Similarity Index)")
    print("  • PSNR (Peak Signal-to-Noise Ratio)")
    print("  • LPIPS (Learned Perceptual Image Patch Similarity)")
    print("  • OCR Accuracy (pourcentage de caractères correctement reconnus)")
    
    # Mode interactif ou arguments en ligne de commande
    if len(sys.argv) > 1:
        # Mode avec arguments
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--real_dir', type=str, help='Dossier d\'images réelles')
        parser.add_argument('--gen_dir', type=str, help='Dossier d\'images générées')
        parser.add_argument('--texts', type=str, nargs='+', help='Textes de référence')
        parser.add_argument('--use_ocr', action='store_true', help='Utiliser OCR')
        parser.add_argument('--output', type=str, help='Fichier JSON de sortie')
        args = parser.parse_args()
        
        real_images = load_images_from_dir(args.real_dir) if args.real_dir else []
        gen_images = load_images_from_dir(args.gen_dir) if args.gen_dir else []
        ground_truth_texts = args.texts if args.texts else None
        use_ocr = args.use_ocr
    else:
        # Mode interactif
        real_images, gen_images, ground_truth_texts = load_images_interactive()
        use_ocr = False
        if ground_truth_texts is None:
            use_ocr_choice = input("\n   Utiliser OCR pour extraire le texte? (o/n): ").strip().lower()
            use_ocr = (use_ocr_choice == 'o')
    
    # Vérifications
    if not gen_images:
        print("\n❌ Erreur: Aucune image générée chargée!")
        return
    
    if not real_images and not use_ocr and not ground_truth_texts:
        print("\n⚠ Avertissement: Aucune image réelle fournie.")
        print("   Seules les métriques SSIM/PSNR nécessitent des images réelles.")
        print("   Les métriques FID/KID nécessitent des images réelles.")
        print("   Les métriques CER/WER nécessitent des textes de référence ou OCR.")
    
    # Calcul des métriques
    print("\n" + "="*70)
    print("CALCUL EN COURS...")
    print("="*70)
    print("(Cela peut prendre quelques minutes selon le nombre d'images)")
    
    try:
        results = metrics.evaluate_handwriting_metrics(
            real_images=real_images if real_images else [],
            generated_images=gen_images,
            ground_truth_texts=ground_truth_texts,
            use_ocr=use_ocr,
            device=None  # Auto-détection
        )
        
        print_results(results)
        
        # Sauvegarder si demandé
        if len(sys.argv) > 1 and args.output:
            import json
            json_results = {}
            for key, value in results.items():
                if value is None:
                    json_results[key] = None
                elif isinstance(value, float) and (value == float('inf') or value == float('-inf')):
                    json_results[key] = str(value)
                else:
                    json_results[key] = value
            
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(json_results, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Résultats sauvegardés dans {args.output}")
        
    except Exception as e:
        print(f"\n❌ Erreur lors du calcul: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

