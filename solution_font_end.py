"""
Solution FONTS - Génération d'écriture manuscrite IMMÉDIATE
Pas besoin d'entraînement, fonctionne instantanément !
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import urllib.request
import zipfile
import io


class FontBasedGenerator:
    """
    Générateur basé sur des fonts manuscrites
    ✅ Fonctionne IMMÉDIATEMENT
    ✅ Résultats GARANTIS
    ✅ Personnalisable
    """
    
    def __init__(self, fonts_dir='./fonts'):
        self.fonts_dir = Path(fonts_dir)
        self.fonts_dir.mkdir(exist_ok=True)
        
        print("🎨 Initialisation du générateur de fonts...")
        
        # Télécharger des fonts manuscrites si nécessaire
        self.download_fonts()
        
        # Charger les fonts
        self.fonts = self.load_fonts()
        
        if len(self.fonts) == 0:
            print("⚠️ Aucune font manuscrite trouvée, utilisation de la font par défaut")
            self.fonts = [None]  # Font par défaut PIL
        else:
            print(f"✅ {len(self.fonts)} fonts manuscrites chargées")
    
    def download_fonts(self):
        """Télécharge des fonts manuscrites gratuites"""
        
        # Liste de fonts manuscrites Google Fonts (gratuites)
        fonts_to_download = {
            'Caveat': 'https://github.com/google/fonts/raw/main/ofl/caveat/Caveat%5Bwght%5D.ttf',
            'DancingScript': 'https://github.com/google/fonts/raw/main/ofl/dancingscript/DancingScript%5Bwght%5D.ttf',
            'IndieFlower': 'https://github.com/google/fonts/raw/main/ofl/indieflower/IndieFlower-Regular.ttf',
            'Pacifico': 'https://github.com/google/fonts/raw/main/ofl/pacifico/Pacifico-Regular.ttf',
        }
        
        print("\n📥 Téléchargement des fonts manuscrites...")
        
        for name, url in fonts_to_download.items():
            font_path = self.fonts_dir / f'{name}.ttf'
            
            if font_path.exists():
                print(f"✅ {name} déjà téléchargée")
                continue
            
            try:
                print(f"⬇️  Téléchargement de {name}...")
                urllib.request.urlretrieve(url, font_path)
                print(f"✅ {name} téléchargée")
            except Exception as e:
                print(f"⚠️ Erreur téléchargement {name}: {e}")
        
        print()
    
    def load_fonts(self):
        """Charge toutes les fonts disponibles"""
        fonts = []
        
        # Charger les fonts téléchargées
        for font_file in self.fonts_dir.glob('*.ttf'):
            try:
                font = ImageFont.truetype(str(font_file), 48)
                fonts.append(font)
            except Exception as e:
                print(f"⚠️ Impossible de charger {font_file.name}: {e}")
        
        return fonts
    
    def generate(self, text: str, style_idx: int = 0, 
                target_size=(64, 800), variations=True):
        """
        Génère une image d'écriture manuscrite
        
        Args:
            text: Texte à générer
            style_idx: Index de la font à utiliser
            target_size: (height, width) de l'image finale
            variations: Ajouter des variations réalistes
        Returns:
            image: np.array (H, W) image en niveaux de gris
        """
        height, width = target_size
        
        # Créer image blanche
        img = Image.new('L', (width, height), color=255)
        draw = ImageDraw.Draw(img)
        
        # Sélectionner la font
        font = self.fonts[style_idx % len(self.fonts)]
        
        try:
            # Calculer la position
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Centrer verticalement, aligner à gauche
            x = 10
            y = (height - text_height) // 2
            
            # Dessiner le texte
            draw.text((x, y), text, fill=0, font=font)
        
        except Exception as e:
            # Fallback si erreur
            print(f"⚠️ Erreur avec font: {e}, utilisation texte simple")
            draw.text((10, height//3), text, fill=0)
        
        # Convertir en array numpy
        img_array = np.array(img)
        
        # Ajouter des variations si demandé
        if variations:
            img_array = self.add_realistic_variations(img_array)
        
        return img_array
    
    def add_realistic_variations(self, img: np.ndarray):
        """
        Ajoute des variations pour rendre l'écriture plus réaliste
        """
        # 1. Légère inclinaison aléatoire (-3° à +3°)
        angle = np.random.uniform(-3, 3)
        h, w = img.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), 
                            borderValue=255,
                            flags=cv2.INTER_LINEAR)
        
        # 2. Variation d'épaisseur (érosion/dilatation aléatoire)
        operation = np.random.choice(['erode', 'dilate', 'none'])
        kernel_size = np.random.choice([1, 2])
        
        if operation != 'none' and kernel_size > 1:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            if operation == 'erode':
                img = cv2.erode(img, kernel, iterations=1)
            else:
                img = cv2.dilate(img, kernel, iterations=1)
        
        # 3. Bruit gaussien léger
        noise = np.random.normal(0, 3, img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        
        # 4. Légère déformation perspective
        rows, cols = img.shape
        pts1 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
        
        # Petites variations aléatoires
        dx = np.random.randint(-5, 5, 4)
        dy = np.random.randint(-5, 5, 4)
        pts2 = pts1 + np.column_stack([dx, dy]).astype(np.float32)
        
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(img, M, (cols, rows), borderValue=255)
        
        return img
    
    def generate_multiple_lines(self, texts: list, style_idx: int = 0,
                                save_path: str = None):
        """
        Génère plusieurs lignes de texte (comme une lettre)
        
        Args:
            texts: Liste de lignes de texte
            style_idx: Index de la font
            save_path: Chemin pour sauvegarder (optionnel)
        Returns:
            composite_image: Image complète avec toutes les lignes
        """
        line_height = 64
        line_width = 800
        margin = 10
        
        # Générer chaque ligne
        lines = []
        for text in texts:
            line_img = self.generate(text, style_idx, (line_height, line_width))
            lines.append(line_img)
        
        # Créer l'image composite
        total_height = len(lines) * (line_height + margin) + margin
        composite = np.ones((total_height, line_width + 2*margin), dtype=np.uint8) * 255
        
        # Placer chaque ligne
        y_offset = margin
        for line in lines:
            composite[y_offset:y_offset+line_height, margin:margin+line_width] = line
            y_offset += line_height + margin
        
        # Sauvegarder si demandé
        if save_path:
            cv2.imwrite(save_path, composite)
            print(f"💾 Sauvegardé: {save_path}")
        
        return composite
    
    def visualize_all_styles(self, text: str = "Bonjour, comment allez-vous?",
                            save_path: str = None):
        """
        Montre le texte dans tous les styles disponibles
        """
        n_styles = len(self.fonts)
        
        fig, axes = plt.subplots(n_styles, 1, figsize=(14, 2*n_styles))
        
        if n_styles == 1:
            axes = [axes]
        
        for i, font in enumerate(self.fonts):
            img = self.generate(text, style_idx=i, variations=False)
            
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'Style {i+1}', fontsize=12)
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Visualisation sauvegardée: {save_path}")
        
        plt.show()


def demo_font_generator():
    """Démonstration du générateur de fonts"""
    
    print("="*60)
    print("✍️ GÉNÉRATEUR D'ÉCRITURE MANUSCRITE - Solution FONTS")
    print("="*60)
    
    # Initialiser
    generator = FontBasedGenerator()
    
    # Créer le dossier de sortie
    output_dir = Path('./outputs_fonts')
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("📝 EXEMPLE 1: Génération simple")
    print("="*60)
    
    text = "Bonjour, comment allez-vous?"
    
    for style_idx in range(min(3, len(generator.fonts))):
        img = generator.generate(text, style_idx=style_idx)
        
        save_path = output_dir / f'example1_style{style_idx}.png'
        cv2.imwrite(str(save_path), img)
        
        print(f"✅ Généré (style {style_idx}): {save_path}")
    
    print("\n" + "="*60)
    print("📝 EXEMPLE 2: Lettre complète (plusieurs lignes)")
    print("="*60)
    
    letter = [
        "Cher Monsieur,",
        "",
        "Je vous écris pour vous informer",
        "de la situation actuelle concernant",
        "le projet de génération d'écriture.",
        "",
        "Cordialement,",
        "Votre nom"
    ]
    
    composite = generator.generate_multiple_lines(
        letter, 
        style_idx=0,
        save_path=str(output_dir / 'example2_letter.png')
    )
    
    print("\n" + "="*60)
    print("📝 EXEMPLE 3: Comparaison de tous les styles")
    print("="*60)
    
    generator.visualize_all_styles(
        text="Exemple d'écriture manuscrite",
        save_path=str(output_dir / 'example3_all_styles.png')
    )
    
    print("\n" + "="*60)
    print("🎉 DÉMONSTRATION TERMINÉE!")
    print("="*60)
    print(f"\n📁 Tous les résultats sont dans: {output_dir}/")
    print("\n💡 Pour générer votre propre texte:")
    print("   generator = FontBasedGenerator()")
    print("   img = generator.generate('Votre texte ici')")
    print("="*60)


def interactive_mode():
    """Mode interactif pour générer du texte"""
    
    generator = FontBasedGenerator()
    output_dir = Path('./outputs_fonts')
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("🎨 MODE INTERACTIF")
    print("="*60)
    print(f"Styles disponibles: {len(generator.fonts)}")
    print("Tapez 'quit' pour quitter\n")
    
    while True:
        text = input("📝 Entrez votre texte: ")
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("👋 Au revoir!")
            break
        
        if not text.strip():
            continue
        
        try:
            style = int(input(f"🎨 Choisissez un style (0-{len(generator.fonts)-1}): "))
        except:
            style = 0
        
        # Générer
        img = generator.generate(text, style_idx=style)
        
        # Afficher
        plt.figure(figsize=(12, 2))
        plt.imshow(img, cmap='gray')
        plt.title(f'"{text}" - Style {style}')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # Sauvegarder
        save = input("💾 Sauvegarder? (o/n): ")
        if save.lower() in ['o', 'oui', 'y', 'yes']:
            filename = f"generated_{len(list(output_dir.glob('*.png')))}.png"
            save_path = output_dir / filename
            cv2.imwrite(str(save_path), img)
            print(f"✅ Sauvegardé: {save_path}\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        interactive_mode()
    else:
        demo_font_generator()