import streamlit as st
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import urllib.request

# ====================================================
# CLASSE PRINCIPALE
# ====================================================
class FontBasedGenerator:
    def __init__(self, fonts_dir='./fonts'):
        self.fonts_dir = Path(fonts_dir)
        self.fonts_dir.mkdir(exist_ok=True)
        self.download_fonts()
        self.fonts = self.load_fonts()

    def download_fonts(self):
        fonts_to_download = {
            'Caveat': 'https://github.com/google/fonts/raw/main/ofl/caveat/Caveat%5Bwght%5D.ttf',
            'DancingScript': 'https://github.com/google/fonts/raw/main/ofl/dancingscript/DancingScript%5Bwght%5D.ttf',
            'IndieFlower': 'https://github.com/google/fonts/raw/main/ofl/indieflower/IndieFlower-Regular.ttf',
            'Pacifico': 'https://github.com/google/fonts/raw/main/ofl/pacifico/Pacifico-Regular.ttf',
        }
        for name, url in fonts_to_download.items():
            font_path = self.fonts_dir / f'{name}.ttf'
            if not font_path.exists():
                urllib.request.urlretrieve(url, font_path)

    def load_fonts(self):
        fonts = []
        for font_file in self.fonts_dir.glob('*.ttf'):
            try:
                fonts.append(ImageFont.truetype(str(font_file), 48))
            except Exception as e:
                print(f"Erreur chargement {font_file}: {e}")
        return fonts

    def generate(self, text, style_idx=0, target_size=(64, 800), variations=True):
        height, width = target_size
        img = Image.new('L', (width, height), color=255)
        draw = ImageDraw.Draw(img)
        font = self.fonts[style_idx % len(self.fonts)]
        bbox = draw.textbbox((0, 0), text, font=font)
        text_height = bbox[3] - bbox[1]
        y = (height - text_height) // 2
        draw.text((10, y), text, fill=0, font=font)
        img_array = np.array(img)
        if variations:
            img_array = self.add_realistic_variations(img_array)
        return img_array

    def add_realistic_variations(self, img):
        angle = np.random.uniform(-3, 3)
        h, w = img.shape
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderValue=255)
        return img


# ====================================================
# INTERFACE STREAMLIT
# ====================================================
st.set_page_config(page_title="🖊️ Générateur d'écriture manuscrite", layout="wide")

st.title("✍️ Générateur d'écriture manuscrite instantané")
st.write("Générez un texte manuscrit stylisé à partir de **fonts Google gratuites**.")

@st.cache_resource
def load_generator():
    return FontBasedGenerator()

generator = load_generator()

text_input = st.text_input("📝 Entrez votre texte :", "Bonjour, comment allez-vous ?")
style_idx = st.slider("🎨 Choisissez un style :", 0, len(generator.fonts)-1, 0)
variations = st.checkbox("Ajouter des variations réalistes", value=True)
height = st.slider("📏 Hauteur de l'image :", 40, 120, 64)
width = st.slider("📐 Largeur de l'image :", 400, 1200, 800)

if st.button("🚀 Générer l'image"):
    img = generator.generate(text_input, style_idx=style_idx, target_size=(height, width), variations=variations)
    pil_img = Image.fromarray(img)

    st.image(pil_img, caption=f"Style {style_idx}", use_container_width=True)

    output_dir = Path('./outputs_fonts')
    output_dir.mkdir(exist_ok=True)
    save_path = output_dir / f"font_style{style_idx}.png"
    pil_img.save(save_path)

    st.success(f"✅ Image sauvegardée : {save_path}")


st.markdown("---")
st.caption("Créé avec ❤️ | FontBasedGenerator (2025)")
