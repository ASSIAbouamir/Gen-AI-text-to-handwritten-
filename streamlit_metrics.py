"""
Interface Streamlit pour calculer et visualiser les métriques d'évaluation.

Usage:
    streamlit run streamlit_metrics.py
"""

import streamlit as st
from pathlib import Path
from typing import List, Optional
import json

from PIL import Image
import metrics


st.set_page_config(
    page_title="Métriques d'Évaluation",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Métriques d'Évaluation")
st.caption("Calculez FID, KID, CER, WER, SSIM, PSNR, LPIPS et OCR Accuracy pour vos images générées")


@st.cache_resource
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
            except Exception:
                pass
    
    return images


# Sidebar pour les paramètres
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("Images")
    real_dir = st.text_input(
        "Dossier images réelles (optionnel)",
        placeholder="C:/path/to/real/images",
        help="Dossier contenant les images de référence (ground truth)"
    )
    
    gen_dir = st.text_input(
        "Dossier images générées",
        placeholder="C:/path/to/generated/images",
        help="Dossier contenant les images générées à évaluer"
    )
    
    st.subheader("Textes (pour CER/WER)")
    use_texts = st.checkbox("Utiliser des textes de référence")
    
    if use_texts:
        texts_input = st.text_area(
            "Textes de référence",
            placeholder="Entrez un texte par ligne\nExemple:\nHello\nWorld",
            help="Un texte par ligne, correspondant aux images générées"
        )
        ground_truth_texts = [t.strip() for t in texts_input.split('\n') if t.strip()] if texts_input else None
    else:
        ground_truth_texts = None
        use_ocr = st.checkbox("Utiliser OCR pour extraire le texte", help="Nécessite Tesseract installé")
    
    st.subheader("Options avancées")
    device = st.selectbox("Device PyTorch", ["auto", "cuda", "cpu"], index=0)
    device = None if device == "auto" else device


# Chargement des images
real_images = []
gen_images = []

if real_dir and Path(real_dir).exists():
    with st.spinner(f"Chargement des images réelles depuis {real_dir}..."):
        real_images = load_images_from_dir(real_dir)
    st.sidebar.success(f"✓ {len(real_images)} images réelles chargées")

if gen_dir and Path(gen_dir).exists():
    with st.spinner(f"Chargement des images générées depuis {gen_dir}..."):
        gen_images = load_images_from_dir(gen_dir)
    st.sidebar.success(f"✓ {len(gen_images)} images générées chargées")
elif gen_dir:
    st.sidebar.error(f"✗ Dossier non trouvé: {gen_dir}")

# Bouton de calcul
if st.button("🚀 Calculer les métriques", type="primary", use_container_width=True):
    if not gen_images:
        st.error("❌ Veuillez fournir un dossier d'images générées!")
    else:
        with st.spinner("Calcul en cours... (cela peut prendre quelques minutes)"):
            try:
                results = metrics.evaluate_handwriting_metrics(
                    real_images=real_images,
                    generated_images=gen_images,
                    ground_truth_texts=ground_truth_texts if use_texts else None,
                    use_ocr=use_ocr if not use_texts else False,
                    device=device
                )
                
                # Afficher les résultats
                st.success("✓ Calcul terminé!")
                
                # Organiser les résultats en colonnes
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("📊 Métriques d'Images")
                    if 'fid' in results and results['fid'] is not None:
                        st.metric("FID", f"{results['fid']:.4f}", help="Plus bas = meilleur (< 50)")
                    if 'kid' in results and results['kid'] is not None:
                        st.metric("KID", f"{results['kid']:.6f}", help="Plus bas = meilleur")
                
                with col2:
                    st.subheader("📝 Métriques de Texte")
                    if 'cer' in results:
                        cer = results['cer']
                        st.metric("CER", f"{cer*100:.2f}%", help="Plus bas = meilleur (0% = parfait)")
                    if 'wer' in results:
                        wer = results['wer']
                        st.metric("WER", f"{wer*100:.2f}%", help="Plus bas = meilleur (0% = parfait)")
                
                with col3:
                    st.subheader("🖼️ Métriques de Similarité")
                    if 'ssim' in results:
                        st.metric("SSIM", f"{results['ssim']:.4f}", help="Plus haut = meilleur (1.0 = identique)")
                    if 'psnr' in results:
                        psnr = results['psnr']
                        if psnr == float('inf'):
                            st.metric("PSNR", "∞", help="Images identiques")
                        else:
                            st.metric("PSNR", f"{psnr:.2f} dB", help="Plus haut = meilleur (20-50 dB)")
                    if 'lpips' in results and results['lpips'] is not None:
                        st.metric("LPIPS", f"{results['lpips']:.4f}", help="Plus bas = meilleur (0.0 = identique)")
                
                # Nouvelle colonne pour OCR Accuracy
                if 'ocr_accuracy' in results and results['ocr_accuracy'] is not None:
                    st.subheader("🔤 OCR Accuracy")
                    ocr_acc = results['ocr_accuracy']
                    st.metric("OCR Accuracy", f"{ocr_acc*100:.2f}%", help="Pourcentage de caractères correctement reconnus")
                
                # Détails dans un expander
                with st.expander("📋 Détails des résultats"):
                    st.json(results)
                
                # Télécharger les résultats en JSON
                json_results = {}
                for key, value in results.items():
                    if value is None:
                        json_results[key] = None
                    elif isinstance(value, float) and (value == float('inf') or value == float('-inf')):
                        json_results[key] = str(value)
                    else:
                        json_results[key] = value
                
                st.download_button(
                    label="💾 Télécharger les résultats (JSON)",
                    data=json.dumps(json_results, indent=2, ensure_ascii=False),
                    file_name="metrics_results.json",
                    mime="application/json",
                    use_container_width=True
                )
                
                # Stocker dans la session pour réutilisation
                st.session_state['last_results'] = results
                st.session_state['last_gen_images'] = gen_images
                
            except Exception as e:
                st.error(f"❌ Erreur lors du calcul: {e}")
                import traceback
                with st.expander("Détails de l'erreur"):
                    st.code(traceback.format_exc())

# Afficher un aperçu des images si disponibles
if gen_images:
    st.divider()
    st.subheader("🖼️ Aperçu des images générées")
    
    num_preview = min(6, len(gen_images))
    cols = st.columns(3)
    
    for i, img in enumerate(gen_images[:num_preview]):
        with cols[i % 3]:
            st.image(img, caption=f"Image {i+1}", use_container_width=True)
    
    if len(gen_images) > num_preview:
        st.caption(f"... et {len(gen_images) - num_preview} autres images")

# Informations sur les métriques
with st.expander("ℹ️ À propos des métriques"):
    st.markdown("""
    **FID (Fréchet Inception Distance)**
    - Mesure la qualité des images générées
    - Plus bas = meilleur (typiquement < 50)
    - Nécessite PyTorch et des images réelles
    
    **KID (Kernel Inception Distance)**
    - Alternative à FID utilisant un kernel MMD
    - Plus bas = meilleur
    - Nécessite PyTorch et des images réelles
    
    **CER (Character Error Rate)**
    - Taux d'erreur au niveau des caractères
    - Plus bas = meilleur (0% = parfait)
    - Nécessite des textes de référence ou OCR
    
    **WER (Word Error Rate)**
    - Taux d'erreur au niveau des mots
    - Plus bas = meilleur (0% = parfait)
    - Nécessite des textes de référence ou OCR
    
    **SSIM (Structural Similarity Index)**
    - Mesure la similarité structurelle entre images
    - Plus haut = meilleur (1.0 = identique)
    - Nécessite des paires d'images (réelles vs générées)
    
    **PSNR (Peak Signal-to-Noise Ratio)**
    - Mesure la qualité d'image en dB
    - Plus haut = meilleur (typiquement 20-50 dB)
    - Nécessite des paires d'images (réelles vs générées)
    
    **LPIPS (Learned Perceptual Image Patch Similarity)**
    - Mesure la similarité perceptuelle basée sur un réseau neuronal
    - Plus bas = meilleur (0.0 = identique, typiquement < 0.1)
    - Nécessite PyTorch et le package lpips
    - Nécessite des paires d'images (réelles vs générées)
    
    **OCR Accuracy**
    - Pourcentage de caractères correctement reconnus par OCR
    - Plus haut = meilleur (1.0 = 100% correct)
    - Nécessite Tesseract OCR et des textes de référence
    """)




