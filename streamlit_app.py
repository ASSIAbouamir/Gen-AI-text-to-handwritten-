import streamlit as st
import os
from handwriting_renderer import HandwritingRenderer, PAPER_PRESETS, RenderConfig

st.set_page_config(
    page_title="My Handwriting Studio",
    page_icon="✍️",
    layout="wide",
)

st.title("✍️ My Handwriting Studio")
st.caption("Transformez votre texte en écriture manuscrite unique et personnalisée.")


@st.cache_resource(show_spinner=False)
def _load_renderer() -> HandwritingRenderer:
    return HandwritingRenderer(RenderConfig())


renderer = _load_renderer()
font_candidates = ["Auto (détection)"] + renderer.available_fonts()

default_text = (
    "Bonjour !\n"
    "Ce petit projet transforme votre texte en un visuel qui ressemble "
    "à de la vraie écriture manuscrite. Modifiez les paramètres dans la barre "
    "latérale pour ajuster le style."
)

# Initialize session state for history
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Paramètres de style")

    font_choice = st.selectbox("Police manuscrite", font_candidates)
    font_name = None if font_choice.startswith("Auto") else font_choice

    font_size = st.slider("Taille du trait", min_value=32, max_value=96, value=64, step=2)
    paper_style = st.selectbox("Type de papier", list(PAPER_PRESETS.keys()), index=1)
    ink_color = st.color_picker("Couleur de l'encre", "#1F1B16")
    
    with st.expander("⚙️ Ajustements fins"):
        jitter = st.slider(
            "Jitter (tremblement)", min_value=0.0, max_value=4.0, value=1.4, step=0.1
        )
        tilt = st.slider(
            "Inclinaison", min_value=-10.0, max_value=10.0, value=-3.0, step=0.5
        )
        noise = st.slider(
            "Texture papier", min_value=0.0, max_value=0.25, value=0.08, step=0.01
        )
        line_spacing = st.slider(
            "Interligne", min_value=1.1, max_value=2.0, value=1.35, step=0.05
        )

    with st.expander("🎨 Personnalisation (Upload)"):
        st.info("Upload your own assets to personalize the result.")
        uploaded_font = st.file_uploader("Police personnalisée (.ttf, .otf)", type=["ttf", "otf"])
        uploaded_bg = st.file_uploader("Papier personnalisé (Image)", type=["png", "jpg", "jpeg"])

    st.caption("💡 Astuce : activez le menu ⋮ pour télécharger l'image générée.")

# Handle uploads
custom_font_path = None
if uploaded_font:
    os.makedirs("temp", exist_ok=True)
    custom_font_path = os.path.join("temp", uploaded_font.name)
    with open(custom_font_path, "wb") as f:
        f.write(uploaded_font.getbuffer())

custom_bg_path = None
if uploaded_bg:
    os.makedirs("temp", exist_ok=True)
    custom_bg_path = os.path.join("temp", uploaded_bg.name)
    with open(custom_bg_path, "wb") as f:
        f.write(uploaded_bg.getbuffer())


col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Entrez votre texte")
    text = st.text_area(
        label="Zone de texte",
        height=300,
        value=default_text,
        label_visibility="collapsed"
    )
    
    generate_btn = st.button("✨ Générer l'écriture", use_container_width=True, type="primary")
    
    random_seed = st.number_input(
        "Seed aléatoire", min_value=0, max_value=9999, value=1337, step=1
    )

with col2:
    st.subheader("2. Résultat")
    if generate_btn:
        if not text.strip():
            st.warning("Veuillez entrer un texte à transformer.")
        else:
            with st.spinner("Création de l'écriture en cours..."):
                try:
                    image = renderer.render(
                        text.strip(),
                        font_name=font_name,
                        font_size=font_size,
                        ink_color=tuple(int(ink_color[i : i + 2], 16) for i in (1, 3, 5)),
                        paper_style=paper_style,
                        jitter_px=jitter,
                        tilt_degrees=tilt,
                        noise_strength=noise,
                        line_spacing=line_spacing,
                        seed=int(random_seed),
                        custom_font_path=custom_font_path,
                        custom_background_path=custom_bg_path,
                    )
                    
                    # Add to history
                    st.session_state.history.insert(0, image)
                    if len(st.session_state.history) > 5:
                        st.session_state.history.pop()
                        
                    st.image(image, caption="Résultat généré", use_column_width=True)

                    st.download_button(
                        label="📥 Télécharger en PNG",
                        data=renderer.to_bytes(image, fmt="PNG"),
                        file_name="handwriting.png",
                        mime="image/png",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"Une erreur est survenue lors de la génération : {e}")
    
    elif st.session_state.history:
        # Show last image if available and button not pressed
        st.image(st.session_state.history[0], caption="Dernier résultat", use_column_width=True)
    else:
        st.info("Cliquez sur 'Générer' pour voir le résultat ici.")

# History Section
if st.session_state.history:
    st.markdown("---")
    st.subheader("📜 Historique récent")
    cols = st.columns(5)
    for idx, img in enumerate(st.session_state.history):
        if idx < 5:
            with cols[idx]:
                st.image(img, use_column_width=True)

