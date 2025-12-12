import streamlit as st
import torch
import cv2
import numpy as np
from models.model import TRGAN
from params import *
from data.dataset import TextDatasetval
import os
import traceback
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="AI Handwriting Generator",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    .success-box {
        padding: 1rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .info-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    h1 {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        font-weight: 800 !important;
        text-align: center;
        margin-bottom: 0.5rem !important;
    }
    .subtitle {
        text-align: center;
        color: #6c757d;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .style-preview {
        border: 2px solid #e9ecef;
        border-radius: 8px;
        padding: 0.5rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    .style-preview:hover {
        border-color: #667eea;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("# ✍️ AI Handwriting Generator")
st.markdown('<p class="subtitle">Transform your digital text into realistic handwriting using AI</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    
    # Info section
    with st.expander("ℹ️ About This App", expanded=False):
        st.markdown("""
        This application uses **TRGAN** (Transformer-based GAN) to generate 
        realistic handwriting from digital text.
        
        **Features:**
        - 🎨 Multiple handwriting styles
        - 🤖 AI-powered generation
        - 💾 Download results
        - ⚡ Fast processing
        
        **Model:** 141M parameters  
        **Dataset:** IAM Handwriting Database
        """)
    
    st.markdown("---")
    
    # Cache clear button
    if st.button("🔄 Clear Cache & Reload"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

# Model Paths
MODEL_PATH = 'files/iam_model.pth'
DATA_PATH = 'files/IAM-32.pickle'

@st.cache_resource
def load_model():
    """Load the TRGAN model"""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
        return None, None
    
    try:
        model = TRGAN()
        model.netG.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        return model, True
    except Exception as e:
        error_msg = f"{str(e)}\n\nFull traceback:\n{traceback.format_exc()}"
        return None, error_msg

@st.cache_data
def load_style_samples():
    """Load and cache style samples from the dataset"""
    if not os.path.exists(DATA_PATH):
        return None
    
    try:
        TextDatasetObjval = TextDatasetval(base_path=DATA_PATH, num_examples=15)
        datasetval = torch.utils.data.DataLoader(
            TextDatasetObjval,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True, 
            drop_last=True,
            collate_fn=TextDatasetObjval.collate_fn)
        
        data_val = next(iter(datasetval))
        
        # Extract style images for preview
        style_images = []
        simg = data_val['simg']
        
        for i in range(simg.shape[0]):
            style_img = simg[i, 0].numpy()
            style_img = ((style_img + 1) / 2 * 255).astype(np.uint8)
            style_images.append(style_img)
        
        return data_val, style_images
    except Exception as e:
        st.error(f"Error loading styles: {e}")
        return None

# Check for files
if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
    st.error("### ❌ Missing Model Files")
    st.markdown(f"""
    <div class="info-card">
    <strong>Required files not found:</strong><br>
    • <code>{MODEL_PATH}</code><br>
    • <code>{DATA_PATH}</code><br><br>
    Please download them using the command below:
    </div>
    """, unsafe_allow_html=True)
    st.code("!gdown --id 16g9zgysQnWk7-353_tMig92KsZsrcM6k && unzip files.zip && rm files.zip", language="bash")
    st.stop()

# Load model and styles
with st.spinner("🔄 Loading AI model..."):
    model, status = load_model()
    style_data = load_style_samples()

if model is None:
    st.error(f"❌ Failed to load model: {status}")
    st.stop()

if style_data is None:
    st.error("❌ Failed to load style samples.")
    st.stop()

# Success message
st.markdown("""
<div class="success-box">
    ✅ <strong>Model Loaded Successfully!</strong><br>
    Ready to generate handwriting with AI
</div>
""", unsafe_allow_html=True)

data_val, style_images = style_data

# Main content area
# Main content area
st.markdown("### 📝 Input Text")
st.markdown("Enter the text you want to convert to handwriting:")

default_text = "A paragraph is a series of related sentences developing a central idea, called the topic."
text_input = st.text_area(
    "Text to convert:",
    value=default_text,
    height=150,
    label_visibility="collapsed",
    placeholder="Type your text here..."
)

# Character count
char_count = len(text_input)
word_count = len(text_input.split())
st.caption(f"📊 {char_count} characters • {word_count} words")

# Generate button
# Generate button
if st.button("🚀 Generate Handwriting", type="primary"):
    if not text_input.strip():
        st.warning("⚠️ Please enter some text to convert.")
    else:
        try:
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🔤 Encoding text...")
            progress_bar.progress(25)
            
            # Prepare text
            text_encode = [j.encode() for j in text_input.split(' ')]
            eval_text_encode, eval_len_text = model.netconverter.encode(text_encode)
            
            # Debug: Show what's being encoded
            with st.expander("🔍 Debug: Text Encoding Info", expanded=False):
                st.write(f"**Input words:** {text_input.split(' ')}")
                st.write(f"**Number of words:** {len(text_input.split(' '))}")
                st.write(f"**Encoded tensor shape:** {eval_text_encode.shape}")
                st.write(f"**Word lengths:** {eval_len_text.tolist()}")
                
                # Try to decode back
                try:
                    decoded_words = []
                    for i in range(eval_text_encode.shape[0]):
                        decoded = model.netconverter.decode(
                            eval_text_encode[i], 
                            torch.IntTensor([eval_len_text[i]])
                        )
                        decoded_words.append(decoded)
                    st.write(f"**Decoded back:** {' '.join(decoded_words)}")
                except Exception as e:
                    st.write(f"**Decode error:** {e}")
            
            status_text.text("🎨 Preparing style...")
            progress_bar.progress(50)
            
            # Randomly select one style
            style_idx = np.random.randint(0, len(style_images))
            selected_styles = [style_idx]
            
            # Filter to only selected styles
            selected_simg = data_val['simg'][selected_styles].to(DEVICE)
            selected_swids = data_val['swids'][selected_styles]
            
            # Repeat text encoding for selected batch size
            eval_text_encode = eval_text_encode.to(DEVICE).repeat(len(selected_styles), 1, 1)
            
            status_text.text("✨ Generating handwriting...")
            progress_bar.progress(75)
            
            # Temporarily change batch size for generation
            original_batch_size = model.batch_size
            model.batch_size = len(selected_styles)
            
            # Generate
            page_val = model._generate_page(
                selected_simg, 
                selected_swids, 
                eval_text_encode, 
                eval_len_text, 
                gen_only=True
            )
            
            # Restore original batch size
            model.batch_size = original_batch_size
            
            progress_bar.progress(100)
            status_text.text("✅ Generation complete!")
            
            # Process output
            img_np = page_val
            
            if len(img_np.shape) == 3 and img_np.shape[0] == 1:
                img_np = img_np.squeeze(0)
            
            # Scale to 0-255
            img_np = (img_np * 255).astype(np.uint8)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Display result
            st.markdown("---")
            st.markdown("### ✨ Generated Handwriting")
            
            result_col1, result_col2 = st.columns([3, 1])
            
            with result_col1:
                st.image(img_np, use_container_width=True)
            
            with result_col2:
                st.markdown("#### 📊 Details")
                st.markdown(f"""
                - **Style:** Random (Style {selected_styles[0]+1})
                - **Words:** {word_count}
                - **Characters:** {char_count}
                - **Image size:** {img_np.shape[1]}×{img_np.shape[0]}px
                """)
                
                # Download button
                img_pil = Image.fromarray(img_np)
                buf = io.BytesIO()
                img_pil.save(buf, format='PNG')
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="💾 Download PNG",
                    data=byte_im,
                    file_name="handwriting_generated.png",
                    mime="image/png",
                    use_container_width=True
                )
                
                st.success("✅ Ready to download!")
            
        except Exception as e:
            st.error(f"❌ An error occurred during generation")
            with st.expander("🔍 Show error details"):
                st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 2rem 0;">
    <p>Powered by <strong>TRGAN</strong> • Built with <strong>Streamlit</strong> & <strong>PyTorch</strong></p>
    <p style="font-size: 0.9rem;">🤖 141M Parameters • ⚡ Real-time Generation</p>
</div>
""", unsafe_allow_html=True)
