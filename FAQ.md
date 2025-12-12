# ❓ Frequently Asked Questions

## General Questions

### What is AI Handwriting Generator?

AI Handwriting Generator is a deep learning application that converts digital text into realistic handwritten text. It uses a Transformer-based GAN (TRGAN) architecture trained on real handwriting samples.

### How does it work?

The system uses:
1. **Transformers** to understand text structure
2. **GANs** to generate realistic images
3. **Style conditioning** to mimic specific handwriting styles

### Is it free to use?

Yes! This project is open-source under the MIT License.

---

## Technical Questions

### What are the system requirements?

**Minimum**:
- Python 3.8+
- 8GB RAM
- CPU (slower)

**Recommended**:
- Python 3.10+
- 16GB RAM
- NVIDIA GPU with 4GB+ VRAM
- CUDA 11.0+

### Can I run this without a GPU?

Yes, but generation will be slower (5-10x). The application automatically detects and uses GPU if available.

### What languages are supported?

Currently, the model is trained on English text. Support for other languages requires retraining with appropriate datasets.

### How long does generation take?

- **With GPU**: 0.5-2 seconds per sentence
- **With CPU**: 5-15 seconds per sentence

---

## Usage Questions

### How do I select a specific style?

1. Open the sidebar in the web interface
2. View style previews
3. Check the box next to your desired style(s)
4. Click "Generate Handwriting"

### Can I use multiple styles at once?

Yes! Select multiple styles and they will be stacked vertically in the output image.

### What's the maximum text length?

Recommended: **100-200 characters** per generation for best results. Longer texts should be split into multiple generations.

### Why does my generated text look blurry?

Possible causes:
- Input text too long
- Low-quality style reference
- Need to adjust resolution in `params.py`

### Can I add my own handwriting style?

Yes, but it requires:
1. Collecting handwriting samples
2. Preparing training data
3. Retraining the model
See [TRAINING.md](TRAINING.md) for details.

---

## Model Questions

### How big is the model?

- **Total parameters**: 141 million
- **Model file size**: ~550MB
- **Memory usage**: ~4GB VRAM during inference

### What dataset was it trained on?

The IAM Handwriting Database, containing:
- 1,539 pages of handwritten text
- 657 writers
- 13,353 text lines

### How accurate is the generated text?

The OCR accuracy on generated text is typically **>95%**, meaning the text is highly readable.

### Can the model generate cursive writing?

Yes! Some of the pre-trained styles include cursive/italic handwriting.

---

## Installation Questions

### Where do I download the model files?

```bash
gdown --id 16g9zgysQnWk7-353_tMig92KsZsrcM6k
unzip files.zip
```

This downloads:
- `iam_model.pth` (model weights)
- `IAM-32.pickle` (style samples)

### I get "CUDA out of memory" error. What should I do?

Solutions:
1. Reduce batch size in `params.py`
2. Use CPU instead: Set `DEVICE = 'cpu'` in `params.py`
3. Close other GPU-intensive applications
4. Generate shorter texts

### Installation fails with "No module named 'torch'"

Install PyTorch first:
```bash
pip install torch torchvision
```

For GPU support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Output Questions

### How do I save the generated image?

Click the "💾 Download PNG" button in the web interface, or use the Python API to save directly.

### What format is the output?

- **Format**: PNG
- **Color**: Grayscale
- **Size**: 32px height, variable width
- **DPI**: 72 (can be adjusted)

### Can I change the output resolution?

Yes, modify in `params.py`:
```python
IMG_HEIGHT = 64  # Increase for higher resolution
resolution = 16  # Increase for wider output
```

### Why is my output image so small?

The default output is 32px height. To increase:
1. Edit `params.py`
2. Increase `IMG_HEIGHT`
3. Restart the application

---

## Performance Questions

### How can I make generation faster?

1. **Use GPU** (10x faster than CPU)
2. **Reduce batch size** if memory limited
3. **Select fewer styles** (1 instead of 8)
4. **Shorten input text**

### The application is using too much memory

Solutions:
- Reduce `batch_size` in `params.py`
- Select fewer styles
- Close other applications
- Use CPU mode (slower but less memory)

### Can I run this on Google Colab?

Yes! Upload the code and run:
```python
!pip install -r requirements.txt
!streamlit run streamlit_app.py &
```

---

## Customization Questions

### Can I change the color scheme of the interface?

Yes, edit the CSS in `streamlit_app.py`:
```python
st.markdown("""
    <style>
    /* Your custom CSS here */
    </style>
""", unsafe_allow_html=True)
```

### How do I add more styles?

You need to:
1. Collect new handwriting samples
2. Add them to the dataset
3. The model will automatically use them

### Can I fine-tune the model on my own data?

Yes! See [TRAINING.md](TRAINING.md) for instructions on:
- Preparing your dataset
- Training configuration
- Fine-tuning process

---

## Troubleshooting

### The web interface won't load

1. Check if Streamlit is installed: `pip install streamlit`
2. Verify port 8501 is not in use
3. Try: `streamlit run streamlit_app.py --server.port 8502`

### Generated text is gibberish

Possible causes:
- Model weights not loaded correctly
- Corrupted model file
- Incompatible PyTorch version

Solution: Re-download model files

### Styles don't appear in sidebar

1. Check if `IAM-32.pickle` exists in `files/`
2. Clear cache: Click "Clear Cache & Reload" button
3. Restart the application

### Error: "AttributeError: 'TRGAN' object has no attribute..."

This usually means:
- Model definition doesn't match saved weights
- Need to update model code
- Try clearing Python cache: `rm -rf __pycache__`

---

## Contributing

### How can I contribute?

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

### I found a bug. Where do I report it?

Open an issue on GitHub with:
- Description of the bug
- Steps to reproduce
- Error messages
- System information

---

## Legal & Licensing

### Can I use this commercially?

Yes! The MIT License allows commercial use.

### Do I need to credit the original authors?

While not required, attribution is appreciated.

### Can I modify and redistribute?

Yes, under the MIT License terms.

---

## Still have questions?

- 📧 Email: your-email@example.com
- 💬 GitHub Issues: [Open an issue](https://github.com/yourusername/handwriting-transformers/issues)
- 📖 Documentation: See other `.md` files in the repository
