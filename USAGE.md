# 📚 Usage Guide

## Quick Start

### Basic Usage

1. **Launch the application**:
```bash
streamlit run streamlit_app.py
```

2. **Open your browser** to `http://localhost:8501`

3. **Enter your text** in the text area

4. **Select a style** from the sidebar

5. **Click "Generate Handwriting"**

6. **Download** your result!

---

## Detailed Usage

### Web Interface

#### 1. Text Input

The text input area accepts:
- **Alphanumeric characters**: a-z, A-Z, 0-9
- **Punctuation**: .,!?;:-'"()
- **Special characters**: @#$%&*

**Tips**:
- Keep sentences under 100 words for best results
- Use proper punctuation for natural-looking output
- Avoid excessive special characters

#### 2. Style Selection

**Available Styles**: 8 pre-trained styles

Each style represents a unique handwriting pattern:
- **Style 1-2**: Regular, upright handwriting
- **Style 3-4**: Slightly cursive
- **Style 5-6**: Italic/slanted
- **Style 7-8**: Bold/thick strokes

**How to select**:
1. View style previews in the sidebar
2. Check/uncheck styles you want to use
3. Multiple styles will be stacked vertically in output

#### 3. Generation Options

**Single Style**: Select one style for consistent output
**Multiple Styles**: Compare different styles side-by-side

---

## Python API

### Basic Example

```python
import torch
from models.model import TRGAN
from params import DEVICE

# Load model
model = TRGAN()
model.netG.load_state_dict(torch.load('files/iam_model.pth', map_location=DEVICE))
model.eval()

# Prepare text
text = "Hello, World!"
text_encode = [word.encode() for word in text.split()]
eval_text_encode, eval_len_text = model.netconverter.encode(text_encode)

# Load style
from data.dataset import TextDatasetval
dataset = TextDatasetval(base_path='files/IAM-32.pickle', num_examples=15)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
data = next(iter(dataloader))

# Generate
with torch.no_grad():
    output = model._generate_page(
        data['simg'].to(DEVICE),
        data['swids'],
        eval_text_encode.to(DEVICE),
        eval_len_text,
        gen_only=True
    )

# Save result
import cv2
cv2.imwrite('output.png', output * 255)
```

### Advanced Example: Batch Processing

```python
import torch
from models.model import TRGAN
from params import DEVICE
import cv2

# Load model
model = TRGAN()
model.netG.load_state_dict(torch.load('files/iam_model.pth'))
model.eval()

# Process multiple texts
texts = [
    "First sentence to convert.",
    "Second sentence to convert.",
    "Third sentence to convert."
]

for idx, text in enumerate(texts):
    # Encode text
    text_encode = [word.encode() for word in text.split()]
    eval_text_encode, eval_len_text = model.netconverter.encode(text_encode)
    
    # Generate (assuming data is loaded)
    with torch.no_grad():
        output = model._generate_page(
            data['simg'].to(DEVICE),
            data['swids'],
            eval_text_encode.to(DEVICE),
            eval_len_text,
            gen_only=True
        )
    
    # Save
    cv2.imwrite(f'output_{idx}.png', output * 255)
    print(f"Generated: output_{idx}.png")
```

---

## Command Line Interface

### Generate from Text File

```bash
python generate.py --input text.txt --output result.png --style 7
```

**Arguments**:
- `--input`: Path to text file
- `--output`: Output image path
- `--style`: Style index (0-7)
- `--batch-size`: Number of styles to use

### Batch Processing

```bash
python batch_generate.py --input-dir texts/ --output-dir results/ --styles 0,3,7
```

---

## Configuration

### Modify Generation Parameters

Edit `params.py`:

```python
# Image resolution
IMG_HEIGHT = 32  # Height in pixels
resolution = 8   # Width multiplier

# Model parameters
TN_HIDDEN_DIM = 512  # Transformer hidden size
TN_NHEADS = 8        # Number of attention heads
```

### Custom Styles

To use your own handwriting style:

1. Prepare training data (see TRAINING.md)
2. Train the model with your data
3. Load the new weights

---

## Tips & Best Practices

### For Best Results

✅ **DO**:
- Use clear, grammatically correct text
- Select styles that match your desired aesthetic
- Keep text length reasonable (< 200 characters)
- Use proper punctuation

❌ **DON'T**:
- Use excessive special characters
- Input very long paragraphs (split them)
- Mix multiple languages in one generation
- Expect perfect results with unusual characters

### Performance Optimization

**For faster generation**:
- Use GPU (CUDA)
- Reduce batch size if memory limited
- Select fewer styles

**For better quality**:
- Use higher resolution settings
- Select appropriate styles
- Post-process images if needed

---

## Troubleshooting

### Common Issues

**Issue**: "Out of memory error"
**Solution**: Reduce batch size or use CPU

**Issue**: "Generated text is unreadable"
**Solution**: Check input text for special characters, try different style

**Issue**: "Slow generation"
**Solution**: Enable GPU acceleration, reduce text length

**Issue**: "Style doesn't match preview"
**Solution**: Clear cache and reload model

---

## Examples

### Example 1: Formal Letter

```
Input: "Dear Sir/Madam, I am writing to express my interest in..."
Style: 1 (Regular, professional)
Output: Clean, formal-looking handwriting
```

### Example 2: Personal Note

```
Input: "Hey! Don't forget about dinner tonight at 7pm :)"
Style: 5 (Casual, italic)
Output: Friendly, relaxed handwriting
```

### Example 3: Quote

```
Input: "The only way to do great work is to love what you do. - Steve Jobs"
Style: 3 (Elegant cursive)
Output: Beautiful, flowing handwriting
```

---

## API Reference

See [API.md](API.md) for complete API documentation.

---

## Support

For issues or questions:
- Check [FAQ.md](FAQ.md)
- Open an issue on GitHub
- Contact: your-email@example.com
