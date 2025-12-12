# 🎨 AI Handwriting Generator

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Transform digital text into realistic handwriting using AI**

[Demo](#demo) • [Features](#features) • [Installation](#installation) • [Usage](#usage) • [Documentation](#documentation)

</div>

---

## 📖 Overview

**AI Handwriting Generator** is a powerful deep learning application that converts digital text into realistic handwritten text. Built on the TRGAN (Transformer-based GAN) architecture, this project combines the power of Transformers and Generative Adversarial Networks to produce authentic-looking handwriting in multiple styles.

### 🎯 Key Highlights

- **141 Million Parameters** - State-of-the-art deep learning model
- **8 Unique Styles** - Multiple handwriting styles to choose from
- **Real-time Generation** - Fast processing with GPU acceleration
- **User-friendly Interface** - Beautiful Streamlit web application
- **High Quality Output** - Realistic and natural-looking handwriting

---

## ✨ Features

### 🤖 AI-Powered Generation
- Advanced Transformer architecture for text encoding
- GAN-based image generation for realistic output
- Style-conditioned generation for multiple handwriting styles

### 🎨 Multiple Styles
- 8 pre-trained handwriting styles from real writers
- Style selection and preview functionality
- Consistent style application across entire text

### 💻 Modern Web Interface
- Clean and intuitive Streamlit interface
- Real-time progress indicators
- Style preview and selection
- One-click download of generated images

### ⚡ Performance
- GPU acceleration support
- Efficient batch processing
- Optimized inference pipeline

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
CUDA 11.0+ (for GPU support)
8GB+ RAM
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/handwriting-transformers.git
cd handwriting-transformers
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download pre-trained models**
```bash
# Download model files (IAM dataset)
gdown --id 16g9zgysQnWk7-353_tMig92KsZsrcM6k
unzip files.zip
rm files.zip
```

4. **Run the application**
```bash
streamlit run streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`

---

## 💡 Usage

### Web Interface

1. **Enter your text** in the text area
2. **Select handwriting styles** from the sidebar
3. **Click "Generate Handwriting"**
4. **Download** your generated image

### Python API

```python
from models.model import TRGAN
import torch

# Load model
model = TRGAN()
model.netG.load_state_dict(torch.load('files/iam_model.pth'))
model.eval()

# Generate handwriting
text = "Hello, World!"
# ... (see documentation for full example)
```

---

## 🏗️ Architecture

### Model Components

```
Input Text → Transformer Encoder → Style Memory
                                         ↓
Style Reference → Feature Extractor → Attention
                                         ↓
                                   Decoder → GAN Generator → Output Image
```

### Key Technologies

- **Transformers**: Text encoding and attention mechanisms
- **ResNet-18**: Style feature extraction
- **BigGAN**: High-quality image generation
- **OCR Network**: Text recognition for training

---

## 📊 Model Details

| Component | Details |
|-----------|---------|
| **Architecture** | TRGAN (Transformer + GAN) |
| **Parameters** | 141 Million |
| **Training Dataset** | IAM Handwriting Database |
| **Input** | Text + Style Reference |
| **Output** | Handwritten Image (Variable Size) |
| **Styles** | 8 Pre-trained Styles |

---

## 📁 Project Structure

```
handwriting-transformers/
├── models/                 # Model architectures
│   ├── model.py           # Main TRGAN model
│   ├── transformer.py     # Transformer components
│   ├── BigGAN_networks.py # GAN architecture
│   └── OCR_network.py     # OCR for training
├── data/                  # Data loading utilities
│   └── dataset.py         # Dataset classes
├── util/                  # Utility functions
│   ├── util.py           # General utilities
│   └── misc.py           # Miscellaneous helpers
├── files/                 # Model weights (download separately)
│   ├── iam_model.pth     # Pre-trained weights
│   └── IAM-32.pickle     # Dataset samples
├── streamlit_app.py      # Web application
├── params.py             # Configuration parameters
└── requirements.txt      # Python dependencies
```

---

## 🎓 Training

To train your own model:

```bash
python train.py --dataset IAM --epochs 100 --batch_size 8
```

See [TRAINING.md](docs/TRAINING.md) for detailed training instructions.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **IAM Handwriting Database** for providing training data
- **Original TRGAN Paper** for the architecture inspiration
- **PyTorch Team** for the deep learning framework
- **Streamlit** for the amazing web framework

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

## 🌟 Star History

If you find this project useful, please consider giving it a star ⭐

---

<div align="center">

**Made with ❤️ using PyTorch and Streamlit**

[⬆ Back to Top](#-ai-handwriting-generator)

</div>
