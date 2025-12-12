"""
AI Handwriting Generator - Configuration Parameters

This module contains all configuration parameters for the TRGAN model
and training pipeline. Modify these values to customize the model behavior.

Author: Your Name
Date: December 2025
License: MIT
"""

import torch

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

# Automatically detect and use GPU if available, otherwise use CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Using device: {DEVICE}")

# ============================================================================
# MODEL ARCHITECTURE PARAMETERS
# ============================================================================

# Transformer Configuration
TN_HIDDEN_DIM = 512          # Hidden dimension for transformer layers
TN_NHEADS = 8                # Number of attention heads
TN_DIM_FEEDFORWARD = 512     # Dimension of feedforward network
TN_DROPOUT = 0.1             # Dropout rate for regularization
TN_ENC_LAYERS = 3            # Number of encoder layers
TN_DEC_LAYERS = 3            # Number of decoder layers

# Vocabulary and Text Configuration
VOCAB_SIZE = 80              # Size of character vocabulary
# IAM alphabet - MUST match the alphabet used during model training
ALPHABET = 'Only thewigsofrcvdampbkuq.A-210xT5\'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/%'

# Image Configuration
IMG_HEIGHT = 32              # Height of generated handwriting images
resolution = 32              # Resolution for GAN architecture (must match D_arch/G_arch keys)

# Dataset Configuration
DATASET_PATHS = 'files/IAM-32.pickle'  # Path to dataset pickle file

# ============================================================================
# TRAINING PARAMETERS
# ============================================================================

# Batch Configuration
batch_size = 8               # Number of samples per batch
NUM_EXAMPLES = 15            # Number of style examples per sample
NUM_WORDS = 8                # Number of words to generate
NUM_WRITERS = 500            # Number of unique writers in dataset (for Writer Discriminator)

# Learning Rates
G_LR = 0.0001               # Generator learning rate
D_LR = 0.0004               # Discriminator learning rate
OCR_LR = 0.0001             # OCR network learning rate
W_LR = 0.0001               # Writer discriminator learning rate

# ============================================================================
# MODEL FEATURES
# ============================================================================

# Feature Flags
IS_SEQ = True               # Use sequential processing
ALL_CHARS = False           # Generate all characters at once
ADD_NOISE = False           # Add noise during generation (disabled for deterministic output)
IS_KLD = False              # Use KL divergence loss

# ============================================================================
# DISPLAY CONFIGURATION
# ============================================================================

def print_config():
    """Print current configuration for debugging"""
    print("\n" + "="*60)
    print("🎨 AI HANDWRITING GENERATOR - CONFIGURATION")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {batch_size}")
    print(f"Transformer Layers: {TN_ENC_LAYERS} encoder, {TN_DEC_LAYERS} decoder")
    print(f"Hidden Dimension: {TN_HIDDEN_DIM}")
    print(f"Attention Heads: {TN_NHEADS}")
    print(f"Vocabulary Size: {VOCAB_SIZE}")
    print(f"Image Height: {IMG_HEIGHT}px")
    print("="*60 + "\n")

# Uncomment to print config on import
# print_config()
