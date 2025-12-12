"""
Miscellaneous Utility Functions for AI Handwriting Generator

This module provides various helper functions for image processing,
text manipulation, and general utilities used throughout the project.

Author: Your Name
Project: AI Handwriting Generator
License: MIT
"""

import numpy as np
import cv2
import torch
from typing import List, Tuple, Optional, Union
import warnings


# ============================================================================
# IMAGE PROCESSING UTILITIES
# ============================================================================

def normalize_image(img: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize image to [0, 1] or [-1, 1] range.
    
    Args:
        img: Input image array
        method: Normalization method ('minmax' or 'standard')
    
    Returns:
        Normalized image array
    """
    if method == 'minmax':
        img_min, img_max = img.min(), img.max()
        if img_max - img_min > 0:
            return (img - img_min) / (img_max - img_min)
        return img
    elif method == 'standard':
        return (img - img.mean()) / (img.std() + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def resize_image(img: np.ndarray, target_height: int, 
                 maintain_aspect: bool = True) -> np.ndarray:
    """
    Resize image to target height while optionally maintaining aspect ratio.
    
    Args:
        img: Input image
        target_height: Desired height in pixels
        maintain_aspect: Whether to maintain aspect ratio
    
    Returns:
        Resized image
    """
    if maintain_aspect:
        h, w = img.shape[:2]
        aspect_ratio = w / h
        target_width = int(target_height * aspect_ratio)
        return cv2.resize(img, (target_width, target_height))
    else:
        return cv2.resize(img, (img.shape[1], target_height))


def pad_image(img: np.ndarray, target_width: int, 
              pad_value: float = 1.0) -> np.ndarray:
    """
    Pad image to target width with specified value.
    
    Args:
        img: Input image
        target_width: Target width after padding
        pad_value: Value to use for padding
    
    Returns:
        Padded image
    """
    h, w = img.shape[:2]
    if w >= target_width:
        return img
    
    pad_width = target_width - w
    if len(img.shape) == 2:
        padding = np.ones((h, pad_width)) * pad_value
        return np.concatenate([img, padding], axis=1)
    else:
        padding = np.ones((h, pad_width, img.shape[2])) * pad_value
        return np.concatenate([img, padding], axis=1)


# ============================================================================
# TEXT PROCESSING UTILITIES
# ============================================================================

def clean_text(text: str, allowed_chars: Optional[str] = None) -> str:
    """
    Clean text by removing or replacing invalid characters.
    
    Args:
        text: Input text string
        allowed_chars: String of allowed characters (None = all printable)
    
    Returns:
        Cleaned text string
    """
    if allowed_chars is None:
        # Default to alphanumeric and common punctuation
        allowed_chars = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?-'
    
    return ''.join(c for c in text if c in allowed_chars)


def split_text_into_lines(text: str, max_words_per_line: int = 8) -> List[str]:
    """
    Split text into lines with maximum words per line.
    
    Args:
        text: Input text
        max_words_per_line: Maximum number of words per line
    
    Returns:
        List of text lines
    """
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        current_line.append(word)
        if len(current_line) >= max_words_per_line:
            lines.append(' '.join(current_line))
            current_line = []
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines


# ============================================================================
# TENSOR UTILITIES
# ============================================================================

def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to NumPy array.
    
    Args:
        tensor: Input PyTorch tensor
    
    Returns:
        NumPy array
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()


def to_tensor(array: np.ndarray, device: str = 'cpu') -> torch.Tensor:
    """
    Convert NumPy array to PyTorch tensor.
    
    Args:
        array: Input NumPy array
        device: Target device ('cpu' or 'cuda')
    
    Returns:
        PyTorch tensor
    """
    if isinstance(array, torch.Tensor):
        return array.to(device)
    return torch.from_numpy(array).to(device)


# ============================================================================
# FILE I/O UTILITIES
# ============================================================================

def save_image(img: np.ndarray, filepath: str, normalize: bool = True) -> bool:
    """
    Save image to file with optional normalization.
    
    Args:
        img: Image array
        filepath: Output file path
        normalize: Whether to normalize to [0, 255]
    
    Returns:
        True if successful, False otherwise
    """
    try:
        if normalize:
            img = (img * 255).astype(np.uint8)
        cv2.imwrite(filepath, img)
        return True
    except Exception as e:
        warnings.warn(f"Failed to save image: {e}")
        return False


def load_image(filepath: str, grayscale: bool = True) -> Optional[np.ndarray]:
    """
    Load image from file.
    
    Args:
        filepath: Input file path
        grayscale: Whether to load as grayscale
    
    Returns:
        Image array or None if failed
    """
    try:
        if grayscale:
            return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        else:
            return cv2.imread(filepath)
    except Exception as e:
        warnings.warn(f"Failed to load image: {e}")
        return None


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def validate_image_shape(img: np.ndarray, expected_channels: Optional[int] = None) -> bool:
    """
    Validate image shape and dimensions.
    
    Args:
        img: Image array to validate
        expected_channels: Expected number of channels (None = any)
    
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(img, np.ndarray):
        return False
    
    if len(img.shape) not in [2, 3]:
        return False
    
    if expected_channels is not None and len(img.shape) == 3:
        if img.shape[2] != expected_channels:
            return False
    
    return True


def validate_text(text: str, max_length: Optional[int] = None) -> Tuple[bool, str]:
    """
    Validate input text.
    
    Args:
        text: Text to validate
        max_length: Maximum allowed length (None = unlimited)
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not text or not text.strip():
        return False, "Text is empty"
    
    if max_length and len(text) > max_length:
        return False, f"Text exceeds maximum length of {max_length}"
    
    return True, ""


# ============================================================================
# DEBUGGING UTILITIES
# ============================================================================

def print_tensor_info(tensor: torch.Tensor, name: str = "Tensor") -> None:
    """
    Print detailed information about a tensor for debugging.
    
    Args:
        tensor: PyTorch tensor
        name: Name to display
    """
    print(f"\n{'='*50}")
    print(f"📊 {name} Info")
    print(f"{'='*50}")
    print(f"Shape: {tensor.shape}")
    print(f"Dtype: {tensor.dtype}")
    print(f"Device: {tensor.device}")
    print(f"Min: {tensor.min().item():.4f}")
    print(f"Max: {tensor.max().item():.4f}")
    print(f"Mean: {tensor.mean().item():.4f}")
    print(f"Std: {tensor.std().item():.4f}")
    print(f"{'='*50}\n")


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class ImageProcessingError(Exception):
    """Custom exception for image processing errors"""
    pass


class TextProcessingError(Exception):
    """Custom exception for text processing errors"""
    pass


# ============================================================================
# MODULE INFO
# ============================================================================

__all__ = [
    'normalize_image',
    'resize_image',
    'pad_image',
    'clean_text',
    'split_text_into_lines',
    'to_numpy',
    'to_tensor',
    'save_image',
    'load_image',
    'validate_image_shape',
    'validate_text',
    'print_tensor_info',
    'ImageProcessingError',
    'TextProcessingError',
]
