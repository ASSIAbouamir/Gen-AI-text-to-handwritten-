"""
Simple synthetic datasets for training and validation used by `train.py`.

This module provides two lightweight datasets that generate random image
 tensors and random target token sequences so that the training pipeline
 (forward/backward/optimizer) can be smoke-tested without real data.

Exports:
 - TextDataset
 - TextDatasetval

These classes implement a `collate_fn` that pads token sequences to the
maximum length in the batch and returns a dict with keys `simg` and `swids`.
"""

import torch
from torch.utils.data import Dataset
import numpy as np

try:
    from util.params import VOCAB_SIZE
except Exception:
    VOCAB_SIZE = 37


class TextDataset(Dataset):
    """Synthetic training dataset.

    Each item is a dict:
      - 'simg': torch.FloatTensor of shape (1, H, W)
      - 'swids': 1D torch.LongTensor of token ids (variable length)
    """

    def __init__(self, num_examples=100, img_height=32, img_width=256, vocab_size=None):
        self.num_examples = int(num_examples)
        self.img_height = int(img_height)
        self.img_width = int(img_width)
        self.vocab_size = int(vocab_size) if vocab_size is not None else VOCAB_SIZE
        self.max_len = 25

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        # Synthetic grayscale image tensor
        simg = torch.randn(1, self.img_height, self.img_width, dtype=torch.float32)

        # Random token sequence length in [5, max_len]
        text_len = np.random.randint(5, self.max_len + 1)
        swids = torch.randint(0, self.vocab_size, (text_len,), dtype=torch.long)

        return {'simg': simg, 'swids': swids}

    def collate_fn(self, batch):
        """Pad variable-length token sequences and stack images.

        Returns dict {'simg': images, 'swids': padded_word_ids}
        """
        images = torch.stack([item['simg'] for item in batch], dim=0)

        max_len = max(item['swids'].shape[0] for item in batch)
        word_ids = torch.zeros(len(batch), max_len, dtype=torch.long)

        for i, item in enumerate(batch):
            wid_len = item['swids'].shape[0]
            word_ids[i, :wid_len] = item['swids']

        return {'simg': images, 'swids': word_ids}


class TextDatasetval(Dataset):
    """Synthetic validation dataset (same behaviour as TextDataset)."""

    def __init__(self, num_examples=50, img_height=32, img_width=256, vocab_size=None):
        self.num_examples = int(num_examples)
        self.img_height = int(img_height)
        self.img_width = int(img_width)
        self.vocab_size = int(vocab_size) if vocab_size is not None else VOCAB_SIZE
        self.max_len = 25

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        simg = torch.randn(1, self.img_height, self.img_width, dtype=torch.float32)
        text_len = np.random.randint(5, self.max_len + 1)
        swids = torch.randint(0, self.vocab_size, (text_len,), dtype=torch.long)
        return {'simg': simg, 'swids': swids}

    def collate_fn(self, batch):
        images = torch.stack([item['simg'] for item in batch], dim=0)
        max_len = max(item['swids'].shape[0] for item in batch)
        word_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
        for i, item in enumerate(batch):
            wid_len = item['swids'].shape[0]
            word_ids[i, :wid_len] = item['swids']
        return {'simg': images, 'swids': word_ids}
