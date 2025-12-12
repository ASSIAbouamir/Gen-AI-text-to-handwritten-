"""
Debug script to understand text encoding and generation
"""
import torch
from models.model import TRGAN
from params import *
from data.dataset import TextDatasetval

# Load model
print("Loading model...")
model = TRGAN()
model.netG.load_state_dict(torch.load('files/iam_model.pth', map_location=DEVICE))
model.eval()

# Test text
test_text = "Hello World"
print(f"\nInput text: '{test_text}'")

# Encode text
text_encode = [word.encode() for word in test_text.split(' ')]
print(f"\nText split into words: {test_text.split(' ')}")
print(f"Encoded bytes: {text_encode}")

# Use converter
eval_text_encode, eval_len_text = model.netconverter.encode(text_encode)
print(f"\nEncoded tensor shape: {eval_text_encode.shape}")
print(f"Encoded tensor: {eval_text_encode}")
print(f"Text lengths: {eval_len_text}")

# Decode back
decoded = model.netconverter.decode(eval_text_encode[0], torch.IntTensor([eval_len_text[0]]))
print(f"\nDecoded text: '{decoded}'")

# Check alphabet
print(f"\nAlphabet: {ALPHABET}")
print(f"Alphabet length: {len(ALPHABET)}")
print(f"VOCAB_SIZE: {VOCAB_SIZE}")

# Check what indices are being used
print(f"\nCharacter to index mapping:")
for word in test_text.split(' '):
    print(f"  '{word}':")
    for char in word:
        if char in model.netconverter.dict:
            print(f"    '{char}' -> {model.netconverter.dict[char]}")
        else:
            print(f"    '{char}' -> NOT IN ALPHABET!")
