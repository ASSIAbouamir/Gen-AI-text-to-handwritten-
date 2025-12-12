"""
Script to test if the model actually generates the correct text
"""
import torch
import numpy as np
from models.model import TRGAN
from params import *
from data.dataset import TextDatasetval
import cv2

print("Loading model and data...")
model = TRGAN(batch_size=1)  # Use batch size 1 for testing
model.netG.load_state_dict(torch.load('files/iam_model.pth', map_location=DEVICE))
model.netG.eval()

# Load style data
TextDatasetObjval = TextDatasetval(base_path='files/IAM-32.pickle', num_examples=15)
datasetval = torch.utils.data.DataLoader(
    TextDatasetObjval,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=True, 
    drop_last=True,
    collate_fn=TextDatasetObjval.collate_fn)

data_val = next(iter(datasetval))

# Test with simple text
test_text = "hello world"
print(f"\n{'='*60}")
print(f"Input text: '{test_text}'")
print(f"{'='*60}")

# Encode text
text_encode = [word.encode() for word in test_text.split(' ')]
print(f"\nWords: {test_text.split(' ')}")
print(f"Encoded bytes: {text_encode}")

eval_text_encode, eval_len_text = model.netconverter.encode(text_encode)
print(f"\nEncoded tensor shape: {eval_text_encode.shape}")
print(f"Encoded tensor:\n{eval_text_encode}")
print(f"Text lengths: {eval_len_text}")

# Decode to verify
decoded_words = []
for i in range(eval_text_encode.shape[0]):
    decoded = model.netconverter.decode(
        eval_text_encode[i], 
        torch.IntTensor([eval_len_text[i]])
    )
    decoded_words.append(decoded)
    print(f"Word {i}: '{decoded}'")

print(f"\nDecoded text: '{' '.join(decoded_words)}'")

# Now generate with the model
print(f"\n{'='*60}")
print("Generating handwriting...")
print(f"{'='*60}")

selected_simg = data_val['simg'][:1].to(DEVICE)
selected_swids = data_val['swids'][:1]
eval_text_encode_device = eval_text_encode.to(DEVICE).unsqueeze(0)  # Add batch dimension

print(f"\nStyle image shape: {selected_simg.shape}")
print(f"Style widths: {selected_swids}")
print(f"Text encode shape for generation: {eval_text_encode_device.shape}")

# Set batch size
model.batch_size = 1

# Generate
with torch.no_grad():
    page_val = model._generate_page(
        selected_simg, 
        selected_swids, 
        eval_text_encode_device, 
        eval_len_text, 
        gen_only=True
    )

print(f"\nGenerated image shape: {page_val.shape}")

# Save the result
img_np = (page_val * 255).astype(np.uint8)
cv2.imwrite('test_generation.png', img_np)
print(f"\n✅ Saved to test_generation.png")

# Now let's check what the Generator.Eval actually receives
print(f"\n{'='*60}")
print("Checking Generator.Eval inputs...")
print(f"{'='*60}")

print(f"\nQRS shape (text encoding): {eval_text_encode_device.shape}")
print(f"QRS content:\n{eval_text_encode_device}")

# Check if the issue is in how words are being processed
print(f"\n{'='*60}")
print("Analyzing word-by-word generation...")
print(f"{'='*60}")

for idx in range(eval_text_encode_device.shape[1]):
    word_indices = eval_text_encode_device[0, idx, :]
    word_len = eval_len_text[idx]
    print(f"\nWord {idx}:")
    print(f"  Indices: {word_indices[:word_len]}")
    print(f"  Length: {word_len}")
    
    # Decode this word
    decoded = model.netconverter.decode(word_indices, torch.IntTensor([word_len]))
    print(f"  Decoded: '{decoded}'")
