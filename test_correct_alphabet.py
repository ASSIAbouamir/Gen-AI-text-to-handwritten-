"""
Test generation with the CORRECT IAM alphabet
"""
import torch
import numpy as np
from models.model import TRGAN
from params import *
from data.dataset import TextDatasetval
import cv2

print(f"Testing with CORRECT IAM alphabet")
print(f"ALPHABET: {ALPHABET}")
print(f"ALPHABET length: {len(ALPHABET)}")
print(f"VOCAB_SIZE: {VOCAB_SIZE}")

print("\nLoading model and data...")
model = TRGAN(batch_size=1)
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

print(f"\n✅ Decoded text: '{' '.join(decoded_words)}'")

# Generate
selected_simg = data_val['simg'][:1].to(DEVICE)
selected_swids = data_val['swids'][:1]
eval_text_encode_device = eval_text_encode.to(DEVICE).unsqueeze(0)

model.batch_size = 1

print("\nGenerating with CORRECT alphabet...")
with torch.no_grad():
    page_val = model._generate_page(
        selected_simg, 
        selected_swids, 
        eval_text_encode_device, 
        eval_len_text, 
        gen_only=True
    )

print(f"Generated image shape: {page_val.shape}")

# Save the result
img_np = (page_val * 255).astype(np.uint8)
cv2.imwrite('test_CORRECT_alphabet.png', img_np)
print(f"\n✅ Saved to test_CORRECT_alphabet.png")
print("\n🎉 The generated handwriting should now match 'hello world'!")
