"""
Test with ALL_CHARS = True to see if it improves text accuracy
"""
import torch
import numpy as np
from models.model import TRGAN
from params import *
from data.dataset import TextDatasetval
import cv2

# Temporarily override ALL_CHARS
import params
params.ALL_CHARS = True

print(f"Testing with ALL_CHARS = {params.ALL_CHARS}")
print(f"ADD_NOISE = {params.ADD_NOISE}")

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
print(f"Text lengths: {eval_len_text}")

# Generate
selected_simg = data_val['simg'][:1].to(DEVICE)
selected_swids = data_val['swids'][:1]
eval_text_encode_device = eval_text_encode.to(DEVICE).unsqueeze(0)

model.batch_size = 1

print("\nGenerating with ALL_CHARS=True...")
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
cv2.imwrite('test_generation_ALL_CHARS.png', img_np)
print(f"\n✅ Saved to test_generation_ALL_CHARS.png")
