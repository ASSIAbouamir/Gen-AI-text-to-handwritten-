import sys
sys.path.insert(0, r'c:\Users\user1\Desktop\Handwriting-Transformers')

from models.model import TRGAN
import torch

print("Testing TRGAN model loading...")
try:
    model = TRGAN()
    print("✓ Model instantiated successfully!")
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    import traceback
    traceback.print_exc()
