"""
Inspect the model checkpoint to understand its structure
"""
import torch

checkpoint_path = 'files/iam_model.pth'
print(f"Loading checkpoint: {checkpoint_path}\n")

checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("="*60)
print("CHECKPOINT STRUCTURE")
print("="*60)

if isinstance(checkpoint, dict):
    print("\nCheckpoint is a dictionary with keys:")
    for key in checkpoint.keys():
        print(f"  - {key}")
        if isinstance(checkpoint[key], torch.Tensor):
            print(f"    Shape: {checkpoint[key].shape}")
else:
    print("\nCheckpoint is a state_dict (OrderedDict)")
    print(f"Number of parameters: {len(checkpoint)}")
    
print("\n" + "="*60)
print("PARAMETER NAMES AND SHAPES")
print("="*60)

if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

# Group by module
modules = {}
for name, param in state_dict.items():
    module_name = name.split('.')[0]
    if module_name not in modules:
        modules[module_name] = []
    modules[module_name].append((name, param.shape))

for module_name, params in sorted(modules.items()):
    print(f"\n{module_name}:")
    for name, shape in params[:5]:  # Show first 5 params
        print(f"  {name}: {shape}")
    if len(params) > 5:
        print(f"  ... and {len(params)-5} more parameters")

# Check query_embed specifically
print("\n" + "="*60)
print("QUERY EMBED (Character Embeddings)")
print("="*60)

query_embed_keys = [k for k in state_dict.keys() if 'query_embed' in k]
if query_embed_keys:
    for key in query_embed_keys:
        print(f"\n{key}:")
        print(f"  Shape: {state_dict[key].shape}")
        print(f"  First 5 values: {state_dict[key][:5, :5]}")
else:
    print("No query_embed found in checkpoint!")

# Check if there's any metadata
print("\n" + "="*60)
print("CHECKING FOR METADATA")
print("="*60)

if isinstance(checkpoint, dict):
    for key in checkpoint.keys():
        if key != 'state_dict' and not isinstance(checkpoint[key], torch.Tensor):
            print(f"\n{key}: {checkpoint[key]}")
