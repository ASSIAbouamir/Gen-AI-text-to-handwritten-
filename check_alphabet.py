"""
Compare alphabets and check character mapping
"""
from params import ALPHABET, VOCAB_SIZE
from models.OCR_network import strLabelConverter

print("="*60)
print("CURRENT ALPHABET CONFIGURATION")
print("="*60)

print(f"\nVOCAB_SIZE: {VOCAB_SIZE}")
print(f"ALPHABET length: {len(ALPHABET)}")
print(f"\nALPHABET: '{ALPHABET}'")

# Create converter
converter = strLabelConverter(ALPHABET)

print("\n" + "="*60)
print("CHARACTER TO INDEX MAPPING")
print("="*60)

# Test specific characters
test_chars = "hello world"
print(f"\nTest string: '{test_chars}'")
print("\nCharacter mappings:")
for char in test_chars:
    if char in converter.dict:
        idx = converter.dict[char]
        print(f"  '{char}' -> index {idx}")
    else:
        print(f"  '{char}' -> NOT IN ALPHABET!")

# Check if there's a different alphabet that was used during training
print("\n" + "="*60)
print("CHECKING FOR ALTERNATIVE ALPHABETS")
print("="*60)

# Common IAM alphabet
iam_alphabet = 'Only thewigsofrcvdampbkuq.A-210xT5\'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/%'
print(f"\nCommon IAM alphabet: '{iam_alphabet}'")
print(f"IAM alphabet length: {len(iam_alphabet)}")

if iam_alphabet != ALPHABET:
    print("\n⚠️  WARNING: Current ALPHABET differs from common IAM alphabet!")
    print("\nTesting with IAM alphabet:")
    
    iam_converter = strLabelConverter(iam_alphabet)
    for char in "hello world":
        if char in iam_converter.dict:
            idx = iam_converter.dict[char]
            print(f"  '{char}' -> index {idx}")
        else:
            print(f"  '{char}' -> NOT IN IAM ALPHABET!")
else:
    print("\n✅ Current ALPHABET matches common IAM alphabet")

# Check what characters map to the indices we saw in debug
print("\n" + "="*60)
print("REVERSE MAPPING (Index to Character)")
print("="*60)

# From our debug: "hello" -> [18, 15, 22, 22, 25]
test_indices = [18, 15, 22, 22, 25]
print(f"\nIndices {test_indices} map to:")
for idx in test_indices:
    if idx < len(ALPHABET):
        char = ALPHABET[idx]
        print(f"  {idx} -> '{char}'")
    else:
        print(f"  {idx} -> OUT OF RANGE!")

# Also check with IAM alphabet
if iam_alphabet != ALPHABET:
    print(f"\nWith IAM alphabet, indices {test_indices} map to:")
    for idx in test_indices:
        if idx < len(iam_alphabet):
            char = iam_alphabet[idx]
            print(f"  {idx} -> '{char}'")
        else:
            print(f"  {idx} -> OUT OF RANGE!")
