import os
import numpy as np

# Check if data directory exists
data_dir = 'data/processed/'

print("Checking data directory:", data_dir)
print("Directory exists:", os.path.exists(data_dir))

if os.path.exists(data_dir):
    print("\nFiles in directory:")
    files = os.listdir(data_dir)
    for f in files:
        print(f"  - {f}")
    
    # Check the expected data files
    data_cols = ['x', 'x_len', 'c', 'c_len']
    print("\nChecking expected data files:")
    for col in data_cols:
        filepath = os.path.join(data_dir, f'{col}.npy')
        if os.path.exists(filepath):
            try:
                data = np.load(filepath)
                print(f"  ✓ {col}.npy exists - shape: {data.shape}, dtype: {data.dtype}")
            except Exception as e:
                print(f"  ✗ {col}.npy exists but cannot be loaded: {e}")
        else:
            print(f"  ✗ {col}.npy NOT FOUND")
else:
    print("\nData directory does not exist!")
    print("You need to:")
    print("  1. Create the 'data/processed/' directory")
    print("  2. Place the following .npy files in it:")
    print("     - x.npy (stroke data)")
    print("     - x_len.npy (sequence lengths)")
    print("     - c.npy (text/character data)")
    print("     - c_len.npy (text lengths)")