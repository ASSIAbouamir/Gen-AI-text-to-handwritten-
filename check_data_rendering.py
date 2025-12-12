import numpy as np
import os
import drawing

def check():
    data_dir = 'data/processed'
    if not os.path.exists(data_dir):
        print(f"Data dir {data_dir} does not exist")
        return

    print(f"Loading data from {data_dir}...")
    try:
        x = np.load(os.path.join(data_dir, 'x.npy'), allow_pickle=True)
        x_len = np.load(os.path.join(data_dir, 'x_len.npy'), allow_pickle=True)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    os.makedirs('debug_render', exist_ok=True)
    
    for i in range(5):
        print(f"Rendering sample {i}...")
        strokes = x[i]
        stroke_len = x_len[i]
        strokes = strokes[:stroke_len]
        
        save_path = f"debug_render/sample_{i}.png"
        try:
            drawing.draw(strokes, save_file=save_path)
            print(f"Saved to {save_path}")
        except Exception as e:
            print(f"Error rendering sample {i}: {e}")

if __name__ == "__main__":
    check()
