import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import prepare_data
import drawing
from handwriting_renderer import HandwritingRenderer, RenderConfig

def prepare_evaluation_data(num_samples=50, output_dir='evaluation'):
    """
    Generates 'real' (from strokes) and 'generated' (from text) images for evaluation.
    """
    print(f"Preparing {num_samples} samples for evaluation...")
    
    # Create directories
    real_dir = os.path.join(output_dir, 'real')
    gen_dir = os.path.join(output_dir, 'gen')
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(gen_dir, exist_ok=True)
    
    # Load data
    # We need to temporarily suppress print output from prepare_data if it's too verbose
    print("Loading IAM data...")
    if not prepare_data.check_dataset_exists():
        print("IAM dataset missing. Cannot proceed.")
        return

    stroke_fnames, transcriptions, writer_ids = prepare_data.collect_data()
    
    if not stroke_fnames:
        print("No data found.")
        return

    # Shuffle and select subset
    indices = list(range(len(stroke_fnames)))
    random.shuffle(indices)
    selected_indices = indices[:num_samples]
    
    # Initialize renderer
    renderer = HandwritingRenderer(RenderConfig())
    
    import metrics

    real_images_list = []
    gen_images_list = []
    texts_list = []

    count = 0
    for idx in selected_indices:
        stroke_fname = stroke_fnames[idx]
        text = transcriptions[idx]
        
        # 1. Generate "Real" Image from strokes
        try:
            # Get strokes
            offsets = prepare_data.get_stroke_sequence(stroke_fname)
            
            # Save path
            real_path = os.path.join(real_dir, f"sample_{idx}.png")
            
            # Render using drawing.draw
            drawing.draw(
                offsets, 
                ascii_seq=None, # Don't put title on image
                align_strokes=True, 
                denoise_strokes=True, 
                save_file=real_path
            )
            
            # Load back as PIL image for metrics
            real_img = Image.open(real_path).convert("RGB")
            real_images_list.append(real_img)
            
        except Exception as e:
            print(f"Error generating real image for {idx}: {e}")
            continue

        # 2. Generate "Fake" Image from text
        try:
            # Decode text from numpy array of indices to string
            # text is a numpy array of indices into drawing.alphabet
            decoded_text = "".join([drawing.alphabet[i] for i in text if i < len(drawing.alphabet)])
            # Remove null characters if any
            decoded_text = decoded_text.replace('\x00', '')
            
            # Render
            # We use a random font or default
            image = renderer.render(
                decoded_text,
                font_size=64,
                paper_style='plain', # Clean background to match plot
                noise_strength=0.0,  # Less noise for fair comparison with plot
                line_spacing=1.5
            )
            
            # Save
            gen_path = os.path.join(gen_dir, f"sample_{idx}.png")
            image.save(gen_path)
            
            gen_images_list.append(image)
            texts_list.append(decoded_text)
            
        except Exception as e:
            print(f"Error generating fake image for {idx}: {e}")
            # If fake fails, we should remove the corresponding real image to keep lists aligned
            if len(real_images_list) > len(gen_images_list):
                real_images_list.pop()
            continue
            
        count += 1
        if count % 10 == 0:
            print(f"Processed {count}/{num_samples}")

    print(f"Done! Generated {count} pairs of images.")
    
    print("\nCalculating metrics...")
    results = metrics.evaluate_handwriting_metrics(
        real_images=real_images_list,
        generated_images=gen_images_list,
        ground_truth_texts=texts_list,
        use_ocr=False, # We provide texts, so no need for OCR
        device='cpu' # Use CPU to be safe
    )
    
    print("\nRESULTS:")
    print(results)
    
    # Save results
    import json
    with open('metrics_results_full.json', 'w') as f:
        # Handle inf/nan
        json_results = {}
        for k, v in results.items():
            if isinstance(v, float) and (v == float('inf') or v == float('-inf')):
                json_results[k] = str(v)
            else:
                json_results[k] = v
        json.dump(json_results, f, indent=2)


if __name__ == "__main__":
    prepare_evaluation_data()
