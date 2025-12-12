import os
from prepare_data import RAW_BASE_DIR, get_ascii_sequences

ascii_root = os.path.join(RAW_BASE_DIR, 'ascii')

count = 0
max_samples = 20
print(f"Diagnostic run: ascii_root={ascii_root}\nScanning up to {max_samples} sample files...\n")
for dirpath, dirnames, filenames in os.walk(ascii_root):
    # collect only leaf dirs (same logic as prepare_data.collect_data)
    if dirnames:
        continue
    for filename in filenames:
        if filename.startswith('.'):
            continue
        fname = os.path.join(dirpath, filename)
        print(f"--- Sample #{count}: {fname}")

        head, tail = os.path.split(fname)
        last_letter = os.path.splitext(fname)[0][-1]
        last_letter = last_letter if last_letter.isalpha() else ''

        line_stroke_dir = head.replace('ascii', 'lineStrokes')
        line_stroke_fname_prefix = os.path.split(head)[-1] + last_letter + '-'

        print(f" head={head}")
        print(f" tail={tail}")
        print(f" last_letter='{last_letter}'")
        print(f" expected line_stroke_dir={line_stroke_dir}")
        print(f" expected stroke prefix={line_stroke_fname_prefix}")

        exists_ls_dir = os.path.isdir(line_stroke_dir)
        print(f" line_stroke_dir exists: {exists_ls_dir}")
        if exists_ls_dir:
            matches = sorted([f for f in os.listdir(line_stroke_dir) if f.startswith(line_stroke_fname_prefix)])
            print(f" matching stroke files: {len(matches)} (showing up to 5): {matches[:5]}")

        original_dir = head.replace('ascii', 'original')
        original_xml = os.path.join(original_dir, 'strokes' + last_letter + '.xml')
        print(f" expected original_xml={original_xml}")
        print(f" original_xml exists: {os.path.exists(original_xml)}")

        try:
            ascii_sequences = get_ascii_sequences(fname)
            print(f" ascii sequences parsed: {len(ascii_sequences)}")
        except Exception as e:
            print(f" ascii parse error: {e}")

        print()
        count += 1
        if count >= max_samples:
            break
    if count >= max_samples:
        break

print('Diagnostic complete.')
