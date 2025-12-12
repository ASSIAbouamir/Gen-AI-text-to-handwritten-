import os
from xml.etree import ElementTree
from prepare_data import RAW_BASE_DIR, get_ascii_sequences

ascii_root = os.path.join(RAW_BASE_DIR, 'ascii')

c_no_ls_dir = 0
c_no_matches = 0
c_no_original = 0
c_mismatch_counts = 0
c_success = 0
examples = {'no_ls_dir':[], 'no_matches':[], 'no_original':[], 'mismatch_counts':[], 'success':[]}

fnames = []
for dirpath, dirnames, filenames in os.walk(ascii_root):
    if dirnames:
        continue
    for filename in filenames:
        if filename.startswith('.'):
            continue
        fnames.append(os.path.join(dirpath, filename))

for i, fname in enumerate(fnames):
    head, tail = os.path.split(fname)
    last_letter = os.path.splitext(fname)[0][-1]
    last_letter = last_letter if last_letter.isalpha() else ''

    line_stroke_dir = head.replace('ascii', 'lineStrokes')
    line_stroke_fname_prefix = os.path.split(head)[-1] + last_letter + '-'

    if not os.path.isdir(line_stroke_dir):
        c_no_ls_dir += 1
        if len(examples['no_ls_dir']) < 3:
            examples['no_ls_dir'].append((fname, line_stroke_dir))
        continue

    line_stroke_fnames = sorted([f for f in os.listdir(line_stroke_dir) if f.startswith(line_stroke_fname_prefix)])
    if not line_stroke_fnames:
        c_no_matches += 1
        if len(examples['no_matches']) < 3:
            examples['no_matches'].append((fname, line_stroke_dir, line_stroke_fname_prefix))
        continue

    # Build original path the same way as prepare_data to match 'original-xml/original'
    ascii_base = os.path.join(RAW_BASE_DIR, 'ascii')
    rel = os.path.relpath(head, ascii_base)
    parts = rel.split(os.sep)
    if parts and parts[0] == 'ascii':
        parts = parts[1:]
    rel_under_original = os.path.join(*parts) if parts else ''
    original_dir = os.path.join(RAW_BASE_DIR, 'original-xml', 'original', rel_under_original)
    original_xml = os.path.join(original_dir, 'strokes' + last_letter + '.xml')
    if not os.path.exists(original_xml):
        c_no_original += 1
        if len(examples['no_original']) < 3:
            examples['no_original'].append((fname, original_xml))
        continue

    try:
        ascii_sequences = get_ascii_sequences(fname)
    except Exception as e:
        # treat parse error as mismatch
        c_mismatch_counts += 1
        if len(examples['mismatch_counts']) < 3:
            examples['mismatch_counts'].append((fname, 'parse_error', str(e)))
        continue

    if len(ascii_sequences) != len(line_stroke_fnames):
        c_mismatch_counts += 1
        if len(examples['mismatch_counts']) < 3:
            examples['mismatch_counts'].append((fname, len(ascii_sequences), len(line_stroke_fnames)))
        continue

    # passed all checks
    c_success += 1
    if len(examples['success']) < 3:
        examples['success'].append((fname, len(ascii_sequences)))

print('Total ascii files:', len(fnames))
print('line stroke dir missing:', c_no_ls_dir)
print('no matching stroke files:', c_no_matches)
print('original xml missing:', c_no_original)
print('mismatch counts / parse errors:', c_mismatch_counts)
print('success (would append):', c_success)
print('\nExamples:')
for k, v in examples.items():
    print(f"\n{k} (first {len(v)}):")
    for ex in v:
        print(' ', ex)
