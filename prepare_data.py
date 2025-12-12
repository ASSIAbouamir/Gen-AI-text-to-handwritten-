from __future__ import print_function
import os
from xml.etree import ElementTree

import numpy as np

import drawing

# Choose which base data directory to use. The original script expects
# data under 'data/raw/' but some setups (including this workspace)
# have the extracted folders directly under 'data/'. Use 'data/raw'
# when present, otherwise fall back to 'data'. This makes the script
# work with both layouts.
RAW_BASE_DIR = 'data/raw' if os.path.exists('data/raw') else 'data'
print(f"Using data base dir: {RAW_BASE_DIR}")


def check_dataset_exists():
    """Check if the IAM dataset exists in the expected location"""
    required_dirs = [
        'data/ascii/ascii',
        'data/lineStrokes/lineStrokes',
        'data/original-xml/original'
    ]
    
    missing = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing.append(dir_path)
    
    if missing:
        print("=" * 70)
        print("ERREUR: Dataset IAM introuvable!")
        print("=" * 70)
        print("\nRépertoires manquants:")
        for path in missing:
            print(f"  ✗ {path}")
        
        print("\n" + "=" * 70)
        print("INSTRUCTIONS POUR OBTENIR LE DATASET IAM:")
        print("=" * 70)
        print("\n1. Inscrivez-vous sur:")
        print("   https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database")
        print("\n2. Téléchargez les fichiers suivants:")
        print("   - ascii-all.tar.gz")
        print("   - lineStrokes-all.tar.gz") 
        print("   - original-xml-all.tar.gz")
        print("\n3. Extrayez les archives dans le répertoire 'data/raw/':")
        print("   tar -xzf ascii-all.tar.gz -C data/raw/")
        print("   tar -xzf lineStrokes-all.tar.gz -C data/raw/")
        print("   tar -xzf original-xml-all.tar.gz -C data/raw/")
        print("\n4. La structure finale devrait être:")
        print("   data/raw/")
        print("   ├── ascii/")
        print("   ├── lineStrokes/")
        print("   └── original/")
        print("\n5. Relancez ce script après avoir extrait les données")
        print("=" * 70)
        print("\nALTERNATIVE: Pour tester le code sans le dataset réel:")
        print("  python create_dummy_data.py")
        print("=" * 70)
        return False
    
    # Check if there's actual data in the directories
    ascii_files = []
    ascii_root = os.path.join(RAW_BASE_DIR, 'ascii')
    for dirpath, dirnames, filenames in os.walk(ascii_root):
        ascii_files.extend([f for f in filenames if not f.startswith('.')])
    
    if not ascii_files:
        print("=" * 70)
        print("ERREUR: Les répertoires existent mais sont vides!")
        print("=" * 70)
        print("\nLes répertoires data/raw/ existent mais ne contiennent pas de données.")
        print("Veuillez extraire les archives téléchargées du dataset IAM.")
        print("=" * 70)
        return False
    
    return True


def get_stroke_sequence(filename):
    tree = ElementTree.parse(filename).getroot()
    strokes = [i for i in tree if i.tag == 'StrokeSet'][0]

    coords = []
    for stroke in strokes:
        for i, point in enumerate(stroke):
            coords.append([
                int(point.attrib['x']),
                -1*int(point.attrib['y']),
                int(i == len(stroke) - 1)
            ])
    coords = np.array(coords)

    coords = drawing.align(coords)
    coords = drawing.denoise(coords)
    offsets = drawing.coords_to_offsets(coords)
    offsets = offsets[:drawing.MAX_STROKE_LEN]
    offsets = drawing.normalize(offsets)
    return offsets


def get_ascii_sequences(filename):
    sequences = open(filename, 'r').read()
    sequences = sequences.replace(r'%%%%%%%%%%%', '\n')
    sequences = [i.strip() for i in sequences.split('\n')]
    lines = sequences[sequences.index('CSR:') + 2:]
    lines = [line.strip() for line in lines if line.strip()]
    lines = [drawing.encode_ascii(line)[:drawing.MAX_CHAR_LEN] for line in lines]
    return lines


def collect_data():
    fnames = []
    for dirpath, dirnames, filenames in os.walk(os.path.join(RAW_BASE_DIR, 'ascii')):
        if dirnames:
            continue
        for filename in filenames:
            if filename.startswith('.'):
                continue
            fnames.append(os.path.join(dirpath, filename))

    # low quality samples (selected by collecting samples to
    # which the trained model assigned very low likelihood)
    blacklist_path = 'data/blacklist.npy'
    if os.path.exists(blacklist_path):
        blacklist = set(np.load(blacklist_path, allow_pickle=True))
    else:
        print("Note: blacklist.npy not found, proceeding without it...")
        blacklist = set()

    stroke_fnames, transcriptions, writer_ids = [], [], []
    for i, fname in enumerate(fnames):
        print(i, fname)
        if fname == 'data/raw/ascii/z01/z01-000/z01-000z.txt':
            continue

        head, tail = os.path.split(fname)
        last_letter = os.path.splitext(fname)[0][-1]
        last_letter = last_letter if last_letter.isalpha() else ''

        line_stroke_dir = head.replace('ascii', 'lineStrokes')
        line_stroke_fname_prefix = os.path.split(head)[-1] + last_letter + '-'

        if not os.path.isdir(line_stroke_dir):
            continue
        line_stroke_fnames = sorted([f for f in os.listdir(line_stroke_dir)
                                     if f.startswith(line_stroke_fname_prefix)])
        if not line_stroke_fnames:
            continue

        # The original XML files live under 'original-xml/original' in the
        # downloaded IAM dataset. Build the original path from the portion
        # of `head` that follows the ascii base so we don't accidentally
        # replace every 'ascii' substring (which caused duplicate parts).
        ascii_base = os.path.join(RAW_BASE_DIR, 'ascii')
        rel = os.path.relpath(head, ascii_base)
        # rel may start with an extra 'ascii' component (e.g. 'ascii/a01/...')
        parts = rel.split(os.sep)
        if parts and parts[0] == 'ascii':
            parts = parts[1:]
        rel_under_original = os.path.join(*parts) if parts else ''
        original_dir = os.path.join(RAW_BASE_DIR, 'original-xml', 'original', rel_under_original)
        original_xml = os.path.join(original_dir, 'strokes' + last_letter + '.xml')

        if not os.path.exists(original_xml):
            continue

        tree = ElementTree.parse(original_xml)
        root = tree.getroot()

        general = root.find('General')
        if general is not None:
            writer_id = int(general[0].attrib.get('writerID', '0'))
        else:
            writer_id = int('0')

        try:
            ascii_sequences = get_ascii_sequences(fname)
        except Exception as e:
            print(f"Warning: Could not parse {fname}: {e}")
            continue
            
        if len(ascii_sequences) != len(line_stroke_fnames):
            print(f"Warning: Mismatch in {fname}, skipping...")
            continue

        for ascii_seq, line_stroke_fname in zip(ascii_sequences, line_stroke_fnames):
            if line_stroke_fname in blacklist:
                continue

            stroke_fnames.append(os.path.join(line_stroke_dir, line_stroke_fname))
            transcriptions.append(ascii_seq)
            writer_ids.append(writer_id)

    return stroke_fnames, transcriptions, writer_ids


if __name__ == '__main__':
    # Check if dataset exists before proceeding
    if not check_dataset_exists():
        exit(1)
    
    print('traversing data directory...')
    stroke_fnames, transcriptions, writer_ids = collect_data()
    
    if not stroke_fnames:
        print("\n" + "=" * 70)
        print("ERREUR: Aucune donnée trouvée!")
        print("=" * 70)
        print("Les fichiers du dataset semblent corrompus ou incomplets.")
        print("Veuillez re-télécharger et extraire le dataset IAM.")
        print("=" * 70)
        exit(1)

    print(f'\nTrouvé {len(stroke_fnames)} échantillons')
    print('dumping to numpy arrays...')
    
    x = np.zeros([len(stroke_fnames), drawing.MAX_STROKE_LEN, 3], dtype=np.float32)
    x_len = np.zeros([len(stroke_fnames)], dtype=np.int16)
    c = np.zeros([len(stroke_fnames), drawing.MAX_CHAR_LEN], dtype=np.int8)
    c_len = np.zeros([len(stroke_fnames)], dtype=np.int8)
    w_id = np.zeros([len(stroke_fnames)], dtype=np.int16)
    valid_mask = np.zeros([len(stroke_fnames)], dtype=bool)

    for i, (stroke_fname, c_i, w_id_i) in enumerate(zip(stroke_fnames, transcriptions, writer_ids)):
        if i % 200 == 0:
            print(i, '\t', '/', len(stroke_fnames))
        
        try:
            x_i = get_stroke_sequence(stroke_fname)
            valid_mask[i] = ~np.any(np.linalg.norm(x_i[:, :2], axis=1) > 60)

            x[i, :len(x_i), :] = x_i
            x_len[i] = len(x_i)

            c[i, :len(c_i)] = c_i
            c_len[i] = len(c_i)

            w_id[i] = w_id_i
        except Exception as e:
            print(f"Warning: Error processing {stroke_fname}: {e}")
            valid_mask[i] = False

    if not os.path.isdir('data/processed'):
        os.makedirs('data/processed')

    num_valid = np.sum(valid_mask)
    print(f"\nSauvegarde de {num_valid} échantillons valides sur {len(stroke_fnames)} total...")
    
    np.save('data/processed/x.npy', x[valid_mask])
    np.save('data/processed/x_len.npy', x_len[valid_mask])
    np.save('data/processed/c.npy', c[valid_mask])
    np.save('data/processed/c_len.npy', c_len[valid_mask])
    np.save('data/processed/w_id.npy', w_id[valid_mask])
    
    print("\n" + "=" * 70)
    print("✓ PRÉTRAITEMENT TERMINÉ AVEC SUCCÈS!")
    print("=" * 70)
    print(f"Fichiers créés dans data/processed/:")
    print(f"  - x.npy: {x[valid_mask].shape}")
    print(f"  - x_len.npy: {x_len[valid_mask].shape}")
    print(f"  - c.npy: {c[valid_mask].shape}")
    print(f"  - c_len.npy: {c_len[valid_mask].shape}")
    print(f"  - w_id.npy: {w_id[valid_mask].shape}")
    print("\nVous pouvez maintenant lancer: python rnn.py")
    print("=" * 70)