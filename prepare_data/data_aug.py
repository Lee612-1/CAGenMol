from genmol.score import docking
from tqdm import tqdm
import os
import torch
import pickle

# ------------------------------------------------------------
# Load dataset containing pocket / ligand information
# ------------------------------------------------------------
train_data = torch.load(
    '/data/project/pocket_ligand_surface_2.pt',
    weights_only=False
)
train_keys = list(train_data.keys())


# ------------------------------------------------------------
# Load existing docking scores (for resuming interrupted runs)
# ------------------------------------------------------------
docking_scores = {}
docking_scores['error_key'] = []

with open('/data/project/docking_scores.pkl', 'rb') as f:
    saved_scores = pickle.load(f)

# Number of already-processed entries
saved_len = len(saved_scores) - 1


# ------------------------------------------------------------
# Main loop: run docking for remaining samples
# ------------------------------------------------------------
for i, k in enumerate(tqdm(train_keys[saved_len:])):

    # Handle test samples separately (pre-fixed receptors)
    if k.startswith('t_'):
        receptor_file = os.path.join(
            '/data/project/',
            train_data[k]['protein_filename'].replace('.pdb', '_fixed2.pdb')
        )

    # Training / crossdocked samples
    else:
        in_path = os.path.join(
            '/data/project/crossdocked',
            train_data[k]['protein_filename'].split('_rec')[0] + '_rec.pdb'
        )

        out_path = in_path.replace('.pdb', '_fixed2.pdb')

        # Rewrite PDB to fix formatting issues (skip HETATM records)
        with open(in_path, "r", encoding="utf-8", errors="ignore") as fin, \
             open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.rstrip()
                # PDB record type is stored in the first 6 columns
                if not line.startswith("HETATM"):
                    fixed_line = line[:75] + '  ' + line[77:] + '\n'
                    fout.write(fixed_line)

        receptor_file = out_path

    # Docking setup
    box_center = train_data[k]['ligand_center_of_mass'].tolist()
    smiles = train_data[k]['ligand_smiles']

    docking_score = docking(
        smiles,
        receptor_file=receptor_file,
        box_center=box_center
    )

    # Docking failure is encoded as score == 99.9
    if docking_score == 99.9:
        print(k, smiles, receptor_file, box_center)
        docking_scores[k] = 99.9
        docking_scores['error_key'].append(k)
    else:
        docking_scores[k] = docking_score

    # Periodically save intermediate results
    if i % 1000 == 0:
        with open('/data/project/docking_scores.pkl', 'wb') as f:
            pickle.dump(docking_scores, f)


# ------------------------------------------------------------
# Final save
# ------------------------------------------------------------
with open('/data/project/docking_scores.pkl', 'wb') as f:
    pickle.dump(docking_scores, f)


    