import os
import pickle
import lmdb
import torch
from tqdm import tqdm

# ------------------------------------------------------------
# Load precomputed training data (surface representations etc.)
# ------------------------------------------------------------
train_data = torch.load('/data/project/pocket_ligand_surface.pt')
train_keys = list(train_data.keys())


# ------------------------------------------------------------
# Compute residue-level centers for pocket residues from a PDB
# Notes:
#   - Only ATOM / HETATM records are considered
#   - Output order strictly follows pocket_res_id for each chain
# ------------------------------------------------------------
def get_pocket_res_centers(pocket_path, pocket_res_id):
    """
    Compute mean atomic coordinates for each pocket residue.

    Returns:
        dict:
            chain_id -> list of [x, y, z] (same order as pocket_res_id[chain_id])
    """

    # Accumulator:
    # chain -> resseq -> [sum_x, sum_y, sum_z, atom_count]
    accum = {ch: {} for ch in pocket_res_id}

    with open(pocket_path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue

            chain = line[21].strip() or "_"
            if chain not in pocket_res_id:
                continue

            try:
                resseq = int(line[22:26].strip())
            except ValueError:
                continue

            # Only keep residues that are part of the pocket
            if resseq not in pocket_res_id[chain]:
                continue

            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue

            if resseq not in accum[chain]:
                accum[chain][resseq] = [0.0, 0.0, 0.0, 0]

            accum[chain][resseq][0] += x
            accum[chain][resseq][1] += y
            accum[chain][resseq][2] += z
            accum[chain][resseq][3] += 1

    # Compute mean coordinates, keeping the original residue order
    centers = {}
    for ch in pocket_res_id:
        centers[ch] = []
        for resseq in pocket_res_id[ch]:
            if resseq in accum[ch] and accum[ch][resseq][3] > 0:
                sx, sy, sz, cnt = accum[ch][resseq]
                centers[ch].append([sx / cnt, sy / cnt, sz / cnt])
            else:
                # Should be rare; keep placeholder for alignment
                centers[ch].append(None)

    return centers


# ------------------------------------------------------------
# Load pocket / chain / residue metadata
# ------------------------------------------------------------
with open('/data/project/metadata/crossdocked_pocket_info.pkl', 'rb') as f:
    results_all = pickle.load(f)


# ------------------------------------------------------------
# Open processed LMDB (read-only)
# ------------------------------------------------------------
db = lmdb.open(
    '/data/project/lmdb/crossdocked_processed.lmdb',
    map_size=10 * (1024 ** 3),
    create=False,
    subdir=False,
    readonly=True,
    lock=False,
    readahead=False,
    meminit=False,
)


# ------------------------------------------------------------
# Main loop:
#   - compute pocket residue centers
#   - attach residue types
#   - load ligand center of mass
# ------------------------------------------------------------
for k in tqdm(train_keys):
    pocket_coords = []
    pocket_aa = []

    chains = results_all[k.encode()][0]

    # Collect pocket residue indices per chain
    pocket_res_id = {
        chain_id: chains[chain_id]["pocket_res_id"]
        for chain_id in chains
    }

    pocket_file = os.path.join(
        '/data/project/pockets',
        train_data[k]['protein_filename']
    )

    pocket_centers = get_pocket_res_centers(pocket_file, pocket_res_id)

    for chain_id in chains:
        pocket_coords.extend(pocket_centers[chain_id])

        seq = chains[chain_id]["seq"]
        res_idx_list = chains[chain_id]["res_index"]

        # Map residue index to sequence position
        index_map = {res_idx_list[i]: i for i in range(len(res_idx_list))}

        pocket_aa.extend(
            [seq[index_map[r]] for r in chains[chain_id]["pocket_res_id"]]
        )

    pocket_coords = torch.tensor(pocket_coords)

    # Sanity checks: must align with existing representations
    assert len(pocket_coords) == len(train_data[k]['rep'])
    assert len(pocket_aa) == len(train_data[k]['rep'])

    train_data[k]['pocket_coords'] = pocket_coords
    train_data[k]['pocket_aa'] = pocket_aa

    # Load ligand center of mass from LMDB
    d = pickle.loads(db.begin().get(k.encode()))
    train_data[k]['ligand_center_of_mass'] = d['ligand_center_of_mass']


# ------------------------------------------------------------
# Merge test set samples into the same dictionary
# ------------------------------------------------------------
with open('/data/project/splits/test_set.pkl', 'rb') as f:
    test_set_dict = pickle.load(f)

for k, v in test_set_dict.items():
    train_data[k] = v


# ------------------------------------------------------------
# Compute distance-based weights w.r.t. ligand center of mass
# ------------------------------------------------------------
def distance_weights(points, center, eps=1e-6, p=2):
    """
    Assign higher weights to pocket points closer to the ligand center.
    """
    diff = points - center.unsqueeze(0)
    dist = torch.norm(diff, dim=1)

    inv = 1.0 / ((dist + eps) ** p)
    return inv / inv.sum()


for k in tqdm(train_data.keys()):
    pocket_coords = train_data[k]['pocket_coords']
    ligand_center = torch.tensor(train_data[k]['ligand_center_of_mass'])

    train_data[k]['dist_weights'] = distance_weights(
        pocket_coords, ligand_center
    )


# ------------------------------------------------------------
# Save final dataset
# ------------------------------------------------------------
torch.save(
    train_data,
    '/data/project/output/pocket_ligand_surface_final.pt'
)
