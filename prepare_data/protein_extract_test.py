import os
import pickle
import lmdb
from tqdm import tqdm

# ------------------------------------------------------------
# Basic residue mappings / filters
# ------------------------------------------------------------
AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O"
}
WAT = {"HOH", "WAT", "H2O"}


def get_res_index_from_pocket(pocket_path):
    """
    Read a pocket PDB and collect residue indices per chain.
    Returns:
        dict: chain_id -> [resseq1, resseq2, ...] (unique, in file order)
    """
    chains_res = {}
    with open(pocket_path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue

            chain = line[21].strip() or "_"
            resseq_str = line[22:26].strip()
            try:
                resseq = int(resseq_str)
            except ValueError:
                continue

            if chain not in chains_res:
                chains_res[chain] = []
            if resseq not in chains_res[chain]:
                chains_res[chain].append(resseq)

    return chains_res


def get_pocket_res_centers(pocket_path, pocket_res_id):
    """
    Compute mean coordinates for each pocket residue.
    Output order strictly follows pocket_res_id[chain].

    Returns:
        dict: chain_id -> [[x, y, z], ...]  (or None placeholders)
    """
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

            # Only consider residues that are in pocket_res_id
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

    # Convert sums into means, preserving pocket_res_id ordering
    centers = {}
    for ch in pocket_res_id:
        centers[ch] = []
        for resseq in pocket_res_id[ch]:
            if resseq in accum[ch] and accum[ch][resseq][3] > 0:
                sx, sy, sz, cnt = accum[ch][resseq]
                centers[ch].append([sx / cnt, sy / cnt, sz / cnt])
            else:
                centers[ch].append(None)

    return centers


def extract_seq_and_residx(pdb_path, pocket_res_id):
    """
    Extract chain sequences and residue indices from a full protein PDB,
    restricted to chains that appear in pocket_res_id.

    Returns:
        dict: chain_id -> {
            "seq": str,
            "res_index": [int, ...],
            "pocket_res_id": [...],
        }
    """
    keep_chains = list(pocket_res_id.keys())

    out = {}
    seen = {ch: set() for ch in keep_chains}
    tmp_seq = {ch: [] for ch in keep_chains}
    tmp_idx = {ch: [] for ch in keep_chains}

    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue

            # Keep only blank altloc or 'A'
            alt = line[16].strip()
            if alt not in ("", "A"):
                continue

            resname = line[17:20].strip().upper()
            if resname in WAT or resname not in AA3_TO_1:
                continue

            chain = line[21].strip() or "_"
            if chain not in keep_chains:
                continue

            try:
                resseq = int(line[22:26].strip())
            except ValueError:
                continue

            # Only record once per resseq (ignore insertion codes)
            if resseq not in seen[chain]:
                seen[chain].add(resseq)
                tmp_seq[chain].append(AA3_TO_1[resname])
                tmp_idx[chain].append(resseq)

    for ch in keep_chains:
        out[ch] = {
            "seq": "".join(tmp_seq[ch]),
            "res_index": tmp_idx[ch],
            "pocket_res_id": pocket_res_id[ch],
            # pocket_center is filled in the main loop
        }

    return out


# ------------------------------------------------------------
# Build extracted test data: (per-chain info, ligand SMILES)
# ------------------------------------------------------------
results_all = {}

with open('/data/project/splits/test_set_surf_2.pkl', 'rb') as f:
    test_set = pickle.load(f)

for key in tqdm(list(test_set.keys())):
    data = test_set[key]

    # Pocket PDB (derived from ligand filename)
    pocket_file = os.path.join(
        '/data/project/pockets',
        data['ligand_filename']
            .replace('test_set/test_set/', '')
            .replace('.sdf', '_pocket10.pdb')
    )

    # Full receptor PDB (derived from protein filename)
    pdb_file = os.path.join(
        '/data/project/crossdocked',
        data['protein_filename']
            .replace('test_set/test_set/', '')
            .split('_rec')[0] + '_rec.pdb'
    )

    # Pocket residue indices per chain
    pocket_res_id = get_res_index_from_pocket(pocket_file)

    # Per-residue mean coordinates inside the pocket
    pocket_centers = get_pocket_res_centers(pocket_file, pocket_res_id)

    # Sequence + residue index extraction from the full receptor PDB
    result = extract_seq_and_residx(pdb_file, pocket_res_id)

    # Attach pocket centers to each chain entry
    for ch in pocket_res_id:
        if ch in result:
            result[ch]["pocket_center"] = pocket_centers.get(ch, [])

    # Store: (chain dict, ligand smiles)
    results_all[key] = (result, data['ligand_smiles'])


# ------------------------------------------------------------
# Save extracted structure
# ------------------------------------------------------------
output_path = '/data/project/metadata/crossdocked_pocket_extracted_test.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(results_all, f)

