import os
import pickle
import lmdb
from tqdm import tqdm

# ------------------------------------------------------------
# Open LMDB (read-only) and pull all keys
# ------------------------------------------------------------
LMDB_PATH = "/data/project/lmdb/crossdocked_processed.lmdb"

db = lmdb.open(
    LMDB_PATH,
    map_size=10 * (1024 ** 3),
    create=False,
    subdir=False,
    readonly=True,
    lock=False,
    readahead=False,
    meminit=False,
)

with db.begin() as txn:
    keys = list(txn.cursor().iternext(values=False))


# ------------------------------------------------------------
# Residue mapping / filters
# ------------------------------------------------------------
AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O",
}
WAT = {"HOH", "WAT", "H2O"}


def get_res_index_from_pocket(pocket_path):
    """
    Read pocket PDB and collect residue indices per chain.
    Returns:
        dict: chain_id -> [resseq, ...] (unique, in file order)
    """
    chains_res = {}
    with open(pocket_path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue

            chain = line[21].strip() or "_"
            try:
                resseq = int(line[22:26].strip())
            except ValueError:
                continue

            if chain not in chains_res:
                chains_res[chain] = []
            if resseq not in chains_res[chain]:
                chains_res[chain].append(resseq)

    return chains_res


def extract_seq_and_residx(pdb_path, pocket_res_id):
    """
    Extract per-chain sequence and residue indices from the full receptor PDB,
    restricted to chains that exist in pocket_res_id.

    Returns:
        dict: chain_id -> {
            "seq": str,
            "res_index": [int, ...],
            "pocket_res_id": [...]
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

            # Keep blank altloc or 'A' only
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

            # Record each residue once (ignore insertion codes)
            if resseq not in seen[chain]:
                seen[chain].add(resseq)
                tmp_seq[chain].append(AA3_TO_1[resname])
                tmp_idx[chain].append(resseq)

    for ch in keep_chains:
        out[ch] = {
            "seq": "".join(tmp_seq[ch]),
            "res_index": tmp_idx[ch],
            "pocket_res_id": pocket_res_id[ch],
        }

    return out


# ------------------------------------------------------------
# Build extracted dict: key -> (chain_dict, ligand_smiles)
# ------------------------------------------------------------
POCKET_PDB_ROOT = "/data/project/pockets"          # pocket10 pdbs
RECEPTOR_PDB_ROOT = "/data/project/crossdocked"   # full receptors

results_all = {}

with db.begin() as txn:
    for key in tqdm(keys):
        data = pickle.loads(txn.get(key))

        pocket_file = os.path.join(
            POCKET_PDB_ROOT,
            data["protein_filename"]
        )

        pdb_file = os.path.join(
            RECEPTOR_PDB_ROOT,
            data["protein_filename"].split("_rec")[0] + "_rec.pdb"
        )

        pocket_res_id = get_res_index_from_pocket(pocket_file)
        result = extract_seq_and_residx(pdb_file, pocket_res_id)

        results_all[key] = (result, data["ligand_smiles"])


# ------------------------------------------------------------
# Save output
# ------------------------------------------------------------
output_path = "/data/project/metadata/crossdocked_pocket_extracted.pkl"
with open(output_path, "wb") as f:
    pickle.dump(results_all, f)
