import pickle
import torch
import esm
from collections import defaultdict
from tqdm import tqdm
import lmdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
# 1) Load ESM-2 model
# ------------------------------------------------------------
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model = model.to(device)
model.eval()  # disable dropout for deterministic embeddings


# ------------------------------------------------------------
# 2) Load extracted pocket metadata
# ------------------------------------------------------------
with open(
    '/data/project/metadata/crossdocked_pocket_extracted.pkl',
    'rb'
) as f:
    results_all = pickle.load(f)

batch_size = 64


# ------------------------------------------------------------
# Flatten chain-level entries for batching
# Each chain becomes one ESM input sequence
# ------------------------------------------------------------
entries = []
ligand_map = {}  # top_key -> ligand SMILES

for key, (result, ligand_smiles) in tqdm(results_all.items()):
    key_str = key.decode("utf-8") if isinstance(key, (bytes, bytearray)) else str(key)
    ligand_map[key_str] = ligand_smiles

    for ch, d in result.items():
        entries.append({
            "top_key": key_str,
            "chain": ch,
            "seq": d["seq"],
            "res_index": d["res_index"],          # aligns with token sequence (no BOS/EOS)
            "pocket_res_id": d["pocket_res_id"],  # residue IDs to extract (same ID system)
        })


# ------------------------------------------------------------
# Accumulator: collect chain-level pocket reps per top_key
# ------------------------------------------------------------
accum_reps = defaultdict(list)


def process_batch(batch_entries):
    if not batch_entries:
        return

    # Prepare ESM batch
    data = [(f"{e['top_key']}_{e['chain']}", e["seq"]) for e in batch_entries]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        outputs = model(batch_tokens, repr_layers=[33], return_contacts=False)

    # token_representations shape: [B, L, H]
    token_representations = outputs["representations"][33]

    for i, entry in enumerate(batch_entries):
        # Remove BOS and EOS so indexing matches the raw sequence
        L = batch_lens[i].item()
        seq_repr = token_representations[i, 1:L - 1]  # [seq_len, H]

        # res_index may be non-contiguous; build explicit mapping
        res_idx_list = entry["res_index"]
        index_map = {res_idx_list[pos]: pos for pos in range(len(res_idx_list))}

        # Extract pocket residues in the given order
        pocket_positions = [index_map[r] for r in entry["pocket_res_id"]]

        chain_rep = seq_repr[pocket_positions]  # [num_pocket_res, H]

        # Move to CPU for accumulation / saving
        accum_reps[entry["top_key"]].append(chain_rep.detach().cpu())


# ------------------------------------------------------------
# 4) Run ESM over all entries in batches
# ------------------------------------------------------------
cur_batch = []
for e in tqdm(entries):
    cur_batch.append(e)
    if len(cur_batch) == batch_size:
        process_batch(cur_batch)
        cur_batch = []

# Process final partial batch
process_batch(cur_batch)


# ------------------------------------------------------------
# 5) Merge chain-level reps and attach metadata from LMDB
# ------------------------------------------------------------
processed_path = '/data/project/lmdb/crossdocked_processed.lmdb'
db = lmdb.open(
    processed_path,
    map_size=10 * (1024 ** 3),
    create=False,
    subdir=False,
    readonly=True,
    lock=False,
    readahead=False,
    meminit=False,
)

final_dict = {}

for top_key, rep_list in tqdm(accum_reps.items()):
    byte_key = top_key.encode('utf-8')

    info_data = pickle.loads(db.begin().get(byte_key))

    # Concatenate along residue dimension
    # Order: chain order from entries, residues follow pocket_res_id order
    rep = torch.cat(rep_list, dim=0)  # [total_pocket_res, H]

    final_dict[top_key] = {
        "rep": rep,                              # torch.Tensor (CPU)
        "ligand_smiles": ligand_map[top_key],
        "protein_filename": info_data["protein_filename"],
        "ligand_filename": info_data["ligand_filename"],
    }


# If some top_key has no matched pocket residues, you can either skip it
# or insert an empty tensor. Here we keep the original behavior unchanged.


# ------------------------------------------------------------
# 6) Save final representations
# ------------------------------------------------------------
torch.save(
    final_dict,
    '/data/project/output/pocket_ligand.pt'
)
