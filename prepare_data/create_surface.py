import os
import pickle
import lmdb
import subprocess
from tqdm import tqdm
from Bio import PDB
from multiprocessing import Pool, cpu_count


def check_residue_ids_continuity(pdb_file):
    """
    Quick sanity check: make sure residue IDs in each chain go 1-by-1 without gaps.
    (Not used right now, but kept around for debugging.)
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_file)

    for model in structure:
        for chain in model:
            prev_id = None
            for residue in chain:
                cur_id = residue.id[1]
                if prev_id is not None and cur_id != prev_id + 1:
                    return False
                prev_id = cur_id

    return True


def generate_surface(pdb_file, output_dir):
    """
    Generate MSMS surface files for a single PDB.
    Produces: <prefix>.vert (and originally .face, which we delete).
    """
    pdb_name = os.path.basename(pdb_file)
    prefix = pdb_name.split(".")[0]

    xyzrn_file = os.path.join(output_dir, prefix + ".xyzrn")
    out_prefix = os.path.join(output_dir, prefix)

    # pdb_to_xyzrn writes to stdout, so we redirect to a file
    pdb_to_xyzrn_cmd = f"pdb_to_xyzrn {pdb_file} > {xyzrn_file}"
    msms_cmd = ["msms", "-if", xyzrn_file, "-of", out_prefix]

    try:
        subprocess.run(pdb_to_xyzrn_cmd, shell=True, stdout=subprocess.DEVNULL, check=False)
        subprocess.run(msms_cmd, stdout=subprocess.DEVNULL, check=True)
    finally:
        # Clean up temporary/intermediate files if they exist
        if os.path.exists(xyzrn_file):
            os.remove(xyzrn_file)
        face_file = out_prefix + ".face"
        if os.path.exists(face_file):
            os.remove(face_file)

    vert_file = out_prefix + ".vert"
    assert os.path.exists(vert_file), f"Missing output: {vert_file}"


def process_pdb_file(rel_pdb_path):
    """
    Worker function (runs in each process).
    rel_pdb_path is the relative path stored in LMDB: e.g. 'xxxx/yyyy.pdb'
    """
    pdb_path = os.path.join(PDB_ROOT, rel_pdb_path)

    subdir = os.path.dirname(rel_pdb_path)
    out_dir = os.path.join(SURFACE_OUT_ROOT, subdir)
    os.makedirs(out_dir, exist_ok=True)

    # If you want to filter weird structures, re-enable this:
    # if check_residue_ids_continuity(pdb_path):
    generate_surface(pdb_path, out_dir)


if __name__ == "__main__":
    # ---- anonymized paths ----
    LMDB_PATH = "/data/project/lmdb/crossdocked_processed.lmdb"
    PDB_ROOT = "/data/project/pockets"
    SURFACE_OUT_ROOT = "/data/project/pockets_surface"
    # --------------------------

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

    # Pull protein filenames from LMDB once
    pdb_files = []
    with db.begin() as txn:
        keys = list(txn.cursor().iternext(values=False))
        for key in tqdm(keys, desc="Reading LMDB"):
            data = pickle.loads(txn.get(key))
            pdb_files.append(data["protein_filename"])

    # Run surface generation in parallel
    with Pool(processes=cpu_count()) as pool:
        list(tqdm(pool.imap(process_pdb_file, pdb_files), total=len(pdb_files), desc="MSMS"))
