from tqdm import tqdm
from utils_surface import extract_vertix, octree, get_voxel_dict
import numpy as np
import random
import torch
import safe as sf

# 3-letter to 1-letter mapping (only keep standard residues)
amino_acid_dict = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
}

# ------------------------------------------------------------
# Surface extraction / cleanup
#   - load pocket-level embeddings + metadata
#   - attach SAFE strings for ligands
#   - read MSMS .vert surface, normalize, optionally downsample
#   - sort vertices by residue index and store as strings
# ------------------------------------------------------------
data_path = "/data/project/output/pocket_ligand.pt"
data = torch.load(data_path)

# Important: we delete bad entries, so iterate over a static key list
keys = list(data.keys())

for k in tqdm(keys):
    try:
        protein_filename = data[k]["protein_filename"]
        smiles = data[k]["ligand_smiles"]

        # Convert SMILES to SAFE (stereo ignored on purpose)
        safe_str = sf.SAFEConverter(ignore_stereo=True).encoder(smiles, allow_empty=True)
        data[k]["ligand_safe"] = safe_str

        # Surface file produced by MSMS (.vert)
        surface_file = (
            "/data/project/pockets_surface/"
            + protein_filename.replace(".pdb", ".vert")
        )

        vertices, atoms = extract_vertix(surface_file)
        if len(vertices) <= 1:
            raise ValueError(f"Surface file {surface_file} has no vertices.")

        # Normalize: center and scale by max span (simple global normalization)
        center = np.mean(vertices, axis=0)
        max_ = np.max(vertices, axis=0)
        min_ = np.min(vertices, axis=0)
        length = np.max(max_ - min_)
        vertices = (vertices - center) / length

        # Convert to line format: "x y z atom_label"
        lines = []
        for vertex, atom in zip(vertices, atoms):
            line = " ".join([str(term) for term in vertex])
            line = line + " " + atom + "\n"
            lines.append(line)

        # If the surface is too dense, downsample roughly to 5k points
        # The octree/voxel split keeps some spatial coverage.
        if len(vertices) > 5000:
            all_lines = lines
            ids = octree(np.array(vertices))
            voxel_dict = get_voxel_dict(ids, all_lines)

            total_points = len(all_lines)
            ratio = min(1.0, float(5000) / total_points)

            lines = []
            for voxel_id in voxel_dict:
                points = voxel_dict[voxel_id]
                number = int(len(points) * ratio)
                samples = random.sample(points, number)
                lines.extend(samples)

        # Parse residue info from atom labels and keep only standard residues
        # vert_dict maps line -> residue index so we can sort later.
        vert_dict = {}
        for line in lines:
            phrases = line.strip().split()
            aa = phrases[-1].split("_")[1]
            index = int(phrases[-1].split("_")[2])

            if aa not in amino_acid_dict:
                continue

            coor = np.array([float(phrases[0]), float(phrases[1]), float(phrases[2])])
            if np.any(np.isnan(coor)):
                continue

            vert_dict[line] = index

        # Sort vertices by residue index (stable ordering for serialization)
        new_lines = sorted(vert_dict.items(), key=lambda item: item[1])

        coor = []
        atom = []
        for item in new_lines:
            line = item[0]
            phrases = line.strip().split()
            coor.extend(phrases[:3])
            atom.append(phrases[-1])

        # Store as space-separated strings (compact, easy to dump)
        coor = " ".join(coor)
        atom = " ".join(atom)

        data[k]["surface"] = {"vertice": coor, "atom": atom}

    except Exception:
        print(f"Error processing {k}, removing it from dataset.")
        del data[k]

torch.save(data, "/data/project/output/pocket_ligand_surface.pt")
