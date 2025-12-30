import os
import sys
sys.path.append(os.path.realpath("."))

from time import time
import pickle
import multiprocessing

import pandas as pd
from tdc import Oracle, Evaluator
from tqdm import tqdm

from score import docking


# ------------------------------------------------------------
# Basic property oracles / diversity evaluator (TDC)
# ------------------------------------------------------------
evaluator = Evaluator("diversity")
oracle_qed = Oracle("qed")
oracle_sa = Oracle("sa")


# ------------------------------------------------------------
# Load generated molecules
# Expected structure: list of dicts, each has r["res"] as a smiles list
# ------------------------------------------------------------
with open("/data/project/input/res_surfprop_attn_44.pkl", "rb") as f:
    res = pickle.load(f)

samples = []
for r in res:
    samples.extend(r["res"])


# ------------------------------------------------------------
# Compute QED / SA and basic stats
# Note: SA in TDC is usually higher = worse; rescale to [0,1] with (10-sa)/9
# ------------------------------------------------------------
df = pd.DataFrame({
    "smiles": samples,
    "qed": oracle_qed(samples),
    "sa": oracle_sa(samples),
})
df["sa"] = (10 - df["sa"]) / 9

df2 = df.drop_duplicates("smiles")
print(f"Diversity:\t{evaluator(df2['smiles'])}")
print(f"QED mean: {df['qed'].mean():.4f}, median: {df['qed'].median():.4f}")
print(f"SA mean: {df['sa'].mean():.4f}, median: {df['sa'].median():.4f}")


# ------------------------------------------------------------
# Optional multiprocessing wrapper (not used below)
# ------------------------------------------------------------
def get_scores(smiles, receptor_file, box_center, n_process=25):
    smiles_groups = []
    group_size = len(smiles) / n_process
    for i in range(n_process):
        smiles_groups.append(smiles[int(i * group_size): int((i + 1) * group_size)])

    temp_data = []
    pool = multiprocessing.Pool(processes=n_process)
    for index in range(n_process):
        temp_data.append(
            pool.apply_async(
                get_scores_subproc,
                args=(smiles_groups[index], receptor_file, box_center),
            )
        )
    pool.close()
    pool.join()

    scores = []
    for index in range(n_process):
        scores += temp_data[index].get()
    return scores


def get_scores_subproc(smiles, receptor_file, box_center):
    scores = []
    for i in tqdm(range(len(smiles))):
        docking_score = docking(smiles[i], receptor_file=receptor_file, box_center=box_center)
        scores.append(docking_score)
    return scores


# ------------------------------------------------------------
# Run docking per target and collect all scores in one list
# Assumes receptor files are already "fixed2" versions
# ------------------------------------------------------------
all_vina = []
for r in tqdm(res):
    smiles = r["res"]

    receptor_file = os.path.join(
        "/data/project/",
        r["protein_filename"].replace(".pdb", "_fixed2.pdb"),
    )
    box_center = r["ligand_center_of_mass"]

    # Current behavior: run in this process (no multiprocessing)
    vina_dock = get_scores_subproc(smiles, receptor_file, box_center)

    v = pd.Series(vina_dock)
    print(
        f'Vina docking for {r["protein_filename"]}: '
        f"mean={v.mean():.4f}, median={v.median():.4f}"
    )

    all_vina.extend(vina_dock)

df["vina"] = all_vina


# ------------------------------------------------------------
# Success definition (keep thresholds exactly as-is)
# ------------------------------------------------------------
success_condition = (df["qed"] >= 0.25) & (df["sa"] >= 0.59) & (df["vina"] <= -8.18)
success_rate = success_condition.mean()

print(f"\nOverall Vina mean: {df['vina'].mean():.4f}, median: {df['vina'].median():.4f}")
print(f"Success rate: {success_rate * 100:.2f}%")


# ------------------------------------------------------------
# Save per-molecule evaluation table
# ------------------------------------------------------------
output_path = "/data/project/output/evaluation_results_surfprop_attn_44.csv"
df.to_csv(output_path, index=False)


