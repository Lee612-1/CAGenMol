from minimol import Minimol
import os
import pickle
import argparse
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
from tqdm import tqdm


# ------------------------------------------------------------
# Task config: (task_type, log_transform_flag)
# ------------------------------------------------------------
benchmark_config = {
    "caco2_wang": ("regression", False),
    "bioavailability_ma": ("binary", False),
    "lipophilicity_astrazeneca": ("regression", False),
    "solubility_aqsoldb": ("regression", False),
    "hia_hou": ("binary", False),
    "pgp_broccatelli": ("binary", False),
    "bbb_martins": ("binary", False),
    "ppbr_az": ("regression", False),
    "vdss_lombardo": ("regression", True),
    "cyp2c9_veith": ("binary", False),
    "cyp2d6_veith": ("binary", False),
    "cyp3a4_veith": ("binary", False),
    "cyp2c9_substrate_carbonmangels": ("binary", False),
    "cyp2d6_substrate_carbonmangels": ("binary", False),
    "cyp3a4_substrate_carbonmangels": ("binary", False),
    "half_life_obach": ("regression", True),
    "clearance_hepatocyte_az": ("regression", True),
    "clearance_microsome_az": ("regression", True),
    "ld50_zhu": ("regression", False),
    "herg": ("binary", False),
    "ames": ("binary", False),
    "dili": ("binary", False),
}


class Model:
    def __init__(self):
        pass

    def predict(mol, task):
        pass

    def predict_batch(mol_lst, task):
        pass


class MiniMol_Model(Model):
    """
    Loads Minimol task heads from disk and provides a tiny predict API.

    Expected checkpoint layout:
        ckpt_dir/<task>/fold1_best.pt
        ckpt_dir/<task>/fold2_best.pt
        ...

    For binary tasks:
        sigmoid is applied on the logits.

    For regression tasks:
        raw output is returned.
    """

    def __init__(
        self,
        tasks,
        ckpt_dir: str = "/data/project/ckpt/ckpt_minimol",
        seeds: list = [1],
        device=None,
    ):
        self.ckpt_dir = ckpt_dir
        self.tasks = tasks
        self.benchmark_config = benchmark_config
        self.device = (
            torch.device(device)
            if isinstance(device, str)
            else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        )

        self.models: dict[tuple[str, int], nn.Module] = {}
        self.seeds_by_task: dict[str, list[int]] = {t: [] for t in self.benchmark_config}

        self.featuriser = Minimol()

        class TaskHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.dense1 = nn.Linear(512, 512)
                self.dense2 = nn.Linear(512, 512)
                self.final_dense = nn.Linear(1024, 1)
                self.bn1 = nn.BatchNorm1d(512)
                self.bn2 = nn.BatchNorm1d(512)
                self.dropout = nn.Dropout(0.10)

            def forward(self, x):
                original_x = x
                x = self.dense1(x)
                x = self.bn1(x)
                x = F.relu(x)
                x = self.dropout(x)

                x = self.dense2(x)
                x = self.bn2(x)
                x = F.relu(x)
                x = self.dropout(x)

                x = torch.cat((x, original_x), dim=1)
                x = self.final_dense(x)  # logits / regression value
                return x

        self._TaskHead = TaskHead

        # Load all requested tasks / folds
        for task in self.tasks:
            for seed in seeds:
                model_path = os.path.join(ckpt_dir, f"{task}/fold{seed}_best.pt")
                if not os.path.exists(model_path):
                    print(f"[MiniMol_Model] missing: {model_path}")
                    continue

                model = TaskHead().to(self.device)
                state = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state)
                model.eval()

                self.models[(task, seed)] = model
                self.seeds_by_task[task].append(seed)

    def create_feature(self, mol_lst):
        # Minimol returns a list of tensors, each is (512,)
        return self.featuriser(list(mol_lst))

    def predict_batch(self, mol_lst, task: str, seed: int = 1, X_lst=None):
        if task not in self.benchmark_config:
            raise ValueError(f"Unknown task '{task}'")

        task_type = self.benchmark_config[task][0]

        # If features are given, we don't need mol strings at all
        if X_lst is None:
            if mol_lst is None:
                raise ValueError("Either mol_lst or X_lst must be provided.")
            X_lst = self.featuriser(list(mol_lst))  # list of (512,) tensors

        X_input = torch.stack(X_lst, dim=0).to(self.device)

        batch_size = 2048
        preds_all = []

        with torch.no_grad():
            key = (task, seed)
            for i in range(0, X_input.size(0), batch_size):
                X_batch = X_input[i : i + batch_size]

                out = self.models[key](X_batch).squeeze()
                if task_type == "binary":
                    out = torch.sigmoid(out)

                preds = out.detach().cpu().tolist()
                if isinstance(preds, float):
                    preds = [preds]
                preds_all.extend(preds)

        return preds_all

    def predict(self, mol: str, task: str, seed: int = 1):
        return float(self.predict_batch([mol], task=task, seed=seed)[0])


def load_smiles(smiles_file=None, smiles_list=None):
    """
    Load SMILES either from a pickle file or from a python list.
    """
    if smiles_list is not None:
        return smiles_list

    if smiles_file is not None:
        with open(smiles_file, "rb") as f:
            return pickle.load(f)

    raise ValueError("Need either --smiles_file or --smiles_list.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    task_id = 2

    # Input SMILES as a pickled python list
    parser.add_argument(
        "--smiles_file",
        type=str,
        default=f"/data/project/input/res_admet_{task_id}_post.pkl",
        help="Pickle file containing a SMILES list",
    )

    # Or pass list directly (string that eval() can parse)
    parser.add_argument(
        "--smiles_list",
        type=str,
        default=None,
        help='SMILES list string, e.g. "[\'CCC\', \'CCO\']"',
    )

    args = parser.parse_args()

    tasks_list = [
        ["hia_hou", "bbb_martins", "ames"],
        ["lipophilicity_astrazeneca", "cyp3a4_substrate_carbonmangels", "dili"],
        ["solubility_aqsoldb", "pgp_broccatelli", "herg"],
        ["herg", "ames", "dili"],
    ]

    task_names = tasks_list[task_id]
    model = MiniMol_Model(task_names)

    smiles_list = None
    if args.smiles_list is not None:
        smiles_list = eval(args.smiles_list)

    smiles_list = load_smiles(smiles_file=args.smiles_file, smiles_list=smiles_list)
    smiles_list = list(set(smiles_list))  # cheap dedup

    # Build features once and reuse across tasks
    X_lst = model.create_feature(smiles_list)

    pred_list = []
    for task in tqdm(task_names):
        pred = model.predict_batch(mol_lst=None, task=task, X_lst=X_lst)
        pred_list.append(pred)

    print(pred_list)

    output_path = f"/data/project/output/evaluation_results_admet_{task_id}_post.csv"
    df = pd.DataFrame(list(zip(*pred_list)), columns=task_names)
    df.to_csv(output_path, index=False)
