# minimol_wrapper.py
from minimol import Minimol
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import List, Dict, Tuple

benchmark_config = {
    'caco2_wang': ('regression', False),
    'bioavailability_ma': ('binary', False),
    'lipophilicity_astrazeneca': ('regression', False),
    'solubility_aqsoldb': ('regression', False),
    'hia_hou': ('binary', False),
    'pgp_broccatelli': ('binary', False),
    'bbb_martins': ('binary', False),
    'ppbr_az': ('regression', False),
    'vdss_lombardo': ('regression', True),
    'cyp2c9_veith': ('binary', False),
    'cyp2d6_veith': ('binary', False),
    'cyp3a4_veith': ('binary', False),
    'cyp2c9_substrate_carbonmangels': ('binary', False),
    'cyp2d6_substrate_carbonmangels': ('binary', False),
    'cyp3a4_substrate_carbonmangels': ('binary', False),
    'half_life_obach': ('regression', True),
    'clearance_hepatocyte_az': ('regression', True),
    'clearance_microsome_az': ('regression', True),
    'ld50_zhu': ('regression', False),
    'herg': ('binary', False),
    'ames': ('binary', False),
    'dili': ('binary', False)
}


class MiniMol_Model:
    """
    只初始化一次，把所有 task / fold 的头都 load 进来，
    后面就可以反复调用 predict/predict_batch 而不用重新加载参数。
    """
    def __init__(
        self,
        tasks,
        ckpt_dir: str = '/hpc2ssd/JH_DATA/spooler/yli106/myproject/genmol/ckpt_minimol',
        seeds: list = [1],
        device=None
    ):
        self.ckpt_dir = ckpt_dir
        self.tasks = tasks
        self.benchmark_config = benchmark_config
        self.device = (
            torch.device(device)
            if isinstance(device, str)
            else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        )
        self.models: Dict[Tuple[str, int], nn.Module] = {}
        self.seeds_by_task: Dict[str, list] = {t: [] for t in self.benchmark_config}
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
                x = self.final_dense(x)
                return x   # raw logits / value

        self._TaskHead = TaskHead

        # 只在初始化时 load 一次
        for task in self.tasks:
            for seed in seeds:
                model_path = os.path.join(ckpt_dir, f'{task}/fold{seed}_best.pt')
                if not os.path.exists(model_path):
                    print(f'[MiniMol_Model] {model_path} not found')
                    continue
                model = TaskHead().to(self.device)
                state = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state)
                model.eval()
                self.models[(task, seed)] = model
                self.seeds_by_task[task].append(seed)

    def create_feature(self, mol_lst: List[str]):
        X_lst = self.featuriser(list(mol_lst))
        return X_lst

    def predict_batch(self, mol_lst, task: str, seed: int = 1, X_lst=None):
        if task not in self.benchmark_config:
            raise ValueError(f"Unknown task '{task}'")

        task_type = self.benchmark_config[task][0]

        # 可以预先算好 feature 传进来，避免重复 featuriser 调用
        if X_lst is None:
            X_lst = self.featuriser(list(mol_lst))  # (N, 512)

        X_input = torch.stack(X_lst, dim=0).to(self.device)
        batch_size = 2048
        preds_all = []

        key = (task, seed)
        if key not in self.models:
            raise ValueError(f"Model for (task={task}, seed={seed}) not loaded")

        with torch.no_grad():
            for i in range(0, X_input.size(0), batch_size):
                X_batch = X_input[i:i + batch_size]
                out = self.models[key](X_batch).squeeze()

                if task_type == 'binary':
                    out = torch.sigmoid(out)

                preds = out.detach().cpu().tolist()
                if isinstance(preds, float):
                    preds = [preds]
                preds_all.extend(preds)

        return preds_all

    def predict(self, mol: str, task: str, seed: int = 1):
        return float(self.predict_batch([mol], task=task, seed=seed)[0])
