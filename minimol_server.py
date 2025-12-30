# minimol_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from threading import Lock

from minimol_wrapper import MiniMol_Model

app = FastAPI()

# Task bundles used in your previous scripts (task_id selects one bundle)
TASKS_LIST = [
    ["hia_hou", "bbb_martins", "ames"],
    ["lipophilicity_astrazeneca", "cyp3a4_substrate_carbonmangels", "dili"],
    ["solubility_aqsoldb", "pgp_broccatelli", "herg"],
    ["herg", "ames", "dili"],
]

# Cache models so we don't reload checkpoints on every request
MODEL_CACHE: Dict[int, MiniMol_Model] = {}
_CACHE_LOCK = Lock()


def get_model(task_id: int) -> MiniMol_Model:
    """
    Lazy-load a model for the given task bundle and keep it in memory.
    A small lock is enough here to avoid duplicate loads under concurrency.
    """
    if task_id < 0 or task_id >= len(TASKS_LIST):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task_id={task_id}. Valid range: 0..{len(TASKS_LIST)-1}",
        )

    with _CACHE_LOCK:
        if task_id not in MODEL_CACHE:
            task_names = TASKS_LIST[task_id]
            print(f"[MiniMol_Server] loading task_id={task_id}, tasks={task_names}")
            MODEL_CACHE[task_id] = MiniMol_Model(task_names)

    return MODEL_CACHE[task_id]


class PredictRequest(BaseModel):
    task_id: int
    smiles: List[str]


class PredictResponse(BaseModel):
    task_names: List[str]
    predictions: Dict[str, List[float]]  # task -> list of predictions (same order as input smiles)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Input:
        task_id + a batch of SMILES
    Output:
        predictions for every task in the selected task bundle
    """
    model = get_model(req.task_id)
    task_names = TASKS_LIST[req.task_id]

    # Featurize once, reuse for all heads
    X_lst = model.create_feature(req.smiles)

    preds_dict: Dict[str, List[float]] = {}
    for task in task_names:
        # predict_batch can run directly from precomputed features
        preds_dict[task] = model.predict_batch(mol_lst=None, task=task, X_lst=X_lst)

    return PredictResponse(task_names=task_names, predictions=preds_dict)
