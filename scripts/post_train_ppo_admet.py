import os
import argparse
import pickle
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from genmol.model import GenMol
from genmol.sampler import Sampler
import itertools
import random
from genmol.utils.utils_chem import safe_to_smiles
from genmol.utils.bracket_safe_converter import bracketsafe2safe
from torch.distributions import Categorical
import requests
from tdc import Oracle

# ======================================================
# Remote ADMET prediction server
# ======================================================

SERVER_URL = "http://0.0.0.0:8000/predict"


def call_minimol(task_id, smiles_list):
    """
    Call external MiniMol service for ADMET property prediction.
    """
    payload = {
        "task_id": task_id,
        "smiles": smiles_list
    }
    resp = requests.post(SERVER_URL, json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data["task_names"], data["predictions"]


def get_scores_admet(smiles, task_id):
    """
    Compute reward scores combining:
    - ADMET prediction scores
    - QED
    - SA score
    """
    scores = []

    oracle_QED = Oracle(name='QED')
    oracle_SA = Oracle(name='SA')

    _, predictions = call_minimol(task_id, smiles)
    qed = oracle_QED(smiles)
    sa = oracle_SA(smiles)

    for i in tqdm(range(len(smiles))):
        prop_score = [preds[i] for preds in predictions.values()]

        # Task-specific reward shaping
        if task_id == 0:
            prop_score = [
                3 * prop_score[0],
                3 * prop_score[1],
                6 * (1 - prop_score[2])
            ]
        elif task_id == 1:
            prop_score = [
                min(0.5 * prop_score[0] ** 2 - 8, 4.5),
                6 * (1 - prop_score[1]) ** 2,
                6 * (1 - prop_score[2]) ** 2
            ]
        elif task_id == 2:
            prop_score = [
                9 - prop_score[0] ** 2,
                1 - prop_score[1],
                6 * (1 - prop_score[2])
            ]

        qed_score = qed[i]
        sa_score = (10 - sa[i]) / 9

        main_score = sum(prop_score) / 3 + 0.1 * (qed_score + sa_score)

        scores.append([
            main_score,
            prop_score[0],
            prop_score[1],
            prop_score[2],
            qed_score,
            sa_score
        ])

    return scores


# ======================================================
# Model utilities
# ======================================================

def load_model_from_path(path):
    """
    Load GenMol model from checkpoint and apply EMA weights if available.
    """
    model = GenMol.load_from_checkpoint(path)
    model.backbone.eval()

    if model.ema:
        model.ema.store(itertools.chain(model.backbone.parameters()))
        model.ema.copy_to(itertools.chain(model.backbone.parameters()))

    return model


def insert_mask(model, x, num_samples, min_add_len=18):
    """
    Insert a block of [MASK] tokens between BOS and EOS.

    The mask length is sampled from a precomputed length distribution,
    with a minimum enforced length.
    """
    with open("/PATH/TO/DATA/len.pk", 'rb') as f:
        seq_len_list = pickle.load(f)

    x = x[0]
    x_new = []

    for _ in range(num_samples):
        add_len = max(random.choice(seq_len_list) - len(x), min_add_len)
        x_new.append(
            torch.hstack([
                x[:-1],
                torch.full((add_len,), model.mask_index, dtype=torch.long),
                x[-1:]
            ])
        )

    pad_len = max(len(xx) for xx in x_new)
    x_new = [
        torch.hstack([
            xx,
            torch.full((pad_len - len(xx),), model.tokenizer.pad_token_id, dtype=torch.long)
        ]) for xx in x_new
    ]

    return torch.stack(x_new)


# ======================================================
# Diffusion generation with trajectory recording
# ======================================================

def generate_with_trajectory_step_conf(
    model,
    x_init,
    softmax_temp=1.0,
    randomness=1.0,
    condition=None,
    fix=True,
):
    """
    Perform diffusion generation using step_confidence.

    During generation, record:
    - state trajectory
    - action trajectory
    - mask transition positions
    - per-step log probabilities

    Returns:
        final_smiles,
        states,
        actions,
        action_masks,
        logprobs,
        logits_traj,
        valid_indices
    """
    device = model.device
    x = x_init.to(device)

    T_max = max(model.mdlm.get_num_steps_confidence(x), 2)

    states, actions, action_masks, logprobs, logits_traj = [], [], [], [], []

    with torch.no_grad():
        for t in range(T_max):
            if not (x == model.mask_index).any():
                break

            x_old = x.clone()
            attention_mask = x_old != model.tokenizer.pad_token_id
            logits = model(x_old, attention_mask, condition=condition)

            logits_traj.append(logits.cpu())

            log_p_x0 = model.mdlm._subs_parameterization(logits, x_old)
            probs = torch.softmax(log_p_x0 / softmax_temp, dim=-1)
            dist = Categorical(probs=probs)

            x_new = model.mdlm.step_confidence(
                logits=logits,
                xt=x_old,
                curr_step=t,
                num_steps=T_max,
                logit_temperature=softmax_temp,
                randomness=randomness,
            )

            changed_mask = (x_old == model.mask_index) & (x_new != model.mask_index)
            logprob = (dist.log_prob(x_new) * changed_mask).sum(-1)

            states.append(x_old.cpu())
            actions.append(x_new.cpu())
            action_masks.append(changed_mask.cpu())
            logprobs.append(logprob.cpu())

            x = x_new

        samples = model.tokenizer.batch_decode(x, skip_special_tokens=True)

        final_smiles, valid_indices = [], []
        for idx, s in enumerate(samples):
            if not s:
                continue

            if model.config.training.get('use_bracket_safe'):
                s = safe_to_smiles(bracketsafe2safe(s), fix=fix)
            else:
                s = safe_to_smiles(s, fix=fix)

            if not s:
                continue

            final_smiles.append(sorted(s.split('.'), key=len)[-1])
            valid_indices.append(idx)

    if not states:
        return [], None, None, None, [], [], []

    return (
        final_smiles,
        torch.stack(states),
        torch.stack(actions),
        torch.stack(action_masks),
        torch.stack(logprobs),
        logits_traj,
        valid_indices
    )


# ======================================================
# PPO update
# ======================================================

def ppo_update(
    model,
    optimizer,
    states,
    actions,
    action_masks,
    old_logprobs,
    rewards,
    condition,
    valid_mask,
    ppo_epochs=1,
    clip_range=0.1,
    entropy_coef=0.001,
    softmax_temp=1.0,
):
    """
    PPO update that only considers positions where
    MASK → non-MASK transitions occurred.
    """
    device = model.device
    T, B, L = actions.shape

    rewards = rewards.to(device)
    valid_mask = valid_mask.to(device)

    valid_rewards = rewards[valid_mask]
    if valid_rewards.numel() == 0:
        return

    adv = torch.zeros(B, device=device)
    adv_valid = (valid_rewards - valid_rewards.mean()) / (valid_rewards.std() + 1e-8)
    adv[valid_mask] = adv_valid

    valid_steps = [t for t in range(T) if action_masks[t].any()]
    if not valid_steps:
        return

    for _ in range(ppo_epochs):
        optimizer.zero_grad()

        for t in valid_steps:
            x_t = states[t].to(device)
            a_t = actions[t].to(device)
            mask_t = action_masks[t].to(device)
            old_lp = old_logprobs[t].to(device)

            attention_mask = x_t != model.tokenizer.pad_token_id
            logits = model(x_t, attention_mask, condition=condition)
            log_p_x0 = model.mdlm._subs_parameterization(logits, x_t)

            probs = torch.softmax(log_p_x0 / softmax_temp, dim=-1)
            dist = Categorical(probs=probs)

            logprob = (dist.log_prob(a_t) * mask_t).sum(-1)
            entropy = (dist.entropy() * mask_t).sum(-1)

            ratio = (logprob - old_lp).exp()
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * adv

            vm = valid_mask.float()
            loss = -(torch.min(surr1, surr2) * vm).sum() / vm.sum()
            loss += entropy_coef * (entropy * vm).sum() / vm.sum()

            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


# ======================================================
# Main training loop
# ======================================================

if __name__ == "__main__":

    task_id = 2

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default="train_ppo_admet_task2")
    parser.add_argument('--ckpt_load_path', type=str,
                        default="/PATH/TO/CHECKPOINTS/admet_task2.ckpt")
    parser.add_argument('--n_steps', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--ppo_epochs', type=int, default=1)
    parser.add_argument('--clip_range', type=float, default=0.1)
    parser.add_argument('--entropy_coef', type=float, default=0.001)
    parser.add_argument('--min_add_len', type=int, default=40)
    parser.add_argument('--softmax_temp', type=float, default=0.5)
    parser.add_argument('--randomness', type=float, default=1.0)

    args = parser.parse_args()

    writer = SummaryWriter("/PATH/TO/LOGS/" + args.run_name)
    writer.add_text("configs", str(args))

    agent = load_model_from_path(args.ckpt_load_path)
    agent.eval()

    optimizer = optim.AdamW(agent.parameters(), lr=args.learning_rate)

    device = "cuda"

    prop_list = [
        [1.0, 1.0, 0.0],
        [5.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0]
    ]

    condition = torch.tensor(prop_list[task_id]).unsqueeze(0).to(device)
    condition = agent.prop_emb_encoder(condition).repeat(args.batch_size, 1).detach()

    for step in tqdm(range(args.n_steps)):
        agent.eval()

        input_x = torch.tensor([[agent.bos_index, agent.eos_index]])
        input_x = insert_mask(agent, input_x, args.batch_size, args.min_add_len).to(device)

        smiles, states, actions, masks, old_logprobs, _, valid_idx = \
            generate_with_trajectory_step_conf(
                agent, input_x,
                args.softmax_temp,
                args.randomness,
                condition
            )

        rewards = torch.zeros(args.batch_size, device=device)
        valid_mask = torch.zeros(args.batch_size, dtype=torch.bool, device=device)

        if smiles:
            scores = get_scores_admet(smiles, task_id)
            for i, b in enumerate(valid_idx):
                rewards[b] = scores[i][0]
                valid_mask[b] = True

        if states is not None:
            agent.train()
            ppo_update(
                agent, optimizer,
                states, actions, masks, old_logprobs,
                rewards, condition, valid_mask,
                args.ppo_epochs,
                args.clip_range,
                args.entropy_coef,
                args.softmax_temp
            )

        if agent.ema:
            agent.ema.update(itertools.chain(agent.backbone.parameters()))

    agent.eval()
    sampler = Sampler("/PATH/TO/CHECKPOINTS/admet_task2.ckpt")
    sampler.model = agent

    samples = sampler.conditional_generation(
        {"input_prop": torch.tensor(prop_list[task_id]).unsqueeze(0)},
        3000,
        softmax_temp=0.5,
        randomness=1.0
    )

    with open("/PATH/TO/RESULTS/admet_task2_post.pkl", "wb") as f:
        pickle.dump(samples, f)
