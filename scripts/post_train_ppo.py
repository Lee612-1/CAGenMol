import os
import argparse
import pickle
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from genmol.score import get_scores_subproc
from genmol.model import GenMol
import itertools
import random
from genmol.utils.utils_chem import safe_to_smiles
from genmol.utils.bracket_safe_converter import bracketsafe2safe
from torch.distributions import Categorical


# ======================================================
# Utility functions
# ======================================================

def load_model_from_path(path):
    """
    Load GenMol checkpoint and apply EMA weights if available.
    """
    model = GenMol.load_from_checkpoint(path)
    model.backbone.eval()
    if model.ema:
        model.ema.store(itertools.chain(model.backbone.parameters()))
        model.ema.copy_to(itertools.chain(model.backbone.parameters()))
    return model


def read_coor_data(line):
    """
    Parse a whitespace-separated coordinate line into a 1D float tensor.
    Expected format: x1 y1 z1 x2 y2 z2 ...
    """
    words = line.strip().split()
    tokens = []
    for i in range(0, len(words), 3):
        coor = [float(words[i]), float(words[i + 1]), float(words[i + 2])]
        tokens.extend(coor)
    return torch.tensor(tokens, dtype=torch.float32)


def read_aa_dict():
    """
    Map single-letter amino acids to indices.
    """
    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    aa2index = {}
    for aa in alphabet:
        aa2index[aa] = len(aa2index)
    return aa2index


def read_aa_data(line):
    """
    Parse residue tokens from a line. Each token contains residue and residue id.
    Returns:
        aa_indices: LongTensor [N]
        aa_ids:     LongTensor [N]
    """
    amino_acid_dict = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
    }
    aa2index = read_aa_dict()
    words = line.strip().split()
    aas = [amino_acid_dict[word.strip().split("_")[-3].strip()] for word in words]
    tokens = [aa2index[a] for a in aas]
    aa_id = [int(word.strip().split("_")[-2].strip()) for word in words]
    return torch.tensor(tokens, dtype=torch.long), torch.tensor(aa_id, dtype=torch.long)


def get_surface_aa_feature(pocket_aa):
    """
    Convert pocket amino acids (single-letter) into per-residue physicochemical features.
    Output: FloatTensor [N, 5]
    """
    HYDROPATHY = {
        "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8,
        "W": -0.9, "G": -0.4, "T": -0.7, "S": -0.8, "Y": -1.3, "P": -1.6,
        "H": -3.2, "N": -3.5, "D": -3.5, "Q": -3.5, "E": -3.5, "K": -3.9, "R": -4.5
    }
    CHARGE = {**{'R': 1, 'K': 1, 'D': -1, 'E': -1, 'H': 0.1}, **{x: 0 for x in 'ABCFGIJLMNOPQSTUVWXYZ'}}
    POLARITY = {**{x: 1 for x in 'RNDQEHKSTY'}, **{x: 0 for x in "ACGILMFPWV"}}
    ACCEPTOR = {**{x: 1 for x in 'DENQHSTY'}, **{x: 0 for x in "RKWACGILMFPV"}}
    DONOR = {**{x: 1 for x in 'RKWNQHSTY'}, **{x: 0 for x in "DEACGILMFPV"}}

    def PMAP(x):
        return [HYDROPATHY[x] / 5, CHARGE[x], POLARITY[x], ACCEPTOR[x], DONOR[x]]

    aa_features = [PMAP(aa) for aa in pocket_aa]
    return torch.tensor(aa_features, dtype=torch.float32)


def insert_mask(model, x, num_samples, min_add_len=18):
    """
    Given an initial sequence x = [BOS, EOS], insert a [MASK] span between BOS and EOS.

    The mask span length is sampled from a precomputed length distribution (len.pk),
    with a minimum enforced length (min_add_len).
    """
    with open("/PATH/TO/DATA/len.pk", 'rb') as f:
        seq_len_list = pickle.load(f)

    x = x[0]  # shape: [2]
    x_new = []
    for _ in range(num_samples):
        add_seq_len = max(random.choice(seq_len_list) - len(x), min_add_len)
        x_new.append(
            torch.hstack([
                x[:-1],
                torch.full((add_seq_len,), model.mask_index, dtype=torch.long),
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
    return torch.stack(x_new)  # [B, L]


# ======================================================
# Diffusion generation with trajectory recording (step_confidence)
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
    Step-wise unmasking using model.mdlm.step_confidence.

    During generation:
      - No gradients are kept (torch.no_grad).
      - Each step:
          * record xt as state_t
          * compute logits and sample xt+1 via step_confidence
          * compute action_mask: positions that changed from MASK -> non-MASK
          * compute per-step logprob on changed positions
      - Finally decode to SMILES (keep only valid ones)

    Returns:
      final_smiles:  list of valid SMILES (only valid samples)
      states:        [T, B, L] CPU
      actions:       [T, B, L] CPU
      action_masks:  [T, B, L] bool CPU
      logprobs:      [T, B] CPU
      logits_traj:   list[T] of [B, L, V] CPU
      valid_indices: indices in batch dimension corresponding to valid SMILES
    """
    device = model.device
    x = x_init.to(device)  # [B, L]

    # Determine maximum number of steps
    T_max = max(model.mdlm.get_num_steps_confidence(x), 2)

    states, actions, action_masks, logprobs, logits_traj = [], [], [], [], []

    with torch.no_grad():
        for t in range(T_max):
            # Stop early if no MASK tokens remain
            if not (x == model.mask_index).any():
                break

            x_old = x.clone()

            attention_mask = x_old != model.tokenizer.pad_token_id
            logits = model(x_old, attention_mask, condition=condition)  # [B, L, V]
            logits_traj.append(logits.detach().cpu())

            # Use the same parameterization as step_confidence internally
            log_p_x0 = model.mdlm._subs_parameterization(logits, x_old)
            probs = torch.softmax(log_p_x0 / softmax_temp, dim=-1)
            dist = Categorical(probs=probs)

            # Apply step_confidence to update tokens
            x_new = model.mdlm.step_confidence(
                logits=logits,
                xt=x_old,
                curr_step=t,
                num_steps=T_max,
                logit_temperature=softmax_temp,
                randomness=randomness,
            )

            # Positions that are actually generated at this step
            changed_mask = (x_old == model.mask_index) & (x_new != model.mask_index)  # [B, L]

            # Log-probabilities only on generated positions
            logprob_full = dist.log_prob(x_new)  # [B, L]
            logprob_selected = (logprob_full * changed_mask).sum(-1)  # [B]

            states.append(x_old.detach().cpu())
            actions.append(x_new.detach().cpu())
            action_masks.append(changed_mask.detach().cpu())
            logprobs.append(logprob_selected.detach().cpu())

            x = x_new

        # Decode final tokens into SMILES
        samples = model.tokenizer.batch_decode(x, skip_special_tokens=True)

        final_smiles, valid_indices = [], []
        for idx, s in enumerate(samples):
            if not s:
                continue
            if model.config.training.get('use_bracket_safe'):
                s_proc = safe_to_smiles(bracketsafe2safe(s), fix=fix)
            else:
                s_proc = safe_to_smiles(s, fix=fix)
            if not s_proc:
                continue
            main = sorted(s_proc.split('.'), key=len)[-1]
            final_smiles.append(main)
            valid_indices.append(idx)

    if len(states) == 0:
        # No steps executed
        return [], None, None, None, [], [], []

    states = torch.stack(states, dim=0)          # [T, B, L]
    actions = torch.stack(actions, dim=0)        # [T, B, L]
    action_masks = torch.stack(action_masks, 0)  # [T, B, L]
    logprobs = torch.stack(logprobs, dim=0)      # [T, B]

    return final_smiles, states, actions, action_masks, logprobs, logits_traj, valid_indices


# ======================================================
# PPO update (logprob only on MASK->non-MASK positions)
# ======================================================

def ppo_update(
    model,
    optimizer,
    states,         # [T, B, L] on CPU
    actions,        # [T, B, L] on CPU
    action_masks,   # [T, B, L] on CPU, bool
    old_logprobs,   # [T, B] on CPU
    rewards,        # [B] on device or CPU
    condition,
    valid_mask,     # [B] bool, True indicates valid samples
    ppo_epochs=1,
    clip_range=0.2,
    entropy_coef=0.01,
    softmax_temp=1.0,
):
    """
    PPO update with valid_mask:
    - Only valid samples contribute to the PPO loss.
    - Only positions that changed from MASK -> non-MASK contribute to logprob terms.
    """
    device = model.device
    T, B, L = actions.shape

    valid_mask = valid_mask.to(device)
    if rewards.device != device:
        rewards = rewards.to(device)

    # Advantage computed only on valid samples
    valid_rewards = rewards[valid_mask]
    if valid_rewards.numel() == 0:
        return

    if valid_rewards.std() < 1e-8:
        adv_valid = torch.zeros_like(valid_rewards)
    else:
        adv_valid = (valid_rewards - valid_rewards.mean()) / (valid_rewards.std() + 1e-8)

    adv = torch.zeros(B, device=device)
    adv[valid_mask] = adv_valid

    # Effective steps: only those with any generated positions
    valid_steps = [t for t in range(T) if action_masks[t].any()]
    if not valid_steps:
        return
    num_effective_steps = len(valid_steps)

    for _ in range(ppo_epochs):
        optimizer.zero_grad()

        for t in tqdm(valid_steps):
            mask_t = action_masks[t].to(device)       # [B, L]
            x_t = states[t].to(device)                # [B, L]
            a_t = actions[t].to(device)               # [B, L]
            old_logprob_t = old_logprobs[t].to(device)  # [B]

            attention_mask = x_t != model.tokenizer.pad_token_id
            logits = model(x_t, attention_mask, condition=condition)  # [B, L, V]

            # Use MDLM parameterization if available
            if hasattr(model, "mdlm") and hasattr(model.mdlm, "_subs_parameterization"):
                log_p_x0 = model.mdlm._subs_parameterization(logits, x_t)
            else:
                log_p_x0 = logits

            probs = torch.softmax(log_p_x0 / softmax_temp, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)

            logprob_full = dist.log_prob(a_t)                  # [B, L]
            logprob_selected = (logprob_full * mask_t).sum(-1) # [B]
            entropy = (dist.entropy() * mask_t).sum(-1)        # [B]

            ratio = (logprob_selected - old_logprob_t).exp()   # [B]
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv

            vm = valid_mask.float()
            policy_loss = -(torch.min(surr1, surr2) * vm).sum() / vm.sum()
            entropy_loss = -(entropy * vm).sum() / vm.sum()

            loss = policy_loss + entropy_coef * entropy_loss
            (loss / num_effective_steps).backward()

            # Free GPU memory
            del x_t, a_t, mask_t, logits, log_p_x0, probs, dist
            torch.cuda.empty_cache()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


# ======================================================
# Final generation helper: generate N molecules with final weights
# ======================================================

def generate_n_smiles_final(
    model,
    condition_vec,
    n_target=100,
    batch_size=128,
    min_add_len=40,
    softmax_temp=0.5,
    randomness=0.5,
    fix=True,
    max_rounds=20,
):
    """
    Generate n_target valid SMILES using the final trained model weights.
    If one round yields fewer than needed, run multiple rounds and de-duplicate.
    """
    model.eval()
    device = model.device

    collected = []
    seen = set()

    # condition_vec should be [batch_size, D]
    # We will regenerate/resize it each round as needed.
    for _ in range(max_rounds):
        if len(collected) >= n_target:
            break

        # Prepare BOS/EOS then insert mask
        input_x = torch.hstack([
            torch.full((1, 1), model.bos_index, dtype=torch.long),
            torch.full((1, 1), model.eos_index, dtype=torch.long)
        ])  # [1, 2]
        input_x = insert_mask(model, input_x, batch_size, min_add_len=min_add_len).to(device)

        # Ensure condition size matches batch_size
        cond = condition_vec
        if cond.shape[0] != batch_size:
            if cond.shape[0] == 1:
                cond = cond.repeat(batch_size, 1)
            else:
                cond = cond[:batch_size]

        smiles, *_ = generate_with_trajectory_step_conf(
            model=model,
            x_init=input_x,
            softmax_temp=softmax_temp,
            randomness=randomness,
            condition=cond,
            fix=fix
        )

        for s in smiles:
            if s not in seen:
                seen.add(s)
                collected.append(s)
                if len(collected) >= n_target:
                    break

    return collected[:n_target]


# ======================================================
# Main program
# ======================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Run naming
    parser.add_argument('--run_name', type=str, default="train_ppo_stepconf")

    # Paths (anonymized placeholders)
    parser.add_argument('--data_path', type=str, default="/PATH/TO/DATA/test_set_surf.pkl")
    parser.add_argument('--ckpt_load_path', type=str, default="/PATH/TO/CHECKPOINTS/model.ckpt")
    parser.add_argument('--log_dir', type=str, default="/PATH/TO/LOGS/post_train/log_generation/")
    parser.add_argument('--out_dir', type=str, default="/PATH/TO/OUTPUT/post_train/generated_mols_ppo/")

    # RL training
    parser.add_argument('--n_steps', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=1e-5)

    # PPO
    parser.add_argument('--ppo_epochs', type=int, default=1)
    parser.add_argument('--clip_range', type=float, default=0.2)
    parser.add_argument('--entropy_coef', type=float, default=0.01)

    # Diffusion sampling
    parser.add_argument('--min_add_len', type=int, default=40)
    parser.add_argument('--softmax_temp', type=float, default=0.5)
    parser.add_argument('--randomness', type=float, default=0.5)

    # Final generation
    parser.add_argument('--final_gen_n', type=int, default=100)

    args = parser.parse_args()

    # TensorBoard logging
    writer = SummaryWriter(os.path.join(args.log_dir, args.run_name))
    writer.add_text("configs", str(args))

    # Load conditional dataset
    with open(args.data_path, 'rb') as f:
        test_set = pickle.load(f)
    test_key = list(test_set.keys())

    # You originally looped i in range(53, 62). Keep the same range.
    for i in range(53, 62):
        # Load agent for each target
        agent = load_model_from_path(args.ckpt_load_path)

        # Freeze encoders as in your original script
        for param in agent.rep_emb_encoder.parameters():
            param.requires_grad = False
        for param in agent.surf_emb_encoder.parameters():
            param.requires_grad = False

        agent.eval()

        optimizer = optim.AdamW(
            agent.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )

        device = 'cuda'
        agent.to(device)

        # Build conditioning vectors
        d = test_set[test_key[i]]
        _ = read_coor_data(d['surface']['vertice'])   # kept for parity (not used downstream)
        _ = read_aa_data(d['surface']['atom'])        # kept for parity (not used downstream)

        cond_dict = {}
        cond_dict['surfprop'] = get_surface_aa_feature(d['pocket_aa']).mean(dim=0).unsqueeze(0)
        cond_dict['input_rep'] = d['rep'].unsqueeze(0)

        for k, v in cond_dict.items():
            if isinstance(v, torch.Tensor):
                cond_dict[k] = v.to(device)

        with torch.no_grad():
            surf_rep = cond_dict['surfprop']
            input_rep = cond_dict['input_rep']
            sample_length = torch.tensor([input_rep.shape[1]], device=device)

            input_rep = agent.rep_emb_pooling(input_rep, sample_length)
            cond_vec = agent.rep_emb_encoder(input_rep) + agent.surf_emb_encoder(surf_rep)

            # Training condition: [batch_size, D]
            condition_train = cond_vec.repeat(args.batch_size, 1).detach()

            # Final generation condition: [final_batch_size, D]
            # We'll construct on the fly in generate_n_smiles_final, but keep a base here.
            condition_base = cond_vec.detach()

        # Build receptor file path (anonymized base directory)
        receptor_file = os.path.join(
            "/PATH/TO/RECEPTORS/",
            d['protein_filename'].replace('.pdb', '_fixed2.pdb')
        )
        box_center = d['ligand_center_of_mass']
        data_tag = str(i) + '-' + d['protein_filename'].split('_rec')[0].replace('/', '-')

        # =========================
        # Training loop
        # =========================
        for step in tqdm(range(args.n_steps)):
            agent.eval()

            # 1) Create all-MASK initial sequence
            input_x = torch.hstack([
                torch.full((1, 1), agent.bos_index, dtype=torch.long),
                torch.full((1, 1), agent.eos_index, dtype=torch.long)
            ])  # [1, 2]

            input_x = insert_mask(agent, input_x, args.batch_size, min_add_len=args.min_add_len).to(device)

            # 2) Diffusion generation + trajectory recording
            smiles, states, actions, action_masks, old_logprobs, logits_traj, valid_indices = \
                generate_with_trajectory_step_conf(
                    model=agent,
                    x_init=input_x,
                    softmax_temp=args.softmax_temp,
                    randomness=args.randomness,
                    condition=condition_train,
                    fix=True
                )

            # 3) Compute rewards only for valid SMILES
            rewards = torch.zeros(args.batch_size, dtype=torch.float32, device=device)
            valid_mask = torch.zeros(args.batch_size, dtype=torch.bool, device=device)

            if len(smiles) > 0:
                scores = get_scores_subproc(smiles, receptor_file, box_center)
                main_scores_valid = [s[0] for s in scores]

                writer.add_scalar('Mean score ' + data_tag, float(np.mean(main_scores_valid)), step)

                # Scatter rewards back to batch positions
                for local_idx, bidx in enumerate(valid_indices):
                    rewards[bidx] = main_scores_valid[local_idx]
                    valid_mask[bidx] = True
            else:
                writer.add_scalar('Mean score ' + data_tag, 0.0, step)

            # 4) PPO update (skip if no trajectory)
            if states is not None:
                agent.train()
                ppo_update(
                    model=agent,
                    optimizer=optimizer,
                    states=states,
                    actions=actions,
                    action_masks=action_masks,
                    old_logprobs=old_logprobs,
                    rewards=rewards,
                    condition=condition_train,
                    valid_mask=valid_mask,
                    ppo_epochs=args.ppo_epochs,
                    clip_range=args.clip_range,
                    entropy_coef=args.entropy_coef,
                    softmax_temp=args.softmax_temp,
                )

            # 5) EMA update
            if getattr(agent, "ema", None) is not None:
                agent.ema.update(itertools.chain(agent.backbone.parameters()))

            torch.cuda.empty_cache()

        # =========================
        # Final generation with the last trained weights
        # =========================
        agent.eval()

        gen_smiles = generate_n_smiles_final(
            model=agent,
            condition_vec=condition_base,   # [1, D], will be repeated
            n_target=args.final_gen_n,
            batch_size=max(args.batch_size, args.final_gen_n),
            min_add_len=args.min_add_len,
            softmax_temp=args.softmax_temp,
            randomness=args.randomness,
            fix=True
        )

        # Optionally score the final generated molecules
        final_scores = []
        if len(gen_smiles) > 0:
            try:
                scored = get_scores_subproc(gen_smiles, receptor_file, box_center)
                final_scores = scored
            except Exception:
                final_scores = []

        # Save final outputs
        out_dir = os.path.join(args.out_dir, args.run_name + "_" + data_tag)
        os.makedirs(out_dir, exist_ok=True)

        # Save generated SMILES (and scores if available)
        out_csv = os.path.join(out_dir, f"final_generated_{args.final_gen_n}.csv")
        with open(out_csv, "w") as f:
            if final_scores:
                f.write("smiles,score,docking_scores,qed_scores,sa_scores\n")
                for s, sc in zip(gen_smiles, final_scores):
                    # sc format assumed: [main_score, docking, qed, sa, ...] depending on your implementation
                    # Your earlier code suggests: [main, docking, qed, sa]
                    f.write(f"{s},{sc[0]},{sc[1]},{sc[2]},{sc[3]}\n")
            else:
                f.write("smiles\n")
                for s in gen_smiles:
                    f.write(f"{s}\n")

        # Save final model weights (optional but usually useful)
        out_ckpt = os.path.join(out_dir, "final_model.ckpt")
        try:
            # If using lightning checkpoint saving elsewhere, you can skip this.
            torch.save(agent.state_dict(), os.path.join(out_dir, "final_model_state_dict.pt"))
        except Exception:
            pass

        print(f"[DONE] Target {data_tag}: generated {len(gen_smiles)} molecules -> {out_csv}")
