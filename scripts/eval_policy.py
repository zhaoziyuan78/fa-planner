import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
from fa_planner.envs.windynav import WindyNavEnv
from fa_planner.models.vqvae import VQVAE
from fa_planner.models.state_prior import StatePrior
from fa_planner.models.action_prior import ActionPrior
from fa_planner.models.adapter import Adapter
from fa_planner.models.line_adapter import LineAdapter
from fa_planner.models.scratch_policy import ScratchPolicy
from fa_planner.models.state_only import StateOnlyPolicy
from fa_planner.utils.config import load_config
from fa_planner.utils.action import action_token_to_continuous
from fa_planner.utils.vision import image_to_tensor


def stopping_distance(v, a_max, gamma, dt):
    dist = 0.0
    v_curr = max(v, 0.0)
    while v_curr > 1e-4:
        v_curr = gamma * v_curr - a_max * dt
        if v_curr < 0.0:
            v_curr = 0.0
        dist += v_curr * dt
    return dist


def line_to_goal_action(p, v, goal, direction, a_max, v_max, gamma, dt):
    remaining = float(np.dot(goal - p, direction))
    v_par = float(np.dot(v, direction))
    if remaining <= 0.0:
        if abs(v_par) < 1e-3:
            a_par = 0.0
        else:
            a_par = -np.sign(v_par) * a_max
    else:
        stop_dist = stopping_distance(v_par, a_max, gamma, dt)
        if stop_dist >= remaining:
            a_par = -a_max
        else:
            a_par = a_max
    return np.clip(a_par * direction, -a_max, a_max)


def discretize_action(action, bins, a_max):
    edges = np.linspace(-a_max, a_max, bins)
    ix = np.argmin(np.abs(action[0] - edges))
    iy = np.argmin(np.abs(action[1] - edges))
    return ix * bins + iy


def build_action_context_sequence(ego_tokens, action_tokens, codebook_size):
    seq = []
    for t in range(len(ego_tokens)):
        seq.extend(ego_tokens[t].reshape(-1).tolist())
        if t < len(action_tokens):
            seq.append(codebook_size + int(action_tokens[t]))
    return torch.tensor(seq, dtype=torch.long)


def run_episode(env, vqvae, model_bundle, cfg, device, model_type):
    L = cfg["action_prior"]["L"]
    action_bins = cfg["action_prior"]["action_bins"]
    a_max = cfg["env"]["a_max"]
    codebook = cfg["tokenizer"]["codebook_size"]
    mid = action_bins // 2
    zero_action_id = mid * action_bins + mid

    world_tokens = []
    ego_tokens = []
    action_tokens = []
    traj = []
    actions = []
    winds = []
    frames = []

    env.reset()

    start_pos = env.state[:2].copy()
    for t in range(env.T):
        world = env.render_world(show_goal=True)
        ego = env.render_ego(world)
        frames.append(world)

        world_t = vqvae.encode(image_to_tensor(world).unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
        ego_t = vqvae.encode(image_to_tensor(ego).unsqueeze(0).to(device)).squeeze(0).cpu().numpy()

        world_tokens.append(world_t)
        ego_tokens.append(ego_t)
        if len(world_tokens) > L:
            world_tokens = world_tokens[-L:]
            ego_tokens = ego_tokens[-L:]
            action_tokens = action_tokens[-L:]

        if len(world_tokens) < L:
            pad = [world_tokens[0]] * (L - len(world_tokens))
            world_seq = pad + world_tokens
            ego_seq = [ego_tokens[0]] * (L - len(ego_tokens)) + ego_tokens
        else:
            world_seq = world_tokens[-L:]
            ego_seq = ego_tokens[-L:]
        if len(action_tokens) < L - 1:
            pad_actions = [zero_action_id] * (L - 1 - len(action_tokens)) + action_tokens
        else:
            pad_actions = action_tokens[-(L - 1):]

        if model_type == "scratch":
            tokens = torch.from_numpy(np.stack(world_seq, axis=0)).unsqueeze(0).to(device)
            goal = torch.from_numpy(env.goal).unsqueeze(0).to(device)
            logits = model_bundle["scratch"](tokens, goal)
            action_id = torch.argmax(logits, dim=-1).item()
        elif model_type == "state_only":
            tokens = torch.from_numpy(np.stack(world_seq, axis=0)).unsqueeze(0).to(device)
            summary = model_bundle["state"].hidden_summary(tokens)
            logits = model_bundle["state_only"](summary)
            action_id = torch.argmax(logits, dim=-1).item()
        elif model_type == "action_only":
            seq = build_action_context_sequence(ego_seq, pad_actions, codebook)
            logits, _ = model_bundle["action"](seq.unsqueeze(0).to(device))
            action_logits = logits[:, -1, codebook:]
            action_id = torch.argmax(action_logits, dim=-1).item()
        elif model_type == "full":
            tokens = torch.from_numpy(np.stack(world_seq, axis=0)).unsqueeze(0).to(device)
            summary = model_bundle["state"].hidden_summary(tokens)
            goal = torch.from_numpy(env.goal).unsqueeze(0).to(device)
            seq = build_action_context_sequence(ego_seq, pad_actions, codebook)
            logits, hidden = model_bundle["action"](seq.unsqueeze(0).to(device))
            action_summary = hidden[:, -1, :]
            adapter_logits = model_bundle["adapter"](summary, action_summary, goal)
            action_id = torch.argmax(adapter_logits, dim=-1).item()
        else:
            tokens = torch.from_numpy(np.stack(world_seq, axis=0)).unsqueeze(0).to(device)
            summary = model_bundle["state"].hidden_summary(tokens)
            goal = torch.from_numpy(env.goal).unsqueeze(0).to(device)
            direction = env.goal - start_pos
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                direction = np.array([1.0, 0.0], dtype=np.float32)
            else:
                direction = direction / norm
            action_hint = line_to_goal_action(
                env.state[:2], env.state[2:4], env.goal, direction, a_max, env.v_max, env.gamma, env.dt
            )
            action_hint = torch.from_numpy(action_hint.astype(np.float32)).unsqueeze(0).to(device)
            adapter_logits = model_bundle["line_adapter"](summary, action_hint, goal)
            action_id = torch.argmax(adapter_logits, dim=-1).item()

        action_id = int(np.clip(action_id, 0, action_bins * action_bins - 1))
        action = action_token_to_continuous(action_id, action_bins, a_max)
        action = action.astype(np.float32)
        state, success, done, info = env.step(action)
        action_tokens.append(action_id)
        traj.append(state[:2].copy())
        actions.append(action.copy())
        winds.append(info["wind"].copy())
        if done or success:
            break

    traj = np.array(traj, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    winds = np.array(winds, dtype=np.float32)
    frames = np.array(frames, dtype=np.uint8)
    return traj, actions, winds, frames, success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model", type=str, choices=["full", "scratch", "action_only", "state_only", "line_adapter"], required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--vqvae", type=str, required=True)
    parser.add_argument("--state", type=str)
    parser.add_argument("--action", type=str)
    parser.add_argument("--adapter", type=str)
    parser.add_argument("--line_adapter", type=str)
    parser.add_argument("--scratch", type=str)
    parser.add_argument("--state_only", type=str)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    env_cfg = cfg["env"].copy()
    env_cfg["image_size"] = cfg["tokenizer"]["image_size"]
    env_cfg["ego_crop"] = 32
    env = WindyNavEnv(env_cfg)

    vqvae = VQVAE(
        in_channels=3,
        hidden=cfg["tokenizer"]["latent_dim"],
        n_codes=cfg["tokenizer"]["codebook_size"],
        code_dim=cfg["tokenizer"]["latent_dim"],
    ).to(args.device)
    vqvae.load_state_dict(torch.load(args.vqvae, map_location=args.device))
    vqvae.eval()

    model_bundle = {}
    if args.model in ["full", "state_only", "line_adapter"]:
        state_prior = StatePrior(
            vocab_size=cfg["tokenizer"]["codebook_size"],
            d_model=cfg["state_prior"]["d_model"],
            n_layers=cfg["state_prior"]["n_layers"],
            n_heads=cfg["state_prior"]["n_heads"],
            d_ff=cfg["state_prior"]["d_ff"],
            dropout=cfg["state_prior"]["dropout"],
            context_frames=cfg["state_prior"]["L"],
        ).to(args.device)
        state_prior.load_state_dict(torch.load(args.state, map_location=args.device))
        state_prior.eval()
        model_bundle["state"] = state_prior

    if args.model in ["full", "action_only"]:
        action_state = torch.load(args.action, map_location=args.device)
        needed_len = cfg["action_prior"]["L"] * 64 + (cfg["action_prior"]["L"] - 1)
        pos_weight = action_state.get("pos_embed.weight")
        pos_len = pos_weight.shape[0] if pos_weight is not None else 0
        max_seq_len = pos_len if pos_len >= needed_len else needed_len
        action_prior = ActionPrior(
            vocab_size=cfg["tokenizer"]["codebook_size"] + cfg["action_prior"]["action_vocab"],
            codebook_size=cfg["tokenizer"]["codebook_size"],
            d_model=cfg["action_prior"]["d_model"],
            n_layers=cfg["action_prior"]["n_layers"],
            n_heads=cfg["action_prior"]["n_heads"],
            d_ff=cfg["action_prior"]["d_ff"],
            dropout=cfg["action_prior"]["dropout"],
            max_seq_len=max_seq_len,
        ).to(args.device)
        if pos_weight is not None and pos_len != max_seq_len:
            new_weight = pos_weight.new_empty((max_seq_len, pos_weight.shape[1]))
            copy_len = min(pos_len, max_seq_len)
            new_weight[:copy_len] = pos_weight[:copy_len]
            if max_seq_len > pos_len:
                new_weight[copy_len:] = pos_weight[-1:].repeat(max_seq_len - copy_len, 1)
            action_state["pos_embed.weight"] = new_weight
        time_weight = action_state.get("time_embed.weight")
        if time_weight is not None:
            target_steps = max_seq_len // action_prior.step_size + 2
            time_len = time_weight.shape[0]
            if time_len != target_steps:
                new_weight = time_weight.new_empty((target_steps, time_weight.shape[1]))
                copy_len = min(time_len, target_steps)
                new_weight[:copy_len] = time_weight[:copy_len]
                if target_steps > time_len:
                    new_weight[copy_len:] = time_weight[-1:].repeat(target_steps - copy_len, 1)
                action_state["time_embed.weight"] = new_weight
        strict = all(
            key in action_state
            for key in ["type_embed.weight", "time_embed.weight", "slot_embed.weight"]
        )
        action_prior.load_state_dict(action_state, strict=strict)
        action_prior.eval()
        model_bundle["action"] = action_prior

    if args.model == "full":
        adapter = Adapter(
            d_model=cfg["action_prior"]["d_model"],
            action_vocab=cfg["action_prior"]["action_vocab"],
            hidden=cfg["adapter"]["mlp_hidden"],
        ).to(args.device)
        adapter.load_state_dict(torch.load(args.adapter, map_location=args.device))
        adapter.eval()
        model_bundle["adapter"] = adapter
    if args.model == "line_adapter":
        line_adapter = LineAdapter(
            d_model=cfg["state_prior"]["d_model"],
            action_vocab=cfg["action_prior"]["action_vocab"],
            hidden=cfg["adapter"]["mlp_hidden"] * 2,
        ).to(args.device)
        line_adapter.load_state_dict(torch.load(args.line_adapter, map_location=args.device))
        line_adapter.eval()
        model_bundle["line_adapter"] = line_adapter

    if args.model == "scratch":
        scratch = ScratchPolicy(
            vocab_size=cfg["tokenizer"]["codebook_size"],
            action_vocab=cfg["action_prior"]["action_vocab"],
            d_model=cfg["action_prior"]["d_model"],
            n_layers=cfg["action_prior"]["n_layers"],
            n_heads=cfg["action_prior"]["n_heads"],
            d_ff=cfg["action_prior"]["d_ff"],
            dropout=cfg["action_prior"]["dropout"],
            context_frames=cfg["action_prior"]["L"],
        ).to(args.device)
        scratch.load_state_dict(torch.load(args.scratch, map_location=args.device))
        scratch.eval()
        model_bundle["scratch"] = scratch

    if args.model == "state_only":
        state_only = StateOnlyPolicy(
            d_model=cfg["state_prior"]["d_model"],
            action_vocab=cfg["action_prior"]["action_vocab"],
        ).to(args.device)
        state_only.load_state_dict(torch.load(args.state_only, map_location=args.device))
        state_only.eval()
        model_bundle["state_only"] = state_only

    os.makedirs(args.out, exist_ok=True)
    success_count = 0
    results = []
    for ep in range(args.episodes):
        traj, actions, winds, frames, success = run_episode(
            env, vqvae, model_bundle, cfg, args.device, args.model
        )
        success_count += int(success)
        results.append({
            "traj": traj,
            "actions": actions,
            "winds": winds,
            "frames": frames,
            "success": success,
            "goal": env.goal.copy(),
        })
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}/{args.episodes}")

    out_path = os.path.join(args.out, f"eval_{args.model}.npz")
    np.savez_compressed(out_path, results=results, success_rate=success_count / args.episodes)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
