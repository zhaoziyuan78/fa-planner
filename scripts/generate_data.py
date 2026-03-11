import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import imageio
from fa_planner.envs.windynav import WindyNavEnv
from fa_planner.utils.config import load_config


def expert_action(p, v, g, w, a_max, kp=2.0, kd=0.8, kw=1.0):
    a = kp * (g - p) + kd * (0 - v) - kw * w
    return np.clip(a, -a_max, a_max)


def rollout_episode(env, mode, seed=None):
    world_frames = []
    ego_frames = []
    actions = []
    sim_state = []
    wind = []

    if mode == "action":
        env.reset(seed=seed, wind=(np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32)))
        start = env.state[:2].copy()
        goal = env.goal.copy()
        direction = goal - start
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            direction = np.array([1.0, 0.0], dtype=np.float32)
        else:
            direction = direction / norm
        a_const = np.clip(env.a_max * direction, -env.a_max, env.a_max)
        for t in range(env.T):
            world = env.render_world(show_goal=True)
            ego = env.render_ego(world)
            a = a_const
            state, success, done, info = env.step(a)
            world_frames.append(world)
            ego_frames.append(ego)
            actions.append(a)
            sim_state.append(state)
            wind.append(info["wind"])
            if success:
                break
        return world_frames, ego_frames, actions, sim_state, wind, env.goal

    if mode == "state":
        env.reset(seed=seed, start_mode="random")
        for t in range(env.T):
            world = env.render_world(show_goal=True)
            ego = env.render_ego(world)
            a = np.zeros(2, dtype=np.float32)
            state, _, done, info = env.step(a)
            world_frames.append(world)
            ego_frames.append(ego)
            actions.append(a)
            sim_state.append(state)
            wind.append(info["wind"])
            if done:
                break
        return world_frames, ego_frames, actions, sim_state, wind, env.goal

    if mode == "align":
        env.reset(seed=seed)
        for t in range(env.T):
            world = env.render_world(show_goal=True)
            ego = env.render_ego(world)
            x, y, vx, vy = env.state
            w = env.current_wind()
            a = expert_action(np.array([x, y]), np.array([vx, vy]), env.goal, w, env.a_max)
            state, _, done, info = env.step(a)
            world_frames.append(world)
            ego_frames.append(ego)
            actions.append(a)
            sim_state.append(state)
            wind.append(info["wind"])
            if done:
                break
        return world_frames, ego_frames, actions, sim_state, wind, env.goal

    raise ValueError(f"Unknown mode: {mode}")


def save_episode(path, world_frames, ego_frames, actions, sim_state, wind, goal, w0, k):
    np.savez_compressed(
        path,
        world_frames=np.asarray(world_frames, dtype=np.uint8),
        ego_frames=np.asarray(ego_frames, dtype=np.uint8),
        actions=np.asarray(actions, dtype=np.float32),
        goal=np.asarray(goal, dtype=np.float32),
        sim_state=np.asarray(sim_state, dtype=np.float32),
        wind=np.asarray(wind, dtype=np.float32),
        meta={"w0": w0.astype(np.float32), "k": k.astype(np.float32)},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["action", "state", "align"], required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    env_cfg = cfg["env"].copy()
    env_cfg["image_size"] = cfg["tokenizer"]["image_size"]
    env_cfg["ego_crop"] = 32
    env = WindyNavEnv(env_cfg)

    os.makedirs(args.out, exist_ok=True)
    np.random.seed(args.seed)

    sample_saved = False
    for ep in range(args.episodes):
        world, ego, actions, sim_state, wind, goal = rollout_episode(
            env, args.mode, seed=args.seed + ep
        )
        w0 = env.region_w0
        k = env.region_k
        path = os.path.join(args.out, f"episode_{ep:06d}.npz")
        save_episode(path, world, ego, actions, sim_state, wind, goal, w0, k)
        if not sample_saved:
            sample_path = os.path.join(args.out, f"sample_{args.mode}.gif")
            imageio.mimsave(sample_path, world, duration=0.1)
            sample_saved = True
        if (ep + 1) % 100 == 0:
            print(f"Generated {ep+1}/{args.episodes}")


if __name__ == "__main__":
    main()
