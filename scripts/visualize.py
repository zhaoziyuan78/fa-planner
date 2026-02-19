import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio


def plot_trajectories(results, out_path, num=20):
    plt.figure(figsize=(6, 6))
    for i, item in enumerate(results[:num]):
        traj = item["traj"]
        goal = item["goal"]
        plt.plot(traj[:, 0], traj[:, 1], alpha=0.7)
        plt.scatter(traj[0, 0], traj[0, 1], c="blue", s=10)
        plt.scatter(traj[-1, 0], traj[-1, 1], c="red", s=10)
        plt.scatter(goal[0], goal[1], c="green", s=20, marker="x")
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.title("Trajectories")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_timeseries(item, out_path):
    actions = item["actions"]
    winds = item["winds"]
    traj = item["traj"]
    goal = item["goal"]
    dist = np.linalg.norm(traj - goal[None, :], axis=-1)

    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(dist)
    axes[0].set_ylabel("dist")
    axes[1].plot(actions[:, 0], label="ax")
    axes[1].plot(actions[:, 1], label="ay")
    axes[1].legend()
    axes[1].set_ylabel("action")
    axes[2].plot(winds[:, 0], label="wx")
    axes[2].plot(winds[:, 1], label="wy")
    axes[2].legend()
    axes[2].set_ylabel("wind")
    axes[2].set_xlabel("t")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def render_gif(item, out_path):
    frames = item["frames"]
    imageio.mimsave(out_path, frames, duration=0.1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--num", type=int, default=20)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    data = np.load(args.eval, allow_pickle=True)
    results = data["results"]

    os.makedirs(args.out, exist_ok=True)
    plot_trajectories(results, os.path.join(args.out, "trajectories.png"), num=args.num)
    plot_timeseries(results[args.index], os.path.join(args.out, "timeseries.png"))
    render_gif(results[args.index], os.path.join(args.out, "rollout.gif"))


if __name__ == "__main__":
    main()
