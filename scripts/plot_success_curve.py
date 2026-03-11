import argparse
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def pad_to_length(values, length):
    if values.shape[0] >= length:
        return values
    pad_value = values[-1]
    pad_width = length - values.shape[0]
    return np.pad(values, (0, pad_width), mode="constant", constant_values=pad_value)


def load_distance_and_success(eval_path, success_radius):
    data = np.load(eval_path, allow_pickle=True)
    results = data["results"]
    max_len = max(len(item["traj"]) for item in results)
    dist_mat = np.zeros((len(results), max_len), dtype=np.float32)
    for i, item in enumerate(results):
        traj = item["traj"]
        goal = item["goal"]
        dist = np.linalg.norm(traj - goal[None, :], axis=-1)
        dist = pad_to_length(dist, max_len)
        dist_mat[i] = dist
    mean_dist = dist_mat.mean(axis=0)
    success_mask = dist_mat <= success_radius
    success_rate = np.maximum.accumulate(success_mask, axis=1).mean(axis=0)
    return mean_dist, success_rate


def derive_out_path(path, suffix):
    base, ext = os.path.splitext(path)
    if not ext:
        ext = ".png"
    return f"{base}_{suffix}{ext}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=str, required=True)
    parser.add_argument("--action_only", type=str, required=True)
    parser.add_argument("--state_only", type=str, required=True)
    parser.add_argument("--scratch", type=str, required=True)
    parser.add_argument("--line_adapter", type=str)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--success_radius", type=float, default=0.08)
    args = parser.parse_args()

    labels = ["full", "action_only", "state_only", "scratch"]
    paths = {
        "full": args.full,
        "action_only": args.action_only,
        "state_only": args.state_only,
        "scratch": args.scratch,
    }
    if args.line_adapter:
        labels.append("line_adapter")
        paths["line_adapter"] = args.line_adapter
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {label: colors[i % len(colors)] for i, label in enumerate(labels)}

    dist_curves = {}
    success_curves = {}
    for label in labels:
        mean_dist, success_rate = load_distance_and_success(paths[label], args.success_radius)
        dist_curves[label] = mean_dist
        success_curves[label] = success_rate

    plt.figure(figsize=(6, 4))
    for label in labels:
        curve = dist_curves[label]
        xs = np.arange(curve.shape[0])
        plt.plot(xs, curve, label=label, color=color_map[label])
    plt.xlabel("step")
    plt.ylabel("mean distance to goal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out)
    plt.close()

    success_out = derive_out_path(args.out, "success")
    plt.figure(figsize=(6, 4))
    for label in labels:
        curve = success_curves[label]
        xs = np.arange(curve.shape[0])
        plt.plot(xs, curve, label=label, color=color_map[label])
    plt.xlabel("step")
    plt.ylabel("success rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(success_out)
    plt.close()


if __name__ == "__main__":
    main()
