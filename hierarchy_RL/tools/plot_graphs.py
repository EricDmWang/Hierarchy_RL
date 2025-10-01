#!/usr/bin/env python3
import os
import sys
import csv
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_training(run_dir: Path):
    p = run_dir / "training_logs.csv"
    if not p.exists():
        return None
    data = {}
    with open(p, "r") as f:
        rd = csv.reader(f)
        header = next(rd, None)
        if not header:
            return None
        fields = [
            "episode", "steps", "epsilon", "return_total", "return_mean", "return_std",
            "policy_loss_mean", "critic_loss", "avg_q_value", "buffer_size", "learning_rate"
        ]
        for name in fields:
            data[name] = [] if name in header else None
        idx = {name: (header.index(name) if name in header else None) for name in fields}
        for row in rd:
            for name in fields:
                j = idx.get(name)
                if j is None:
                    continue
                try:
                    val = float(row[j]) if name != "episode" and name != "steps" else int(float(row[j]))
                    data[name].append(val)
                except Exception:
                    pass
    out = {}
    for name in fields:
        if data.get(name) is not None and len(data[name]) > 0:
            dtype = np.int32 if name in ("episode", "steps") else np.float32
            out[name] = np.asarray(data[name], dtype=dtype)
    if not out:
        return None
    return out


def read_strategic(run_dir: Path):
    p = run_dir / "strategic_training_logs.csv"
    if not p.exists():
        return None
    with open(p, "r") as f:
        rd = csv.reader(f)
        header = next(rd, None)
        if not header:
            return None
        data = {}
        fields_current = ["episode", "strategic_return", "num_intervals", "actor_loss_mean", "critic_loss_mean", "avg_q_value"]
        fields_legacy = ["s_episode", "interval_total_reward"]
        if all(n in header for n in fields_current[:2]):
            fields = fields_current
        elif all(n in header for n in fields_legacy):
            fields = fields_legacy
        else:
            return None
        for name in fields:
            data[name] = []
        idx = {name: header.index(name) for name in fields}
        for row in rd:
            for name in fields:
                j = idx[name]
                try:
                    val = float(row[j])
                    if name in ("episode", "s_episode", "num_intervals"):
                        val = int(val)
                    data[name].append(val)
                except Exception:
                    pass
    out = {}
    for name, arr in data.items():
        if len(arr) > 0:
            dtype = np.int32 if name in ("episode", "s_episode", "num_intervals") else np.float32
            out[name] = np.asarray(arr, dtype=dtype)
    return out if out else None


def align_and_combine(series_list, key_name: str, limit_len: int):
    if not series_list:
        return None, None, None
    min_len = min(limit_len, min(s.shape[0] for s in series_list))
    arr = np.stack([s[:min_len] for s in series_list], axis=0)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    x = np.arange(1, min_len + 1)
    return x, mean, std


def moving_average(arr: np.ndarray, window: int = 10):
    if window <= 1 or arr.size == 0:
        return arr
    if arr.size < window:
        return arr
    c = np.cumsum(np.insert(arr, 0, 0.0))
    out = (c[window:] - c[:-window]) / float(window)
    pad = np.empty_like(arr)
    pad[:window-1] = arr[:window-1]
    pad[window-1:] = out
    return pad


def plot_layers(x, per_run_series, mean, std, out_path, smooth_window=10, xlabel="Episode", ylabel="Value"):
    plt.figure(figsize=(8,5))
    for s in per_run_series:
        plt.plot(x[:len(s)], s[:len(x)], color="#6b7280", alpha=0.12, linewidth=1.0)
    for s in per_run_series:
        sm = moving_average(s, smooth_window)
        plt.plot(x[:len(sm)], sm[:len(x)], color="#9ca3af", alpha=0.25, linewidth=1.2)
    if std is not None:
        plt.fill_between(x, mean-std, mean+std, color="#93c5fd", alpha=0.45)
    plt.plot(x, mean, color="#2563eb", linewidth=2.3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Plot combined metrics without labels")
    ap.add_argument("--runs", nargs="+", required=False, default=[
        "/home/dongmingwang/project/Hierarchy/hierarchy_RL/results/hierarchy_20250930_014218",
        "/home/dongmingwang/project/Hierarchy/hierarchy_RL/results/hierarchy_20250930_015814",
        "/home/dongmingwang/project/Hierarchy/hierarchy_RL/results/hierarchy_20250930_021420",
        "/home/dongmingwang/project/Hierarchy/hierarchy_RL/results/hierarchy_20250930_023021",
        "/home/dongmingwang/project/Hierarchy/hierarchy_RL/results/hierarchy_20250930_024549",
    ], help="Run directories to combine")
    ap.add_argument("--out_dir", type=str, required=False, default="/home/dongmingwang/project/Hierarchy/hierarchy_RL/results/final results", help="Output directory (will be created)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_out = out_dir / "training"
    strat_out = out_dir / "strategic"
    train_out.mkdir(parents=True, exist_ok=True)
    strat_out.mkdir(parents=True, exist_ok=True)

    ylabels = {
        "return_total": "Return",
        "epsilon": "Epsilon",
        "avg_q_value": "Avg Q",
        "policy_loss_mean": "Policy Loss",
        "critic_loss": "Critic Loss",
        "buffer_size": "Buffer Size",
        "learning_rate": "Learning Rate",
        "strategic_return": "Strategic Return",
        "actor_loss_mean": "Actor Loss",
        "critic_loss_mean": "Critic Loss",
        "avg_q_value_strat": "Strategic Avg Q"
    }

    train_list = []
    for rp in args.runs:
        tr = read_training(Path(rp))
        if tr is not None:
            train_list.append(tr)
    metrics = [
        ("return_total", "return_total"),
        ("epsilon", "epsilon"),
        ("avg_q_value", "avg_q_value"),
        ("policy_loss_mean", "policy_loss_mean"),
        ("critic_loss", "critic_loss"),
        ("buffer_size", "buffer_size"),
        ("learning_rate", "learning_rate"),
    ]
    return_x = return_mean = return_std = None
    for name, base in metrics:
        per_run_series = []
        for tr in train_list:
            if name in tr and tr[name] is not None and tr[name].size > 0:
                per_run_series.append(tr[name])
        if per_run_series:
            x, mean, std = align_and_combine(per_run_series, name, limit_len=400)
            if x is not None:
                plot_layers(x, per_run_series, mean, std, train_out / f"{base}_0_400.png", smooth_window=10, xlabel="Episode", ylabel=ylabels.get(name, name))
                if name == "return_total":
                    return_x, return_mean, return_std = x, mean, std
                np.savetxt(train_out / f"{base}_0_400.csv", np.c_[x, mean, std], delimiter=",", header="x,mean,std", comments="")
    if return_x is not None:
        plot_layers(return_x, [tr["return_total"] for tr in train_list if "return_total" in tr], return_mean, return_std, train_out / "combined_return_total.png", smooth_window=10, xlabel="Episode", ylabel=ylabels.get("return_total"))

    for name, base in metrics:
        per_run_series = []
        for tr in train_list:
            if name in tr and tr[name] is not None and tr[name].size > 0:
                per_run_series.append(tr[name])
        if per_run_series:
            full_len = min(s.shape[0] for s in per_run_series)
            x_full, mean_full, std_full = align_and_combine(per_run_series, name, limit_len=full_len)
            if x_full is not None:
                plot_layers(x_full, per_run_series, mean_full, std_full, train_out / f"{base}_full.png", smooth_window=10, xlabel="Episode", ylabel=ylabels.get(name, name))
                np.savetxt(train_out / f"{base}_full.csv", np.c_[x_full, mean_full, std_full], delimiter=",", header="x,mean,std", comments="")

    strat_list = []
    for rp in args.runs:
        st = read_strategic(Path(rp))
        if st is not None:
            strat_list.append(st)

    def collect_strat_series(name, target_eps):
        per_run = []
        for st in strat_list:
            if name not in st:
                continue
            ep_key = 'episode' if 'episode' in st else ('s_episode' if 's_episode' in st else None)
            if ep_key is None:
                continue
            mapping = {int(e): float(v) for e, v in zip(st[ep_key], st[name])}
            vals = []
            for e in target_eps:
                if e in mapping:
                    vals.append(mapping[e])
                else:
                    keys = [k for k in mapping.keys() if k <= e]
                    vals.append(mapping[max(keys)] if keys else 0.0)
            per_run.append(np.asarray(vals, dtype=np.float32))
        return per_run

    strat_metrics = [
        ("strategic_return", "strategic_return", ylabels.get("strategic_return")),
        ("actor_loss_mean", "strategic_actor_loss", ylabels.get("actor_loss_mean")),
        ("critic_loss_mean", "strategic_critic_loss", ylabels.get("critic_loss_mean")),
        ("avg_q_value", "strategic_avg_q_value", ylabels.get("avg_q_value_strat")),
    ]
    # Reindex strategic to 1..80 for first window
    target_eps_400 = np.arange(1, 401, 5)
    for name, base, ylab in strat_metrics:
        per_run = collect_strat_series(name, target_eps_400)
        if per_run:
            x, mean, std = align_and_combine(per_run, name, limit_len=len(target_eps_400))
            if x is not None:
                x_plot = np.arange(1, len(x) + 1)  # reindex 1..80
                plot_layers(x_plot, per_run, mean, std, strat_out / f"{base}_every5_0_400.png", smooth_window=6, xlabel="Index", ylabel=ylab or name)
                np.savetxt(strat_out / f"{base}_every5_0_400.csv", np.c_[x_plot, mean, std], delimiter=",", header="x,mean,std", comments="")

    if strat_list:
        max_eps = []
        for st in strat_list:
            ep_key = 'episode' if 'episode' in st else ('s_episode' if 's_episode' in st else None)
            if ep_key is not None:
                max_eps.append(int(np.max(st[ep_key])))
        if max_eps:
            common_max = min(max_eps)
            target_eps_full = np.arange(1, common_max + 1, 5)
            for name, base, ylab in strat_metrics:
                per_run = collect_strat_series(name, target_eps_full)
                if per_run:
                    x, mean, std = align_and_combine(per_run, name, limit_len=len(target_eps_full))
                    if x is not None:
                        x_plot = np.arange(1, len(x) + 1)
                        plot_layers(x_plot, per_run, mean, std, strat_out / f"{base}_every5_full.png", smooth_window=6, xlabel="Index", ylabel=ylab or name)
                        np.savetxt(strat_out / f"{base}_every5_full.csv", np.c_[x_plot, mean, std], delimiter=",", header="x,mean,std", comments="")

    print(f"All graphs saved under: {out_dir}")


if __name__ == "__main__":
    main()


