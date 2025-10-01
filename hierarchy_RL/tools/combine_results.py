#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
import csv
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_training_log(run_dir: Path):
    p = run_dir / "training_logs.csv"
    if not p.exists():
        return None
    episodes = []
    returns = []
    epsilons = []
    try:
        with open(p, "r") as f:
            rd = csv.reader(f)
            header = next(rd, None)
            # Expected columns include: episode,steps,epsilon,return_total,...
            if header is None:
                return None
            idx_ep = header.index("episode") if "episode" in header else 0
            idx_eps = header.index("epsilon") if "epsilon" in header else 2
            # return_total could be at 3
            idx_ret = header.index("return_total") if "return_total" in header else 3
            for row in rd:
                try:
                    episodes.append(int(row[idx_ep]))
                    epsilons.append(float(row[idx_eps]))
                    returns.append(float(row[idx_ret]))
                except Exception:
                    continue
        return {
            "episodes": np.asarray(episodes, dtype=np.int32),
            "return_total": np.asarray(returns, dtype=np.float32),
            "epsilon": np.asarray(epsilons, dtype=np.float32)
        }
    except Exception:
        return None


def read_strategic_log(run_dir: Path):
    p = run_dir / "strategic_training_logs.csv"
    if not p.exists():
        return None
    steps = []
    sret = []
    try:
        with open(p, "r") as f:
            rd = csv.reader(f)
            header = next(rd, None)
            if header is None:
                return None
            # recent format: episode, strategic_return, num_intervals, actor_loss_mean, critic_loss_mean, avg_q_value
            # legacy: s_episode, global_step, epsilon, interval_total_reward, ...
            if "episode" in header and "strategic_return" in header:
                idx_se = header.index("episode")
                idx_sr = header.index("strategic_return")
                for row in rd:
                    try:
                        steps.append(int(row[idx_se]))
                        sret.append(float(row[idx_sr]))
                    except Exception:
                        continue
            else:
                # fallback: treat s_episode as x and interval reward as y (approx)
                if "s_episode" in header and "interval_total_reward" in header:
                    idx_se = header.index("s_episode")
                    idx_sr = header.index("interval_total_reward")
                    for row in rd:
                        try:
                            steps.append(int(row[idx_se]))
                            sret.append(float(row[idx_sr]))
                        except Exception:
                            continue
        if not steps:
            return None
        return {
            "s_episode": np.asarray(steps, dtype=np.int32),
            "strategic_return": np.asarray(sret, dtype=np.float32)
        }
    except Exception:
        return None


def moving_average(arr: np.ndarray, window: int = 10):
    if window <= 1:
        return arr
    if arr.size < window:
        return arr
    c = np.cumsum(np.insert(arr, 0, 0.0))
    out = (c[window:] - c[:-window]) / float(window)
    pad = np.empty_like(arr)
    pad[:window-1] = arr[:window-1]
    pad[window-1:] = out
    return pad


def combine_series(series_list, x_key, y_key, smooth=10):
    # Align by min length
    min_len = min(s[y_key].shape[0] for s in series_list)
    ys = np.stack([moving_average(s[y_key][:min_len], smooth) for s in series_list], axis=0)
    xs = series_list[0][x_key][:min_len]
    mean = ys.mean(axis=0)
    std = ys.std(axis=0)
    return xs, mean, std


def plot_band(x, mean, std, title, ylabel, out_path):
    plt.figure(figsize=(8,5))
    plt.plot(x, mean, color="#2563eb", label="mean")
    plt.fill_between(x, mean-std, mean+std, color="#93c5fd", alpha=0.4, label="±1 std")
    plt.grid(True, alpha=0.3)
    plt.xlabel("episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Combine multiple run results and plot smoothed curves")
    ap.add_argument("--runs", nargs="+", required=True, help="Paths to run directories to combine")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for combined results")
    ap.add_argument("--smooth", type=int, default=10, help="Moving average window")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Training logs
    tr_series = []
    st_series = []
    for rp in args.runs:
        rd = Path(rp)
        tr = read_training_log(rd)
        if tr is not None and tr.get("return_total") is not None and tr["return_total"].size > 0:
            tr_series.append(tr)
        st = read_strategic_log(rd)
        if st is not None and st.get("strategic_return") is not None and st["strategic_return"].size > 0:
            st_series.append(st)

    report = {"combined_from": args.runs, "smooth_window": args.smooth}

    if tr_series:
        x, mean, std = combine_series(tr_series, "episodes", "return_total", smooth=args.smooth)
        plot_band(x, mean, std, "Return (total) – combined", "return_total", out_dir / "return_total_combined.png")
        np.savetxt(out_dir / "return_total_mean.csv", np.c_[x, mean], delimiter=",", header="episode,mean", comments="")
    else:
        report["warning_training"] = "No valid training logs found"

    if st_series:
        x, mean, std = combine_series(st_series, "s_episode", "strategic_return", smooth=args.smooth)
        plot_band(x, mean, std, "Strategic Return – combined", "strategic_return", out_dir / "strategic_return_combined.png")
        np.savetxt(out_dir / "strategic_return_mean.csv", np.c_[x, mean], delimiter=",", header="s_episode,mean", comments="")
    else:
        report["warning_strategic"] = "No valid strategic logs found"

    with open(out_dir / "combine_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"Combined results saved under: {out_dir}")


if __name__ == "__main__":
    main()


