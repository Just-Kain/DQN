"""
plot_training.py -- Training log analysis for DQN / DDQN / Q-Learning

Usage:
    python plot_training.py                     # all algos found in checkpoints/
    python plot_training.py --algo ddqn         # specific algo
    python plot_training.py --algo ddqn dqn     # multiple algos on one chart
    python plot_training.py --save              # save PNG instead of showing
    python plot_training.py --window 500        # smoothing window (default 200)

Output: 7 panels
    1. Win Rate (smoothed)
    2. Total Reward (smoothed)
    3. Steps per Episode (smoothed)
    4. Loss (smoothed, non-zero only)
    5. Epsilon
    6. Win Rate per Phase (box plot)
    7. Reward distribution per Phase (violin)
"""

import argparse
import glob
import io
import os
import sys

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams.update({
    "figure.facecolor": "#0f0f0f",
    "axes.facecolor":   "#1a1a1a",
    "axes.edgecolor":   "#444",
    "axes.labelcolor":  "#ccc",
    "axes.titlecolor":  "#fff",
    "xtick.color":      "#888",
    "ytick.color":      "#888",
    "grid.color":       "#2a2a2a",
    "grid.linewidth":   0.8,
    "text.color":       "#ccc",
    "legend.facecolor": "#1e1e1e",
    "legend.edgecolor": "#444",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.titleweight": "bold",
})

PHASE_COLORS = ["#4fc3f7", "#81c784", "#ffb74d", "#f06292", "#ce93d8"]
PHASE_LABELS = {
    0: "Ph0: 16x16 no-enemy",
    1: "Ph1: 16x16 enemy",
    2: "Ph2: 20x20",
    3: "Ph3: 24x24",
    4: "Ph4: 32x32",
}
ALGO_COLORS  = ["#4fc3f7", "#ff8a65", "#a5d6a7", "#ce93d8", "#fff176"]

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


# ── helpers ──────────────────────────────────────────────────────────────────

def load_log(algo: str) -> pd.DataFrame | None:
    path = os.path.join(CKPT_DIR, f"train_log_{algo}.csv")
    if not os.path.exists(path):
        print(f"[plot] Not found: {path}")
        return None
    with open(path, "rb") as f:
        raw = f.read().replace(b"\x00", b"")
    try:
        df = pd.read_csv(io.StringIO(raw.decode("utf-8", errors="replace")))
    except Exception as e:
        print(f"[plot] Failed to parse {path}: {e}")
        return None
    df = df.dropna(subset=["episode"]).copy()
    df["episode"] = df["episode"].astype(int)
    print(f"[plot] {algo}: {len(df)} episodes, phases {sorted(df['phase'].unique())}")
    return df


def smooth(series: np.ndarray, w: int) -> np.ndarray:
    """Simple centered moving average."""
    if w <= 1 or len(series) < w:
        return series
    kernel = np.ones(w) / w
    pad    = np.pad(series, (w // 2, w // 2), mode="edge")
    return np.convolve(pad, kernel, mode="valid")[: len(series)]


def phase_boundaries(df: pd.DataFrame):
    """Returns list of (episode, phase_idx) where phase changes."""
    phases = df[["episode", "phase"]].copy()
    shifted = phases["phase"].shift(1)
    changes = phases[phases["phase"] != shifted]
    return list(zip(changes["episode"], changes["phase"]))


def draw_phase_bands(ax, df: pd.DataFrame, alpha: float = 0.07):
    """Shade background by phase and draw vertical transition lines."""
    bounds = phase_boundaries(df)
    eps    = list(df["episode"])
    max_ep = eps[-1]
    for i, (start_ep, ph) in enumerate(bounds):
        end_ep = bounds[i + 1][0] if i + 1 < len(bounds) else max_ep + 1
        color  = PHASE_COLORS[int(ph) % len(PHASE_COLORS)]
        ax.axvspan(start_ep, end_ep, color=color, alpha=alpha, linewidth=0)
        if i > 0:
            ax.axvline(start_ep, color=color, lw=1.0, alpha=0.5, linestyle="--")


def phase_legend_handles(df: pd.DataFrame):
    phases = sorted(df["phase"].unique())
    return [
        mpatches.Patch(
            color=PHASE_COLORS[int(p) % len(PHASE_COLORS)],
            label=PHASE_LABELS.get(int(p), f"Phase {p}"),
            alpha=0.7,
        )
        for p in phases
    ]


# ── main plot ─────────────────────────────────────────────────────────────────

def plot_single(algo: str, df: pd.DataFrame, window: int, save: bool):
    """Full 7-panel dashboard for one algorithm."""
    ep   = df["episode"].values
    wr   = df["win_rate"].values
    rw   = df["total_reward"].values
    st   = df["steps"].values
    lo   = df["loss_mean"].values
    eps  = df["epsilon"].values

    # mask zero-loss rows (buffer not warm yet)
    lo_mask = lo > 0.0
    ep_lo   = ep[lo_mask]
    lo_val  = lo[lo_mask]

    fig, axes = plt.subplots(3, 3, figsize=(18, 13))
    fig.suptitle(f"Training Dashboard — {algo.upper()}   ({len(df):,} episodes)",
                 fontsize=14, fontweight="bold", color="#fff", y=0.98)

    axs = axes.flatten()

    # ── 1. Win Rate ──────────────────────────────────────────────────────────
    ax = axs[0]
    ax.plot(ep, wr * 100, color="#555", lw=0.4, alpha=0.4)
    ax.plot(ep, smooth(wr, window) * 100, color="#4fc3f7", lw=1.8, label=f"smooth w={window}")
    draw_phase_bands(ax, df)
    ax.set_title("Win Rate (%)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("%")
    ax.set_ylim(-2, 105)
    ax.axhline(50, color="#fff", lw=0.6, ls=":", alpha=0.3)
    ax.legend(handles=phase_legend_handles(df), fontsize=8, loc="upper left")
    ax.grid(True)

    # ── 2. Total Reward ──────────────────────────────────────────────────────
    ax = axs[1]
    ax.plot(ep, rw, color="#555", lw=0.3, alpha=0.3)
    ax.plot(ep, smooth(rw, window), color="#81c784", lw=1.8)
    draw_phase_bands(ax, df)
    ax.set_title("Total Reward (per episode)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.axhline(0, color="#fff", lw=0.5, ls=":", alpha=0.3)
    ax.grid(True)

    # ── 3. Steps per Episode ─────────────────────────────────────────────────
    ax = axs[2]
    ax.plot(ep, st, color="#555", lw=0.3, alpha=0.3)
    ax.plot(ep, smooth(st, window), color="#ffb74d", lw=1.8)
    draw_phase_bands(ax, df)
    ax.set_title("Steps per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.grid(True)

    # ── 4. Loss ──────────────────────────────────────────────────────────────
    ax = axs[3]
    if len(ep_lo) > 0:
        ax.plot(ep_lo, lo_val, color="#555", lw=0.3, alpha=0.3)
        ax.plot(ep_lo, smooth(lo_val, window), color="#f06292", lw=1.8)
    draw_phase_bands(ax, df)
    ax.set_title("Loss (SmoothL1, non-zero)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    ax.grid(True)

    # ── 5. Epsilon ───────────────────────────────────────────────────────────
    ax = axs[4]
    ax.plot(ep, eps, color="#ce93d8", lw=1.6)
    draw_phase_bands(ax, df)
    ax.set_title("Epsilon (exploration rate)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True)

    # ── 6. Win Rate by Phase (box) ───────────────────────────────────────────
    ax = axs[5]
    phases = sorted(df["phase"].unique())
    data   = [df[df["phase"] == p]["win_rate"].values * 100 for p in phases]
    labels = [PHASE_LABELS.get(int(p), f"Ph{p}") for p in phases]
    colors = [PHASE_COLORS[int(p) % len(PHASE_COLORS)] for p in phases]
    bp = ax.boxplot(data, patch_artist=True, labels=labels,
                    medianprops=dict(color="#fff", lw=2))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_title("Win Rate by Phase")
    ax.set_ylabel("%")
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(True, axis="y")

    # ── 7. Reward Distribution by Phase (violin) ─────────────────────────────
    ax = axs[6]
    data_r = [df[df["phase"] == p]["total_reward"].values for p in phases]
    parts  = ax.violinplot(data_r, positions=range(len(phases)),
                           showmedians=True, showextrema=True)
    for i, (pc, c) in enumerate(zip(parts["bodies"], colors)):
        pc.set_facecolor(c)
        pc.set_alpha(0.6)
    parts["cmedians"].set_color("#fff")
    parts["cmins"].set_color("#888")
    parts["cmaxes"].set_color("#888")
    parts["cbars"].set_color("#888")
    ax.set_xticks(range(len(phases)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("Reward Distribution by Phase")
    ax.set_ylabel("Reward")
    ax.axhline(0, color="#fff", lw=0.5, ls=":", alpha=0.3)
    ax.grid(True, axis="y")

    # ── 8. Cumulative Wins ───────────────────────────────────────────────────
    ax = axs[7]
    wins_cumsum = np.cumsum((rw > 50).astype(int))
    ax.plot(ep, wins_cumsum, color="#fff176", lw=1.6)
    draw_phase_bands(ax, df)
    ax.set_title("Cumulative Wins (reward > 50)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Wins")
    ax.grid(True)

    # ── 9. Reward vs Steps scatter (last 2000 ep) ────────────────────────────
    ax = axs[8]
    tail = df.tail(2000)
    sc = ax.scatter(tail["steps"], tail["total_reward"],
                    c=tail["win_rate"], cmap="RdYlGn",
                    s=4, alpha=0.5, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label="win_rate")
    ax.set_title("Reward vs Steps (last 2000 ep)")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Reward")
    ax.axhline(0, color="#fff", lw=0.5, ls=":", alpha=0.3)
    ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save:
        out = os.path.join(CKPT_DIR, f"dashboard_{algo}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[plot] Saved: {out}")
    else:
        plt.show()

    plt.close(fig)


def plot_compare(algos: list[str], dfs: list[pd.DataFrame], window: int, save: bool):
    """Overlay win_rate and reward curves for multiple algos."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Algorithm Comparison", fontsize=13, fontweight="bold",
                 color="#fff", y=1.01)

    for i, (algo, df) in enumerate(zip(algos, dfs)):
        color = ALGO_COLORS[i % len(ALGO_COLORS)]
        ep = df["episode"].values
        wr = df["win_rate"].values
        rw = df["total_reward"].values

        axes[0].plot(ep, smooth(wr, window) * 100,
                     color=color, lw=1.8, label=algo.upper())
        axes[1].plot(ep, smooth(rw, window),
                     color=color, lw=1.8, label=algo.upper())

    for ax, title, ylabel in zip(
        axes,
        ["Win Rate (%)", "Total Reward"],
        ["%", "Reward"],
    ):
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)
        ax.axhline(0, color="#fff", lw=0.5, ls=":", alpha=0.3)

    plt.tight_layout()

    if save:
        out = os.path.join(CKPT_DIR, "dashboard_compare.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[plot] Saved: {out}")
    else:
        plt.show()

    plt.close(fig)


# ── per-phase summary table ───────────────────────────────────────────────────

def print_summary(algo: str, df: pd.DataFrame):
    print(f"\n{'='*65}")
    print(f"  SUMMARY: {algo.upper()}   ({len(df):,} episodes)")
    print(f"{'='*65}")
    phases = sorted(df["phase"].unique())
    for p in phases:
        sub = df[df["phase"] == p]
        wins = (sub["total_reward"] > 50).sum()
        print(f"\n  Phase {p}  ({PHASE_LABELS.get(int(p), '?')}) — {len(sub)} episodes")
        print(f"    win_rate  : max={sub['win_rate'].max():.1%}  "
              f"last={sub['win_rate'].iloc[-1]:.1%}  "
              f"mean={sub['win_rate'].mean():.1%}")
        print(f"    reward    : max={sub['total_reward'].max():.1f}  "
              f"mean={sub['total_reward'].mean():.1f}  "
              f"min={sub['total_reward'].min():.1f}")
        print(f"    steps     : mean={sub['steps'].mean():.0f}  "
              f"max={sub['steps'].max()}")
        print(f"    wins (r>50): {wins}  ({wins/len(sub):.1%})")
        print(f"    eps range : {sub['epsilon'].max():.3f} -> {sub['epsilon'].min():.3f}")
        if sub['loss_mean'].max() > 0:
            nz = sub[sub['loss_mean'] > 0]['loss_mean']
            print(f"    loss      : mean={nz.mean():.4f}  max={nz.max():.4f}")
    print()


# ── entry point ──────────────────────────────────────────────────────────────

def discover_algos() -> list[str]:
    found = []
    for path in glob.glob(os.path.join(CKPT_DIR, "train_log_*.csv")):
        name = os.path.basename(path)
        algo = name.replace("train_log_", "").replace(".csv", "")
        if not algo.startswith("fail"):
            found.append(algo)
    return sorted(found)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algo",   nargs="+", default=None,
                        help="Algo(s) to plot. Default: all found in checkpoints/")
    parser.add_argument("--window", type=int, default=200,
                        help="Smoothing window size (default: 200)")
    parser.add_argument("--save",   action="store_true",
                        help="Save PNG files instead of showing interactive window")
    args = parser.parse_args()

    algos = args.algo if args.algo else discover_algos()
    if not algos:
        print(f"[plot] No train_log_*.csv files found in {CKPT_DIR}")
        sys.exit(1)

    print(f"[plot] Algos: {algos}  |  window: {args.window}  |  save: {args.save}")

    loaded = []
    for algo in algos:
        df = load_log(algo)
        if df is not None:
            print_summary(algo, df)
            loaded.append((algo, df))

    if not loaded:
        print("[plot] No data to plot.")
        sys.exit(1)

    # Individual dashboards
    for algo, df in loaded:
        plot_single(algo, df, args.window, args.save)

    # Comparison overlay (if multiple algos)
    if len(loaded) > 1:
        plot_compare([a for a, _ in loaded],
                     [d for _, d in loaded],
                     args.window, args.save)


if __name__ == "__main__":
    main()
