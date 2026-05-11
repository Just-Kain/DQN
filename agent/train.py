"""
train.py - Main DQN training loop with curriculum learning

Usage:
    cd agent
    pip install -r requirements.txt
    python train.py                           # ddqn, full curriculum
    python train.py --algo dqn                # vanilla DQN
    python train.py --algo qlearn             # Q-Learning (no buffer)
    python train.py --algo ddqn --resume      # resume DDQN
    python train.py --fixed-seed 75           # fixed seed (one map)

Algorithms (--algo):
    qlearn - Neural Q-Learning: no buffer, no target net,
             update after every step. Most unstable.
    dqn    - Vanilla DQN: buffer + target net. Max Q_target.
    ddqn   - Double DQN (default): buffer + target net.
             Online selects action, target evaluates -> less overestimation.

Checkpoint files (algo-specific):
    checkpoints/best_{algo}.pt
    checkpoints/last_{algo}.pt
    checkpoints/best_episode_{algo}.pkl

Curriculum (auto-advance on win_rate, min MIN_PHASE_EPISODES per phase):
    Phase 0 (16x16, NO enemies, 15hp): win_rate >= 50% -> agent learns navigation
    Phase 1 (16x16, enemies,    15hp): win_rate >= 35% -> add combat
    Phase 2 (18x18, enemies,    20hp): win_rate >= 30% -> intermediate step
    Phase 3 (20x20, enemies,    25hp): win_rate >= 50%
    Phase 4 (24x24, enemies,    30hp): win_rate >= 65%
    Phase 5 (32x32, enemies,    50hp): final difficulty

Epsilon reset on phase transition:
    eps = max(EPS_RESET_MIN, eps * EPS_RESET_FACTOR)
    This forces re-exploration of the new, larger map.

Stagnation detection (win_rate-based):
    Every STAGNATION_CHECK (300) episodes we compare win_rate of the last
    STAGNATION_HALF episodes vs the preceding STAGNATION_HALF episodes.
    If improvement < STAGNATION_THRESHOLD and eps < EPS_BUMP_MAX,
    epsilon is bumped up by EPS_BUMP_FACTOR (x2.0) to escape local optima.
    A cooldown (STAGNATION_COOLDOWN=600) prevents repeated bumps too fast.
"""

import argparse
import collections
import os
import pickle
import time

import numpy as np

from agent          import make_agent, ALGO_CHOICES
from game_interface import GameInterface

# ── Curriculum ────────────────────────────────────────────────────────────────
CURRICULUM = [
    {"map_size": 16, "no_enemies": True,  "player_hp": 15, "win_threshold": 0.50, "name": "Phase 0 (16x16, no enemies)"},
    {"map_size": 16, "no_enemies": False, "player_hp": 15, "win_threshold": 0.35, "name": "Phase 1 (16x16, enemies)"},
    {"map_size": 18, "no_enemies": False, "player_hp": 20, "win_threshold": 0.30, "name": "Phase 2 (18x18, enemies)"},
    {"map_size": 20, "no_enemies": False, "player_hp": 25, "win_threshold": 0.50, "name": "Phase 3 (20x20)"},
    {"map_size": 24, "no_enemies": False, "player_hp": 30, "win_threshold": 0.65, "name": "Phase 4 (24x24)"},
    {"map_size": 32, "no_enemies": False, "player_hp": 50, "win_threshold": 999.0,"name": "Phase 5 (32x32)"},
]
WIN_WINDOW         = 200   # sliding window for win_rate
MIN_PHASE_EPISODES = 500   # minimum episodes per phase before advancing
EPS_RESET_FACTOR   = 2.0   # multiply eps by this on phase advance
EPS_RESET_MIN      = 0.30  # eps floor after reset (never below 30%)

# ── Stagnation detection ──────────────────────────────────────────────────────
# Checked every STAGNATION_CHECK episodes.
# Compares win_rate of last STAGNATION_HALF eps vs preceding STAGNATION_HALF.
# Triggers only when eps < EPS_BUMP_MAX (no point bumping if already exploring).
# STAGNATION_COOLDOWN prevents repeated bumps in quick succession.
STAGNATION_CHECK    = 300    # check interval (episodes)
STAGNATION_HALF     = 100    # half-window for before/after comparison
STAGNATION_THRESHOLD = 0.01  # min win_rate improvement to NOT trigger (+1%)
EPS_BUMP_FACTOR     = 2.0    # multiply eps by this on stagnation
EPS_BUMP_MAX        = 0.15   # only bump if eps is below this value
EPS_BUMP_CEIL       = 0.50   # eps never bumped above this value
STAGNATION_COOLDOWN = 600    # episodes to wait before next possible bump

CKPT_DIR = "checkpoints"


# ── Checkpoint paths (algo-specific) ─────────────────────────────────────────
def ckpt_paths(algo):
    return {
        "best_model":   os.path.join(CKPT_DIR, "best_{}.pt".format(algo)),
        "last_model":   os.path.join(CKPT_DIR, "last_{}.pt".format(algo)),
        "log_file":     os.path.join(CKPT_DIR, "train_log_{}.csv".format(algo)),
        "best_episode": os.path.join(CKPT_DIR, "best_episode_{}.pkl".format(algo)),
    }


# ── Arguments ─────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__
    )
    p.add_argument("--algo",       type=str,  default="ddqn",
                   choices=ALGO_CHOICES,
                   help="Algorithm: qlearn | dqn | ddqn  (default: ddqn)")
    p.add_argument("--episodes",   type=int,  default=50_000)
    p.add_argument("--device",     type=str,  default="cpu")
    p.add_argument("--resume",     action="store_true",
                   help="Resume from last_{algo}.pt")
    p.add_argument("--fixed-seed", type=int,  default=None,
                   metavar="SEED",
                   help="Fix one seed for all episodes. Example: --fixed-seed 75")
    return p.parse_args()


def is_win(done, last_state, ep_reward):
    """Win = episode ended, player alive, total reward > 50."""
    return done and last_state.get("player_hp", 0) > 0 and ep_reward > 50.0


def check_stagnation(ep, win_history, agent, last_bump_ep, algo):
    """
    Compares win_rate of last STAGNATION_HALF episodes vs preceding half.
    If improvement < STAGNATION_THRESHOLD and eps is already low,
    bumps epsilon to force re-exploration.

    Returns updated last_bump_ep.
    """
    # Need enough history and cooldown elapsed
    if len(win_history) < STAGNATION_HALF * 2:
        return last_bump_ep
    if ep - last_bump_ep < STAGNATION_COOLDOWN:
        return last_bump_ep
    if agent.eps >= EPS_BUMP_MAX:
        return last_bump_ep

    hist = list(win_history)
    old_wr  = sum(hist[:STAGNATION_HALF])  / STAGNATION_HALF
    new_wr  = sum(hist[-STAGNATION_HALF:]) / STAGNATION_HALF
    delta   = new_wr - old_wr

    if delta < STAGNATION_THRESHOLD:
        old_eps   = agent.eps
        agent.eps = min(EPS_BUMP_CEIL, agent.eps * EPS_BUMP_FACTOR)
        print("[train/{}] Stagnation: win_rate {:.1%} -> {:.1%} (delta={:+.1%}) "
              "| eps {:.3f} -> {:.3f}".format(
                algo, old_wr, new_wr, delta, old_eps, agent.eps))
        return ep

    return last_bump_ep


# ── Main loop ─────────────────────────────────────────────────────────────────
def train():
    args  = parse_args()
    paths = ckpt_paths(args.algo)
    os.makedirs(CKPT_DIR, exist_ok=True)

    # Auto-fallback: if CUDA requested but unavailable -> CPU
    import torch
    device = args.device
    if device == "cuda":
        if torch.cuda.is_available():
            print("[train] CUDA available: {}".format(torch.cuda.get_device_name(0)))
        else:
            print("[train] CUDA not available. Falling back to CPU.")
            device = "cpu"

    # Create agent
    agent = make_agent(args.algo, device=device)

    # Curriculum + env
    phase_idx      = 0
    phase          = CURRICULUM[phase_idx]
    phase_start_ep = 1
    env            = GameInterface(map_size=phase["map_size"],
                                   no_enemies=phase["no_enemies"],
                                   player_hp=phase["player_hp"])
    win_history    = collections.deque(maxlen=WIN_WINDOW)
    last_bump_ep   = -STAGNATION_COOLDOWN  # allow first check immediately

    # Resume
    if args.resume and os.path.exists(paths["last_model"]):
        agent.load(paths["last_model"])
        print("[train] Resumed from {}, eps={:.3f}".format(paths["last_model"], agent.eps))

    # Log file
    if not os.path.exists(paths["log_file"]):
        with open(paths["log_file"], "w") as f:
            f.write("episode,total_reward,steps,epsilon,loss_mean,"
                    "win_rate,phase,map_size,no_enemies,ep_seed\n")

    algo_label = {
        "qlearn": "Q-Learning  (no buffer, no target)",
        "dqn":    "Vanilla DQN (buffer + target, vanilla max)",
        "ddqn":   "Double DQN  (buffer + target, online selects)",
    }[args.algo]

    fixed_seed    = args.fixed_seed
    enemies_label = "NO enemies" if phase["no_enemies"] else "with enemies"

    print("[train] Algorithm: {}".format(algo_label))
    print("[train] Device: {} | Episodes: {}".format(device, args.episodes))
    print("[train] Start: {} ({})".format(phase["name"], enemies_label))
    if fixed_seed is not None:
        print("[train] Fixed seed: {}  (one map for all training)".format(fixed_seed))
    print("[train] Checkpoints: best={}".format(paths["best_model"]))
    print("[train] Phase min episodes: {}  eps_reset: x{}  eps_reset_min: {}".format(
        MIN_PHASE_EPISODES, EPS_RESET_FACTOR, EPS_RESET_MIN))
    print("[train] Stagnation: check=/{} half={} thr={:.0%} bump=x{} cooldown={}".format(
        STAGNATION_CHECK, STAGNATION_HALF, STAGNATION_THRESHOLD,
        EPS_BUMP_FACTOR, STAGNATION_COOLDOWN))
    print("-" * 70)

    t0 = time.time()

    for ep in range(1, args.episodes + 1):
        if fixed_seed is not None:
            obs = env.reset_to_seed(fixed_seed)
        else:
            obs = env.reset()
        ep_seed    = env.episode_seed
        done       = False
        ep_reward  = 0.0
        ep_steps   = 0
        ep_losses  = []
        ep_actions = []

        while not done:
            action = agent.select_action(obs)
            ep_actions.append(int(action))
            next_obs, reward, done = env.step(action)

            agent.store(obs, action, reward, next_obs, done)

            # PER: duplicate winning transition x10 (buffer agents only)
            if agent.has_replay_buffer and done and reward >= 80.0:
                for _ in range(9):
                    agent.store(obs, action, reward, next_obs, done)

            loss = agent.train_step()
            if loss is not None:
                ep_losses.append(loss)

            obs        = next_obs
            ep_reward += reward
            ep_steps  += 1

        agent.decay_epsilon()

        loss_mean = float(np.mean(ep_losses)) if ep_losses else 0.0

        # Win rate
        won = is_win(done, env.last_state, ep_reward)
        win_history.append(int(won))
        win_rate = sum(win_history) / len(win_history) if win_history else 0.0

        # Best model checkpoint
        if ep_reward > agent.best_reward:
            agent.best_reward = ep_reward
            agent.save(paths["best_model"])
            with open(paths["best_episode"], "wb") as f:
                pickle.dump({
                    "seed":         ep_seed,
                    "actions":      ep_actions,
                    "total_reward": ep_reward,
                    "steps":        ep_steps,
                    "episode":      ep,
                    "map_size":     phase["map_size"],
                    "algo":         args.algo,
                }, f)
            print("[train/{}] * Best: {:.2f}  (ep={}, seed={}, steps={}, win_rate={:.1%})".format(
                args.algo, ep_reward, ep, ep_seed, ep_steps, win_rate))

        # Periodic checkpoint
        if ep % 100 == 0:
            agent.save(paths["last_model"])

        # Log
        with open(paths["log_file"], "a") as f:
            f.write("{},{:.3f},{},{:.4f},{:.6f},{:.4f},{},{},{},{}\n".format(
                ep, ep_reward, ep_steps, agent.eps,
                loss_mean, win_rate,
                phase_idx, phase["map_size"],
                int(phase["no_enemies"]), ep_seed))

        # Console output
        if ep % 100 == 0:
            elapsed = time.time() - t0
            print("[ep {:>6}/{}]  [{}]  reward={:>7.2f}  steps={:>4}  "
                  "eps={:.3f}  loss={:.5f}  win_rate={:.1%}  t={:.0f}s  seed={}".format(
                ep, args.episodes, args.algo,
                ep_reward, ep_steps,
                agent.eps, loss_mean, win_rate, elapsed, ep_seed))
        elif ep % 10 == 0 and won:
            print("[ep {:>6}]  [{}] WIN  reward={:.2f}  steps={}  win_rate={:.1%}".format(
                ep, args.algo, ep_reward, ep_steps, win_rate))

        # Stagnation detection (every STAGNATION_CHECK episodes)
        if ep % STAGNATION_CHECK == 0:
            last_bump_ep = check_stagnation(
                ep, win_history, agent, last_bump_ep, args.algo)

        # Curriculum advance
        phase_episodes = ep - phase_start_ep + 1
        if (phase_idx < len(CURRICULUM) - 1
                and len(win_history) >= WIN_WINDOW
                and win_rate >= phase["win_threshold"]
                and phase_episodes >= MIN_PHASE_EPISODES):

            phase_idx += 1
            phase = CURRICULUM[phase_idx]
            enemies_label = "NO enemies" if phase["no_enemies"] else "with enemies"

            old_eps   = agent.eps
            agent.eps = max(EPS_RESET_MIN, agent.eps * EPS_RESET_FACTOR)
            phase_start_ep = ep + 1
            last_bump_ep   = ep  # reset cooldown after phase transition

            print("\n[train] == Curriculum -> {} ({}) (win_rate={:.1%}) ==".format(
                phase["name"], enemies_label, win_rate))
            print("[train]    eps: {:.3f} -> {:.3f}  (reset for exploration)".format(
                old_eps, agent.eps))
            print("[train]    Phase ran for {} episodes".format(phase_episodes))

            agent.clear_buffer()
            print()

            env.close()
            env = GameInterface(map_size=phase["map_size"],
                                no_enemies=phase["no_enemies"],
                                player_hp=phase["player_hp"])
            win_history.clear()

    env.close()
    agent.save(paths["last_model"])
    print("\n[train/{}] Done. Best reward: {:.2f}".format(args.algo, agent.best_reward))
    print("[train/{}] Files: {}".format(args.algo, paths["best_model"]))


if __name__ == "__main__":
    train()
