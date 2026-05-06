"""
train.py — Главный обучающий цикл DQN с курикулум-обучением

Запуск:
    cd agent
    pip install -r requirements.txt
    python train.py                      # ddqn по умолчанию
    python train.py --algo dqn           # vanilla DQN
    python train.py --algo qlearn        # Q-Learning без буфера
    python train.py --algo ddqn --resume # продолжить обучение DDQN

Алгоритмы (--algo):
    qlearn — нейросетевой Q-Learning: нет буфера, нет target-сети,
             обновление после каждого шага. Самый нестабильный.
    dqn    — Vanilla DQN: буфер + target-сеть. Max Q_target для цели.
    ddqn   — Double DQN (умолчание): буфер + target-сеть.
             Online выбирает действие, target оценивает → меньше overestimation.

Файлы чекпоинтов (algo-specific):
    checkpoints/best_{algo}.pt
    checkpoints/last_{algo}.pt
    checkpoints/best_episode_{algo}.pkl

Курикулум (автоматический переход при достижении win_rate):
    Фаза 1 (16×16): win_rate < 30%   → простая карта, враги есть
    Фаза 2 (20×20): win_rate 30–50%  → карта чуть больше
    Фаза 3 (24×24): win_rate 50–65%  → средняя карта
    Фаза 4 (32×32): win_rate ≥ 65%   → полная сложность
"""

import argparse
import collections
import os
import pickle
import time

import numpy as np

from agent          import make_agent, ALGO_CHOICES
from game_interface import GameInterface

# ── Курикулум ──────────────────────────────────────────────────────────────────
CURRICULUM = [
    {"map_size": 16, "win_threshold": 999.0,  "name": "Фаза 1 (16×16)"},
    # {"map_size": 20, "win_threshold": 0.50,  "name": "Фаза 2 (20×20)"},
    # {"map_size": 24, "win_threshold": 0.65,  "name": "Фаза 3 (24×24)"},
    # {"map_size": 32, "win_threshold": 999.0, "name": "Фаза 4 (32×32)"},
]
WIN_WINDOW = 200   # скользящее окно для win_rate

CKPT_DIR   = "checkpoints"


# ── Имена файлов чекпоинтов (зависят от алгоритма) ───────────────────────────
def ckpt_paths(algo: str) -> dict[str, str]:
    """Возвращает пути чекпоинтов для выбранного алгоритма."""
    return {
        "best_model":    os.path.join(CKPT_DIR, f"best_{algo}.pt"),
        "last_model":    os.path.join(CKPT_DIR, f"last_{algo}.pt"),
        "log_file":      os.path.join(CKPT_DIR, f"train_log_{algo}.csv"),
        "best_episode":  os.path.join(CKPT_DIR, f"best_episode_{algo}.pkl"),
    }


# ── Аргументы ─────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__
    )
    p.add_argument("--algo",     type=str,  default="ddqn",
                   choices=ALGO_CHOICES,
                   help="Алгоритм: qlearn | dqn | ddqn  (default: ddqn)")
    p.add_argument("--episodes", type=int,  default=50_000)
    p.add_argument("--device",   type=str,  default="cpu")
    p.add_argument("--resume",   action="store_true",
                   help="Продолжить обучение с last_{algo}.pt")
    return p.parse_args()


def is_win(done: bool, last_state: dict, ep_reward: float) -> bool:
    """Победа = эпизод завершён, игрок жив, суммарная награда > 50."""
    return done and last_state.get("player_hp", 0) > 0 and ep_reward > 50.0


# ── Основной цикл ─────────────────────────────────────────────────────────────
def train():
    args  = parse_args()
    paths = ckpt_paths(args.algo)
    os.makedirs(CKPT_DIR, exist_ok=True)

    # Создаём агента нужного типа
    agent = make_agent(args.algo, device=args.device)

    # Курикулум и env
    phase_idx   = 0
    phase       = CURRICULUM[phase_idx]
    env         = GameInterface(map_size=phase["map_size"])
    win_history: collections.deque[int] = collections.deque(maxlen=WIN_WINDOW)

    # Resume
    if args.resume and os.path.exists(paths["last_model"]):
        agent.load(paths["last_model"])
        print(f"[train] Продолжение с {paths['last_model']}, ε={agent.eps:.3f}")

    # Лог-файл (algo-specific)
    if not os.path.exists(paths["log_file"]):
        with open(paths["log_file"], "w") as f:
            f.write("episode,total_reward,steps,epsilon,loss_mean,"
                    "win_rate,phase,map_size,ep_seed\n")

    algo_label = {
        "qlearn": "Q-Learning  (нет буфера, нет target)",
        "dqn":    "Vanilla DQN (буфер + target, vanilla max)",
        "ddqn":   "Double DQN  (буфер + target, online выбирает)",
    }[args.algo]

    print(f"[train] Алгоритм: {algo_label}")
    print(f"[train] Устройство: {args.device} | Эпизодов: {args.episodes}")
    print(f"[train] Курикулум: {phase['name']}")
    print(f"[train] Чекпоинты: best={paths['best_model']}")
    print("-" * 70)

    t0 = time.time()

    for ep in range(1, args.episodes + 1):
        obs      = env.reset()
        ep_seed  = env.episode_seed
        done     = False
        ep_reward  = 0.0
        ep_steps   = 0
        ep_losses  = []
        ep_actions: list[int] = []

        while not done:
            action = agent.select_action(obs)
            ep_actions.append(int(action))
            next_obs, reward, done = env.step(action)

            # ── Обычное сохранение в буфер ──────────────────────────────────
            agent.store(obs, action, reward, next_obs, done)

            # ── PER: дублируем победный переход ×10 (только с буфером) ──────
            # Q-Learning не использует буфер — PER там бессмысленен.
            if agent.has_replay_buffer and done and reward >= 80.0:
                for _ in range(9):   # 1 обычный + 9 дублей = ×10
                    agent.store(obs, action, reward, next_obs, done)

            loss = agent.train_step()
            if loss is not None:
                ep_losses.append(loss)

            obs        = next_obs
            ep_reward += reward
            ep_steps  += 1

        agent.decay_epsilon()

        loss_mean = float(np.mean(ep_losses)) if ep_losses else 0.0

        # ── Win rate ─────────────────────────────────────────────────────────
        won = is_win(done, env.last_state, ep_reward)
        win_history.append(int(won))
        win_rate = sum(win_history) / len(win_history) if win_history else 0.0

        # ── Лучшая модель + запись эпизода ───────────────────────────────────
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
            print(f"[train/{args.algo}] ★ Рекорд: {ep_reward:.2f}  "
                  f"(ep={ep}, seed={ep_seed}, шагов={ep_steps}, "
                  f"win_rate={win_rate:.1%})")

        # Периодический чекпоинт
        if ep % 100 == 0:
            agent.save(paths["last_model"])

        # ── Лог ─────────────────────────────────────────────────────────────
        with open(paths["log_file"], "a") as f:
            f.write(
                f"{ep},{ep_reward:.3f},{ep_steps},{agent.eps:.4f},"
                f"{loss_mean:.6f},{win_rate:.4f},"
                f"{phase_idx+1},{phase['map_size']},{ep_seed}\n"
            )

        # ── Консольный вывод ─────────────────────────────────────────────────
        if ep % 100 == 0:
            elapsed = time.time() - t0
            print(
                f"[ep {ep:>6}/{args.episodes}]  [{args.algo}]  "
                f"reward={ep_reward:>7.2f}  steps={ep_steps:>4}  "
                f"ε={agent.eps:.3f}  loss={loss_mean:.5f}  "
                f"win_rate={win_rate:.1%}  t={elapsed:.0f}s  seed={ep_seed}"
            )
        elif ep % 10 == 0 and won:
            print(f"[ep {ep:>6}]  [{args.algo}] ПОБЕДА  "
                  f"reward={ep_reward:.2f}  steps={ep_steps}  "
                  f"win_rate={win_rate:.1%}")

        # ── Курикулум ────────────────────────────────────────────────────────
        if (phase_idx < len(CURRICULUM) - 1
                and len(win_history) >= WIN_WINDOW
                and win_rate >= phase["win_threshold"]):
            phase_idx += 1
            phase = CURRICULUM[phase_idx]
            print(
                f"\n[train] ══ Курикулум → {phase['name']} "
                f"(win_rate={win_rate:.1%}) ══\n"
            )
            env.close()
            env = GameInterface(map_size=phase["map_size"])
            win_history.clear()

    env.close()
    agent.save(paths["last_model"])
    print(f"\n[train/{args.algo}] Готово. Лучшая награда: {agent.best_reward:.2f}")
    print(f"[train/{args.algo}] Файлы: {paths['best_model']}")


if __name__ == "__main__":
    train()
