"""
train.py — Главный обучающий цикл DQN

Запуск:
    cd agent
    pip install -r requirements.txt
    python train.py

Параметры командной строки (опционально):
    --episodes 2000   — число эпизодов (по умолчанию 2000)
    --device   cuda   — устройство PyTorch (cpu / cuda)
    --resume          — продолжить с последнего чекпоинта
"""

import argparse
import os
import sys
import time

import numpy as np

from agent          import DQNAgent
from game_interface import GameInterface

# ── Пути ──────────────────────────────────────────────────────────────────────
CKPT_DIR   = "checkpoints"
BEST_MODEL = os.path.join(CKPT_DIR, "best.pt")
LAST_MODEL = os.path.join(CKPT_DIR, "last.pt")
LOG_FILE   = os.path.join(CKPT_DIR, "train_log.csv")

# ── Аргументы ─────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int,  default=2000)
    p.add_argument("--device",   type=str,  default="cpu")
    p.add_argument("--resume",   action="store_true")
    return p.parse_args()


# ── Основной цикл ─────────────────────────────────────────────────────────────
def train():
    args = parse_args()
    os.makedirs(CKPT_DIR, exist_ok=True)

    agent = DQNAgent(device=args.device)
    env   = GameInterface(seed_start=0)

    if args.resume and os.path.exists(LAST_MODEL):
        agent.load(LAST_MODEL)
        print(f"[train] Продолжение с {LAST_MODEL}, ε={agent.eps:.3f}")

    # Лог-файл
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write("episode,total_reward,steps,epsilon,loss_mean\n")

    print(f"[train] Устройство: {args.device} | Эпизодов: {args.episodes}")
    print("-" * 60)

    global_step = 0
    t0 = time.time()

    for ep in range(1, args.episodes + 1):
        obs   = env.reset()
        done  = False
        ep_reward  = 0.0
        ep_steps   = 0
        ep_losses  = []

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, done = env.step(action)

            agent.store(obs, action, reward, next_obs, done)
            loss = agent.train_step()
            if loss is not None:
                ep_losses.append(loss)

            obs         = next_obs
            ep_reward  += reward
            ep_steps   += 1
            global_step += 1

        # Затухание ε раз в эпизод
        agent.decay_epsilon()

        loss_mean = float(np.mean(ep_losses)) if ep_losses else 0.0

        # Сохраняем лучшую модель
        if ep_reward > agent.best_reward:
            agent.best_reward = ep_reward
            agent.save(BEST_MODEL)

        # Периодически сохраняем последний чекпоинт
        if ep % 50 == 0:
            agent.save(LAST_MODEL)

        # Лог
        with open(LOG_FILE, "a") as f:
            f.write(f"{ep},{ep_reward:.3f},{ep_steps},{agent.eps:.4f},{loss_mean:.6f}\n")

        # Консольный вывод каждые 10 эпизодов
        if ep % 10 == 0:
            elapsed = time.time() - t0
            print(
                f"[ep {ep:>5}]  reward={ep_reward:>7.2f}  "
                f"steps={ep_steps:>4}  ε={agent.eps:.3f}  "
                f"loss={loss_mean:.5f}  elapsed={elapsed:.0f}s"
            )

    env.close()
    agent.save(LAST_MODEL)
    print(f"\n[train] Готово. Лучшая награда: {agent.best_reward:.2f}")
    print(f"[train] Лучшая модель: {BEST_MODEL}")


if __name__ == "__main__":
    train()
