"""
play.py — Воспроизведение обученной модели

Загружает best.pt и запускает игру в C# (с окном SFML),
отображая действия агента через keyboard-эмуляцию.

Запуск:
    python play.py                   — использовать best.pt
    python play.py --model checkpoints/last.pt
    python play.py --episodes 5      — сыграть 5 эпизодов
    python play.py --greedy          — без случайности (ε=0)

Требование: установлен пакет pynput
    pip install pynput
"""

import argparse
import time
import os

import numpy as np
import torch

from model          import DQN, NUM_ACTIONS, OBS_SIZE
from game_interface import GameInterface

# ── Маппинг действий на клавиши для визуального режима ───────────────────────
# (Используется только в --visual режиме, пока не реализован)
ACTION_NAMES = ["Up", "Down", "Left", "Right", "Melee", "Arrow", "Dash", "Idle"]

DEFAULT_MODEL = os.path.join("checkpoints", "best.pt")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",    type=str, default=DEFAULT_MODEL)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--greedy",   action="store_true",
                   help="ε=0 — полностью жадная политика")
    p.add_argument("--eps",      type=float, default=0.05,
                   help="ε для исследования во время воспроизведения")
    return p.parse_args()


def load_model(path: str, device: str = "cpu") -> DQN:
    net = DQN(OBS_SIZE, NUM_ACTIONS)
    ckpt = torch.load(path, map_location=device)
    net.load_state_dict(ckpt["online"])
    net.eval()
    print(f"[play] Загружена модель: {path}")
    print(f"       Лучшая награда при обучении: {ckpt.get('best_reward', '?'):.2f}")
    return net


def select_action(net: DQN, obs: np.ndarray, eps: float, device: str) -> int:
    if np.random.rand() < eps:
        return np.random.randint(NUM_ACTIONS)
    t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        return int(net(t).argmax(dim=1).item())


def play():
    args   = parse_args()
    device = "cpu"
    eps    = 0.0 if args.greedy else args.eps

    if not os.path.exists(args.model):
        print(f"[play] Модель не найдена: {args.model}")
        print("       Сначала запустите train.py")
        return

    net = load_model(args.model, device)
    env = GameInterface()

    total_rewards = []

    for ep in range(1, args.episodes + 1):
        obs  = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps  = 0

        while not done:
            action = select_action(net, obs, eps, device)
            obs, reward, done = env.step(action)
            ep_reward += reward
            ep_steps  += 1

            # Небольшая задержка — чтобы можно было наблюдать в окне C#
            time.sleep(0.05)

        total_rewards.append(ep_reward)
        print(
            f"[ep {ep:>3}]  reward={ep_reward:>7.2f}  steps={ep_steps:>4}  "
            f"action={ACTION_NAMES[action]}"
        )

    env.close()
    print(f"\nСредняя награда за {args.episodes} эпизодов: {np.mean(total_rewards):.2f}")


if __name__ == "__main__":
    play()
