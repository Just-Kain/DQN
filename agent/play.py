"""
play.py — Запуск обученной модели с визуальным окном игры

Загружает чекпоинт и запускает C# с SFML-окном (--ai-visual),
выводя в консоль каждый шаг: действие, награду, HP, расстояние до выхода.

Использование:
    cd agent
    python play.py                          # best_ddqn.pt, 5 эпизодов, delay=0.15
    python play.py --algo dqn               # best_dqn.pt
    python play.py --episodes 10            # 10 эпизодов
    python play.py --delay 0.3             # медленнее (удобно для наблюдения)
    python play.py --delay 0               # максимальная скорость
    python play.py --eps 0.0               # без случайности (чисто жадная политика)
    python play.py --map-size 16           # тест на маленькой карте
    python play.py --seed 14553            # конкретный seed (лучший эпизод)
    python play.py --replay                # воспроизвести best_episode_{algo}.pkl

Параметры карты по умолчанию:
    map_size=32, player_hp=50 — соответствуют фазе 5, на которой обучена модель.
"""

import argparse
import os
import pickle
import time

import numpy as np
import torch

from agent          import make_agent, ALGO_CHOICES
from game_interface import GameInterface

ACTION_NAMES = {
    0: "Up    ",
    1: "Down  ",
    2: "Left  ",
    3: "Right ",
    4: "Melee ",
    5: "Arrow ",
    6: "Dash  ",
}

CKPT_DIR = "checkpoints"


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--algo",       type=str,   default="ddqn", choices=ALGO_CHOICES)
    p.add_argument("--episodes",   type=int,   default=5)
    p.add_argument("--delay",      type=float, default=0.05,
                   help="Задержка между ходами (сек). 0 = максимальная скорость.")
    p.add_argument("--eps",        type=float, default=0.30,
                   help="Epsilon для случайных действий (0 = полностью жадная политика)")
    p.add_argument("--map-size",   type=int,   default=32)
    p.add_argument("--player-hp",  type=int,   default=50)
    p.add_argument("--no-enemies", action="store_true")
    p.add_argument("--seed",       type=int,   default=None,
                   help="Фиксированный seed для всех эпизодов")
    p.add_argument("--replay",     action="store_true",
                   help="Воспроизвести лучший записанный эпизод (best_episode_{algo}.pkl)")
    p.add_argument("--last",       action="store_true",
                   help="Загрузить last_{algo}.pt вместо best_{algo}.pt")
    return p.parse_args()


def load_best_episode(algo):
    path = os.path.join(CKPT_DIR, f"best_episode_{algo}.pkl")
    if not os.path.exists(path):
        print(f"[play] Файл не найден: {path}")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def play():
    args = parse_args()

    # ── Выбор чекпоинта ───────────────────────────────────────────────────
    prefix     = "last" if args.last else "best"
    model_path = os.path.join(CKPT_DIR, f"{prefix}_{args.algo}.pt")

    if not os.path.exists(model_path):
        print(f"[play] Модель не найдена: {model_path}")
        print("       Сначала запустите train.py")
        return

    # ── Режим воспроизведения записанного эпизода ──────────────────────────
    if args.replay:
        ep_data = load_best_episode(args.algo)
        if ep_data is None:
            return

        print(f"\n[play] === Replay лучшего эпизода ({args.algo}) ===")
        print(f"       seed={ep_data['seed']}  reward={ep_data['total_reward']:.2f}"
              f"  steps={ep_data['steps']}  ep=#{ep_data['episode']}"
              f"  map={ep_data['map_size']}x{ep_data['map_size']}")
        print()

        env = GameInterface(
            map_size=ep_data["map_size"],
            no_enemies=False,
            player_hp=args.player_hp,
            visual_mode=True,
        )

        obs  = env.reset_to_seed(ep_data["seed"])
        done = False

        for step_i, action in enumerate(ep_data["actions"], start=1):
            state  = env.last_state
            hp     = state.get("player_hp", "?")
            max_hp = state.get("max_hp", args.player_hp)

            obs, reward, done = env.step(action)

            aname = ACTION_NAMES.get(action, str(action))
            print(f"  step {step_i:>3} | {aname} | reward={reward:>7.3f}"
                  f" | HP={hp}/{max_hp}")

            if args.delay > 0:
                time.sleep(args.delay)
            if done:
                break

        final  = env.last_state
        hp     = final.get("player_hp", 0)
        result = "WIN" if hp > 0 else "DEAD"
        print(f"\n  Результат: {result}")
        env.close()
        return

    # ── Обычный режим: агент играет сам ───────────────────────────────────
    agent = make_agent(args.algo, device="cpu")
    agent.load(model_path)
    agent.eps = args.eps

    ckpt               = torch.load(model_path, map_location="cpu")
    best_reward_trained = ckpt.get("best_reward", "?")

    print(f"\n[play] Алгоритм  : {args.algo.upper()}")
    print(f"[play] Модель    : {model_path}")
    if isinstance(best_reward_trained, float):
        print(f"[play] Best reward (при обучении): {best_reward_trained:.2f}")
    print(f"[play] Карта     : {args.map_size}x{args.map_size}"
          f"  HP={args.player_hp}"
          f"  enemies={'нет' if args.no_enemies else 'да'}")
    print(f"[play] Epsilon   : {agent.eps}")
    print(f"[play] Delay     : {args.delay}s между ходами")
    print(f"[play] Эпизодов  : {args.episodes}")
    print("-" * 60)

    env = GameInterface(
        map_size=args.map_size,
        no_enemies=args.no_enemies,
        player_hp=args.player_hp,
        visual_mode=True,
    )

    results = []

    for ep in range(1, args.episodes + 1):
        if args.seed is not None:
            obs  = env.reset_to_seed(args.seed)
            seed = args.seed
        else:
            obs  = env.reset()
            seed = env.episode_seed

        done      = False
        ep_reward = 0.0
        ep_steps  = 0

        print(f"\n[ep {ep}/{args.episodes}] seed={seed}")
        print(f"  {'step':>4} | {'action':<8} | {'reward':>8} | {'HP':>6} | dist")
        print(f"  {'-'*4}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+------")

        while not done:
            state  = env.last_state
            hp     = state.get("player_hp", "?")
            max_hp = state.get("max_hp", args.player_hp)
            px     = state.get("player_x", 0)
            py     = state.get("player_y", 0)
            ex     = state.get("exit_x", 0)
            ey     = state.get("exit_y", 0)
            dist   = abs(ex - px) + abs(ey - py)

            action = agent.select_action(obs)
            obs, reward, done = env.step(action)

            ep_reward += reward
            ep_steps  += 1

            aname = ACTION_NAMES.get(action, str(action))
            print(f"  {ep_steps:>4} | {aname} | {reward:>8.3f} |"
                  f" {hp}/{max_hp:>2} | {dist:>3}")

            if args.delay > 0:
                time.sleep(args.delay)

        final  = env.last_state
        hp_end = final.get("player_hp", 0)
        if hp_end > 0 and ep_reward > 50:
            result = "WIN"
        elif hp_end <= 0:
            result = "DEAD"
        else:
            result = "TIMEOUT"
        results.append(result == "WIN")

        print(f"\n  [{result}]  total_reward={ep_reward:.2f}  steps={ep_steps}")

    env.close()

    wins = sum(results)
    print("\n" + "=" * 60)
    print(f"Итого: {wins}/{args.episodes} побед"
          f"  ({wins / args.episodes:.0%} win_rate)")
    for i, (won, r) in enumerate(zip(results,
            [None] * args.episodes), start=1):
        mark = "WIN " if won else "LOSE"
        print(f"  ep {i}: {mark}")


if __name__ == "__main__":
    play()
