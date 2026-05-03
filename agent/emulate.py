"""
emulate.py — Режим эмуляции: SFML игра + живая визуализация весов/Q-значений.

Что делает:
  1. Запускает C# игру в режиме --ai-visual → открывается SFML-окно.
  2. Загружает best.pt (или указанную модель) и играет автоматически.
  3. В том же окне Python (matplotlib) показывает в реальном времени:
       • Текущие Q-значения для каждого действия (bar chart)
       • 9×9 heatmap внимания первого слоя (статичный, обновляется при старте)
       • Историю наград эпизода

Запуск:
    cd agent
    python emulate.py                          # best.pt, 10 эпизодов
    python emulate.py --model checkpoints/last.pt
    python emulate.py --episodes 5 --delay 0.1
    python emulate.py --greedy                 # ε=0

Зависимости: pip install matplotlib
"""

import argparse
import os
import time

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from model          import DQN, OBS_SIZE, NUM_ACTIONS
from game_interface import GameInterface

matplotlib.rcParams.update({"font.size": 9})

ACTION_NAMES = ["Up", "Down", "Left", "Right", "Melee", "Arrow", "Dash", "Idle"]
DEFAULT_MODEL = os.path.join("checkpoints", "best.pt")
VIEW = 9


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",    type=str,   default=DEFAULT_MODEL)
    p.add_argument("--episodes", type=int,   default=10)
    p.add_argument("--delay",    type=float, default=0.05,
                   help="Задержка между шагами (сек), чтобы видеть игру")
    p.add_argument("--greedy",   action="store_true",
                   help="ε=0 — полностью жадная политика")
    p.add_argument("--eps",      type=float, default=0.02)
    return p.parse_args()


def load_model(path: str) -> DQN:
    net  = DQN(OBS_SIZE, NUM_ACTIONS)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net.load_state_dict(ckpt["online"])
    net.eval()
    best = ckpt.get("best_reward", "?")
    print(f"[emulate] Модель: {path}  |  Лучшая награда: {best}")
    return net


def select_action(net: DQN, obs: np.ndarray, eps: float) -> tuple[int, np.ndarray]:
    """Возвращает (action_idx, q_values)."""
    t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q = net(t).numpy().flatten()
    if np.random.rand() < eps:
        return np.random.randint(NUM_ACTIONS), q
    return int(q.argmax()), q


def get_attention_map(net: DQN) -> np.ndarray:
    """Усреднённое |W₁| для первых 81 признаков (9×9)."""
    layers = [m for m in net.net if isinstance(m, torch.nn.Linear)]
    W1 = layers[0].weight.detach().numpy()       # (256, 85)
    spatial = np.abs(W1[:, :VIEW * VIEW]).mean(axis=0).reshape(VIEW, VIEW)
    return spatial


# ─────────────────────────────────────────────────────────────────────────────
class LivePlot:
    """
    Matplotlib окно с тремя панелями, обновляемое каждый шаг.
    """
    def __init__(self, att_map: np.ndarray):
        plt.ion()
        self.fig = plt.figure(figsize=(12, 7))
        self.fig.suptitle("DQN Emulation — live", fontweight="bold")

        gs = gridspec.GridSpec(2, 2, figure=self.fig,
                               hspace=0.45, wspace=0.35)

        # ── Карта внимания (статичная) ──────────────────────────────────────
        self.ax_att = self.fig.add_subplot(gs[0, 0])
        self.im_att = self.ax_att.imshow(att_map, cmap="hot",
                                          interpolation="nearest")
        self.ax_att.set_title("Внимание W₁ (9×9)")
        self.ax_att.set_xlabel("dx")
        self.ax_att.set_ylabel("dy")
        ticks = list(range(VIEW))
        labels = [str(x) for x in range(-(VIEW // 2), VIEW // 2 + 1)]
        self.ax_att.set_xticks(ticks); self.ax_att.set_xticklabels(labels, fontsize=7)
        self.ax_att.set_yticks(ticks); self.ax_att.set_yticklabels(labels, fontsize=7)
        plt.colorbar(self.im_att, ax=self.ax_att, fraction=0.046)

        # ── Q-значения (динамические) ────────────────────────────────────────
        self.ax_q = self.fig.add_subplot(gs[0, 1])
        self._q_vals = np.zeros(NUM_ACTIONS)
        self._bar_colors = ["#2196F3"] * NUM_ACTIONS
        self.bars = self.ax_q.bar(ACTION_NAMES, self._q_vals,
                                   color=self._bar_colors)
        self.ax_q.set_title("Q-значения (текущий шаг)")
        self.ax_q.set_ylabel("Q")
        self.ax_q.tick_params(axis="x", rotation=30)
        self._q_text = [
            self.ax_q.text(b.get_x() + b.get_width() / 2, 0, "",
                           ha="center", va="bottom", fontsize=7)
            for b in self.bars
        ]

        # ── История наград текущего эпизода ──────────────────────────────────
        self.ax_rew = self.fig.add_subplot(gs[1, :])
        self.ax_rew.set_title("Награды по эпизодам")
        self.ax_rew.set_xlabel("Эпизод")
        self.ax_rew.set_ylabel("Суммарная награда")
        self.ax_rew.grid(alpha=0.3)
        self._ep_rewards: list[float] = []
        self._line, = self.ax_rew.plot([], [], "o-", color="#1565C0",
                                        linewidth=1.5, markersize=4)

        self.fig.canvas.draw()
        plt.pause(0.01)

    def update_q(self, q_vals: np.ndarray, chosen: int) -> None:
        best_q = q_vals.max()
        for i, (bar, val) in enumerate(zip(self.bars, q_vals)):
            color = "#4CAF50" if i == chosen else (
                "#FF5722" if val == best_q else "#2196F3")
            bar.set_color(color)
            bar.set_height(val)
            self._q_text[i].set_position(
                (bar.get_x() + bar.get_width() / 2,
                 val + abs(q_vals.max() - q_vals.min()) * 0.02))
            self._q_text[i].set_text(f"{val:.2f}")

        # Авто-масштаб по Y
        margin = max(0.1, (q_vals.max() - q_vals.min()) * 0.15)
        self.ax_q.set_ylim(q_vals.min() - margin, q_vals.max() + margin)

    def end_episode(self, ep_reward: float) -> None:
        self._ep_rewards.append(ep_reward)
        xs = list(range(1, len(self._ep_rewards) + 1))
        self._line.set_xdata(xs)
        self._line.set_ydata(self._ep_rewards)
        self.ax_rew.relim()
        self.ax_rew.autoscale_view()

    def refresh(self) -> None:
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self) -> None:
        plt.ioff()
        plt.close(self.fig)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    eps  = 0.0 if args.greedy else args.eps

    if not os.path.exists(args.model):
        print(f"[emulate] Модель не найдена: {args.model}")
        print("          Сначала запустите train.py")
        return

    net     = load_model(args.model)
    att_map = get_attention_map(net)

    # ── Запуск C# в visual-режиме (откроется SFML-окно) ──────────────────────
    print("[emulate] Запуск C# игры в --ai-visual режиме…")
    env  = GameInterface(visual_mode=True)
    plot = LivePlot(att_map)

    all_rewards = []

    try:
        for ep in range(1, args.episodes + 1):
            obs      = env.reset()
            done     = False
            ep_reward = 0.0
            ep_steps  = 0

            while not done:
                action, q_vals = select_action(net, obs, eps)

                # Обновляем matplotlib
                plot.update_q(q_vals, action)
                plot.refresh()

                obs, reward, done = env.step(action)
                ep_reward += reward
                ep_steps  += 1

                if args.delay > 0:
                    time.sleep(args.delay)

            plot.end_episode(ep_reward)
            plot.refresh()
            all_rewards.append(ep_reward)

            print(
                f"[ep {ep:>3}/{args.episodes}]  "
                f"reward={ep_reward:>7.2f}  steps={ep_steps:>4}"
            )

    except KeyboardInterrupt:
        print("\n[emulate] Прервано пользователем.")

    finally:
        env.close()
        plot.close()

    if all_rewards:
        print(f"\nСредняя награда: {np.mean(all_rewards):.2f}  "
              f"(лучшая: {max(all_rewards):.2f})")


if __name__ == "__main__":
    main()
