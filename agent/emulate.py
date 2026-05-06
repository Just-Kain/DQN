"""
emulate.py — Режим эмуляции: SFML игра + живая визуализация весов/Q-значений.

Режим 1 — живая игра (по умолчанию):
  Загружает best.pt и играет N эпизодов, отображая в matplotlib
  Q-значения, карту внимания и историю наград.

Режим 2 — воспроизведение лучшего эпизода (--replay):
  Загружает записанную последовательность действий из best_episode.pkl,
  отправляет C# ровно тот же сид и те же действия.

Запуск:
    cd agent
    python emulate.py                              # best.pt, 10 эпизодов
    python emulate.py --model checkpoints/last.pt
    python emulate.py --episodes 5 --delay 0.1
    python emulate.py --greedy                     # ε=0
    python emulate.py --replay                     # воспроизвести лучший эпизод
    python emulate.py --replay --replay-path checkpoints/best_episode.pkl
"""

import argparse
import os
import pickle
import time

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from model          import DQN, OBS_SIZE, NUM_ACTIONS
from game_interface import GameInterface

matplotlib.rcParams.update({"font.size": 9})

# Idle исключён из пространства действий агента (NUM_ACTIONS=7)
ACTION_NAMES    = ["Up", "Down", "Left", "Right", "Melee", "Arrow", "Dash"]
DEFAULT_MODEL   = os.path.join("checkpoints", "best_ddqn.pt")
DEFAULT_EPISODE = os.path.join("checkpoints", "best_episode_ddqn.pkl")

VIEW      = 17
MINI_SIZE = 8


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Emulate / replay DQN agent in SFML + matplotlib"
    )
    p.add_argument("--algo",         type=str,   default="ddqn",
                   choices=("qlearn", "dqn", "ddqn"),
                   help="Алгоритм: выбирает best_{algo}.pt и best_episode_{algo}.pkl "
                        "если --model / --replay-path не заданы явно")
    p.add_argument("--model",        type=str,   default=None,
                   help="Явный путь к .pt (переопределяет --algo)")
    p.add_argument("--episodes",     type=int,   default=10)
    p.add_argument("--delay",        type=float, default=0.05)
    p.add_argument("--greedy",       action="store_true")
    p.add_argument("--eps",          type=float, default=0.02)
    p.add_argument("--map-size",     type=int,   default=16,
                   help="Размер карты (должен совпадать с тем, на чём обучалась модель)")
    p.add_argument("--replay",       action="store_true")
    p.add_argument("--replay-path",  type=str,   default=None,
                   help="Явный путь к .pkl (переопределяет --algo)")
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
    """Усреднённое |W₁| для первых 289 признаков (17×17)."""
    layers = [m for m in net.net if isinstance(m, torch.nn.Linear)]
    W1     = layers[0].weight.detach().numpy()         # (512, 357)
    n_sp   = VIEW * VIEW                               # 289
    spatial = np.abs(W1[:, :n_sp]).mean(axis=0).reshape(VIEW, VIEW)
    return spatial


# ─────────────────────────────────────────────────────────────────────────────
class LivePlot:
    """
    Matplotlib окно с тремя панелями.
    Поддерживает отображение «записанного» действия vs. выбор модели.
    """
    def __init__(self, att_map: np.ndarray, title: str = "DQN Emulation — live"):
        plt.ion()
        self.fig = plt.figure(figsize=(12, 7))
        self.fig.suptitle(title, fontweight="bold")

        gs = gridspec.GridSpec(2, 2, figure=self.fig,
                               hspace=0.45, wspace=0.35)

        # ── Карта внимания (17×17) ──────────────────────────────────────────
        self.ax_att = self.fig.add_subplot(gs[0, 0])
        self.im_att = self.ax_att.imshow(att_map, cmap="hot",
                                          interpolation="nearest")
        self.ax_att.set_title(f"Внимание W₁ ({VIEW}×{VIEW})")
        self.ax_att.set_xlabel("dx")
        self.ax_att.set_ylabel("dy")
        half   = VIEW // 2
        ticks  = list(range(VIEW))
        labels = [str(x) for x in range(-half, half + 1)]
        self.ax_att.set_xticks(ticks); self.ax_att.set_xticklabels(labels, fontsize=6)
        self.ax_att.set_yticks(ticks); self.ax_att.set_yticklabels(labels, fontsize=6)
        plt.colorbar(self.im_att, ax=self.ax_att, fraction=0.046)

        # ── Q-значения ───────────────────────────────────────────────────────
        self.ax_q = self.fig.add_subplot(gs[0, 1])
        self._q_vals = np.zeros(NUM_ACTIONS)
        self.bars = self.ax_q.bar(ACTION_NAMES, self._q_vals,
                                   color=["#2196F3"] * NUM_ACTIONS)
        self.ax_q.set_title("Q-значения (текущий шаг)")
        self.ax_q.set_ylabel("Q")
        self.ax_q.tick_params(axis="x", rotation=30)
        self._q_text = [
            self.ax_q.text(b.get_x() + b.get_width() / 2, 0, "",
                           ha="center", va="bottom", fontsize=7)
            for b in self.bars
        ]

        # ── История наград ────────────────────────────────────────────────────
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

    def update_q(self, q_vals: np.ndarray, chosen: int,
                 model_choice: int | None = None) -> None:
        """
        Цветовая схема:
          Зелёный   — выбранное и совпадает с моделью
          Оранжевый — записанное действие (модель выбрала бы другое)
          Красный   — что модель выбрала бы сейчас
          Синий     — прочие
        """
        if model_choice is None:
            model_choice = chosen

        for i, (bar, val) in enumerate(zip(self.bars, q_vals)):
            if i == chosen and i == model_choice:
                color = "#4CAF50"
            elif i == chosen:
                color = "#FF9800"
            elif i == model_choice:
                color = "#F44336"
            else:
                color = "#2196F3"

            bar.set_color(color)
            bar.set_height(val)
            self._q_text[i].set_position(
                (bar.get_x() + bar.get_width() / 2,
                 val + abs(q_vals.max() - q_vals.min()) * 0.02))
            self._q_text[i].set_text(f"{val:.2f}")

        margin = max(0.1, (q_vals.max() - q_vals.min()) * 0.15)
        self.ax_q.set_ylim(q_vals.min() - margin, q_vals.max() + margin)

    def end_episode(self, ep_reward: float) -> None:
        self._ep_rewards.append(ep_reward)
        xs = list(range(1, len(self._ep_rewards) + 1))
        self._line.set_xdata(xs)
        self._line.set_ydata(self._ep_rewards)
        self.ax_rew.relim()
        self.ax_rew.autoscale_view()

    def set_q_title(self, title: str) -> None:
        self.ax_q.set_title(title)

    def refresh(self) -> None:
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self) -> None:
        plt.ioff()
        plt.close(self.fig)


# ─────────────────────────────────────────────────────────────────────────────
def run_live(env: GameInterface, net: DQN, plot: LivePlot,
             episodes: int, eps: float, delay: float) -> list[float]:
    all_rewards = []

    for ep in range(1, episodes + 1):
        obs       = env.reset()
        done      = False
        ep_reward = 0.0
        ep_steps  = 0

        while not done:
            action, q_vals = select_action(net, obs, eps)
            plot.update_q(q_vals, action)
            plot.refresh()

            obs, reward, done = env.step(action)
            ep_reward += reward
            ep_steps  += 1

            if delay > 0:
                time.sleep(delay)

        plot.end_episode(ep_reward)
        plot.refresh()
        all_rewards.append(ep_reward)
        print(f"[ep {ep:>3}/{episodes}]  reward={ep_reward:>7.2f}  steps={ep_steps:>4}")

    return all_rewards


# ─────────────────────────────────────────────────────────────────────────────
def run_replay(env: GameInterface, net: DQN, plot: LivePlot,
               replay_path: str, delay: float) -> None:
    if not os.path.exists(replay_path):
        print(f"\n[replay] Файл записи не найден: {replay_path}")
        print("         Сначала запустите train.py")
        return

    with open(replay_path, "rb") as f:
        data = pickle.load(f)

    seed         = data["seed"]
    actions      = data["actions"]
    total_reward = data["total_reward"]
    episode      = data.get("episode", "?")
    steps        = data.get("steps", len(actions))
    map_size     = data.get("map_size", 16)

    print(f"\n[replay] ═══════════════════════════════════════")
    print(f"[replay]  Лучший эпизод:")
    print(f"[replay]    Эпизод    : {episode}")
    print(f"[replay]    Сид       : {seed}")
    print(f"[replay]    Размер карты: {map_size}×{map_size}")
    print(f"[replay]    Награда   : {total_reward:.2f}")
    print(f"[replay]    Шагов     : {steps}")
    print(f"[replay] ═══════════════════════════════════════\n")

    plot.set_q_title("Q-значения  |  🟢 совпадение  🟠 запись  🔴 модель")

    obs  = env.reset_to_seed(seed)
    done = False
    ep_reward = 0.0
    diverged  = 0

    for step_num, recorded_action in enumerate(actions):
        model_action, q_vals = select_action(net, obs, eps=0.0)

        if model_action != recorded_action:
            diverged += 1

        plot.update_q(q_vals, chosen=recorded_action, model_choice=model_action)
        plot.refresh()

        obs, reward, done = env.step(recorded_action)
        ep_reward += reward

        if delay > 0:
            time.sleep(delay)

        if done:
            break

    plot.end_episode(ep_reward)
    plot.refresh()

    match_pct = 100.0 * (1 - diverged / max(1, step_num + 1))
    print(f"[replay] Готово. Награда: {ep_reward:.2f}  ({step_num + 1} шагов)")
    print(f"[replay] Совпадение модели с записью: {match_pct:.1f}%  "
          f"({step_num + 1 - diverged}/{step_num + 1})")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    eps  = 0.0 if args.greedy else args.eps

    # Разрешаем пути: явный --model переопределяет --algo
    model_path   = args.model   or os.path.join("checkpoints", f"best_{args.algo}.pt")
    replay_path  = args.replay_path or os.path.join("checkpoints", f"best_episode_{args.algo}.pkl")

    if not os.path.exists(model_path):
        print(f"[emulate] Модель не найдена: {model_path}")
        print(f"          Запустите:  python train.py --algo {args.algo}")
        return

    net     = load_model(model_path)
    att_map = get_attention_map(net)

    # При --replay берём map_size из pkl если возможно
    replay_map_size = args.map_size
    if args.replay and os.path.exists(replay_path):
        try:
            with open(replay_path, "rb") as f:
                d = pickle.load(f)
            replay_map_size = d.get("map_size", args.map_size)
        except Exception:
            pass

    algo_label = args.algo.upper()
    mode_name  = (f"[{algo_label}] Воспроизведение лучшего эпизода"
                  if args.replay else f"[{algo_label}] DQN Emulation — live")
    print(f"[emulate/{args.algo}] Модель: {model_path}")
    print(f"[emulate] Запуск C# в --ai-visual режиме (map_size={replay_map_size})…")
    env  = GameInterface(visual_mode=True, map_size=replay_map_size)
    plot = LivePlot(att_map, title=mode_name)

    try:
        if args.replay:
            run_replay(env, net, plot,
                       replay_path=replay_path,
                       delay=args.delay)
        else:
            all_rewards = run_live(env, net, plot,
                                   episodes=args.episodes,
                                   eps=eps,
                                   delay=args.delay)
            if all_rewards:
                print(f"\nСредняя награда: {np.mean(all_rewards):.2f}  "
                      f"(лучшая: {max(all_rewards):.2f})")

    except KeyboardInterrupt:
        print("\n[emulate] Прервано пользователем.")

    finally:
        env.close()
        plot.close()


if __name__ == "__main__":
    main()
