"""
visualize.py — Статическая визуализация обученной DQN-модели.

Показывает три панели:
  1. Карта весов первого слоя (9×9 spatial heatmap)
     Усреднённая абсолютная величина весов |W₁| по первым 81 входам
     (локальный вид 9×9), свёрнутая в 2D-карту.
     Яркие пиксели → нейрон «обращает внимание» на эту позицию вида.

  2. Карта полных весов первого слоя (каждый нейрон — отдельный 9×9)
     Первые N_SHOW нейронов из 256 отображаются как мини-плитки.

  3. Гистограмма Q-значений для случайного состояния из replay-buffer
     (только если есть train_log.csv — иначе для нулевого состояния).

Запуск:
    cd agent
    python visualize.py                        # использует best.pt
    python visualize.py --model checkpoints/last.pt
    python visualize.py --show-neurons 32      # показать 32 нейрона (4×8)
"""

import argparse
import os

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from model import DQN, OBS_SIZE, NUM_ACTIONS

matplotlib.rcParams.update({"font.size": 9})

ACTION_NAMES = ["Up", "Down", "Left", "Right", "Melee", "Arrow", "Dash", "Idle"]
DEFAULT_MODEL = os.path.join("checkpoints", "best.pt")

VIEW = 9   # 9×9 локальный вид


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",        type=str, default=DEFAULT_MODEL)
    p.add_argument("--show-neurons", type=int, default=16,
                   help="Сколько нейронов первого слоя показать (квадратная сетка)")
    return p.parse_args()


def load_model(path: str) -> DQN:
    net  = DQN(OBS_SIZE, NUM_ACTIONS)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net.load_state_dict(ckpt["online"])
    net.eval()
    return net


def get_layer_weights(net: DQN):
    """Возвращает веса первого и последнего линейных слоёв."""
    layers = [m for m in net.net if isinstance(m, torch.nn.Linear)]
    W1 = layers[0].weight.detach().numpy()   # (256, 85)
    Wout = layers[-1].weight.detach().numpy() # (8, 128)
    return W1, Wout


def spatial_attention_map(W1: np.ndarray) -> np.ndarray:
    """
    Усреднённая |W1| по первым 81 признакам (9×9 вид) → форма (9, 9).
    Показывает, куда 'смотрит' первый слой в локальном виде игрока.
    """
    spatial = np.abs(W1[:, :VIEW * VIEW])     # (256, 81)
    mean_map = spatial.mean(axis=0)            # (81,)
    return mean_map.reshape(VIEW, VIEW)


def neuron_weight_tiles(W1: np.ndarray, n: int):
    """
    Возвращает n карт весов (9×9) для первых n нейронов первого слоя.
    Каждый нейрон: W1[i, :81].reshape(9,9).
    """
    spatial = W1[:n, :VIEW * VIEW]             # (n, 81)
    return spatial.reshape(n, VIEW, VIEW)


def load_reward_history(log_path: str):
    """Читает reward из CSV-лога обучения."""
    if not os.path.exists(log_path):
        return None, None
    eps, rewards = [], []
    with open(log_path) as f:
        next(f)   # пропускаем заголовок
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            try:
                eps.append(int(parts[0]))
                rewards.append(float(parts[1]))
            except ValueError:
                pass
    return np.array(eps), np.array(rewards)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    if not os.path.exists(args.model):
        print(f"[visualize] Модель не найдена: {args.model}")
        print("            Сначала запустите train.py")
        return

    net = load_model(args.model)
    W1, Wout = get_layer_weights(net)

    n_neurons = args.show_neurons
    cols = max(1, int(np.sqrt(n_neurons)))
    rows = (n_neurons + cols - 1) // cols

    eps_hist, rew_hist = load_reward_history(
        os.path.join("checkpoints", "train_log.csv"))

    # ── Компоновка фигуры ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(f"DQN weight analysis  |  {args.model}", fontweight="bold")

    gs_top = gridspec.GridSpec(1, 3, figure=fig, top=0.90, bottom=0.48,
                               wspace=0.35)
    gs_bot = gridspec.GridSpec(1, 2, figure=fig, top=0.42, bottom=0.07,
                               wspace=0.35)

    # ── 1. Spatial attention (верх, левая) ───────────────────────────────────
    ax_att = fig.add_subplot(gs_top[0, 0])
    att = spatial_attention_map(W1)
    im1 = ax_att.imshow(att, cmap="hot", interpolation="nearest")
    ax_att.set_title("Внимание первого слоя\n(среднее |W₁| по 9×9 виду)")
    ax_att.set_xlabel("dx от игрока")
    ax_att.set_ylabel("dy от игрока")
    ax_att.set_xticks(range(VIEW))
    ax_att.set_xticklabels(range(-(VIEW // 2), VIEW // 2 + 1), fontsize=7)
    ax_att.set_yticks(range(VIEW))
    ax_att.set_yticklabels(range(-(VIEW // 2), VIEW // 2 + 1), fontsize=7)
    plt.colorbar(im1, ax=ax_att, fraction=0.046, pad=0.04)

    # ── 2. Q-values для нулевого входа (верх, центр) ────────────────────────
    ax_q = fig.add_subplot(gs_top[0, 1])
    dummy_obs = torch.zeros(1, OBS_SIZE)
    with torch.no_grad():
        q_vals = net(dummy_obs).numpy().flatten()

    colors = ["#4CAF50" if v == q_vals.max() else "#2196F3" for v in q_vals]
    bars = ax_q.bar(ACTION_NAMES, q_vals, color=colors)
    ax_q.set_title("Q-значения (нулевое состояние)\nзелёный = лучшее действие")
    ax_q.set_ylabel("Q-value")
    ax_q.tick_params(axis="x", rotation=30)
    for bar, val in zip(bars, q_vals):
        ax_q.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + (abs(q_vals.max() - q_vals.min()) * 0.02),
                  f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    # ── 3. Выходные веса (верх, правая) ─────────────────────────────────────
    ax_wout = fig.add_subplot(gs_top[0, 2])
    im3 = ax_wout.imshow(Wout, cmap="RdBu", aspect="auto",
                          interpolation="nearest")
    ax_wout.set_title(f"Выходной слой W_out\n(8 действий × 128 нейронов)")
    ax_wout.set_xlabel("Нейрон скрытого слоя (128)")
    ax_wout.set_ylabel("Действие")
    ax_wout.set_yticks(range(NUM_ACTIONS))
    ax_wout.set_yticklabels(ACTION_NAMES, fontsize=7)
    plt.colorbar(im3, ax=ax_wout, fraction=0.046, pad=0.04)

    # ── 4. Сетка нейронов первого слоя (низ, левая) ─────────────────────────
    ax_grid = fig.add_subplot(gs_bot[0, 0])
    ax_grid.axis("off")
    ax_grid.set_title(f"Первые {n_neurons} нейронов W₁ (9×9)")

    tiles = neuron_weight_tiles(W1, n_neurons)
    vmax  = np.abs(tiles).max()
    pad   = 1
    h     = rows * VIEW + (rows - 1) * pad
    w     = cols * VIEW + (cols - 1) * pad
    canvas = np.zeros((h, w))
    for i in range(n_neurons):
        r, c = divmod(i, cols)
        ry = r * (VIEW + pad)
        cx = c * (VIEW + pad)
        canvas[ry:ry + VIEW, cx:cx + VIEW] = tiles[i]

    ax_grid.imshow(canvas, cmap="RdBu", vmin=-vmax, vmax=vmax,
                   interpolation="nearest")
    ax_grid.set_title(
        f"Пространственные веса первых {n_neurons} нейронов W₁\n"
        f"красный=+, синий=−")

    # ── 5. История наград (низ, правая) ─────────────────────────────────────
    ax_rew = fig.add_subplot(gs_bot[0, 1])
    if eps_hist is not None and len(eps_hist) > 1:
        # Сглаживание скользящим средним
        window = max(1, len(rew_hist) // 50)
        smoothed = np.convolve(rew_hist,
                               np.ones(window) / window, mode="valid")
        ep_smooth = eps_hist[:len(smoothed)]
        ax_rew.plot(eps_hist, rew_hist, alpha=0.25, color="#90CAF9", linewidth=0.8)
        ax_rew.plot(ep_smooth, smoothed, color="#1565C0", linewidth=1.5,
                    label=f"скользящее ср. (окно={window})")
        ax_rew.set_title("История наград за эпизод")
        ax_rew.set_xlabel("Эпизод")
        ax_rew.set_ylabel("Суммарная награда")
        ax_rew.legend(fontsize=8)
        ax_rew.grid(alpha=0.3)
    else:
        ax_rew.text(0.5, 0.5, "train_log.csv не найден\nили менее 2 эпизодов",
                    ha="center", va="center", transform=ax_rew.transAxes)
        ax_rew.set_title("История наград (нет данных)")

    plt.savefig("checkpoints/weights.png", dpi=150, bbox_inches="tight")
    print("[visualize] Сохранено: checkpoints/weights.png")
    plt.show()


if __name__ == "__main__":
    main()
