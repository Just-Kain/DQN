"""
visualize.py — Статическая визуализация обученной DQN-модели.

Показывает три панели:
  1. Карта внимания первого слоя (17×17 spatial heatmap)
     Усреднённая абсолютная величина весов |W₁| по первым 289 входам
     (локальный вид 17×17), свёрнутая в 2D-карту.
     Яркие пиксели → нейрон «обращает внимание» на эту позицию вида.

  2. Гистограмма Q-значений для нулевого состояния.

  3. История наград из train_log.csv.

Запуск:
    cd agent
    python visualize.py                        # использует best.pt
    python visualize.py --model checkpoints/last.pt
    python visualize.py --show-neurons 16      # показать 16 нейронов (4×4)
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

# Idle исключён из пространства действий агента
ACTION_NAMES  = ["Up", "Down", "Left", "Right", "Melee", "Arrow", "Dash"]
DEFAULT_MODEL = os.path.join("checkpoints", "best_ddqn.pt")

VIEW      = 17   # 17×17 локальный вид
MINI_SIZE = 8    # 8×8 мини-карта (признаки [289..352] входного вектора)


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algo",         type=str, default="ddqn",
                   choices=("qlearn", "dqn", "ddqn"),
                   help="Алгоритм (выбирает best_{algo}.pt если --model не задан)")
    p.add_argument("--model",        type=str, default=None,
                   help="Явный путь к .pt (переопределяет --algo)")
    p.add_argument("--show-neurons", type=int, default=16,
                   help="Сколько нейронов первого слоя показать")
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
    W1   = layers[0].weight.detach().numpy()    # (512, 357)
    Wout = layers[-1].weight.detach().numpy()   # (7, 256)
    return W1, Wout


def spatial_attention_map(W1: np.ndarray) -> np.ndarray:
    """
    Усреднённая |W1| по первым 289 признакам (17×17 вид) → форма (17, 17).
    Показывает, куда 'смотрит' первый слой в локальном виде игрока.
    """
    n_spatial = VIEW * VIEW                     # 289
    spatial   = np.abs(W1[:, :n_spatial])       # (512, 289)
    mean_map  = spatial.mean(axis=0)            # (289,)
    return mean_map.reshape(VIEW, VIEW)         # (17, 17)


def neuron_weight_tiles(W1: np.ndarray, n: int):
    """
    Возвращает n карт весов (17×17) для первых n нейронов первого слоя.
    """
    n_spatial = VIEW * VIEW
    spatial   = W1[:n, :n_spatial]             # (n, 289)
    return spatial.reshape(n, VIEW, VIEW)


def load_reward_history(log_path: str):
    """Читает reward и win_rate из CSV-лога обучения."""
    if not os.path.exists(log_path):
        return None, None, None
    eps, rewards, win_rates = [], [], []
    with open(log_path) as f:
        next(f)   # заголовок
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            try:
                eps.append(int(parts[0]))
                rewards.append(float(parts[1]))
                win_rates.append(float(parts[5]) if len(parts) > 5 else 0.0)
            except ValueError:
                pass
    return np.array(eps), np.array(rewards), np.array(win_rates)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    args       = parse_args()
    model_path = args.model or os.path.join("checkpoints", f"best_{args.algo}.pt")
    log_path   = os.path.join("checkpoints", f"train_log_{args.algo}.csv")

    if not os.path.exists(model_path):
        print(f"[visualize] Модель не найдена: {model_path}")
        print(f"            Запустите: python train.py --algo {args.algo}")
        return

    net = load_model(model_path)
    W1, Wout = get_layer_weights(net)

    n_neurons = args.show_neurons
    cols = max(1, int(np.sqrt(n_neurons)))
    rows = (n_neurons + cols - 1) // cols

    eps_hist, rew_hist, win_hist = load_reward_history(log_path)

    # ── Компоновка фигуры ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"DQN weight analysis  |  {args.model}", fontweight="bold")

    gs_top = gridspec.GridSpec(1, 3, figure=fig, top=0.90, bottom=0.50,
                               wspace=0.35)
    gs_bot = gridspec.GridSpec(1, 2, figure=fig, top=0.44, bottom=0.06,
                               wspace=0.35)

    # ── 1. Spatial attention (верх, левая) ───────────────────────────────────
    ax_att = fig.add_subplot(gs_top[0, 0])
    att = spatial_attention_map(W1)
    im1 = ax_att.imshow(att, cmap="hot", interpolation="nearest")
    ax_att.set_title(f"Внимание первого слоя\n(среднее |W₁| по {VIEW}×{VIEW} виду)")
    ax_att.set_xlabel("dx от игрока")
    ax_att.set_ylabel("dy от игрока")
    half = VIEW // 2
    ax_att.set_xticks(range(VIEW))
    ax_att.set_xticklabels(range(-half, half + 1), fontsize=6)
    ax_att.set_yticks(range(VIEW))
    ax_att.set_yticklabels(range(-half, half + 1), fontsize=6)
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
    ax_wout.set_title(f"Выходной слой W_out\n(7 действий × 256 нейронов)")
    ax_wout.set_xlabel("Нейрон скрытого слоя (256)")
    ax_wout.set_ylabel("Действие")
    ax_wout.set_yticks(range(NUM_ACTIONS))
    ax_wout.set_yticklabels(ACTION_NAMES, fontsize=7)
    plt.colorbar(im3, ax=ax_wout, fraction=0.046, pad=0.04)

    # ── 4. Сетка нейронов первого слоя (низ, левая) ─────────────────────────
    ax_grid = fig.add_subplot(gs_bot[0, 0])
    ax_grid.axis("off")

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

    # ── 5. История наград + win_rate (низ, правая) ───────────────────────────
    ax_rew = fig.add_subplot(gs_bot[0, 1])
    if eps_hist is not None and len(eps_hist) > 1:
        window   = max(1, len(rew_hist) // 50)
        smoothed = np.convolve(rew_hist, np.ones(window) / window, mode="valid")
        ep_sm    = eps_hist[:len(smoothed)]

        ax_rew.plot(eps_hist, rew_hist, alpha=0.2, color="#90CAF9", linewidth=0.7)
        ax_rew.plot(ep_sm, smoothed, color="#1565C0", linewidth=1.5,
                    label=f"reward (окно={window})")
        ax_rew.set_xlabel("Эпизод")
        ax_rew.set_ylabel("Суммарная награда", color="#1565C0")

        if win_hist is not None and win_hist.max() > 0:
            ax2 = ax_rew.twinx()
            w_sm = np.convolve(win_hist, np.ones(window) / window, mode="valid")
            ax2.plot(ep_sm, w_sm * 100, color="#E53935", linewidth=1.2,
                     label="win_rate %")
            ax2.set_ylabel("Win Rate %", color="#E53935")
            ax2.tick_params(axis="y", labelcolor="#E53935")

        ax_rew.set_title("История наград и win_rate")
        ax_rew.legend(fontsize=8, loc="upper left")
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
