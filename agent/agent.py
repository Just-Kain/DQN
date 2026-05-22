"""
agent.py — Три алгоритма обучения с подкреплением
══════════════════════════════════════════════════════════════════════════════

╔══════════════════════════════════════════════════════════════════════════╗
║  СРАВНЕНИЕ АЛГОРИТМОВ                                                   ║
╚══════════════════════════════════════════════════════════════════════════╝

  Выбор через --algo {qlearn | dqn | ddqn} в train.py

  Алгоритм │ Буфер │ Target-сеть │ Выбор действия в target │ Оценка Q_next
  ─────────┼───────┼─────────────┼─────────────────────────┼──────────────
  qlearn   │  нет  │     нет     │     —  (online)         │  online
  dqn      │  да   │     да      │   target (max)          │  target
  ddqn     │  да   │     да      │   online (argmax)       │  target

────────────────────────────────────────────────────────────────────────────

1. Q-LEARNING (QLearningAgent)
   ─────────────────────────────
   Классический алгоритм Q-learning с нейросетью вместо таблицы.

   Нет буфера опыта — обновление СРАЗУ после каждого шага:

       Q(s, a) ← Q(s, a) + α · δ
       δ = r + γ · max_{a'} Q(s', a') − Q(s, a)

   В нейросетевой форме (gradient step):
       y  = r + γ · max_{a'} Q_online(s', a') · (1 − done)
       loss = HuberLoss(Q_online(s, a), y)

   Проблемы:
   • Нестабильность: сеть, используемая для вычисления цели y, одновременно
     обновляется — "движущаяся цель" (moving target).
   • Коррелированные переходы: нет разрыва временно́й корреляции.
   • Но: самый быстрый per-step (нет выборки из буфера).

2. VANILLA DQN (DQNAgent)
   ─────────────────────────
   Решает обе проблемы Q-learning:
   • Буфер опыта размером D разрывает корреляции.
   • Замороженная target-сеть θ⁻ стабилизирует цель:

       y = r + γ · max_{a'} Q_target(s', a'; θ⁻) · (1 − done)

   Одна сеть (target) и выбирает a', и оценивает Q(s', a').
   Это приводит к систематическому завышению Q (overestimation bias).

3. EXPLICIT DOUBLE DQN (DDQNAgent)
   ──────────────────────────────────
   Устраняет overestimation, разделяя выбор действия и его оценку:

       a* = argmax_{a'} Q_online(s', a'; θ)     ← online выбирает
       y  = r + γ · Q_target(s', a*; θ⁻) · (1 − done)  ← target оценивает

   Почему это лучше? При обычном max Q_target могут "случайно" получить
   высокое значение неоптимальные действия (estimation noise). Online-сеть
   со своими весами не подтвердит тот же шум → bias снижается.

╔══════════════════════════════════════════════════════════════════════════╗
║  ОБЩИЕ ГИПЕРПАРАМЕТРЫ                                                   ║
╚══════════════════════════════════════════════════════════════════════════╝

   GAMMA       = 0.95    γ: горизонт ~20 шагов (для эпизодов 500–1000 шагов)
   LR          = 1e-4    α: Adam learning rate
   EPS_START   = 1.0     начальная ε (полное исследование)
   EPS_END     = 0.075   минимальная ε
   EPS_DECAY   = 0.9998  медленное затухание (~13000 эп. до EPS_END)
   BATCH_SIZE  = 256     (только DQN/DDQN)
   BUFFER_CAP  = 200_000 (только DQN/DDQN)
   TARGET_UPDATE= 500    (только DQN/DDQN)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model         import DQN, NUM_ACTIONS, OBS_SIZE
from replay_buffer import ReplayBuffer

# ── Гиперпараметры (общие) ────────────────────────────────────────────────────
GAMMA         = 0.95
LR            = 1e-4
EPS_START     = 1.0
EPS_END       = 0.005
EPS_DECAY     = 0.9998   # ~13000 эпизодов до EPS_END

# ── Гиперпараметры (DQN / DDQN) ──────────────────────────────────────────────
BATCH_SIZE    = 256
BUFFER_CAP    = 200_000
TARGET_UPDATE = 500


# ══════════════════════════════════════════════════════════════════════════════
# Базовый класс — общий интерфейс для всех алгоритмов
# ══════════════════════════════════════════════════════════════════════════════
class _BaseAgent:
    """
    Общий интерфейс: select_action, store, train_step, decay_epsilon, save, load.

    Подклассы переопределяют _store_normalized() и train_step().
    has_replay_buffer = False/True указывает train.py нужен ли PER.
    """
    has_replay_buffer: bool = False   # переопределяется в DQN/DDQN

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.eps    = EPS_START

        self.online    = DQN(OBS_SIZE, NUM_ACTIONS).to(self.device)
        self.optimizer = optim.Adam(self.online.parameters(), lr=LR)
        self.loss_fn   = nn.SmoothL1Loss()

        self._steps      = 0
        self.best_reward = -float("inf")

        # Онлайн нормализация наград: σ̂ ← (1−α)σ̂ + α|r|
        self._reward_running_std = 1.0
        self._reward_alpha       = 0.001

    # ── ε-жадный выбор действия ──────────────────────────────────────────────
    def select_action(self, obs: np.ndarray) -> int:
        if np.random.rand() < self.eps:
            return np.random.randint(NUM_ACTIONS)
        t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(self.online(t).argmax(dim=1).item())

    # ── Нормализация + сохранение перехода ───────────────────────────────────
    def store(self, obs: np.ndarray, action: int, reward: float,
              next_obs: np.ndarray, done: bool) -> None:
        # Обновляем бегущую оценку масштаба награды
        self._reward_running_std = (
            (1.0 - self._reward_alpha) * self._reward_running_std
            + self._reward_alpha * abs(reward)
        )
        normalized = reward / max(self._reward_running_std, 1.0)
        self._store_normalized(obs, action, normalized, next_obs, done)

    def _store_normalized(self, obs, action, reward, next_obs, done) -> None:
        raise NotImplementedError

    def train_step(self) -> float | None:
        raise NotImplementedError

    # ── Затухание ε ──────────────────────────────────────────────────────────
    def decay_epsilon(self) -> None:
        self.eps = max(EPS_END, self.eps * EPS_DECAY)

    # ── Сброс буфера при смене фазы ──────────────────────────────────────────
    def clear_buffer(self) -> None:
        """
        Очищает replay buffer при переходе на новую фазу.
        Веса сети сохраняются — переносится знание о механике игры.
        Переходы сбрасываются — старая карта не отравляет обучение на новой.
        Базовый класс (Q-Learning, нет буфера): no-op.
        """
        pass   # Q-Learning без буфера — ничего не делать

    # ── Один gradient step (внутренний хелпер) ───────────────────────────────
    def _gradient_step(self, q_pred: torch.Tensor,
                       q_target: torch.Tensor) -> float:
        loss = self.loss_fn(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        self._steps += 1
        return loss.item()

    # ── Сохранение / загрузка ────────────────────────────────────────────────
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self._state_dict(), path)

    def _state_dict(self) -> dict:
        return {
            "online":              self.online.state_dict(),
            "optimizer":           self.optimizer.state_dict(),
            "eps":                 self.eps,
            "steps":               self._steps,
            "best_reward":         self.best_reward,
            "reward_running_std":  self._reward_running_std,
        }

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.online.load_state_dict(ckpt["online"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.eps                  = ckpt.get("eps",                EPS_END)
        self._steps               = ckpt.get("steps",              0)
        self.best_reward          = ckpt.get("best_reward",        -float("inf"))
        self._reward_running_std  = ckpt.get("reward_running_std", 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Q-Learning
# ══════════════════════════════════════════════════════════════════════════════
class QLearningAgent(_BaseAgent):
    """
    Нейросетевой Q-learning без буфера и без target-сети.

    Каждый шаг немедленно обновляет online-сеть по последнему переходу:

        y    = r + γ · max_{a'} Q_online(s', a') · (1 − done)
        loss = HuberLoss(Q_online(s, a),  y)

    Достоинства: простота, нет задержки прогрева буфера.
    Недостатки:  нестабильность из-за "движущейся цели" и коррелированных данных.
    """
    has_replay_buffer = False

    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        self._last: tuple | None = None   # последний нормализованный переход

    def _store_normalized(self, obs, action, reward, next_obs, done) -> None:
        # Нет буфера — просто запоминаем последний переход
        self._last = (obs, action, reward, next_obs, done)

    def train_step(self) -> float | None:
        if self._last is None:
            return None   # первый вызов до store()

        obs, action, reward, next_obs, done = self._last

        s  = torch.tensor(obs,      dtype=torch.float32, device=self.device).unsqueeze(0)
        ns = torch.tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        a  = torch.tensor([[action]], dtype=torch.int64, device=self.device)

        # TD-цель: та же сеть, что и обновляемая → "движущаяся цель"
        with torch.no_grad():
            q_next  = self.online(ns).max(dim=1).values.item()
            y_val   = reward + GAMMA * q_next * (1.0 - float(done))

        q_pred  = self.online(s).gather(1, a).squeeze()
        y       = torch.tensor(y_val, dtype=torch.float32, device=self.device)

        return self._gradient_step(q_pred, y)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Vanilla DQN
# ══════════════════════════════════════════════════════════════════════════════
class DQNAgent(_BaseAgent):
    """
    Vanilla DQN: буфер опыта + замороженная target-сеть.

    TD-цель:
        y = r + γ · max_{a'} Q_target(s', a'; θ⁻) · (1 − done)

    Та же target-сеть и выбирает максимальное действие, и оценивает его —
    отсюда систематическое завышение Q (overestimation bias).
    Устраняется в DDQN разделением этих ролей.
    """
    has_replay_buffer = True

    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        self.target = DQN(OBS_SIZE, NUM_ACTIONS).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.buffer = ReplayBuffer(BUFFER_CAP)

    def _store_normalized(self, obs, action, reward, next_obs, done) -> None:
        self.buffer.push(obs, action, reward, next_obs, done)

    def clear_buffer(self) -> None:
        """Сбрасывает буфер при смене фазы (переходы с другой карты нерелевантны)."""
        dropped = len(self.buffer.buffer)
        self.buffer.buffer.clear()
        self.buffer.pos = 0
        self._reward_running_std = 1.0   # сбрасываем нормализацию наград вместе с буфером
        print("[agent] Replay buffer cleared ({:,} transitions dropped). "
              "reward_std reset.".format(dropped))

    def train_step(self) -> float | None:
        if not self.buffer.ready:
            return None

        s, a, r, ns, d = self._sample_batch()

        q_pred = self.online(s).gather(1, a).squeeze(1)   # (B,)

        with torch.no_grad():
            # Vanilla DQN: target и выбирает, и оценивает → overestimation
            q_next   = self.target(ns).max(dim=1).values   # (B,)
            q_target = r + GAMMA * q_next * (1.0 - d)     # (B,)

        loss_val = self._gradient_step(q_pred, q_target)
        self._maybe_update_target()
        return loss_val

    def _sample_batch(self):
        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)
        s  = torch.tensor(states,      dtype=torch.float32, device=self.device)
        a  = torch.tensor(actions,     dtype=torch.int64,   device=self.device).unsqueeze(1)
        r  = torch.tensor(rewards,     dtype=torch.float32, device=self.device)
        ns = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        d  = torch.tensor(dones,       dtype=torch.float32, device=self.device)
        return s, a, r, ns, d

    def _maybe_update_target(self) -> None:
        if self._steps % TARGET_UPDATE == 0:
            self.target.load_state_dict(self.online.state_dict())

    # Сохранение: добавляем target-сеть
    def _state_dict(self) -> dict:
        d = super()._state_dict()
        d["target"] = self.target.state_dict()
        return d

    def load(self, path: str) -> None:
        super().load(path)
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if "target" in ckpt:
            self.target.load_state_dict(ckpt["target"])


# ══════════════════════════════════════════════════════════════════════════════
# 3. Explicit Double DQN
# ══════════════════════════════════════════════════════════════════════════════
class DDQNAgent(DQNAgent):
    """
    Явный Double DQN: online выбирает действие, target оценивает его ценность.

    TD-цель:
        a* = argmax Q_online(s', a'; theta)       -- online выбирает
        y  = r + gamma * Q_target(s', a*; theta-)  -- target оценивает

    Разделение ролей устраняет overestimation bias: если online случайно
    завысил Q для некоторого a*, target-сеть с другими весами с высокой
    вероятностью не подтвердит то же завышение.
    """

    def train_step(self) -> float | None:
        if not self.buffer.ready:
            return None

        s, a, r, ns, d = self._sample_batch()

        q_pred = self.online(s).gather(1, a).squeeze(1)

        with torch.no_grad():
            # Шаг 1: online-сеть выбирает лучшее действие в s'
            best_actions = self.online(ns).argmax(dim=1, keepdim=True)   # (B, 1)
            # Шаг 2: target-сеть оценивает это конкретное действие
            q_next       = self.target(ns).gather(1, best_actions).squeeze(1)  # (B,)
            q_target     = r + GAMMA * q_next * (1.0 - d)

        loss_val = self._gradient_step(q_pred, q_target)
        self._maybe_update_target()
        return loss_val


# ══════════════════════════════════════════════════════════════════════════════
# Фабрика агентов
# ══════════════════════════════════════════════════════════════════════════════
ALGO_CHOICES = ("qlearn", "dqn", "ddqn")


def make_agent(algo: str, device: str = "cpu") -> _BaseAgent:
    """
    Создаёт агента по имени алгоритма.

    algo:
      "qlearn" -- QLearningAgent  (без буфера, без target-сети)
      "dqn"    -- DQNAgent        (буфер + target, vanilla)
      "ddqn"   -- DDQNAgent       (буфер + target, explicit double DQN)
    """
    algo = algo.lower()
    if algo == "qlearn":
        return QLearningAgent(device)
    elif algo == "dqn":
        return DQNAgent(device)
    elif algo == "ddqn":
        return DDQNAgent(device)
    else:
        raise ValueError(
            "Unknown algo: '{}'. Choices: {}".format(algo, ALGO_CHOICES)
        )
