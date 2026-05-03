"""
agent.py — DQN-агент

Реализует:
  - ε-жадную политику с линейным затуханием
  - две сети: основная (online) + целевая (target)
  - обновление целевой сети каждые TARGET_UPDATE шагов
  - сохранение/загрузку лучшей модели
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model        import DQN, NUM_ACTIONS, OBS_SIZE
from replay_buffer import ReplayBuffer

# ── Гиперпараметры ────────────────────────────────────────────────────────────
GAMMA         = 0.99      # коэффициент дисконтирования
LR            = 1e-4      # скорость обучения
BATCH_SIZE    = 64
EPS_START     = 1.0       # начальная ε
EPS_END       = 0.05      # минимальная ε
EPS_DECAY     = 0.995     # множитель затухания за эпизод
TARGET_UPDATE = 500       # шагов между обновлениями target-сети
BUFFER_CAP    = 50_000


class DQNAgent:
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.eps    = EPS_START

        self.online = DQN(OBS_SIZE, NUM_ACTIONS).to(self.device)
        self.target = DQN(OBS_SIZE, NUM_ACTIONS).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=LR)
        self.loss_fn   = nn.SmoothL1Loss()   # Huber loss — устойчив к выбросам
        self.buffer    = ReplayBuffer(BUFFER_CAP)

        self._steps = 0             # общий счётчик шагов для обновления target
        self.best_reward = -float("inf")

    # ── Выбор действия (ε-жадная политика) ───────────────────────────────────
    def select_action(self, obs: np.ndarray) -> int:
        if np.random.rand() < self.eps:
            return np.random.randint(NUM_ACTIONS)

        state_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.online(state_t)
        return int(q_vals.argmax(dim=1).item())

    # ── Сохранение перехода в буфер ───────────────────────────────────────────
    def store(
        self,
        obs:      np.ndarray,
        action:   int,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
    ) -> None:
        self.buffer.push(obs, action, reward, next_obs, done)

    # ── Шаг обучения ──────────────────────────────────────────────────────────
    def train_step(self) -> float | None:
        """Возвращает loss или None если буфер ещё не заполнен."""
        if not self.buffer.ready:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)

        s  = torch.tensor(states,      device=self.device)
        a  = torch.tensor(actions,     device=self.device).unsqueeze(1)
        r  = torch.tensor(rewards,     device=self.device)
        ns = torch.tensor(next_states, device=self.device)
        d  = torch.tensor(dones,       device=self.device)

        # Q(s, a) — Q-значение выбранного действия
        q_pred = self.online(s).gather(1, a).squeeze(1)

        # TD-цель: r + γ * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            q_next = self.target(ns).max(dim=1).values
            q_target = r + GAMMA * q_next * (1.0 - d)

        loss = self.loss_fn(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)  # gradient clipping
        self.optimizer.step()

        self._steps += 1
        if self._steps % TARGET_UPDATE == 0:
            self._update_target()

        return loss.item()

    # ── Обновление целевой сети ───────────────────────────────────────────────
    def _update_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    # ── Затухание ε ───────────────────────────────────────────────────────────
    def decay_epsilon(self) -> None:
        self.eps = max(EPS_END, self.eps * EPS_DECAY)

    # ── Сохранение / загрузка ─────────────────────────────────────────────────
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "online":       self.online.state_dict(),
            "target":       self.target.state_dict(),
            "optimizer":    self.optimizer.state_dict(),
            "eps":          self.eps,
            "steps":        self._steps,
            "best_reward":  self.best_reward,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.eps         = ckpt.get("eps",         EPS_END)
        self._steps      = ckpt.get("steps",       0)
        self.best_reward = ckpt.get("best_reward", -float("inf"))
