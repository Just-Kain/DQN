"""
replay_buffer.py — Буфер воспроизведения опыта (Experience Replay)

Хранит кортежи (state, action, reward, next_state, done).
При превышении ёмкости перезаписывает самые старые записи (циклический буфер).
"""

import random
import numpy as np
from typing import Tuple


Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.capacity = capacity
        self.buffer: list[Transition] = []
        self.pos = 0

    # ── Добавление ────────────────────────────────────────────────────────────
    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.pos = (self.pos + 1) % self.capacity

    # ── Выборка мини-батча ────────────────────────────────────────────────────
    def sample(self, batch_size: int) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)

    @property
    def ready(self) -> bool:
        """True если в буфере достаточно переходов для начала обучения."""
        return len(self.buffer) >= 1_000
