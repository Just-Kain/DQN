"""
model.py — DQN нейросеть

Архитектура: полносвязная сеть (MLP)
  Input  → 256 → ReLU → 128 → ReLU → num_actions

Входной вектор (85 признаков):
  - 81 : локальный вид 9×9 вокруг игрока (нормализованные типы тайлов/существ)
  -  2  : (dx_exit, dy_exit) — нормализованное расстояние до выхода
  -  1  : hp_norm — HP игрока / MaxHP
  -  1  : step_norm — шаг / MaxSteps (0..1)
"""

import torch
import torch.nn as nn


OBS_SIZE   = 85    # размер входного вектора
NUM_ACTIONS = 8    # Up Down Left Right MeleeAttack ArrowShot Dash Idle


class DQN(nn.Module):
    def __init__(self, obs_size: int = OBS_SIZE, num_actions: int = NUM_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
