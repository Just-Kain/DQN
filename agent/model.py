"""
model.py — DQN нейросеть

Архитектура: полносвязная сеть (MLP)
  Input → 512 → ReLU → 256 → ReLU → num_actions

Входной вектор (1157 признаков):
  - 1089 : локальный вид 33×33 вокруг игрока (нормализованные коды сущностей / 8)
  -   64 : глобальная мини-карта 8×8 (max-pooling полной карты, нормализована)
  -    2 : (dx_exit, dy_exit) — нормализованное направление до выхода
  -    1 : hp_norm — HP игрока / MaxHP
  -    1 : step_norm — шаг / MaxSteps (0..1)

Изменения по сравнению с базовой версией:
  • OBS_SIZE: 357 → 1157  (33×33 вместо 17×17, max-pool вместо avg-pool)
  • NUM_ACTIONS: 8 → 7  (Idle исключён — всегда штрафуется, агент не должен его выбирать)
  • Ширина сети: 256→128 → 512→256  (больше параметров для более сложного входа)
"""

import torch
import torch.nn as nn


OBS_SIZE    = 1157  # 33×33 + 8×8 + 4 скаляра = 1089 + 64 + 4
NUM_ACTIONS = 7     # Up Down Left Right MeleeAttack ArrowShot Dash  (без Idle)


class DQN(nn.Module):
    def __init__(self, obs_size: int = OBS_SIZE, num_actions: int = NUM_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
