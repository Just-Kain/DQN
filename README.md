# DungeonRL — среда для обучения RL-агента

Процедурно-генерируемый 2D dungeon crawler на C# + SFML без игровых движков.
Используется как среда для обучения с подкреплением (Double DQN).

---

## Технологический стек

| Компонент | Стек |
|-----------|------|
| Игровая логика | C# (.NET), ECS-подобная архитектура |
| Графика | SFML.NET |
| AI-агент | Python 3, PyTorch |
| Протокол | stdin/stdout, JSON (строка в строку) |

---

## Структура проекта

```
DQN/
├── Game/DungeonRL/              — C# проект
│   └── Core/
│       ├── Game/
│       │   ├── DungeonEnv.cs    — основной цикл среды (Reset / Step)
│       │   └── AiProtocol.cs   — JSON-протокол Python <-> C#
│       ├── Systems/
│       │   ├── MovementSystem.cs
│       │   ├── CombatSystem.cs
│       │   ├── EnemySystem.cs
│       │   ├── CleanupSystem.cs
│       │   └── RewardSystem.cs  — расчёт наград
│       ├── Map/
│       │   └── DungeonGenerator.cs — BSP-генератор карт
│       └── Entities/
│           ├── WalkingEnemy.cs
│           ├── FlyingEnemy.cs
│           └── CrawlingEnemy.cs
└── agent/                       — Python агент
    ├── train.py                 — обучающий цикл + curriculum
    ├── agent.py                 — Q-Learning / DQN / DDQN
    ├── model.py                 — MLP нейросеть
    ├── game_interface.py        — интерфейс к C# (subprocess, JSON)
    ├── plot_training.py         — анализ логов (9-панельный дашборд)
    └── checkpoints/             — модели, логи
```

---

## Игровая механика

### Тайлы

| Код | Тип | Описание |
|-----|-----|----------|
| 0 | Empty | за границей карты |
| 1 | Wall | стена, непроходима |
| 2 | Floor | пол, основная поверхность |
| 3 | Player | позиция игрока |
| 4 | Pit | яма (непроходима для наземных) |
| 5 | WalkingEnemy | ходячий враг |
| 6 | FlyingEnemy | летающий враг |
| 7 | CrawlingEnemy | ползающий враг |
| 8 | **Exit** | **выход — вершина иерархии** |

Exit кодируется значением 8 (максимум), что обеспечивает его доминирование
в max-pooling мини-карте над любым типом врагов.

### Действия игрока (7 штук, Idle исключён)

| ID | Действие | Описание |
|----|----------|----------|
| 0 | Up | шаг вверх |
| 1 | Down | шаг вниз |
| 2 | Left | шаг влево |
| 3 | Right | шаг вправо |
| 4 | MeleeAttack | удар (3 клетки вперёд+края, убивает Crawling и Walking) |
| 5 | ArrowShot | выстрел (до стены, убивает Flying и Walking) |
| 6 | Dash | рывок на 2 клетки, перепрыгивает врагов и ямы |

### Враги

| Тип | Движение | Уязвим к |
|-----|----------|----------|
| Walking | Floor, Exit | MeleeAttack, ArrowShot |
| Flying | всё кроме Wall | ArrowShot |
| Crawling | Floor, Exit | MeleeAttack |

### Условия завершения эпизода

- игрок вступил на тайл Exit → **победа** (+100 reward)
- HP игрока ≤ 0 → поражение
- шагов ≥ MaxSteps (500) → таймаут

---

## Генерация карт (BSP)

- BSP-разбиение → комнаты → коридоры
- Exit — в центре случайной комнаты
- Spawn игрока — в точке с максимальным BFS-расстоянием от Exit
- Враги не спавнятся ближе SafeDist от игрока и от Exit
- Масштаб врагов адаптируется к размеру карты

---

## Система наград

| Событие | Награда |
|---------|---------|
| Каждый шаг | −0.05 |
| Убийство врага | +3.0 |
| Бесполезная атака | −0.3 |
| Idle | −2.0 |
| Урон по игроку | −10.0 × hpLost / MaxHP |
| Смерть | −25.0 |
| BFS-шейпинг | ±0.5 × (prevDist − currDist) |
| Достижение выхода | +100.0 |

**Урон в %** от MaxHP: потеря 10% HP = −1.0 reward при любом MaxHP (15, 25, 50).
**BFS-шейпинг** — потенциальный: блуждание по кругу суммарно даёт 0, фарм невозможен.

---

## Curriculum и HP игрока

| Фаза | Карта | Враги | HP | Порог перехода |
|------|-------|-------|-----|----------------|
| Phase 0 | 16×16 | нет | 15 | win_rate ≥ 50% |
| Phase 1 | 16×16 | да | 15 | win_rate ≥ 35% |
| Phase 2 | 18×18 | да | 20 | win_rate ≥ 30% |
| Phase 3 | 20×20 | да | 25 | win_rate ≥ 50% |
| Phase 4 | 24×24 | да | 30 | win_rate ≥ 65% |
| Phase 5 | 32×32 | да | 50 | финальная фаза |

HP передаётся через аргумент `--player-hp:N` при запуске C# процесса.

---

## Протокол Python ↔ C#

```
Python  →  C#  :  "reset <N>"     — начать эпизод с сидом N
Python  →  C#  :  "0"…"6"         — выполнить действие (ActionType)

C#      →  Python :  JSON-строка
{
  "player_x": int, "player_y": int,
  "player_hp": int, "max_hp": int,
  "exit_x": int, "exit_y": int,
  "map": int[][],   // матрица [H][W], кодировка тайлов выше
  "reward": float,
  "done": bool,
  "step": int
}
```

Запуск C# из Python:
```
dotnet run --project <path> -- --ai --map-size:16 --player-hp:15 [--no-enemies]
```

---

## Запуск

```bash
# Режим отладки (визуальный, ручное управление):
cd Game/DungeonRL
dotnet run

# Управление: WASD — движение, E — удар, Q — выстрел, Space — рывок

# Обучение агента:
cd agent
pip install -r requirements.txt
python train.py                   # DDQN, полный curriculum
python train.py --algo dqn        # Vanilla DQN
python train.py --algo qlearn     # Q-Learning
python train.py --resume          # продолжить с last_ddqn.pt
```
