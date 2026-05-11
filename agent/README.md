# DQN Agent — DungeonRL

The agent learns to navigate procedurally-generated dungeons using Deep Q-Networks.
It controls the player character through a JSON protocol with the C# game engine.

---

## Algorithms

Three algorithms are available via `--algo`. All share the same network architecture
and observation format; they differ in how targets are computed and whether a replay
buffer / target network are used.

### Q-Learning (`--algo qlearn`)

Neural Q-Learning: no replay buffer, no target network.
The network is updated after every single step using the Bellman target:

```
target = r  +  gamma * max_a Q(s', a)        (if not done)
target = r                                    (if done)
```

Both the online prediction and the target are computed by the **same** network,
so weights shift every step. Most unstable variant — prone to divergence on complex maps.

### Vanilla DQN (`--algo dqn`)

Adds two stabilising components:

- **Replay buffer** (50 000 transitions): samples random mini-batches, breaking
  temporal correlation between consecutive observations.
- **Target network**: a frozen copy of the online network, synced every
  `TARGET_UPDATE` steps. Targets are stable during training.

Target computation (max Q):
```
target = r  +  gamma * max_a Q_target(s', a)
```

### Double DQN (`--algo ddqn`) — default

Extends DQN to reduce Q-value overestimation. The online network **selects** the
best action; the target network **evaluates** it:

```
a* = argmax_a  Q_online(s', a)
target = r  +  gamma * Q_target(s', a*)
```

Decoupling selection from evaluation prevents the positive bias that accumulates
when the same network both picks and scores actions.

---

## Observation Space  (OBS_SIZE = 1157)

Each observation is a flat float32 vector of 1157 elements:

```
[ local_view (1089)  |  minimap (64)  |  scalars (4) ]
```

### Local view — 33 × 33 = 1089 values

A 33×33 tile window centred on the player (16 tiles in every direction).
Tiles outside the map boundaries are padded with 0 (EMPTY).
Each cell stores the **entity code** normalised to [0, 1] by dividing by ENTITY_MAX (8):

| Code | Entity / Tile   |
|------|-----------------|
| 0    | Empty / out-of-bounds |
| 1    | Wall            |
| 2    | Floor           |
| 3    | Player          |
| 4    | Pit             |
| 5    | Walking Enemy   |
| 6    | Flying Enemy    |
| 7    | Crawling Enemy  |
| 8    | Exit (dominates in max-pool) |

Hierarchy is intentional: Exit (8) > any enemy (5-7) > Pit (4) > Player (3) > Floor (2) > Wall (1).

VIEW = 33 covers full maps up to 24×24 without clipping.
For 32×32 maps the player can reach the centre; peripheral tiles at the edge of
the 33-window show part of the map — sufficient for navigation.

### Minimap — 8 × 8 = 64 values

Max-pooling of the full map down to 8×8 cells.
**Max-pooling** is used (not average) so Exit (value 8, normalised 1.0) always
dominates in its cell regardless of how many wall or floor tiles surround it.
This guarantees the exit is always visible on the minimap.

### Scalars — 4 values

| Index | Value                        | Range  |
|-------|------------------------------|--------|
| 0     | Player HP / MaxHP            | [0, 1] |
| 1     | Steps taken / MAX_STEPS      | [0, 1] |
| 2     | (exit_x - player_x) / map_w  | [-1, 1]|
| 3     | (exit_y - player_y) / map_h  | [-1, 1]|

---

## Action Space  (NUM_ACTIONS = 7)

| Index | Action      | Description                      |
|-------|-------------|----------------------------------|
| 0     | MoveUp      | Move one tile north              |
| 1     | MoveDown    | Move one tile south              |
| 2     | MoveLeft    | Move one tile west               |
| 3     | MoveRight   | Move one tile east               |
| 4     | MeleeAttack | Attack adjacent enemy            |
| 5     | ArrowShot   | Ranged attack (straight line)    |
| 6     | Idle        | Do nothing (−2.0 reward penalty) |

Idle is included but strongly discouraged by a large penalty.

---

## Reward System

| Event                           | Reward                            |
|---------------------------------|-----------------------------------|
| Each step (time penalty)        | −0.05                             |
| Kill enemy                      | +3.0 per kill                     |
| Attack with no kill             | −0.3                              |
| Idle action                     | −2.0                              |
| Player takes damage             | −10.0 × (hpLost / MaxHP)          |
| Player death                    | −25.0                             |
| BFS shaping (potential-based)   | ±0.5 × Δdist_to_exit              |
| Reach exit (win)                | +100.0                            |

**HP damage** is percentage-based: losing 10% of MaxHP always costs −1.0,
regardless of phase. This keeps the avoidance signal consistent as MaxHP
scales from 15 (Phase 0) to 50 (Phase 5).

**BFS shaping** is a potential-based reward: `0.5 × (prev_dist − curr_dist)`.
Moving one tile closer to the exit yields +0.5; moving away yields −0.5.
Circular wandering nets zero — reward farming is impossible.

---

## Network Architecture

```
Input:  1157  (flat float32 observation)
        |
   Linear(1157 → 512)
        |
      ReLU
        |
   Linear(512 → 256)
        |
      ReLU
        |
   Linear(256 → 7)
        |
Output:  7 Q-values (one per action)
```

**Hyperparameters:**

| Parameter       | Value    |
|-----------------|----------|
| GAMMA           | 0.99     |
| LR              | 1e-4     |
| BATCH_SIZE      | 64       |
| BUFFER_SIZE     | 50 000   |
| TARGET_UPDATE   | 500 steps|
| EPS_START       | 1.0      |
| EPS_END         | 0.01     |
| EPS_DECAY       | 0.9995   |

---

## Curriculum Learning

The agent trains through 6 phases of increasing difficulty.
Phase advance requires: win_rate >= threshold over the last 200 episodes
AND at least 500 episodes completed in the current phase.

| Phase | Map     | Enemies | HP | Win threshold |
|-------|---------|---------|-----|---------------|
| 0     | 16×16   | No      | 15  | 50%           |
| 1     | 16×16   | Yes     | 15  | 35%           |
| 2     | 18×18   | Yes     | 20  | 30%           |
| 3     | 20×20   | Yes     | 25  | 50%           |
| 4     | 24×24   | Yes     | 30  | 65%           |
| 5     | 32×32   | Yes     | 50  | (final phase) |

**Phase 0** is navigation-only: no enemies, small map.
The agent learns to find the exit before combat is introduced.

**Phase 2 (18×18)** is an intermediate step added after analysis showed
direct 16×16 → 20×20 transitions resulted in ~4% win_rate (vs 15%+ with 18×18).

**On phase transition:**
- Replay buffer is cleared (removes stale transitions from the smaller map).
- Epsilon is reset: `eps = max(0.30, eps × 2.0)` — forces re-exploration
  of the new, larger map layout.

---

## Stagnation Detection

Every 300 episodes the agent checks for learning plateaus.
It compares win_rate of the **last 100 episodes** vs the **preceding 100 episodes**.

Conditions to trigger an epsilon bump:
- Improvement delta < 1% (STAGNATION_THRESHOLD = 0.01)
- Current eps < 0.15 (EPS_BUMP_MAX — no point bumping if already exploring)
- Cooldown elapsed: at least 600 episodes since last bump (STAGNATION_COOLDOWN)

On trigger: `eps = min(0.50, eps × 2.0)`

This forces re-exploration without fully resetting training progress.

---

## Checkpoints

Files are algo-specific (replace `{algo}` with `qlearn`, `dqn`, or `ddqn`):

| File                              | Contents                              |
|-----------------------------------|---------------------------------------|
| `checkpoints/best_{algo}.pt`      | Model weights at best episode reward  |
| `checkpoints/last_{algo}.pt`      | Latest checkpoint (every 100 episodes)|
| `checkpoints/best_episode_{algo}.pkl` | Seed + action sequence for best run|
| `checkpoints/train_log_{algo}.csv`| Per-episode training log              |

Training log columns: `episode, total_reward, steps, epsilon, loss_mean, win_rate, phase, map_size, no_enemies, ep_seed`

---

## Usage

```bash
cd agent
pip install -r requirements.txt

# Double DQN (default, recommended)
python train.py

# Resume from last checkpoint
python train.py --algo ddqn --resume

# Vanilla DQN
python train.py --algo dqn

# Q-Learning (most unstable)
python train.py --algo qlearn

# Fix one map for all episodes (reproducible training)
python train.py --fixed-seed 75

# Replay best episode visually
python replay.py --algo ddqn

# Watch agent play live (visual mode)
python train.py --algo ddqn  # (use --ai-visual flag on the game binary)
```

---

## Algorithm Comparison

| Property               | Q-Learning       | DQN              | DDQN             |
|------------------------|------------------|------------------|------------------|
| Replay buffer          | No               | Yes              | Yes              |
| Target network         | No               | Yes              | Yes              |
| Q-value overestimation | High             | Medium           | Low              |
| Stability              | Low              | Medium           | High             |
| Sample efficiency      | Low              | Medium           | Medium           |
| Best for               | Debugging only   | Baseline         | Production       |

**Q-Learning** diverges quickly on complex maps because every step shifts the
network, which immediately changes targets for all other states.

**DQN** stabilises training significantly. The replay buffer decorrelates
experience; the target network provides stable TD targets. However, using
`max Q_target` to both select and evaluate actions introduces an upward bias.

**DDQN** keeps all DQN components but separates action selection (online net)
from action evaluation (target net). This eliminates the overestimation bias,
leading to more accurate Q-values and better policies on longer-horizon tasks
like dungeon navigation.

---

## Protocol (game_interface.py ↔ C# engine)

The Python agent communicates with the game via stdin/stdout:

**Python → C#:**
```
reset          # start new episode (C# picks seed)
reset <N>      # start episode with seed N
0 .. 6         # execute action (ActionType index)
```

**C# → Python** (JSON per step):
```json
{
  "player_x": 5,  "player_y": 8,
  "player_hp": 12, "max_hp": 15,
  "exit_x": 13,   "exit_y": 2,
  "map": [[...]], 
  "reward": 0.45, "done": false, "step": 42
}
```

`max_hp` is used for HP normalisation in the observation scalars and
for the percentage-based damage reward.
