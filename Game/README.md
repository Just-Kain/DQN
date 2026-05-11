# DungeonRL — Game Engine

C# dungeon game built on .NET 8 and SFML.Net.
Supports manual play, headless AI mode (stdin/stdout protocol), and visual AI mode.

---

## Project Structure

```
Game/
  DungeonRL/
    Program.cs                  Entry point, launch mode selection
    DungeonRL.csproj            .NET 8 project, SFML.Net 3.0.0
    assets/                     Sprite textures (PNG)
    Core/
      ActionType.cs             8-value action enum
      Direction.cs              4-direction enum (Up/Down/Left/Right)
      HitBox.cs                 Tile-aligned hit detection
      Time.cs                   Step timing utilities
      Entities/
        Entity.cs               Base: X, Y, Xf, Yf, IsAlive, Speed
        Player.cs               Player entity: HP, MaxHP, Facing
        Enemy.cs                Abstract enemy base class
        WalkingEnemy.cs         HP=3, Speed=4, VisionRange=8
        FlyingEnemy.cs          HP=1, Speed=5, VisionRange=12, flies over pits
        CrawlingEnemy.cs        HP=4, Speed=3, VisionRange=6
        EnemyType.cs            Walking / Flying / Crawling enum
      Game/
        AiProtocol.cs           stdin/stdout JSON protocol handler
        DungeonEnv.cs           Episode management (Reset/Step/Terminal)
        GameState.cs            Snapshot: map + player + enemies + step counter
        StepResult.cs           (NextState, Reward, Done) tuple
      Map/
        DungeonGenerator.cs     BSP procedural dungeon generator
        DungeonMap.cs           Tile grid, room list, exit position
        TileType.cs             Empty / Wall / Floor / Pit / Exit
      Systems/
        MovementSystem.cs       Player movement + Dash logic
        CombatSystem.cs         MeleeAttack (arc) + ArrowShot (ray)
        EnemySystem.cs          Enemy AI: movement energy, LOS, attack cooldown
        CollisionSystem.cs      Tile and entity collision detection
        CleanupSystem.cs        Remove dead enemies after each step
        RewardSystem.cs         BFS-shaped reward computation
    Rendering/
      Camera.cs                 Viewport centering
      Renderer.cs               SFML tile/entity rendering
      SpriteAtlas.cs            Texture registry
```

---

## Launch Modes

```bash
# Manual play (1024x1024 SFML window)
dotnet run

# Headless AI mode — JSON protocol over stdin/stdout (no window)
dotnet run -- --ai

# Headless AI with explicit parameters
dotnet run -- --ai --map-size:16 --player-hp:15 --no-enemies

# Visual AI mode — renders the window while the agent plays
dotnet run -- --ai-visual --map-size:20 --player-hp:25
```

### Launch Flags

| Flag               | Description                                      | Default |
|--------------------|--------------------------------------------------|---------|
| `--ai`             | Headless mode, JSON protocol via stdin/stdout    | —       |
| `--ai-visual`      | Visual mode, renders game while AI plays         | —       |
| `--map-size:N`     | Square map size (8–64)                           | 16      |
| `--player-hp:N`    | Player starting HP (1–200)                       | 10      |
| `--no-enemies`     | Spawn no enemies (curriculum Phase 0)            | false   |

---

## Manual Controls

| Key     | Action              |
|---------|---------------------|
| W       | Move up             |
| S       | Move down           |
| A       | Move left           |
| D       | Move right          |
| E       | Melee attack        |
| Q       | Arrow shot          |
| Space   | Dash (jump 2 tiles) |
| R       | Reset / next seed   |
| X       | Quit                |

---

## Action Space

| Index | ActionType  | Description                                          |
|-------|-------------|------------------------------------------------------|
| 0     | Up          | Move one tile north                                  |
| 1     | Down        | Move one tile south                                  |
| 2     | Left        | Move one tile west                                   |
| 3     | Right       | Move one tile east                                   |
| 4     | MeleeAttack | Hit arc: 3 cells in front (centre + 2 diagonals)    |
| 5     | ArrowShot   | Ray forward until first wall; hits first enemy       |
| 6     | Dash        | Jump 2 tiles forward; passes over pits and enemies   |
| 7     | Idle        | Do nothing (not exposed to the AI agent)             |

The AI agent uses actions 0–6 only. Idle (7) exists internally but is
excluded from the AI action space (it carries a −2.0 reward penalty).

---

## Map Generation (BSP)

`DungeonGenerator` builds a new map from a seed in six steps:

1. **Fill** the grid with walls.
2. **BSP split**: recursively partition the interior into leaf sections.
   Depth adapts to map size: 16×16 → depth 2, 32×32 → depth 4.
3. **Carve rooms** inside each leaf (minimum room size: 4×4).
4. **Connect rooms** with L-shaped corridors.
5. **Place Exit** at the centre of a random room.
   Surround Exit with Pit tiles on diagonals — four cardinal approaches remain clear.
6. **BFS from Exit** → find the farthest reachable floor tile → player spawn.

The same seed always produces the same map. The AI agent's `game_interface.py`
uses `reset <seed>` to reproduce specific episodes.

### Tile Types

| TileType | Value (entity code) | Description                      |
|----------|---------------------|----------------------------------|
| Empty    | 0                   | Out-of-bounds / undefined        |
| Wall     | 1                   | Impassable solid tile            |
| Floor    | 2                   | Standard walkable tile           |
| Player   | 3                   | Player position (in entity map)  |
| Pit      | 4                   | Hole — kills walking enemies     |
| Exit     | 8                   | Goal tile — ends the episode     |

Entity codes 5–7 are enemies (see entity encoding in AI protocol below).
Exit is assigned 8 (highest) so max-pooling always makes it visible on the minimap.

---

## Enemy Types

| Type    | HP | Speed | Vision | Movement          | Notes                         |
|---------|-----|-------|--------|-------------------|-------------------------------|
| Walking | 3   | 4/5   | 8      | Floor + Exit only | Most common; avoids pits     |
| Flying  | 1   | 5/5   | 12     | All non-wall tiles| Can stack; spawns over pits  |
| Crawling| 4   | 3/5   | 6      | Floor + Exit only | Slowest; most HP             |

**Speed** is relative to the player's base speed (5). Walking enemies move on
4 out of every 5 player steps; crawling enemies move on 3 out of every 5.

Enemy movement uses an **energy accumulator**: each step `energy += speed / playerSpeed`.
When `energy >= 1.0` the enemy takes a move, then `energy -= 1.0`.

Enemies only pursue the player if:
- Manhattan distance ≤ VisionRange
- Line of sight is unobstructed (Bresenham ray, walls block)

Attack cooldown: 8 steps between hits (~1 second at the default step rate).

---

## Episode Lifecycle

```
env.Reset(seed)          → generates map, spawns entities, returns GameState
loop:
    env.Step(action)     → applies action, updates all systems, returns StepResult
    if result.Done: break
```

**Terminal conditions:**
- Player HP ≤ 0 (death)
- Player stands on Exit tile (win)
- Step count ≥ 500 (timeout)

`MaxSteps = 500` in `DungeonEnv.cs` — must match `MAX_STEPS` in `game_interface.py`.

---

## AI Protocol

Communication is over stdin/stdout as newline-delimited JSON.

### Python → C#

```
reset          # start new episode, C# picks seed
reset <N>      # start episode with specific integer seed
0 .. 6         # perform action (ActionType index, Idle excluded)
```

### C# → Python (one JSON line per step)

```json
{
  "player_x":  5,
  "player_y":  8,
  "player_hp": 12,
  "max_hp":    15,
  "exit_x":    13,
  "exit_y":    2,
  "map":       [[0,1,2,...], ...],
  "reward":    0.45,
  "done":      false,
  "step":      42
}
```

`map` is a 2D array `[height][width]` with entity codes:

| Code | Entity          |
|------|-----------------|
| 0    | Empty / Wall border |
| 1    | Wall            |
| 2    | Floor           |
| 3    | Player          |
| 4    | Pit             |
| 5    | Walking Enemy   |
| 6    | Flying Enemy    |
| 7    | Crawling Enemy  |
| 8    | Exit            |

On each `reset`, the engine sends the initial state immediately.
The engine does **not** send anything until it receives a `reset` command —
this prevents desynchronisation between the two processes.

---

## Reward System

Computed by `RewardSystem.cs` every step:

| Event                        | Reward                              |
|------------------------------|-------------------------------------|
| Each step (time penalty)     | −0.05                               |
| Kill enemy                   | +3.0 per kill                       |
| Attack with no kill          | −0.3                                |
| Idle action                  | −2.0                                |
| Player takes damage          | −10.0 × (hpLost / MaxHP)            |
| Player death                 | −25.0                               |
| BFS shaping                  | ±0.5 × (prevDist − currDist)        |
| Reach exit (win)             | +100.0                              |

**BFS shaping** computes the shortest walkable path to the exit each step.
Moving one tile closer → +0.5; moving away → −0.5; circling → net 0.
Anti-farming guarantee: rewards from wandering always cancel out.

**HP damage** is percentage-based: losing 10% of MaxHP always costs −1.0,
regardless of the phase's MaxHP value (15–50). This keeps avoidance pressure
consistent across all curriculum phases.

---

## Dependencies

- **.NET 8**
- **SFML.Net 3.0.0** (`dotnet restore` handles this automatically)
- Native SFML libraries must be present on the system (Linux: `libsfml-*`, Windows: DLLs in output dir)

```bash
cd Game/DungeonRL
dotnet restore
dotnet build
dotnet run -- --ai   # headless AI mode
```
