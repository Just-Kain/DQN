using System.Collections.Generic;
using System.Linq;

public class DungeonEnv
{
    // ── Системы ───────────────────────────────────────────────────────────────
    private readonly MovementSystem   movement  = new();
    private readonly CombatSystem     combat    = new();
    private readonly EnemySystem      enemies   = new();
    private readonly CleanupSystem    cleanup   = new();
    private readonly RewardSystem     rewards   = new();
    private readonly DungeonGenerator generator = new();

    public const int MaxSteps = 500;

    /// <summary>
    /// Размер карты (квадратная). Используется курикулум-обучением:
    ///   Фаза 1: 16   Фаза 2: 20   Фаза 3: 24   Фаза 4: 32
    /// Меняется через AiProtocol (аргумент --map-size:N).
    /// </summary>
    public int MapSize { get; set; } = 16;

    public GameState State { get; private set; } = null!;

    // ── Reset ─────────────────────────────────────────────────────────────────
    public GameState Reset(int seed = 0)
    {
        var map    = generator.Generate(MapSize, MapSize, seed);
        int spawnX = map.PlayerSpawnX;
        int spawnY = map.PlayerSpawnY;

        State = new GameState
        {
            Map    = map,
            Player = new Player
            {
                X = spawnX, Y = spawnY,
                Xf = spawnX, Yf = spawnY,
                HP = 10, MaxHP = 10,
                Facing = Direction.Down
            },
            Enemies    = SpawnEnemies(map, spawnX, spawnY, seed),
            StepCount  = 0,
            IsTerminal = false
        };

        rewards.Reset();
        return State;
    }

    // ── Step ──────────────────────────────────────────────────────────────────
    public StepResult Step(ActionType action)
    {
        if (State.IsTerminal)
            return new StepResult(State, 0f, true);

        var prev = CloneState(State);

        State.LastAction = action;

        movement.Update(State, action, 0f);
        combat.Update(State, action);
        enemies.Update(State, 0f);
        cleanup.Update(State);

        State.StepCount++;

        bool done = CheckTerminal();
        State.IsTerminal = done;

        float reward = rewards.Compute(prev, State, done);
        return new StepResult(State, reward, done);
    }

    // ── Завершение эпизода ────────────────────────────────────────────────────
    private bool CheckTerminal()
    {
        if (State.Player.HP <= 0) return true;
        if (State.Map.Tiles[State.Player.X, State.Player.Y] == TileType.Exit) return true;
        if (State.StepCount >= MaxSteps) return true;
        return false;
    }

    // ── Спавн врагов ──────────────────────────────────────────────────────────
    /// <summary>
    /// Параметры спавна адаптируются к размеру карты.
    ///
    /// EnemyMinDist       = max(3, MapSize / 6)   — дистанция между врагами
    /// SafeDistFromPlayer = max(4, MapSize / 5)   — безопасная зона вокруг спавна
    /// SafeDistFromExit   = max(2, MapSize / 8)   — безопасная зона вокруг выхода
    ///
    ///   MapSize=16: EnemyMinDist=3, SafePlayer=4, SafeExit=2
    ///   MapSize=20: EnemyMinDist=3, SafePlayer=4, SafeExit=2
    ///   MapSize=24: EnemyMinDist=4, SafePlayer=4, SafeExit=3
    ///   MapSize=32: EnemyMinDist=5, SafePlayer=6, SafeExit=4
    /// </summary>
    private const int MaxEnemiesPerRoom = 2;

    private int EnemyMinDist       => Math.Max(5, MapSize / 6);
    private int SafeDistFromPlayer => Math.Max(4, MapSize / 5);
    private int SafeDistFromExit   => Math.Max(2, MapSize / 8);

    private List<Enemy> SpawnEnemies(DungeonMap map, int spawnX, int spawnY, int seed)
    {
        var rng    = new Random(seed + 9999);
        var result = new List<Enemy>();

        int enemyMinDist       = EnemyMinDist;
        int safeDistFromPlayer = SafeDistFromPlayer;
        int safeDistFromExit   = SafeDistFromExit;

        Func<Enemy>[] factories =
        [
            () => new WalkingEnemy(),
            () => new FlyingEnemy(),
            () => new CrawlingEnemy()
        ];
        int typeIndex = 0;

        var allPlaced = new List<(int x, int y)>();

        foreach (var (roomX, roomY, roomW, roomH) in map.Rooms)
        {
            int capacity = Math.Min(
                MaxEnemiesPerRoom,
                (roomW * roomH) / (enemyMinDist * enemyMinDist));

            if (capacity <= 0) continue;

            var candidates = new List<(int x, int y)>();
            for (int x = roomX; x < roomX + roomW; x++)
            for (int y = roomY; y < roomY + roomH; y++)
            {
                var tile = map.Tiles[x, y];
                if (tile != TileType.Floor && tile != TileType.Pit) continue;
                if (Math.Abs(x - spawnX)    + Math.Abs(y - spawnY)    < safeDistFromPlayer) continue;
                if (Math.Abs(x - map.ExitX) + Math.Abs(y - map.ExitY) < safeDistFromExit)   continue;
                candidates.Add((x, y));
            }
            Shuffle(candidates, rng);

            int placed = 0;
            foreach (var (cx, cy) in candidates)
            {
                if (placed >= capacity) break;

                bool tooClose = false;
                foreach (var (px, py) in allPlaced)
                {
                    if (Math.Abs(px - cx) + Math.Abs(py - cy) < enemyMinDist)
                    { tooClose = true; break; }
                }
                if (tooClose) continue;

                var factory = factories[typeIndex % factories.Length];
                var enemy   = factory();

                if (enemy is not FlyingEnemy && map.Tiles[cx, cy] == TileType.Pit)
                    continue;

                enemy.X = cx; enemy.Xf = cx;
                enemy.Y = cy; enemy.Yf = cy;

                result.Add(enemy);
                allPlaced.Add((cx, cy));
                placed++;
                typeIndex++;
            }
        }

        return result;
    }

    private static void Shuffle<T>(List<T> list, Random rng)
    {
        for (int i = list.Count - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (list[i], list[j]) = (list[j], list[i]);
        }
    }

    // ── Глубокая копия ────────────────────────────────────────────────────────
    private static GameState CloneState(GameState s) => new()
    {
        Map    = s.Map,
        Player = new Player
        {
            X = s.Player.X, Y = s.Player.Y,
            HP = s.Player.HP, MaxHP = s.Player.MaxHP,
            Facing = s.Player.Facing
        },
        Enemies    = s.Enemies.Select(e => e.Clone()).ToList(),
        StepCount  = s.StepCount,
        IsTerminal = s.IsTerminal
    };
}
