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

    public GameState State { get; private set; } = null!;

    // ── Reset ─────────────────────────────────────────────────────────────────
    public GameState Reset(int seed = 0)
    {
        var map    = generator.Generate(32, 32, seed);
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

        State.LastAction = action;   // для визуализации хитбоксов атак

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
    /// Количество врагов в каждой комнате зависит от её площади:
    ///   capacity = clamp(floor(W * H / EnemyMinDist²), 0, MaxEnemiesPerRoom)
    ///
    /// EnemyMinDist — минимальное манхэттенское расстояние между спавн-точками
    /// врагов в одной комнате (параметр "enemy distance = n").
    ///
    /// Типы врагов чередуются циклически: Walking → Flying → Crawling → ...
    /// Flying может появляться над ямами, остальные — только на Floor.
    /// </summary>
    private const int EnemyMinDist       = 4;   // ← «enemy distance = n»
    private const int MaxEnemiesPerRoom  = 3;
    private const int SafeDistFromPlayer = 6;
    private const int SafeDistFromExit   = 4;

    private static List<Enemy> SpawnEnemies(DungeonMap map, int spawnX, int spawnY, int seed)
    {
        var rng    = new Random(seed + 9999);
        var result = new List<Enemy>();

        // Фабрики врагов в порядке чередования
        Func<Enemy>[] factories =
        [
            () => new WalkingEnemy(),
            () => new FlyingEnemy(),
            () => new CrawlingEnemy()
        ];
        int typeIndex = 0;

        // Глобальный список размещённых позиций (враги не накладываются)
        var allPlaced = new List<(int x, int y)>();

        foreach (var (roomX, roomY, roomW, roomH) in map.Rooms)
        {
            // Вместимость комнаты по дистанции между врагами
            int capacity = Math.Min(
                MaxEnemiesPerRoom,
                (roomW * roomH) / (EnemyMinDist * EnemyMinDist));

            if (capacity <= 0) continue;

            // Кандидаты: Floor и Pit-тайлы внутри комнаты,
            // достаточно далеко от игрока и выхода
            var candidates = new List<(int x, int y)>();
            for (int x = roomX; x < roomX + roomW; x++)
            for (int y = roomY; y < roomY + roomH; y++)
            {
                var tile = map.Tiles[x, y];
                if (tile != TileType.Floor && tile != TileType.Pit) continue;
                if (Math.Abs(x - spawnX)    + Math.Abs(y - spawnY)    < SafeDistFromPlayer) continue;
                if (Math.Abs(x - map.ExitX) + Math.Abs(y - map.ExitY) < SafeDistFromExit)   continue;
                candidates.Add((x, y));
            }
            Shuffle(candidates, rng);

            // Размещаем врагов с соблюдением EnemyMinDist
            int placed = 0;
            foreach (var (cx, cy) in candidates)
            {
                if (placed >= capacity) break;

                // Минимальное расстояние до всех уже размещённых врагов
                bool tooClose = false;
                foreach (var (px, py) in allPlaced)
                {
                    if (Math.Abs(px - cx) + Math.Abs(py - cy) < EnemyMinDist)
                    { tooClose = true; break; }
                }
                if (tooClose) continue;

                // Выбираем тип и проверяем совместимость с тайлом
                var factory = factories[typeIndex % factories.Length];
                var enemy   = factory();

                // Не-летающие не могут спавниться на яме
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

    // ── Глубокая копия (использует полиморфный Clone()) ───────────────────────
    private static GameState CloneState(GameState s) => new()
    {
        Map    = s.Map,
        Player = new Player
        {
            X = s.Player.X, Y = s.Player.Y,
            HP = s.Player.HP, MaxHP = s.Player.MaxHP,
            Facing = s.Player.Facing
        },
        // Каждый подкласс знает, как себя копировать
        Enemies    = s.Enemies.Select(e => e.Clone()).ToList(),
        StepCount  = s.StepCount,
        IsTerminal = s.IsTerminal
    };
}
