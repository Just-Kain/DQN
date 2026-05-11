using System;
using System.Text.Json;
using System.Text.Json.Serialization;
using SFML.Graphics;
using SFML.Window;

/// <summary>
/// Протокол взаимодействия с Python DQN-агентом через stdin/stdout.
///
/// Аргумент запуска:
///   --map-size:N    — задать размер карты (по умолчанию 16, используется курикулумом)
///   --player-hp:N   — задать HP игрока (по умолчанию 10, растёт с фазой: 15→50)
///
/// Команды Python → C#:
///   "reset"       — начать новый эпизод (C# управляет seed сам)
///   "reset &lt;N&gt;"   — начать эпизод с конкретным seed N (для воспроизведения)
///   "0" … "6"     — выполнить действие ActionType (Idle исключён из пространства агента)
///
/// Ответ C# → Python (одна JSON-строка):
/// {
///   "player_x": int,    "player_y": int,
///   "player_hp": int,
///   "exit_x": int,      "exit_y": int,
///   "map": int[][],     — матрица [H][W] с закодированными сущностями
///   "reward": float,    "done": bool,  "step": int
/// }
///
/// Кодировка ячеек матрицы map[y][x] (иерархия по важности для max-pool):
///   0 = Empty  (пусто / за границей)
///   1 = Wall   (стена)
///   2 = Floor  (пол)
///   3 = Player (игрок)
///   4 = Pit    (яма)
///   5 = WalkingEnemy  (живой)
///   6 = FlyingEnemy   (живой)
///   7 = CrawlingEnemy (живой)
///   8 = Exit   (выход — вершина иерархии, ENTITY_MAX=8)
/// </summary>
public static class AiProtocol
{
    private static readonly JsonSerializerOptions Opts = new()
    {
        PropertyNamingPolicy        = JsonNamingPolicy.SnakeCaseLower,
        DefaultIgnoreCondition      = JsonIgnoreCondition.WhenWritingNull,
        WriteIndented               = false
    };

    // ── Разбор --map-size:N ───────────────────────────────────────────────────
    private static int ParseMapSize(string[] args, int defaultSize = 16)
    {
        foreach (var arg in args)
        {
            const string prefix = "--map-size:";
            if (arg.StartsWith(prefix, StringComparison.Ordinal)
                && int.TryParse(arg.AsSpan(prefix.Length), out int size)
                && size >= 8 && size <= 64)
            {
                return size;
            }
        }
        return defaultSize;
    }

    // ── Разбор --no-enemies ───────────────────────────────────────────────────
    private static bool ParseNoEnemies(string[] args)
    {
        foreach (var arg in args)
            if (arg.Equals("--no-enemies", StringComparison.Ordinal))
                return true;
        return false;
    }

    // ── Разбор --player-hp:N ─────────────────────────────────────────────────
    private static int ParsePlayerHp(string[] args, int defaultHp = 10)
    {
        foreach (var arg in args)
        {
            const string prefix = "--player-hp:";
            if (arg.StartsWith(prefix, StringComparison.Ordinal)
                && int.TryParse(arg.AsSpan(prefix.Length), out int hp)
                && hp >= 1 && hp <= 200)
            {
                return hp;
            }
        }
        return defaultHp;
    }

    // ── Разбор команды reset / reset <N> ────────────────────────────────────
    private static bool TryParseReset(string line, ref int seed)
    {
        if (line == "reset")
            return true;

        if (line.StartsWith("reset ", StringComparison.Ordinal)
            && int.TryParse(line.AsSpan(6), out int newSeed))
        {
            seed = newSeed;
            return true;
        }

        return false;
    }

    // ── Главный цикл (--ai) ───────────────────────────────────────────────────
    // Протокол: C# НЕ отправляет ничего до получения "reset" от Python.
    // Это исключает десинхронизацию: Python всегда читает ровно тот JSON,
    // который соответствует его последней команде.
    public static void Run(string[] args)
    {
        var env  = new DungeonEnv { MapSize = ParseMapSize(args), NoEnemies = ParseNoEnemies(args), PlayerHp = ParsePlayerHp(args) };
        int seed = 75;

        while (true)
        {
            // Ждём команду "reset [N]" — не сбрасываем среду проактивно
            string? resetLine;
            while (true)
            {
                resetLine = Console.ReadLine();
                if (resetLine is null) return;
                if (TryParseReset(resetLine.Trim(), ref seed)) break;
            }

            // Сброс среды и отправка начального состояния
            var state = env.Reset(seed);
            SendState(state, 0f, false);

            // Обрабатываем действия до конца эпизода
            while (true)
            {
                string? line = Console.ReadLine();
                if (line is null) return;
                line = line.Trim();

                // Если Python прислал "reset" раньше — обрабатываем
                if (TryParseReset(line, ref seed)) break;

                if (!int.TryParse(line, out int idx)) continue;
                idx = Math.Clamp(idx, 0, 7);

                var result = env.Step((ActionType)idx);
                SendState(result.NextState, result.Reward, result.Done);

                if (result.Done) break;  // эпизод завершён — ждём следующий "reset"
            }
        }
    }

    // ── Визуальный AI-режим (--ai-visual) ────────────────────────────────────
    public static void RunVisual(RenderWindow window, Renderer renderer, string[] args)
    {
        var env  = new DungeonEnv { MapSize = ParseMapSize(args), NoEnemies = ParseNoEnemies(args), PlayerHp = ParsePlayerHp(args) };
        int seed = 75;

        while (window.IsOpen)
        {
            window.DispatchEvents();

            // Ждём "reset [N]" от Python
            string? resetLine;
            while (window.IsOpen)
            {
                window.DispatchEvents();
                resetLine = Console.ReadLine();
                if (resetLine is null) return;
                if (TryParseReset(resetLine.Trim(), ref seed)) break;
            }
            if (!window.IsOpen) return;

            var state = env.Reset(seed);
            renderer.Draw(state);
            SendState(state, 0f, false);

            while (window.IsOpen)
            {
                window.DispatchEvents();

                string? line = Console.ReadLine();
                if (line is null) return;
                line = line.Trim();

                if (TryParseReset(line, ref seed)) break;
                if (!int.TryParse(line, out int idx)) continue;
                idx = Math.Clamp(idx, 0, 7);

                var result = env.Step((ActionType)idx);
                renderer.Draw(result.NextState);
                SendState(result.NextState, result.Reward, result.Done);

                if (result.Done) break;
            }
        }
    }

    // ── Сериализация состояния ────────────────────────────────────────────────
    private static void SendState(GameState state, float reward, bool done)
    {
        var dto = new StateDto
        {
            PlayerX  = state.Player.X,
            PlayerY  = state.Player.Y,
            PlayerHp = state.Player.HP,
            MaxHp    = state.Player.MaxHP,
            ExitX    = state.Map.ExitX,
            ExitY    = state.Map.ExitY,
            Map      = BuildEntityMap(state),
            Reward   = reward,
            Done     = done,
            Step     = state.StepCount
        };

        Console.WriteLine(JsonSerializer.Serialize(dto, Opts));
        Console.Out.Flush();
    }

    // ── Построение entity-матрицы ────────────────────────────────────────────
    private static int[][] BuildEntityMap(GameState state)
    {
        var map = state.Map;
        var result = new int[map.Height][];

        for (int y = 0; y < map.Height; y++)
        {
            result[y] = new int[map.Width];
            for (int x = 0; x < map.Width; x++)
            {
                result[y][x] = map.Tiles[x, y] switch
                {
                    TileType.Wall  => 1,
                    TileType.Floor => 2,
                    TileType.Pit   => 4,
                    TileType.Exit  => 8,   // вершина иерархии — доминирует в max-pool
                    _              => 0
                };
            }
        }

        foreach (var e in state.Enemies)
        {
            if (!e.IsAlive) continue;

            int code = e.Type switch
            {
                EnemyType.Walking  => 5,
                EnemyType.Flying   => 6,
                EnemyType.Crawling => 7,
                _                  => 5
            };

            if (e.Y >= 0 && e.Y < map.Height && e.X >= 0 && e.X < map.Width)
                result[e.Y][e.X] = code;
        }

        if (state.Player.Y >= 0 && state.Player.Y < map.Height &&
            state.Player.X >= 0 && state.Player.X < map.Width)
        {
            result[state.Player.Y][state.Player.X] = 3;
        }

        return result;
    }

    // ── DTO ───────────────────────────────────────────────────────────────────
    private sealed class StateDto
    {
        public int     PlayerX  { get; set; }
        public int     PlayerY  { get; set; }
        public int     PlayerHp { get; set; }
        public int     MaxHp    { get; set; }
        public int     ExitX    { get; set; }
        public int     ExitY    { get; set; }
        public int[][] Map      { get; set; } = Array.Empty<int[]>();
        public float   Reward   { get; set; }
        public bool    Done     { get; set; }
        public int     Step     { get; set; }
    }
}
