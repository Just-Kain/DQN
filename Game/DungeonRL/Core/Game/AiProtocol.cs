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
/// Кодировка ячеек матрицы map[y][x]:
///   0 = Empty (пусто / за границей)
///   1 = Wall  (стена)
///   2 = Floor (пол)
///   3 = Exit  (выход)
///   4 = Pit   (яма)
///   5 = WalkingEnemy  (живой)
///   6 = FlyingEnemy   (живой)
///   7 = CrawlingEnemy (живой)
///   8 = Player
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
    public static void Run(string[] args)
    {
        var env  = new DungeonEnv { MapSize = ParseMapSize(args) };
        int seed = 75;

        while (true)
        {
            var state = env.Reset(seed);
            SendState(state, 0f, false);

            while (true)
            {
                string? line = Console.ReadLine();
                if (line is null) return;

                line = line.Trim();

                if (TryParseReset(line, ref seed)) break;

                if (!int.TryParse(line, out int idx)) continue;
                idx = Math.Clamp(idx, 0, 7);

                var result = env.Step((ActionType)idx);
                SendState(result.NextState, result.Reward, result.Done);

                if (result.Done) break;

                if (result.StepIteration >= 1_000_000) break;
            }
        }
    }

    // ── Визуальный AI-режим (--ai-visual) ────────────────────────────────────
    public static void RunVisual(RenderWindow window, Renderer renderer, string[] args)
    {
        var env  = new DungeonEnv { MapSize = ParseMapSize(args) };
        int seed = 75;

        while (window.IsOpen)
        {
            window.DispatchEvents();

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

                if (result.Done)
                {
                    // if (result.Reward > 0) seed++;
                    break;
                }
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
                    TileType.Exit  => 3,
                    TileType.Pit   => 4,
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
            result[state.Player.Y][state.Player.X] = 8;
        }

        return result;
    }

    // ── DTO ───────────────────────────────────────────────────────────────────
    private sealed class StateDto
    {
        public int     PlayerX  { get; set; }
        public int     PlayerY  { get; set; }
        public int     PlayerHp { get; set; }
        public int     ExitX    { get; set; }
        public int     ExitY    { get; set; }
        public int[][] Map      { get; set; } = Array.Empty<int[]>();
        public float   Reward   { get; set; }
        public bool    Done     { get; set; }
        public int     Step     { get; set; }
    }
}
