using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using SFML.Graphics;
using SFML.Window;

/// <summary>
/// Протокол взаимодействия с Python DQN-агентом через stdin/stdout.
///
/// Запуск: dotnet run -- --ai
///
/// Формат:
///   C# → Python : JSON-строка состояния (одна строка)
///   Python → C# : целое число действия (0-7) или "reset"
/// </summary>
public static class AiProtocol
{
    private static readonly JsonSerializerOptions Opts = new()
    {
        PropertyNamingPolicy        = JsonNamingPolicy.SnakeCaseLower,
        DefaultIgnoreCondition      = JsonIgnoreCondition.WhenWritingNull,
        WriteIndented               = false
    };

    // ── Главный цикл ─────────────────────────────────────────────────────────
    public static void Run()
    {
        var env  = new DungeonEnv();
        int seed = 75;

        while (true)
        {
            var state = env.Reset(seed);
            SendState(state, 0f, false);

            while (true)
            {
                string? line = Console.ReadLine();
                if (line is null) return; // stdin закрыт — выходим

                line = line.Trim();

                if (line == "reset") break; // принудительный сброс от агента

                if (!int.TryParse(line, out int idx)) continue;
                idx = Math.Clamp(idx, 0, 7);

                var result = env.Step((ActionType)idx);
                SendState(result.NextState, result.Reward, result.Done);

                if (result.Done)
                {
                    if(result.Reward > 0) seed++;
                    break;
                }

                if(result.StepIteration >= 1_000_000)
                { 
                    break;
                }
            }
        }
    }

    // ── Визуальный AI-режим: SFML окно + JSON-протокол ────────────────────────
    /// <summary>
    /// Запускается по флагу --ai-visual.
    /// Читает действия из stdin (как Run), но рендерит каждый шаг в SFML-окно.
    /// Позволяет наблюдать за игрой агента визуально.
    /// </summary>
    public static void RunVisual(RenderWindow window, Renderer renderer)
    {
        var env  = new DungeonEnv();
        int seed = 75;

        while (window.IsOpen)
        {
            // Диспетчер событий — иначе окно не реагирует и ОС его "убьёт"
            window.DispatchEvents();

            var state = env.Reset(seed);
            renderer.Draw(state);
            SendState(state, 0f, false);

            while (window.IsOpen)
            {
                window.DispatchEvents();

                string? line = Console.ReadLine();
                if (line is null) return;   // stdin закрыт Python-ом
                line = line.Trim();

                if (line == "reset") break;
                if (!int.TryParse(line, out int idx)) continue;
                idx = Math.Clamp(idx, 0, 7);

                var result = env.Step((ActionType)idx);

                // Рендерим КАЖДЫЙ шаг — агент виден в окне
                renderer.Draw(result.NextState);

                SendState(result.NextState, result.Reward, result.Done);

                if (result.Done)
                {
                    if (result.Reward > 0) seed++;
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
            PlayerX      = state.Player.X,
            PlayerY      = state.Player.Y,
            PlayerHp     = state.Player.HP,
            PlayerMaxHp  = state.Player.MaxHP,
            PlayerFacing = (int)state.Player.Facing,
            Enemies      = state.Enemies.Select(e => new EnemyDto
            {
                X     = e.X,
                Y     = e.Y,
                Hp    = e.HP,
                Type  = (int)e.Type,
                Alive = e.IsAlive
            }).ToList(),
            ExitX  = state.Map.ExitX,
            ExitY  = state.Map.ExitY,
            Map    = FlattenMap(state.Map),
            Reward = reward,
            Done   = done,
            Step   = state.StepCount
        };

        Console.WriteLine(JsonSerializer.Serialize(dto, Opts));
        Console.Out.Flush();
    }

    private static int[] FlattenMap(DungeonMap map)
    {
        var flat = new int[map.Width * map.Height];
        for (int y = 0; y < map.Height; y++)
        for (int x = 0; x < map.Width; x++)
            flat[y * map.Width + x] = (int)map.Tiles[x, y];
        return flat;
    }

    // ── DTO-классы ────────────────────────────────────────────────────────────
    private sealed class StateDto
    {
        public int            PlayerX      { get; set; }
        public int            PlayerY      { get; set; }
        public int            PlayerHp     { get; set; }
        public int            PlayerMaxHp  { get; set; }
        public int            PlayerFacing { get; set; }
        public List<EnemyDto> Enemies      { get; set; } = new();
        public int            ExitX        { get; set; }
        public int            ExitY        { get; set; }
        public int[]          Map          { get; set; } = Array.Empty<int>();
        public float          Reward       { get; set; }
        public bool           Done         { get; set; }
        public int            Step         { get; set; }
    }

    private sealed class EnemyDto
    {
        public int  X     { get; set; }
        public int  Y     { get; set; }
        public int  Hp    { get; set; }
        public int  Type  { get; set; }
        public bool Alive { get; set; }
    }
}
