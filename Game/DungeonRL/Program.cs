// DungeonRL — точка входа
// Режимы запуска:
//   dotnet run                     → игровое окно (WASD + E/Q/Space/R)
//   dotnet run -- --ai             → AI-режим: JSON-протокол через stdin/stdout (без окна)
//   dotnet run -- --ai-visual      → AI-режим С SFML-окном (агент виден визуально)

using SFML.Graphics;
using SFML.Window;
using SFML.System;

// ── AI-режим (без окна) ───────────────────────────────────────────────────────
if (args.Contains("--ai"))
{
    AiProtocol.Run();
    return;
}

// ── Общие ресурсы (нужны и для игры, и для --ai-visual) ──────────────────────
static SpriteAtlas CreateAtlas() => new SpriteAtlas
{
    Floor         = new Texture("assets/floor.png"),
    Wall          = new Texture("assets/wall.png"),
    Exit          = new Texture("assets/exit.png"),
    Pit           = new Texture("assets/pit.png"),
    Player        = new Texture("assets/player.png"),
    EnemyWalking  = new Texture("assets/walker.png"),
    EnemyFlying   = new Texture("assets/fly.png"),
    EnemyCrawling = new Texture("assets/crawl.png"),
};

// ── AI-визуальный режим: SFML-окно + JSON-протокол ────────────────────────────
if (args.Contains("--ai-visual"))
{
    var aiWindow = new RenderWindow(
        new VideoMode((1024, 1024)),
        "DungeonRL  [AI-Visual]  —  агент управляет");
    aiWindow.Closed += (_, _) => aiWindow.Close();

    var aiAtlas    = CreateAtlas();
    var aiRenderer = new Renderer(aiWindow, aiAtlas);

    AiProtocol.RunVisual(aiWindow, aiRenderer);
    return;
}

// ── Ручной игровой режим ──────────────────────────────────────────────────────
var window = new RenderWindow(
    new VideoMode((1024, 1024)),
    "DungeonRL  [WASD] движение  [E] удар  [Q] выстрел  [Space] прыжок  [R] сброс");
window.Closed += (_, _) => window.Close();

var atlas    = CreateAtlas();
var env      = new DungeonEnv();
var renderer = new Renderer(window, atlas);
var clock    = new Clock();

int   seed         = 75;
float stepTimer    = 0f;
float stepInterval = 0.15f;

env.Reset(seed);
window.SetTitle($"DungeonRL  WASD/E/Q/Space/R  |  seed={seed}");

while (window.IsOpen)
{
    window.DispatchEvents();

    float dt = clock.Restart().AsSeconds();
    stepTimer += dt;

    var action = ActionType.Idle;
    if (Keyboard.IsKeyPressed(Keyboard.Key.W))     action = ActionType.Up;
    if (Keyboard.IsKeyPressed(Keyboard.Key.S))     action = ActionType.Down;
    if (Keyboard.IsKeyPressed(Keyboard.Key.A))     action = ActionType.Left;
    if (Keyboard.IsKeyPressed(Keyboard.Key.D))     action = ActionType.Right;
    if (Keyboard.IsKeyPressed(Keyboard.Key.E))     action = ActionType.MeleeAttack;
    if (Keyboard.IsKeyPressed(Keyboard.Key.Q))     action = ActionType.ArrowShot;
    if (Keyboard.IsKeyPressed(Keyboard.Key.Space)) action = ActionType.Dash;

    if (Keyboard.IsKeyPressed(Keyboard.Key.R))
    {
        env.Reset(++seed);
        window.SetTitle($"DungeonRL  WASD/E/Q/Space/R  |  seed={seed}");
        stepTimer = 0f;
    }

    if (stepTimer >= stepInterval)
    {
        if (!env.State.IsTerminal)
            env.Step(action);
        stepTimer = 0f;
    }

    renderer.Draw(env.State);
}
