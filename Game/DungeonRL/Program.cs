// See https://aka.ms/new-console-template for more information
using SFML.Graphics;
using SFML.Window;
using SFML.System;

var window = new RenderWindow(new VideoMode((1080, 1080)), "DungeonRL");

var atlas = new SpriteAtlas
{
    Wall   = new Texture("assets/wall.png"),
    Floor  = new Texture("assets/floor.png"),
    Player = new Texture("assets/player.png"),
    Enemy  = new Texture("assets/enemy.png"),
    Exit   = new Texture("assets/door.png")
};

// Закрываем окно стандартным способом (X кнопка)
window.Closed += (_, _) => window.Close();

var env = new DungeonEnv();
env.Reset(42);

var renderer = new Renderer(window, atlas);
var time = new Time();

float stepTimer = 0f;
float stepInterval = 0.15f;

while (window.IsOpen)
{
    window.DispatchEvents();

    stepTimer += time.DeltaTime();
    ActionType action = ActionType.Idle;
    
    if (Keyboard.IsKeyPressed(Keyboard.Key.W)) action = ActionType.Up;
    if (Keyboard.IsKeyPressed(Keyboard.Key.S)) action = ActionType.Down;
    if (Keyboard.IsKeyPressed(Keyboard.Key.A)) action = ActionType.Left;
    if (Keyboard.IsKeyPressed(Keyboard.Key.D)) action = ActionType.Right;
    if (Keyboard.IsKeyPressed(Keyboard.Key.X)) break;
    if (stepTimer >= stepInterval)
    {
        env.Step(action);
        stepTimer = 0f;
    }

    renderer.Draw(env.State);
}