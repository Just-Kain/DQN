using SFML.Graphics;
using SFML.System;
public class Renderer
{
    private RenderWindow window;
    private SpriteAtlas atlas;

    private const int TileSize = 32;
    public bool DebugDrawHitboxes = true;

    public Renderer(RenderWindow window, SpriteAtlas atlas)
    {
        this.window = window;
        this.atlas = atlas;
    }

    public void Draw(GameState state)
    {
         window.Clear();

        DrawMap(state.Map);
        DrawEntities(state);

        if (DebugDrawHitboxes)
        {
            DrawEntityHitboxes(state);
        }

        window.Display();
    }

    // Цвет ямы: глубокий тёмно-фиолетовый
    private static readonly Color PitColor  = new Color(18,  10, 35);
    // Цвет подсветки выхода (накладывается поверх спрайта двери)
    private static readonly Color ExitTint  = new Color(80, 255, 120);

    private void DrawMap(DungeonMap map)
    {
        for (int x = 0; x < map.Width; x++)
        for (int y = 0; y < map.Height; y++)
        {
            var tile = map.Tiles[x, y];

            if (tile == TileType.Pit)
            {
                // Яма — рисуем закрашенный прямоугольник (нет спрайта)
                var pitRect = new RectangleShape(new Vector2f(TileSize, TileSize))
                {
                    Position  = new Vector2f(x * TileSize, y * TileSize),
                    FillColor = PitColor
                };
                window.Draw(pitRect);
                continue;
            }

            // Выбираем текстуру тайла
            Texture tex = tile switch
            {
                TileType.Wall  => atlas.Wall,
                TileType.Exit  => atlas.Exit,
                _              => atlas.Floor   // Floor, Empty
            };

            var sprite = new Sprite(tex)
            {
                Scale    = new Vector2f(TileSize / (float)tex.Size.X,
                                        TileSize / (float)tex.Size.Y),
                Position = new Vector2f(x * TileSize, y * TileSize)
            };

            // Подсветка клетки выхода зелёным оттенком
            if (tile == TileType.Exit)
                sprite.Color = ExitTint;

            window.Draw(sprite);
        }
    }

    private void DrawEntities(GameState state)
    {
        DrawSprite(atlas.Player, state.Player.X, state.Player.Y);

        foreach (var e in state.Enemies)
            DrawSprite(atlas.Enemy, e.X, e.Y);
    }

    private void DrawSprite(Texture tex, int x, int y)
    {
        var sprite = new Sprite(tex);

        float scaleX = TileSize / tex.Size.X;
        float scaleY = TileSize / tex.Size.Y;

        sprite.Scale = new Vector2f(scaleX, scaleY);
        sprite.Position = new Vector2f(x * TileSize, y * TileSize);

        window.Draw(sprite);
    }


    private void DrawHitbox(int x, int y, float tileSize)
    {
        var rect = new RectangleShape(new Vector2f(tileSize, tileSize));

        rect.Position = new Vector2f(x * tileSize, y * tileSize);

        rect.FillColor = Color.Transparent;
        rect.OutlineColor = Color.Red;
        rect.OutlineThickness = 1f;

        window.Draw(rect);
    }
    private void DrawEntityHitboxes(GameState state)
    {
        // игрок
        DrawHitbox(state.Player.X, state.Player.Y, TileSize);

        // враги
        foreach (var e in state.Enemies)
        {
            if (e.IsAlive)
                DrawHitbox(e.X, e.Y, TileSize);
        }
    }
}