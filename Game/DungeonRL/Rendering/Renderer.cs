using SFML.Graphics;
using SFML.System;

/// <summary>
/// Тайловый рендерер на SFML.
///
/// Слои отрисовки (снизу вверх):
///   1. Тайлы карты (Wall, Floor, Exit, Pit)
///   2. Спрайты существ (Player, Enemy ×3 типа)
///   3. Хитбоксы атак (MeleeAttack — жёлтый, ArrowShot — голубой)
///   4. Debug-слой: хитбоксы тайлов + хитбоксы/origin существ (DebugDrawHitboxes=true)
///   5. HUD: HP игрока (поверх всего)
/// </summary>
public class Renderer
{
    private readonly RenderWindow window;
    private readonly SpriteAtlas  atlas;
    private readonly Font?        font;   // null — шрифт не найден, HUD не рисуется

    private const int TileSize = 32;
    private const int HalfTile = TileSize / 2;

    public bool DebugDrawHitboxes = true;

    // ── Цвета debug-слоя ──────────────────────────────────────────────────────
    private static readonly Color DbgFloor    = new(255, 255, 255,  30);
    private static readonly Color DbgWall     = new(255,  80,  80,  80);
    private static readonly Color DbgPit      = new(120,  60, 200, 100);
    private static readonly Color DbgExit     = new( 80, 255, 120, 100);
    private static readonly Color DbgPlayer   = new(255, 255,   0, 180);
    private static readonly Color OriginColor = new(255, 255, 255, 220);

    // Цвета хитбоксов атак
    private static readonly Color MeleeHitFill    = new(255, 200,   0,  55);
    private static readonly Color MeleeHitOutline = new(255, 200,   0, 220);
    private static readonly Color ArrowHitFill    = new(  0, 220, 255,  45);
    private static readonly Color ArrowHitOutline = new(  0, 220, 255, 200);

    // Цвет рамки хитбокса врага совпадает с «фирменным» цветом его класса
    private static Color EnemyHitboxColor(EnemyType t) => t switch
    {
        EnemyType.Flying   => new Color( 80, 140, 255, 200),   // синий
        EnemyType.Crawling => new Color(255, 140,  40, 200),   // оранжевый
        _                  => new Color(255,  60,  60, 200)    // красный
    };

    // ── Конструктор ───────────────────────────────────────────────────────────
    public Renderer(RenderWindow window, SpriteAtlas atlas)
    {
        this.window = window;
        this.atlas  = atlas;
        this.font   = TryLoadFont();
    }

    /// <summary>
    /// Пробует загрузить шрифт из нескольких стандартных мест.
    /// Возвращает null, если ни один файл не найден.
    /// </summary>
    private static Font? TryLoadFont()
    {
        string[] candidates =
        {
            "assets/font.ttf",
            @"C:\Windows\Fonts\arial.ttf",
            @"C:\Windows\Fonts\consola.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        };

        foreach (var path in candidates)
        {
            if (!System.IO.File.Exists(path)) continue;
            try { return new Font(path); }
            catch { /* пробуем следующий */ }
        }
        return null;
    }

    // ── Главный вызов ─────────────────────────────────────────────────────────
    public void Draw(GameState state)
    {
        window.Clear(new Color(10, 8, 18));   // очень тёмный фон

        DrawMap(state.Map);
        DrawEntities(state);
        DrawCombatHitboxes(state);

        if (DebugDrawHitboxes)
            DrawDebugLayer(state);

        DrawHud(state);       // поверх всего — HP игрока

        window.Display();
    }

    // ── 1. Карта ──────────────────────────────────────────────────────────────
    private void DrawMap(DungeonMap map)
    {
        for (int x = 0; x < map.Width; x++)
        for (int y = 0; y < map.Height; y++)
        {
            var tile = map.Tiles[x, y];
            var pos  = new Vector2f(x * TileSize, y * TileSize);

            Texture tex = tile switch
            {
                TileType.Wall => atlas.Wall,
                TileType.Exit => atlas.Exit,
                TileType.Pit  => atlas.Pit,
                _             => atlas.Floor
            };

            window.Draw(MakeSprite(tex, pos));
        }
    }

    // ── 2. Существа ───────────────────────────────────────────────────────────
    private void DrawEntities(GameState state)
    {
        // Игрок
        window.Draw(MakeSprite(atlas.Player,
            new Vector2f(state.Player.X * TileSize, state.Player.Y * TileSize)));

        // Враги — каждый тип получает свою текстуру
        foreach (var e in state.Enemies)
        {
            if (!e.IsAlive) continue;
            window.Draw(MakeSprite(
                atlas.GetEnemyTexture(e.Type),
                new Vector2f(e.X * TileSize, e.Y * TileSize)));
        }
    }

    // ── 3. Хитбоксы атак ─────────────────────────────────────────────────────
    private void DrawCombatHitboxes(GameState state)
    {
        switch (state.LastAction)
        {
            case ActionType.MeleeAttack: DrawMeleeHitboxes(state); break;
            case ActionType.ArrowShot:   DrawArrowHitboxes(state);  break;
        }
    }

    /// <summary>Закрашивает 3 клетки дуги ближнего удара (жёлтый).</summary>
    private void DrawMeleeHitboxes(GameState state)
    {
        var cells = CombatSystem.MeleeArc(
            state.Player.X, state.Player.Y, state.Player.Facing);

        foreach (var (cx, cy) in cells)
        {
            if (cx < 0 || cx >= state.Map.Width || cy < 0 || cy >= state.Map.Height)
                continue;

            window.Draw(new RectangleShape(new Vector2f(TileSize, TileSize))
            {
                Position         = new Vector2f(cx * TileSize, cy * TileSize),
                FillColor        = MeleeHitFill,
                OutlineColor     = MeleeHitOutline,
                OutlineThickness = 2f
            });
        }
    }

    /// <summary>Закрашивает луч стрелы до первой стены или врага (голубой).</summary>
    private void DrawArrowHitboxes(GameState state)
    {
        var map = state.Map;
        var (dx, dy) = MovementSystem.FacingDelta(state.Player.Facing);

        int cx = state.Player.X + dx;
        int cy = state.Player.Y + dy;

        while (cx >= 0 && cx < map.Width && cy >= 0 && cy < map.Height)
        {
            var tile = map.Tiles[cx, cy];
            if (tile == TileType.Wall || tile == TileType.Empty) break;

            window.Draw(new RectangleShape(new Vector2f(TileSize, TileSize))
            {
                Position         = new Vector2f(cx * TileSize, cy * TileSize),
                FillColor        = ArrowHitFill,
                OutlineColor     = ArrowHitOutline,
                OutlineThickness = 2f
            });

            bool hitEnemy = false;
            foreach (var e in state.Enemies)
            {
                if (e.IsAlive && e.X == cx && e.Y == cy) { hitEnemy = true; break; }
            }
            if (hitEnemy) break;

            cx += dx;
            cy += dy;
        }
    }

    // ── 4. Debug-слой: хитбоксы тайлов + хитбоксы/origin существ ─────────────
    private void DrawDebugLayer(GameState state)
    {
        DrawTileHitboxes(state.Map);
        DrawEntityHitbox(state.Player.X, state.Player.Y, DbgPlayer);
        DrawEntityOrigin(state.Player.X, state.Player.Y);

        foreach (var e in state.Enemies)
        {
            if (!e.IsAlive) continue;
            DrawEntityHitbox(e.X, e.Y, EnemyHitboxColor(e.Type));
            DrawEntityOrigin(e.X, e.Y);
        }
    }

    // ── Хитбоксы всех тайлов ─────────────────────────────────────────────────
    private void DrawTileHitboxes(DungeonMap map)
    {
        for (int x = 0; x < map.Width; x++)
        for (int y = 0; y < map.Height; y++)
        {
            Color outlineColor = map.Tiles[x, y] switch
            {
                TileType.Floor => DbgFloor,
                TileType.Wall  => DbgWall,
                TileType.Pit   => DbgPit,
                TileType.Exit  => DbgExit,
                _              => Color.Transparent
            };

            if (outlineColor == Color.Transparent) continue;

            window.Draw(new RectangleShape(new Vector2f(TileSize - 1, TileSize - 1))
            {
                Position         = new Vector2f(x * TileSize, y * TileSize),
                FillColor        = Color.Transparent,
                OutlineColor     = outlineColor,
                OutlineThickness = 1f
            });
        }
    }

    // ── Хитбокс существа (рамка) ──────────────────────────────────────────────
    private void DrawEntityHitbox(int x, int y, Color outlineColor)
    {
        window.Draw(new RectangleShape(new Vector2f(TileSize, TileSize))
        {
            Position         = new Vector2f(x * TileSize, y * TileSize),
            FillColor        = Color.Transparent,
            OutlineColor     = outlineColor,
            OutlineThickness = 1.5f
        });
    }

    // ── Origin: крестик в центре хитбокса ─────────────────────────────────────
    private void DrawEntityOrigin(int x, int y)
    {
        float cx = x * TileSize + HalfTile;
        float cy = y * TileSize + HalfTile;
        const float r = 3f;

        window.Draw(new RectangleShape(new Vector2f(r * 2, 1f))
        {
            Position  = new Vector2f(cx - r, cy),
            FillColor = OriginColor
        });

        window.Draw(new RectangleShape(new Vector2f(1f, r * 2))
        {
            Position  = new Vector2f(cx, cy - r),
            FillColor = OriginColor
        });
    }

    // ── 5. HUD: HP игрока ─────────────────────────────────────────────────────
    private void DrawHud(GameState state)
    {
        if (font == null) return;   // шрифт не загружен — пропускаем

        var p = state.Player;

        // ── Фоновая полоска (чуть прозрачная) ────────────────────────────────
        const float hudH = 36f;
        window.Draw(new RectangleShape(new Vector2f(window.Size.X, hudH))
        {
            Position  = new Vector2f(0, 0),
            FillColor = new Color(0, 0, 0, 160)
        });

        // ── Зелёная полоска HP ────────────────────────────────────────────────
        const float barW = 200f, barH = 16f;
        const float barX = 10f,  barY = 10f;
        float hpFrac = p.MaxHP > 0 ? (float)p.HP / p.MaxHP : 0f;

        // Подложка (серая)
        window.Draw(new RectangleShape(new Vector2f(barW, barH))
        {
            Position  = new Vector2f(barX, barY),
            FillColor = new Color(60, 60, 60, 220)
        });

        // Заливка HP (зелёная → жёлтая → красная по уровню)
        var hpColor = hpFrac > 0.5f
            ? new Color(50, 200, 50)
            : hpFrac > 0.25f
                ? new Color(220, 200, 0)
                : new Color(220, 50, 50);

        window.Draw(new RectangleShape(new Vector2f(barW * hpFrac, barH))
        {
            Position  = new Vector2f(barX, barY),
            FillColor = hpColor
        });

        // ── Текст «HP x / y» ──────────────────────────────────────────────────
        var text = new Text(font, $"HP  {p.HP} / {p.MaxHP}", 16)
        {
            Position  = new Vector2f(barX + barW + 12f, barY),
            FillColor = Color.White
        };
        window.Draw(text);
    }

    // ── Вспомогательный метод для спрайта ─────────────────────────────────────
    private static Sprite MakeSprite(Texture tex, Vector2f pos)
    {
        return new Sprite(tex)
        {
            Scale    = new Vector2f((float)TileSize / tex.Size.X,
                                    (float)TileSize / tex.Size.Y),
            Position = pos
        };
    }
}
