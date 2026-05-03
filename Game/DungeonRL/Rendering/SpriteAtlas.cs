using SFML.Graphics;

/// <summary>
/// Хранит все текстуры игры.
///
/// Тайлы:         Floor, Wall, Exit, Pit
/// Игрок:         Player
/// Враги (3 шт.): EnemyWalking, EnemyFlying, EnemyCrawling
///
/// Метод GetEnemyTexture(EnemyType) возвращает нужную текстуру врага
/// без switch-конструкций в рендерере.
/// </summary>
public class SpriteAtlas
{
    // ── Тайлы ────────────────────────────────────────────────────────────────
    public Texture Floor   = null!;   // floor.png
    public Texture Wall    = null!;   // wall.png
    public Texture Exit    = null!;   // exit.png
    public Texture Pit     = null!;   // pit.png

    // ── Существа ─────────────────────────────────────────────────────────────
    public Texture Player        = null!;   // player.png
    public Texture EnemyWalking  = null!;   // walker.png
    public Texture EnemyFlying   = null!;   // fly.png
    public Texture EnemyCrawling = null!;   // crawl.png

    // ── Хелпер: текстура врага по типу ───────────────────────────────────────
    public Texture GetEnemyTexture(EnemyType type) => type switch
    {
        EnemyType.Flying   => EnemyFlying,
        EnemyType.Crawling => EnemyCrawling,
        _                  => EnemyWalking
    };
}
