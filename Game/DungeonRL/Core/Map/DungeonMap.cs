public class DungeonMap
{
    public TileType[,] Tiles;

    public int Width  => Tiles.GetLength(0);
    public int Height => Tiles.GetLength(1);

    // Координаты выхода и стартовой позиции игрока — заполняются генератором
    public int ExitX        { get; set; }
    public int ExitY        { get; set; }
    public int PlayerSpawnX { get; set; }
    public int PlayerSpawnY { get; set; }

    // Список комнат, созданных генератором (заполняется DungeonGenerator)
    public List<(int X, int Y, int W, int H)> Rooms { get; set; } = new();

    /// <summary>
    /// Игрок может ходить по Floor, Exit и Pit.
    /// Pit проходима, но наносит 2 HP урона (обрабатывается в MovementSystem).
    /// </summary>
    public bool IsWalkable(int x, int y)
    {
        if (x < 0 || x >= Width || y < 0 || y >= Height) return false;
        var t = Tiles[x, y];
        return t == TileType.Floor || t == TileType.Exit || t == TileType.Pit;
    }

    /// <summary>
    /// Только твёрдая поверхность (без Pit) — используется в BFS спавна
    /// и для проверки прозрачности тайлов (линия видимости).
    /// </summary>
    public bool IsSolid(int x, int y)
    {
        if (x < 0 || x >= Width || y < 0 || y >= Height) return false;
        var t = Tiles[x, y];
        return t == TileType.Floor || t == TileType.Exit;
    }
}