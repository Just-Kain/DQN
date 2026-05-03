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

    /// <summary>
    /// Пит — непроходим для пешего движения (можно перепрыгнуть через Dash).
    /// </summary>
    public bool IsWalkable(int x, int y)
    {
        if (x < 0 || x >= Width || y < 0 || y >= Height) return false;
        var t = Tiles[x, y];
        return t == TileType.Floor || t == TileType.Exit;
    }
}