public class CollisionSystem
{
    public bool CanMoveTo(GameState state, int x, int y)
    {
        // границы
        if (x < 0 || y < 0 || x >= state.Map.Width || y >= state.Map.Height)
            return false;

        // тайлы
        if (!state.Map.IsWalkable(x, y))
            return false;

        // враги (нельзя заходить в клетку врага)
        foreach (var e in state.Enemies)
        {
            if (e.IsAlive && e.X == x && e.Y == y)
                return false;
        }

        return true;
    }
}