/// <summary>
/// Дискретное пошаговое движение игрока (1 тайл за шаг).
/// Проверяет коллизии через CollisionSystem.
///
/// Яма (Pit):
///   • WASD — можно войти, отнимается 2 HP.
///   • Dash — перелетает яму без урона (прыжок).
///
/// Dash — прыжок на 2 клетки вперёд.
///   Перелетает ямы и врагов, останавливается перед стеной.
/// </summary>
public class MovementSystem
{
    private readonly CollisionSystem collision = new();

    private const int PitDamage = 2;   // урон за шаг по яме

    public void Update(GameState state, ActionType action, float _dt)
    {
        var player = state.Player;

        int dx = 0, dy = 0;

        switch (action)
        {
            case ActionType.Up:    dy = -1; player.Facing = Direction.Up;    break;
            case ActionType.Down:  dy =  1; player.Facing = Direction.Down;  break;
            case ActionType.Left:  dx = -1; player.Facing = Direction.Left;  break;
            case ActionType.Right: dx =  1; player.Facing = Direction.Right; break;

            case ActionType.Dash:
                ApplyDash(state);
                return;

            default:
                return; // MeleeAttack / ArrowShot / Idle — позицию не меняют
        }

        int newX = player.X + dx;
        int newY = player.Y + dy;

        if (collision.CanMoveTo(state, newX, newY))
        {
            SetPosition(player, newX, newY);

            // Ямы проходимы, но опасны при обычной ходьбе
            if (state.Map.Tiles[newX, newY] == TileType.Pit)
                player.HP = Math.Max(0, player.HP - PitDamage);
        }
    }

    // ── Dash: прыжок на 2 клетки вперёд — без урона от ям ────────────────────
    private static void ApplyDash(GameState state)
    {
        var player = state.Player;
        var (dx, dy) = FacingDelta(player.Facing);

        for (int step = 1; step <= 2; step++)
        {
            int nx = player.X + dx * step;
            int ny = player.Y + dy * step;

            if (nx < 0 || nx >= state.Map.Width ||
                ny < 0 || ny >= state.Map.Height)
                break;

            var tile = state.Map.Tiles[nx, ny];
            if (tile == TileType.Wall || tile == TileType.Empty)
                break;

            // Ямы и врагов перелетаем — позицию обновляем, урон не берём
            SetPosition(player, nx, ny);
        }
    }

    // ── Вспомогательные ───────────────────────────────────────────────────────
    private static void SetPosition(Player p, int x, int y)
    {
        p.X = x; p.Xf = x;
        p.Y = y; p.Yf = y;
    }

    public static (int dx, int dy) FacingDelta(Direction facing) => facing switch
    {
        Direction.Up    => (0, -1),
        Direction.Down  => (0,  1),
        Direction.Left  => (-1, 0),
        Direction.Right => (1,  0),
        _               => (0,  0)
    };
}
