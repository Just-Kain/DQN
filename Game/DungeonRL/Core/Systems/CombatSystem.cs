using System.Collections.Generic;

/// <summary>
/// Боевая система.
///
/// MeleeAttack — удар дугой из 3 клеток перед игроком:
///   центр + 2 боковые (по диагонали относительно направления).
///   Уничтожает: Crawling и Walking врагов.
///
/// ArrowShot — луч в направлении взгляда до первого препятствия.
///   Уничтожает первого уязвимого врага (Flying, Walking).
///   Стены и края карты блокируют луч. Ямы — не блокируют.
/// </summary>
public class CombatSystem
{
    public void Update(GameState state, ActionType action)
    {
        switch (action)
        {
            case ActionType.MeleeAttack: ApplyMelee(state);  break;
            case ActionType.ArrowShot:   ApplyArrow(state);  break;
        }
    }

    // ── Удар: дуга 3 клетки ───────────────────────────────────────────────────
    private static void ApplyMelee(GameState state)
    {
        var hitCells = MeleeArc(state.Player.X, state.Player.Y, state.Player.Facing);

        foreach (var enemy in state.Enemies)
        {
            if (!enemy.IsAlive) continue;
            if (!hitCells.Contains((enemy.X, enemy.Y))) continue;
            if (!enemy.IsVulnerableTo(ActionType.MeleeAttack)) continue;

            enemy.HP--;
            if (enemy.HP <= 0) enemy.IsAlive = false;
        }
    }

    /// <summary>
    /// Возвращает 3 клетки дуги удара:
    ///   прямо, влево-вперёд и вправо-вперёд от направления игрока.
    /// </summary>
    public static List<(int x, int y)> MeleeArc(int px, int py, Direction facing)
    {
        return facing switch
        {
            Direction.Up    => new() { (px, py-1), (px-1, py-1), (px+1, py-1) },
            Direction.Down  => new() { (px, py+1), (px-1, py+1), (px+1, py+1) },
            Direction.Left  => new() { (px-1, py), (px-1, py-1), (px-1, py+1) },
            Direction.Right => new() { (px+1, py), (px+1, py-1), (px+1, py+1) },
            _               => new()
        };
    }

    // ── Выстрел: луч вперёд ───────────────────────────────────────────────────
    private static void ApplyArrow(GameState state)
    {
        var map = state.Map;
        var (dx, dy) = MovementSystem.FacingDelta(state.Player.Facing);

        int cx = state.Player.X + dx;
        int cy = state.Player.Y + dy;

        while (cx >= 0 && cx < map.Width && cy >= 0 && cy < map.Height)
        {
            var tile = map.Tiles[cx, cy];

            // Стена блокирует полёт стрелы
            if (tile == TileType.Wall || tile == TileType.Empty) break;

            // Ищем врага на этой клетке
            foreach (var enemy in state.Enemies)
            {
                if (!enemy.IsAlive) continue;
                if (enemy.X != cx || enemy.Y != cy) continue;

                if (enemy.IsVulnerableTo(ActionType.ArrowShot))
                {
                    enemy.HP--;
                    if (enemy.HP <= 0) enemy.IsAlive = false;
                }
                // Стрела поглощается первым встреченным врагом (вне зависимости от уязвимости)
                return;
            }

            cx += dx;
            cy += dy;
        }
    }
}
