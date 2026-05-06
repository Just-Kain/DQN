/// <summary>
/// Система врагов — дискретное движение + урон с кулдауном + зона видимости + скорость.
///
/// Правила движения (энергетический аккумулятор):
///   Каждый игровой шаг:
///     e.MoveEnergy += e.Speed / player.Speed
///   Если MoveEnergy ≥ 1.0 — враг совершает ход (движение + атака), MoveEnergy -= 1.0.
///   Иначе — только тикает кулдаун атаки.
///
///   Соотношения скоростей:
///     Walking  (4/5 = 0.80) — чуть медленнее игрока, можно убежать.
///     Flying   (5/5 = 1.00) — такая же скорость, как у игрока.
///     Crawling (3/5 = 0.60) — заметно медленнее, легко убежать.
///
///   • Враг движется к игроку ТОЛЬКО если:
///       1. Манхэттенское расстояние ≤ e.VisionRange.
///       2. Нет стены на пути (проверка через Bresenham LOS).
///   • Враг НЕ заходит на клетку игрока — держится вплотную (dist == 1).
///   • Проходимость тайла — e.CanPassTile().
///   • Несколько врагов на одной клетке запрещено, если !e.CanStack.
///
/// Правила урона:
///   • Наносится, когда dist == 1 (вплотную) и враг получил ход в этом шаге.
///   • Не чаще одного раза в AttackCooldownSteps шагов (~1 сек при 0.12 с/шаг).
/// </summary>
public class EnemySystem
{
    private const int AttackCooldownSteps = 8;   // ~1 секунда

    public void Update(GameState state, float _dt)
    {
        float playerSpeed = state.Player.Speed;   // базовая скорость (5)

        foreach (var e in state.Enemies)
        {
            if (!e.IsAlive) continue;

            // Накапливаем энергию движения пропорционально скорости
            e.MoveEnergy += e.Speed / playerSpeed;

            if (e.MoveEnergy >= 1.0f)
            {
                e.MoveEnergy -= 1.0f;          // тратим накопленную энергию

                TryMoveTowardPlayer(state, e);
                TryDamagePlayer(state, e);
            }
            else
            {
                // Враг не двигается в этот шаг, но кулдаун атаки всё равно тикает
                if (e.AttackCooldown > 0) e.AttackCooldown--;
            }
        }
    }

    // ── Движение ──────────────────────────────────────────────────────────────
    private static void TryMoveTowardPlayer(GameState state, Enemy e)
    {
        int px = state.Player.X;
        int py = state.Player.Y;

        int dist = Math.Abs(px - e.X) + Math.Abs(py - e.Y);

        // Враг не видит игрока — стоит на месте
        if (dist > e.VisionRange) return;
        if (!HasLineOfSight(state.Map, e.X, e.Y, px, py)) return;

        int absDx = Math.Abs(px - e.X);
        int absDy = Math.Abs(py - e.Y);

        // Предпочитаем ось с большим расстоянием; при равенстве — горизонталь
        (int dx, int dy)[] candidates = absDx >= absDy
            ? new[] { (Math.Sign(px - e.X), 0), (0, Math.Sign(py - e.Y)) }
            : new[] { (0, Math.Sign(py - e.Y)), (Math.Sign(px - e.X), 0) };

        foreach (var (dx, dy) in candidates)
        {
            if (dx == 0 && dy == 0) continue;

            int nx = e.X + dx;
            int ny = e.Y + dy;

            if (!CanEnemyMoveTo(state, e, nx, ny)) continue;

            e.X = nx; e.Xf = nx;
            e.Y = ny; e.Yf = ny;
            break;
        }
    }

    /// <summary>
    /// Проверяет проходимость клетки (x, y) для врага e.
    /// Запрещены: стены, клетка игрока, другие нестакующиеся враги.
    /// </summary>
    private static bool CanEnemyMoveTo(GameState state, Enemy e, int x, int y)
    {
        if (x < 0 || x >= state.Map.Width || y < 0 || y >= state.Map.Height)
            return false;

        if (!e.CanPassTile(state.Map.Tiles[x, y]))
            return false;

        // Клетка игрока — враг остаётся рядом, не заходит
        if (x == state.Player.X && y == state.Player.Y)
            return false;

        if (!e.CanStack)
        {
            foreach (var other in state.Enemies)
            {
                if (other == e || !other.IsAlive) continue;
                if (other.X == x && other.Y == y) return false;
            }
        }

        return true;
    }

    // ── Линия видимости (Bresenham) ───────────────────────────────────────────
    /// <summary>
    /// Проверяет, есть ли прямая видимость между (x0,y0) и (x1,y1).
    /// Стены (Wall, Empty) прерывают линию.
    /// Ямы прозрачны — летающие враги видят сквозь них.
    /// </summary>
    private static bool HasLineOfSight(DungeonMap map, int x0, int y0, int x1, int y1)
    {
        int dx  = Math.Abs(x1 - x0);
        int dy  = Math.Abs(y1 - y0);
        int sx  = x0 < x1 ? 1 : -1;
        int sy  = y0 < y1 ? 1 : -1;
        int err = dx - dy;

        int cx = x0, cy = y0;

        while (true)
        {
            if (cx == x1 && cy == y1) return true;   // дошли до цели — видим

            // Промежуточные клетки (не стартовая, не цель) проверяем на блок
            if ((cx != x0 || cy != y0) && (cx != x1 || cy != y1))
            {
                var t = map.Tiles[cx, cy];
                if (t == TileType.Wall || t == TileType.Empty) return false;
            }

            int e2 = 2 * err;
            if (e2 > -dy) { err -= dy; cx += sx; }
            if (e2 <  dx) { err += dx; cy += sy; }
        }
    }

    // ── Урон при контакте с кулдауном ────────────────────────────────────────
    private static void TryDamagePlayer(GameState state, Enemy e)
    {
        if (e.AttackCooldown > 0) { e.AttackCooldown--; return; }

        int dist = Math.Abs(e.X - state.Player.X) + Math.Abs(e.Y - state.Player.Y);
        if (dist != 1) return;

        state.Player.HP = Math.Max(0, state.Player.HP - e.DamageOnContact);
        e.AttackCooldown = AttackCooldownSteps;
    }
}
