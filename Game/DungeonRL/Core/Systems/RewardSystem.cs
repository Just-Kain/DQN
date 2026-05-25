/// <summary>
/// Система наград для DQN-агента.
///
/// Сводная таблица сигналов:
///
///   Событие                              Награда
///   ─────────────────────────────────── ──────────────────────────────────────
///   Каждый шаг (штраф за время)          −0.05
///   Убийство врага (за каждого)           +3.0
///   Атака без результата                  −0.3
///   Действие Idle                         −2.0
///   Удар в стену (движение без смещения)  −1.0
///   Урон по игроку (% от MaxHP × 10)       −10 × hpLost/MaxHP
///   Смерть игрока                         −25.0
///   BFS-шейпинг (непрерывный, см. ниже)   ±0.5 × Δdist
///   Достижение выхода (победа)            +100.0
///
/// Почему нет бонуса за исследование:
///   Любой фиксированный бонус «за новую клетку» создаёт стимул бесконечно
///   бродить по карте и фармить очки вместо того, чтобы идти к выходу.
///   Даже условный вариант (новая клетка + стал ближе) ломается при врагах:
///   агент вынужден маневрировать и не получает никакого сигнала.
///
/// Непрерывный BFS-шейпинг (потенциальный):
///   reward_shape = BFS_SCALE × (prev_dist − curr_dist)
///
///   Примеры (BFS_SCALE = 0.5):
///     Приблизился на 1 клетку → +0.5
///     Удалился на 1 клетку    → −0.5   (активный штраф, не просто «нет бонуса»)
///     Детур 3 шага + возврат  → суммарно 0 (фарм невозможен)
///     Стоит на месте          → 0
///
///   Гарантия anti-farming: при блуждании по кругу все дельты компенсируются.
///   Градиент всегда указывает к выходу — даже при обходе врагов.
/// </summary>
public class RewardSystem
{
    private int _prevExitDist = int.MaxValue;

    private const float BFS_SCALE    = 0.5f;   // награда/штраф за 1 шаг к/от выхода
    private const float DAMAGE_SCALE = 10.0f;  // штраф = DAMAGE_SCALE * (hpLost / MaxHP)
                                                // При MaxHP=10: -1 HP → -1.0 (как раньше)
                                                // При MaxHP=50: -5 HP → -1.0 (пропорционально)
    private const float WALL_HIT_PENALTY = -1.0f; // штраф за попытку войти в стену

    // ── Основной расчёт ───────────────────────────────────────────────────────
    public float Compute(GameState prev, GameState current, bool done)
    {
        float reward = 0f;

        // ── Штраф за время ──────────────────────────────────────────────────
        reward -= 0.05f;

        // ── Убийство врагов ──────────────────────────────────────────────────
        int killedThisStep = CountAlive(prev) - CountAlive(current);
        if (killedThisStep > 0)
            reward += 3.0f * killedThisStep;

        // ── Штраф за бесполезную атаку ──────────────────────────────────────
        if (ActionIsAttack(current) && killedThisStep == 0)
            reward -= 0.3f;

        // ── Штраф за простой ────────────────────────────────────────────────
        if (ActionIsIdle(current))
            reward -= 2.0f;

        // ── Штраф за удар в стену ────────────────────────────────────────────
        // Движение (Up/Down/Left/Right), при котором позиция не изменилась —
        // значит агент упёрся в стену или границу карты.
        // Не применяется к Dash (может частично переместиться) и атакам.
        if (ActionIsMove(current) &&
            prev.Player.X == current.Player.X &&
            prev.Player.Y == current.Player.Y)
            reward += WALL_HIT_PENALTY;

        // ── Урон по игроку (% от MaxHP) ──────────────────────────────────────
        // Штраф нормализован: потеря 10% HP = -1.0 вне зависимости от MaxHP.
        // Это выравнивает сигнал между фазами (MaxHP 15 → 50).
        int hpDiff = current.Player.HP - prev.Player.HP;
        if (hpDiff < 0 && current.Player.MaxHP > 0)
            reward -= DAMAGE_SCALE * (-hpDiff) / current.Player.MaxHP;

        // ── Смерть ───────────────────────────────────────────────────────────
        if (current.Player.HP <= 0)
            reward -= 25.0f;

        // ── Непрерывный BFS-шейпинг ──────────────────────────────────────────
        //
        //   Вычисляем кратчайший BFS-путь до выхода в текущем состоянии.
        //   Награда пропорциональна изменению расстояния:
        //     delta > 0  →  агент стал ближе  →  положительная награда
        //     delta < 0  →  агент удалился    →  отрицательная награда (штраф)
        //     delta = 0  →  нет изменений     →  0
        //
        //   Это потенциальный шейпинг: блуждание по кругу не накапливает
        //   ни плюс, ни минус — фарм очков принципиально невозможен.
        //
        int currDist = BfsDist(
            current.Map,
            current.Player.X, current.Player.Y,
            current.Map.ExitX, current.Map.ExitY
        );

        if (_prevExitDist != int.MaxValue && currDist != int.MaxValue)
        {
            int delta = _prevExitDist - currDist;   // >0: ближе, <0: дальше, 0: на месте
            reward += BFS_SCALE * delta;
        }

        _prevExitDist = Math.Max(currDist, _prevExitDist);

        // ── Победа ───────────────────────────────────────────────────────────
        bool onExit = current.Map.Tiles[current.Player.X, current.Player.Y] == TileType.Exit;
        if (done && current.Player.HP > 0 && onExit)
            reward += 100.0f;

        return reward;
    }

    // ── BFS: кратчайший путь по проходимым тайлам ────────────────────────────
    /// <summary>
    /// Возвращает длину кратчайшего пути (в шагах) от (sx, sy) до (tx, ty)
    /// по тайлам, проходимым для игрока (IsWalkable = Floor, Exit, Pit).
    ///
    /// Алгоритм: обход в ширину (BFS) — гарантирует кратчайший путь
    /// в невзвешенном графе за O(V + E) = O(W × H).
    ///
    /// Возвращает int.MaxValue если цель недостижима.
    /// </summary>
    private static int BfsDist(DungeonMap map, int sx, int sy, int tx, int ty)
    {
        if (sx == tx && sy == ty) return 0;

        var queue = new Queue<(int x, int y)>();
        var dist  = new Dictionary<(int, int), int>();

        queue.Enqueue((sx, sy));
        dist[(sx, sy)] = 0;

        Span<int> ddx = stackalloc int[] {  0,  0, 1, -1 };
        Span<int> ddy = stackalloc int[] {  1, -1, 0,  0 };

        while (queue.Count > 0)
        {
            var (cx, cy) = queue.Dequeue();
            int d = dist[(cx, cy)];

            for (int i = 0; i < 4; i++)
            {
                int nx = cx + ddx[i];
                int ny = cy + ddy[i];

                if (!map.IsWalkable(nx, ny))      continue;
                if (dist.ContainsKey((nx, ny)))   continue;

                int nd = d + 1;
                dist[(nx, ny)] = nd;

                if (nx == tx && ny == ty) return nd;

                queue.Enqueue((nx, ny));
            }
        }

        return int.MaxValue;
    }

    // ── Вспомогательные методы ────────────────────────────────────────────────
    private static int CountAlive(GameState state)
    {
        int count = 0;
        foreach (var e in state.Enemies)
            if (e.IsAlive) count++;
        return count;
    }

    private static bool ActionIsAttack(GameState state)
        => state.LastAction is ActionType.MeleeAttack or ActionType.ArrowShot;

    private static bool ActionIsIdle(GameState state)
        => state.LastAction == ActionType.Idle;

    private static bool ActionIsMove(GameState state)
        => state.LastAction is ActionType.Up or ActionType.Down
                            or ActionType.Left or ActionType.Right;

    // ── Сброс между эпизодами ─────────────────────────────────────────────────
    public void Reset()
    {
        _prevExitDist = int.MaxValue;
    }
}
