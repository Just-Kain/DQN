using System.Collections.Generic;

/// <summary>
/// Система наград для DQN-агента.
///
/// Сводная таблица сигналов (нормализована под максимальную награду ~100):
///
///   Событие                          Награда
///   ─────────────────────────────── ──────────────────────────
///   Каждый шаг (штраф за время)      −0.2
///   Новая клетка (исследование)       +0.2
///   Убийство врага (за каждого)       +3.0
///   Атака без результата              −0.3   (штрафуем спам)
///   Действие Idle                     −2.0
///   Урон по игроку (за единицу HP)    −1.0
///   Смерть игрока                     −25.0
///   Приближение к выходу (BFS)        +0.5   (см. ниже)
///   Достижение выхода (победа)        +100.0
///
/// Масштаб наград намеренно уменьшен (max ≈ 100 вместо 1000):
///   большой разброс r ∈ [−1000, +1000] дестабилизирует TD-цель
///   (большие δ = Q_pred − y дают нестабильные градиенты даже при Huber Loss).
///   Диапазон [−25, +100] безопаснее для SmoothL1 без нормировки.
///
/// Награда за приближение к выходу:
///   Каждый шаг вычисляем BFS-расстояние от игрока до выхода
///   по проходимым тайлам (IsWalkable = Floor | Exit | Pit).
///   Если dist_current &lt; dist_prev → агент стал ближе → +0.5.
///   Расстояние int.MaxValue означает недостижимость — награда не начисляется.
///
///   BFS гарантирует КРАТЧАЙШИЙ путь по сетке, что важно:
///   агент получает сигнал ровно тогда, когда реально сокращает
///   расстояние, а не просто движется в общем направлении.
/// </summary>
public class RewardSystem
{
    private HashSet<(int, int)> _visited     = new();
    private int                  _prevExitDist = int.MaxValue;

    // ── Основной расчёт ───────────────────────────────────────────────────────
    public float Compute(GameState prev, GameState current, bool done)
    {
        float reward = 0f;

        // ── Штраф за время ──────────────────────────────────────────────────
        // Небольшой штраф побуждает агента двигаться к выходу быстро,
        // но не настолько большой, чтобы перекрывать сигнал о победе.
        reward -= 0.2f;

        // ── Исследование новых клеток ────────────────────────────────────────
        // Поощряет перемещение по карте, не перекрывает BFS-сигнал.
        var pos = (current.Player.X, current.Player.Y);
        if (!_visited.Contains(pos))
        {
            _visited.Add(pos);
            reward += 0.2f;
        }

        // ── Убийство врагов ──────────────────────────────────────────────────
        int killedThisStep = CountAlive(prev) - CountAlive(current);
        if (killedThisStep > 0)
            reward += 3.0f * killedThisStep;

        // ── Штраф за бесполезную атаку ──────────────────────────────────────
        // Небольшой штраф чтобы агент не спамил атакой впустую.
        // Начисляем ТОЛЬКО если не было убийства (убийство компенсирует).
        if (ActionIsAttack(current) && killedThisStep == 0)
            reward -= 0.3f;

        // ── Штраф за простой ────────────────────────────────────────────────
        if (ActionIsIdle(current))
            reward -= 2.0f;

        // ── Урон по игроку ───────────────────────────────────────────────────
        int hpDiff = current.Player.HP - prev.Player.HP;
        if (hpDiff < 0)
            reward -= 1.0f * (-hpDiff);   // -hpDiff > 0 → вычитаем

        // ── Смерть ───────────────────────────────────────────────────────────
        if (current.Player.HP <= 0)
            reward -= 25.0f;

        // ── Приближение к выходу (BFS по проходимым тайлам) ─────────────────
        //
        //   d_t   = BFS(player_pos_t,   exit_pos)
        //   d_{t-1} сохранён в _prevExitDist
        //
        //   Если d_t < d_{t-1}  →  агент стал ближе к выходу по реальному пути
        //   Начисляем +0.5 именно за реальное сокращение пути.
        //
        int currDist = BfsDist(
            current.Map,
            current.Player.X, current.Player.Y,
            current.Map.ExitX, current.Map.ExitY
        );

        if (_prevExitDist != int.MaxValue   // предыдущая дистанция известна
            && currDist   != int.MaxValue   // путь существует
            && currDist   <  _prevExitDist) // агент реально приблизился
        {
            reward += 0.5f;
        }

        _prevExitDist = currDist;

        // ── Победа ───────────────────────────────────────────────────────────
        // 100 — достаточно большой сигнал (в ~500 раз больше штрафа за шаг),
        // но не 1000 — иначе разброс наград дестабилизирует обучение.
        if (done && current.Player.HP > 0)
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

    // ── Сброс между эпизодами ─────────────────────────────────────────────────
    public void Reset()
    {
        _visited.Clear();
        _prevExitDist = int.MaxValue;
    }
}
