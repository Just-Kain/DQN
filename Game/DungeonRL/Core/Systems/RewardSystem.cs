using System.Collections.Generic;

public class RewardSystem
{
    private HashSet<(int, int)> visited = new();

    public float Compute(GameState prev, GameState current, bool done)
    {
        float reward = 0f;

        // штраф за шаг
        reward -= 0.1f;

        // исследование
        var pos = (current.Player.X, current.Player.Y);
        if (!visited.Contains(pos))
        {
            visited.Add(pos);
            reward += 0.2f;
        }

        // убийство врага
        int prevEnemies = CountAlive(prev);
        int currEnemies = CountAlive(current);

        if (currEnemies < prevEnemies)
            reward += 1.0f;

        // смерть игрока
        if (current.Player.HP <= 0)
            reward -= 10f;

        // завершение
        if (done && current.Player.HP > 0)
            reward += 20f;

        return reward;
    }

    private int CountAlive(GameState state)
    {
        int count = 0;
        foreach (var e in state.Enemies)
            if (e.IsAlive) count++;
        return count;
    }

    public void Reset()
    {
        visited.Clear();
    }
}