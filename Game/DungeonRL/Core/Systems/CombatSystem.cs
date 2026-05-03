public class CombatSystem
{
    public void Update(GameState state, ActionType action)
    {
        if (action != ActionType.Attack) return;

        // foreach (var enemy in state.Enemies)
        // {
        //     if (Math.Abs(enemy.X - state.Player.X) +
        //         Math.Abs(enemy.Y - state.Player.Y) == 1)
        //     {
        //         enemy.HP -= 1;
        //         if (enemy.HP <= 0)
        //             enemy.IsAlive = false;
        //     }
        // }
    }
}