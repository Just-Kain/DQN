public class CleanupSystem
{
    public void Update(GameState state)
    {
        state.Enemies.RemoveAll(e => !e.IsAlive);
    }
}