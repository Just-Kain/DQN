public class EnemySystem
{
    public void Update(GameState state, float dt)
    {
        foreach (var e in state.Enemies)
        {
            float dx = Math.Sign(state.Player.X - e.X);
            float dy = Math.Sign(state.Player.Y - e.Y);

            e.Xf += dx * e.Speed * dt;
            e.Yf += dy * e.Speed * dt;

            e.X = (int)e.Xf;
            e.Y = (int)e.Yf;
            Console.WriteLine($"Enemy is move {e.Xf} {e.Yf} {dx} {dy} {dt}");
        }
    }
}