
public class MovementSystem
{
    public void Update(GameState state, ActionType action, float dt)
    {
        float dx = 0, dy = 0;

        switch (action)
        {
            case ActionType.Up: dy = -1; break;
            case ActionType.Down: dy = 1; break;
            case ActionType.Left: dx = -1; break;
            case ActionType.Right: dx = 1; break;
        }

        state.Player.Xf += dx * state.Player.Speed * dt;
        state.Player.Yf += dy * state.Player.Speed * dt;

        state.Player.X = (int)state.Player.Xf;
        state.Player.Y = (int)state.Player.Yf;
        Console.WriteLine($"Enemy is move {dt}");

    }
}