

public class DungeonEnv
{
    private MovementSystem movement = new();
    private CombatSystem combat = new();
    private EnemySystem enemy = new();
    private CleanupSystem cleanup = new();
    private CollisionSystem collision = new();
    private RewardSystem rewardSystem = new();
    private DungeonGenerator generator = new();
    private float dt = 0f;

    public Time time = new(); 
    public GameState State { get; private set; }

    public GameState Reset(int seed = 0)
    {
        var map = generator.Generate(32, 32, seed);

        // Спавним игрока в точке максимального удаления от выхода (рассчитывает генератор)
        int spawnX = map.PlayerSpawnX;
        int spawnY = map.PlayerSpawnY;

        State = new GameState
        {
            Map = map,
            Player = new Player
            {
                X  = spawnX, Y  = spawnY,
                Xf = spawnX, Yf = spawnY,
                HP = 10
            },
            Enemies = new List<Enemy>
            {
                new Enemy { X = 10, Xf = 10f, Y = 10, Yf = 10f, HP = 3 }
            },
            StepCount  = 0,
            IsTerminal = false
        };

        rewardSystem.Reset();

        return State;
    }

    public StepResult Step(ActionType action)
    {
        if (State.IsTerminal)
            return new StepResult(State, 0, true);

        // копия состояния ДО шага (для reward)
        var prevState = CloneState(State);

        dt = time.DeltaTime();
        // --- системы ---
        ApplyPlayerAction(action);
        combat.Update(State, action);
        enemy.Update(State, dt);
        cleanup.Update(State);

        State.StepCount++;

        bool done = CheckTerminal();
        State.IsTerminal = done;

        float reward = rewardSystem.Compute(prevState, State, done);

        return new StepResult(State, reward, done);
    }

    private void ApplyPlayerAction(ActionType action)
    {
        movement.Update(State, action, dt);
    }

    private bool CheckTerminal()
    {
        // Игрок погиб
        if (State.Player.HP <= 0)
            return true;

        // Игрок достиг выхода
        int px = State.Player.X;
        int py = State.Player.Y;
        if (State.Map.Tiles[px, py] == TileType.Exit)
            return true;

        return false;
    }

    // важно: глубокая копия (минимальная)
    private GameState CloneState(GameState s)
    {
        return new GameState
        {
            Map = s.Map, // можно шарить (если immutable)
            Player = new Player
            {
                X = s.Player.X,
                Y = s.Player.Y,
                HP = s.Player.HP
            },
            Enemies = s.Enemies
                .Select(e => new Enemy
                {
                    X = e.X,
                    Y = e.Y,
                    HP = e.HP,
                    IsAlive = e.IsAlive
                })
                .ToList(),
            StepCount = s.StepCount,
            IsTerminal = s.IsTerminal
        };
    }
}