public class GameState
{
    public DungeonMap Map;

    public Player Player;
    public List<Enemy> Enemies;

    public int StepCount;
    public bool IsTerminal;

    /// <summary>
    /// Последнее действие игрока на этом шаге.
    /// Используется рендерером для отрисовки хитбоксов атак.
    /// </summary>
    public ActionType LastAction = ActionType.Idle;
}