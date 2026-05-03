public class GameState
{
    public DungeonMap Map;

    public Player Player;
    public List<Enemy> Enemies;

    public int StepCount;
    public bool IsTerminal;
}