public struct StepResult
{
    public GameState NextState;
    public float Reward;
    public bool Done;
    public int StepIteration = 0;

    public StepResult(GameState nextState, float reward, bool done)
    {
        NextState = nextState;
        Reward = reward;
        Done = done;
        StepIteration += 1;
    }
}