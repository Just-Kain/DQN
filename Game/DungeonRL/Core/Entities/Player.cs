public class Player : Entity
{
    public int    HP    = 5;
    public int    MaxHP = 5;

    /// <summary>Направление последнего движения — определяет зону атаки.</summary>
    public Direction Facing = Direction.Down;

    public override float Speed { get; set; } = 5f;   // эталон; враги делятся на эту скорость
}