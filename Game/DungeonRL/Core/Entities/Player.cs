public class Player : Entity
{
    public int    HP    = 10;
    public int    MaxHP = 10;

    /// <summary>Направление последнего движения — определяет зону атаки.</summary>
    public Direction Facing = Direction.Down;

    public override float Speed { get; set; } = 10f;
}