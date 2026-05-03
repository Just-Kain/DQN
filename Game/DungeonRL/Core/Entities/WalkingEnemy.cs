/// <summary>
/// Ходячий враг — самый распространённый тип.
///
/// Характеристики:
///   HP  = 3  |  Speed = 4  |  Damage = 1
///
/// Движение:  только по проходимым тайлам (Floor, Exit).
/// Уязвим к:  удару (MeleeAttack) и выстрелу (ArrowShot).
/// </summary>
public sealed class WalkingEnemy : Enemy
{
    public override float     Speed           { get; set; } = 2f;
    public override EnemyType Type            => EnemyType.Walking;
    public override int       DamageOnContact => 1;
    public override int       VisionRange     => 8;    // видит в радиусе 8 клеток

    public override bool IsVulnerableTo(ActionType attack)
        => attack is ActionType.MeleeAttack or ActionType.ArrowShot;

    public override bool CanPassTile(TileType tile)
        => tile is TileType.Floor or TileType.Exit;

    public override Enemy Clone() => new WalkingEnemy
    {
        X = X, Y = Y, Xf = Xf, Yf = Yf,
        HP = HP, MaxHP = MaxHP,
        IsAlive = IsAlive,
        AttackCooldown = AttackCooldown
    };
}
