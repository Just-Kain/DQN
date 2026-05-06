/// <summary>
/// Ходячий враг — самый распространённый тип.
///
/// Характеристики:
///   HP  = 3  |  Speed = 4  |  Damage = 1
///
/// Движение:  только по проходимым тайлам (Floor, Exit).
/// Уязвим к:  любому оружию (MeleeAttack и ArrowShot).
/// </summary>
public sealed class WalkingEnemy : Enemy
{
    public override float     Speed           { get; set; } = 4f;   // 4/5 шагов игрока
    public override EnemyType Type            => EnemyType.Walking;
    public override int       DamageOnContact => 1;
    public override int       VisionRange     => 8;    // видит в радиусе 8 клеток

    public override bool IsVulnerableTo(ActionType attack)
        => attack is ActionType.MeleeAttack or ActionType.ArrowShot or ActionType.Dash;

    public override bool CanPassTile(TileType tile)
        => tile is TileType.Floor or TileType.Exit;

    public override Enemy Clone() => new WalkingEnemy
    {
        X = X, Y = Y, Xf = Xf, Yf = Yf,
        HP = HP, MaxHP = MaxHP,
        IsAlive = IsAlive,
        AttackCooldown = AttackCooldown,
        MoveEnergy = MoveEnergy
    };
}
