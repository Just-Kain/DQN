/// <summary>
/// Летающий враг — игнорирует ямы и может стоять над ними.
///
/// Характеристики:
///   HP  = 2  |  Speed = 5  |  Damage = 1
///
/// Движение:  все тайлы, кроме Wall и Empty (включая Pit и Exit).
///            Может стоять на ямах (CanStack = true — несколько летающих
///            могут занимать одну клетку, т.к. летят на разной высоте).
///
/// Уязвим к:  любому оружию (MeleeAttack и ArrowShot).
///
/// Спавн:     может появляться над ямами (TileType.Pit).
/// </summary>
public sealed class FlyingEnemy : Enemy
{
    public FlyingEnemy() { HP = 1; MaxHP = 1; }

    public override float     Speed           { get; set; } = 5f;   // равен скорости игрока
    public override EnemyType Type            => EnemyType.Flying;
    public override int       DamageOnContact => 1;
    public override bool      CanStack        => true;
    public override int       VisionRange     => 12;   // летучий — лучший обзор

    public override bool IsVulnerableTo(ActionType attack)
        => attack is ActionType.MeleeAttack or ActionType.ArrowShot or ActionType.Dash;

    public override bool CanPassTile(TileType tile)
        => tile is not (TileType.Wall or TileType.Empty);

    public override Enemy Clone() => new FlyingEnemy
    {
        X = X, Y = Y, Xf = Xf, Yf = Yf,
        HP = HP, MaxHP = MaxHP,
        IsAlive = IsAlive,
        AttackCooldown = AttackCooldown,
        MoveEnergy = MoveEnergy
    };
}
