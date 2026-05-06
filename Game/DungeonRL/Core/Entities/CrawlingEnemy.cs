/// <summary>
/// Ползающий враг — медленный, но живучий.
///
/// Характеристики:
///   HP  = 4  |  Speed = 3  |  Damage = 1
///
/// Движение:  только по проходимым тайлам (Floor, Exit); медленнее ходячих.
///
/// Уязвим к:  любому оружию (MeleeAttack и ArrowShot).
/// </summary>
public sealed class CrawlingEnemy : Enemy
{
    public CrawlingEnemy() { HP = 4; MaxHP = 4; }

    public override float     Speed           { get; set; } = 3f;   // 3/5 шагов игрока
    public override EnemyType Type            => EnemyType.Crawling;
    public override int       DamageOnContact => 1;
    public override int       VisionRange     => 6;    // ползун — слабое зрение

    public override bool IsVulnerableTo(ActionType attack)
        => attack is ActionType.MeleeAttack or ActionType.ArrowShot or ActionType.Dash;

    public override bool CanPassTile(TileType tile)
        => tile is TileType.Floor or TileType.Exit;

    public override Enemy Clone() => new CrawlingEnemy
    {
        X = X, Y = Y, Xf = Xf, Yf = Yf,
        HP = HP, MaxHP = MaxHP,
        IsAlive = IsAlive,
        AttackCooldown = AttackCooldown,
        MoveEnergy = MoveEnergy
    };
}
