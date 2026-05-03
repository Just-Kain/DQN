/// <summary>
/// Ползающий враг — медленный, но живучий.
///
/// Характеристики:
///   HP  = 4  |  Speed = 3  |  Damage = 1
///
/// Движение:  только по проходимым тайлам (Floor, Exit); медленнее ходячих.
///
/// Уязвим к:  ТОЛЬКО удару (MeleeAttack).
///            Неуязвим к выстрелу (ArrowShot) — стрела его не задевает
///            (ползёт низко к земле).
/// </summary>
public sealed class CrawlingEnemy : Enemy
{
    public CrawlingEnemy() { HP = 4; MaxHP = 4; }

    public override float     Speed           { get; set; } = 2f;
    public override EnemyType Type            => EnemyType.Crawling;
    public override int       DamageOnContact => 1;
    public override int       VisionRange     => 6;    // ползун — слабое зрение

    public override bool IsVulnerableTo(ActionType attack)
        => attack is ActionType.MeleeAttack;

    public override bool CanPassTile(TileType tile)
        => tile is TileType.Floor or TileType.Exit;

    public override Enemy Clone() => new CrawlingEnemy
    {
        X = X, Y = Y, Xf = Xf, Yf = Yf,
        HP = HP, MaxHP = MaxHP,
        IsAlive = IsAlive,
        AttackCooldown = AttackCooldown
    };
}
