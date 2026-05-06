/// <summary>
/// Абстрактный базовый класс врага.
///
/// Каждый конкретный подкласс определяет:
///   Type           — идентификатор типа (для сериализации / рендера)
///   Speed          — скорость (тайлов в секунду, не используется при дискретном шаге)
///   HP / MaxHP     — здоровье
///   DamageOnContact — урон игроку при контакте за шаг
///   IsVulnerableTo — какие атаки наносят урон
///   CanPassTile    — какие тайлы проходимы (определяет способ движения)
///   CanStack       — могут ли несколько врагов стоять на одной клетке
///   Clone          — глубокая копия для CloneState в DungeonEnv
/// </summary>
public abstract class Enemy : Entity
{
    public int HP    = 1;
    public int MaxHP = 1;

    /// <summary>
    /// Оставшиеся шаги до следующей атаки.
    /// 0 = может атаковать немедленно.
    /// Сбрасывается в AttackCooldownSteps после каждого удара.
    /// </summary>
    public int AttackCooldown = 0;

    /// <summary>
    /// Накопленная энергия движения [0..∞).
    /// Каждый шаг игры прибавляется Speed / Player.Speed.
    /// Когда накопилось ≥ 1.0 — враг совершает ход (движение + атака),
    /// из счётчика вычитается 1.0.
    /// Это позволяет врагам с Speed &lt; Player.Speed двигаться реже,
    /// и игрок может убежать от медленных противников.
    /// </summary>
    public float MoveEnergy = 0f;

    // ── Свойства, определяемые подклассом ────────────────────────────────────
    public abstract EnemyType Type            { get; }
    public abstract int       DamageOnContact { get; }
    public virtual  bool      CanStack        => false;

    /// <summary>
    /// Радиус обзора врага в тайлах (манхэттенское расстояние).
    /// Враг начинает преследование только если видит игрока в этом радиусе
    /// И линия взгляда не перекрыта стенами.
    /// </summary>
    public abstract int VisionRange { get; }

    /// <summary>Возвращает true, если атака наносит этому врагу урон.</summary>
    public abstract bool IsVulnerableTo(ActionType attack);

    /// <summary>
    /// Возвращает true, если враг может наступить на тайл данного типа.
    /// Определяет способ движения: наземный или воздушный.
    /// </summary>
    public abstract bool CanPassTile(TileType tile);

    /// <summary>Глубокая копия для CloneState.</summary>
    public abstract Enemy Clone();
}
