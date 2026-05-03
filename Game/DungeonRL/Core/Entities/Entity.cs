public abstract class Entity
{
    public int X;
    public int Y;
    public float Xf;
    public float Yf;

    public bool IsAlive = true;
    public abstract float Speed {get; set;} // клеток в секунду
}