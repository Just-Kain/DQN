using SFML.System;

public class Time
{
    Clock clock = new Clock();
    public float DeltaTime()
    {
        return clock.Restart().AsSeconds();
    }
}