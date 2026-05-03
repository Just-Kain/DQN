using SFML.Graphics;
using SFML.System;

public class Camera
{
    private View view;
    private RenderWindow window;

    public Camera(RenderWindow window, float width, float height)
    {
        this.window = window;
        view = new View(new FloatRect((0, 0), (width, height)));
    }

    public void Reset()
    {
        window.SetView(window.DefaultView);
    }
}