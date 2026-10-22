using System;
using System.Collections.Generic;
using Godot;

namespace PingPong;

/// <summary>右侧面板的控件（与 Python 版同样的滑条和按钮）：自己画、自己处理鼠标。</summary>
public abstract class Widget
{
    public string[] Pages;
    public abstract bool Handle(InputEvent e);
    public abstract void Draw(Ui ui);
}

public class UiSlider : Widget
{
    public Rect2 Rect, Track;
    public string Label;
    public double VMin, VMax, Value;
    public Color Color;
    public Action<double> OnChange;
    public Func<double, string> Fmt;
    bool _drag;

    public UiSlider(float x, float y, float w, string label, double vmin, double vmax, double value, Color color,
        Action<double> onChange, Func<double, string> fmt = null)
    {
        Rect = new Rect2(x, y, w, 36);
        Track = new Rect2(x, y + 24, w, 6);
        Label = label;
        VMin = vmin;
        VMax = vmax;
        Value = value;
        Color = color;
        OnChange = onChange;
        Fmt = fmt ?? (v => $"{v:F0}");
    }

    void SetFromX(float x)
    {
        double t = Math.Clamp((x - Track.Position.X) / Track.Size.X, 0.0, 1.0);
        Value = VMin + t * (VMax - VMin);
        OnChange(Value);
    }

    public void SetValue(double v)
    {
        Value = v;
        OnChange(v);
    }

    public override bool Handle(InputEvent e)
    {
        if (e is InputEventMouseButton mb && mb.ButtonIndex == MouseButton.Left)
        {
            if (mb.Pressed && Rect.Grow(4).HasPoint(mb.Position))
            {
                _drag = true;
                SetFromX(mb.Position.X);
                return true;
            }
            if (!mb.Pressed && _drag)
            {
                _drag = false;
                return true;
            }
        }
        if (e is InputEventMouseMotion mm && _drag)
        {
            SetFromX(mm.Position.X);
            return true;
        }
        return false;
    }

    public override void Draw(Ui ui)
    {
        ui.Text(Label, Rect.Position, Ui.FONT, Ui.TEXT);
        var val = Fmt(Value);
        ui.TextRight(val, new Vector2(Rect.End.X, Rect.Position.Y), Ui.FONT, Color);
        ui.RoundRect(Track, Gfx.C(55, 60, 75), 3);
        float t = (float)((Value - VMin) / (VMax - VMin));
        var fill = new Rect2(Track.Position, new Vector2(Track.Size.X * t, Track.Size.Y));
        ui.RoundRect(fill, Color, 3);
        ui.DrawCircle(new Vector2(Track.Position.X + fill.Size.X, Track.GetCenter().Y), 8, Gfx.C(240, 240, 245));
    }
}

public class UiButton : Widget
{
    public Rect2 Rect;
    public Func<string> Text;
    public Action OnClick;
    public Func<bool> Active;
    public Color Color;

    public UiButton(Rect2 rect, Func<string> text, Action onClick, Func<bool> active = null, Color? color = null)
    {
        Rect = rect;
        Text = text;
        OnClick = onClick;
        Active = active;
        Color = color ?? Gfx.C(80, 110, 170);
    }

    public UiButton(Rect2 rect, string text, Action onClick, Func<bool> active = null, Color? color = null)
        : this(rect, () => text, onClick, active, color)
    {
    }

    public override bool Handle(InputEvent e)
    {
        if (e is InputEventMouseButton mb && mb.Pressed && mb.ButtonIndex == MouseButton.Left &&
            Rect.HasPoint(mb.Position))
        {
            OnClick();
            return true;
        }
        return false;
    }

    public override void Draw(Ui ui)
    {
        bool on = Active?.Invoke() ?? false;
        bool hover = Rect.HasPoint(ui.GetLocalMousePosition());
        var col = on ? Color : hover ? Gfx.C(70, 76, 92) : Gfx.C(52, 57, 70);
        ui.RoundRect(Rect, col, 6);
        ui.TextCenter(Text(), Rect.GetCenter(), Ui.FONT_S, Ui.TEXT);
    }
}
