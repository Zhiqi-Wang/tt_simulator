using System;
using Godot;

namespace PingPong;

/// <summary>环绕镜头（与 Python 版 Camera 相同的方位角/仰角/距离）：拖动转视角，滚轮缩放。</summary>
public class CameraRig
{
    public double Az, El, Dist;
    public V3 Target;
    public V3 Pos, Fwd, Right, Up;
    readonly Camera3D _cam;

    public CameraRig(Camera3D cam)
    {
        _cam = cam;
        _cam.Fov = 60f;
        _cam.Near = 0.05f;
        _cam.Far = 60f;
        Reset();
    }

    public void Reset()
    {
        Az = -Math.PI / 2;
        El = Mathf.DegToRad(30);
        Dist = 7.4;
        Target = new V3(0.3, 0.0, 0.7);
        Update();
    }

    public void Update()
    {
        El = Math.Clamp(El, Mathf.DegToRad(3), Mathf.DegToRad(85));
        Dist = Math.Clamp(Dist, 2.5, 14.0);
        double ce = Math.Cos(El);
        Pos = Target + Dist * new V3(Math.Sin(Az) * ce, -Math.Cos(Az) * ce, Math.Sin(El));
        Fwd = (Target - Pos).Unit();
        Right = Fwd.Cross(V3.Up).Unit();
        Up = Right.Cross(Fwd);
        _cam.LookAtFromPosition(Pos.G, Target.G, Vector3.Up);
    }

    public Vector2 Screen(V3 p) => _cam.UnprojectPosition(p.G);
    public bool Behind(V3 p) => _cam.IsPositionBehind(p.G);
}
