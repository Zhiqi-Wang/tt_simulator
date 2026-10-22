using System;
using Godot;

namespace PingPong;

/// <summary>模拟坐标系（x 沿球台长边、y 横向、z 向上，米）里的双精度向量。动画、骨架都在这个坐标系里算，画的时候再换到 Godot。</summary>
public struct V3
{
    public double X, Y, Z;

    public V3(double x, double y, double z)
    {
        X = x;
        Y = y;
        Z = z;
    }

    public static readonly V3 Zero = new(0, 0, 0);
    public static readonly V3 Up = new(0, 0, 1);

    public double this[int k]
    {
        get => k == 0 ? X : k == 1 ? Y : Z;
        set
        {
            if (k == 0) X = value;
            else if (k == 1) Y = value;
            else Z = value;
        }
    }

    public static V3 operator +(V3 a, V3 b) => new(a.X + b.X, a.Y + b.Y, a.Z + b.Z);
    public static V3 operator -(V3 a, V3 b) => new(a.X - b.X, a.Y - b.Y, a.Z - b.Z);
    public static V3 operator -(V3 a) => new(-a.X, -a.Y, -a.Z);
    public static V3 operator *(V3 a, double s) => new(a.X * s, a.Y * s, a.Z * s);
    public static V3 operator *(double s, V3 a) => new(a.X * s, a.Y * s, a.Z * s);
    public static V3 operator /(V3 a, double s) => new(a.X / s, a.Y / s, a.Z / s);

    public double Dot(V3 b) => X * b.X + Y * b.Y + Z * b.Z;
    public V3 Cross(V3 b) => new(Y * b.Z - Z * b.Y, Z * b.X - X * b.Z, X * b.Y - Y * b.X);
    public double Norm() => Math.Sqrt(X * X + Y * Y + Z * Z);
    public double NormXY() => Math.Sqrt(X * X + Y * Y);
    public V3 Unit() => this / (Norm() + 1e-9);
    public V2 XY => new(X, Y);
    public V3 WithZ(double z) => new(X, Y, z);

    public static V3 Lerp(V3 a, V3 b, double u) => a + (b - a) * u;

    /// <summary>模拟 (x, y, z-up) → Godot (x, z, -y)：是一个正常旋转，手性不变。</summary>
    public Vector3 G => new((float)X, (float)Z, (float)-Y);

    public static V3 FromG(Vector3 v) => new(v.X, -v.Z, v.Y);

    public override string ToString() => $"({X:F3}, {Y:F3}, {Z:F3})";
}

public struct V2
{
    public double X, Y;

    public V2(double x, double y)
    {
        X = x;
        Y = y;
    }

    public static readonly V2 Zero = new(0, 0);

    public double this[int k]
    {
        get => k == 0 ? X : Y;
        set
        {
            if (k == 0) X = value;
            else Y = value;
        }
    }

    public static V2 operator +(V2 a, V2 b) => new(a.X + b.X, a.Y + b.Y);
    public static V2 operator -(V2 a, V2 b) => new(a.X - b.X, a.Y - b.Y);
    public static V2 operator -(V2 a) => new(-a.X, -a.Y);
    public static V2 operator *(V2 a, double s) => new(a.X * s, a.Y * s);
    public static V2 operator *(double s, V2 a) => new(a.X * s, a.Y * s);
    public static V2 operator /(V2 a, double s) => new(a.X / s, a.Y / s);
    public double Norm() => Math.Sqrt(X * X + Y * Y);
    public double Dot(V2 b) => X * b.X + Y * b.Y;
}

public static class M
{
    public static double Clamp(double v, double lo, double hi) => v < lo ? lo : v > hi ? hi : v;

    public static double Smooth(double u)
    {
        u = Clamp(u, 0.0, 1.0);
        return u * u * (3.0 - 2.0 * u);
    }

    public static double CopySign(double mag, double sign) => Math.CopySign(mag, sign);
}
