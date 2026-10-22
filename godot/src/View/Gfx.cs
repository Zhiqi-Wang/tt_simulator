using System.Collections.Generic;
using Godot;

namespace PingPong;

/// <summary>3D 画面的小工具：材质缓存、把圆柱/球/多面体摆到两点之间。</summary>
public static class Gfx
{
    static readonly Dictionary<(Color, bool, bool), StandardMaterial3D> _mats = new();

    public static Color C(int r, int g, int b, int a = 255) => Color.Color8((byte)r, (byte)g, (byte)b, (byte)a);

    public static Color Mix(Color a, Color b, float t) => a.Lerp(b, t);

    public static StandardMaterial3D Mat(Color c, bool unshaded = false, bool twoSided = false)
    {
        if (_mats.TryGetValue((c, unshaded, twoSided), out var m)) return m;
        m = new StandardMaterial3D
        {
            AlbedoColor = c,
            Roughness = 0.65f,
            ShadingMode = unshaded ? BaseMaterial3D.ShadingModeEnum.Unshaded : BaseMaterial3D.ShadingModeEnum.PerPixel,
            CullMode = twoSided ? BaseMaterial3D.CullModeEnum.Disabled : BaseMaterial3D.CullModeEnum.Back,
        };
        if (c.A < 1f) m.Transparency = BaseMaterial3D.TransparencyEnum.Alpha;
        _mats[(c, unshaded, twoSided)] = m;
        return m;
    }

    public static MeshInstance3D Cylinder(Node parent, float r0, float r1, Material mat, int seg = 14)
    {
        var mi = new MeshInstance3D
        {
            Mesh = new CylinderMesh { BottomRadius = r0, TopRadius = r1, Height = 1f, RadialSegments = seg, Rings = 1 },
            MaterialOverride = mat,
        };
        parent.AddChild(mi);
        return mi;
    }

    public static MeshInstance3D Sphere(Node parent, float r, Material mat, bool hemi = false)
    {
        var mi = new MeshInstance3D
        {
            Mesh = new SphereMesh
            {
                Radius = r, Height = hemi ? r : 2 * r, IsHemisphere = hemi, RadialSegments = 20,
                Rings = hemi ? 6 : 10,
            },
            MaterialOverride = mat,
        };
        parent.AddChild(mi);
        return mi;
    }

    public static MeshInstance3D Box(Node parent, Vector3 size, Vector3 pos, Material mat)
    {
        var mi = new MeshInstance3D { Mesh = new BoxMesh { Size = size }, MaterialOverride = mat, Position = pos };
        parent.AddChild(mi);
        return mi;
    }

    /// <summary>任意一个与 y 轴不平行的单位向量的垂直方向。</summary>
    public static Vector3 Perp(Vector3 y)
    {
        var a = Mathf.Abs(y.Y) < 0.9f ? Vector3.Up : Vector3.Right;
        return y.Cross(a).Normalized();
    }

    /// <summary>把单位高的圆柱摆成从 a 到 b（底面在 a）。</summary>
    public static void Span(MeshInstance3D mi, Vector3 a, Vector3 b)
    {
        var y = b - a;
        float len = y.Length();
        if (len < 1e-5f)
        {
            mi.Visible = false;
            return;
        }
        mi.Visible = true;
        var yu = y / len;
        var x = Perp(yu);
        var z = x.Cross(yu);
        mi.Transform = new Transform3D(new Basis(x, y, z), (a + b) * 0.5f);
    }

    /// <summary>球体单位网格摆到 p，局部 y 轴朝 up。</summary>
    public static void Orient(MeshInstance3D mi, Vector3 p, Vector3 up, Vector3 scale)
    {
        var yu = up.Normalized();
        var x = Perp(yu);
        var z = x.Cross(yu);
        mi.Transform = new Transform3D(new Basis(x * scale.X, yu * scale.Y, z * scale.Z), p);
    }

    /// <summary>六面体（下面 4 个角 + 上面 4 个角，各自绕一圈）：每面法向按离开中心的方向。</summary>
    public static void Hexa(ImmediateMesh im, IList<Vector3> c, Material mat)
    {
        var center = Vector3.Zero;
        foreach (var p in c) center += p;
        center /= c.Count;
        int[][] faces =
        {
            new[] { 0, 1, 2, 3 }, new[] { 4, 5, 6, 7 }, new[] { 0, 1, 5, 4 }, new[] { 1, 2, 6, 5 },
            new[] { 2, 3, 7, 6 }, new[] { 3, 0, 4, 7 },
        };
        im.SurfaceBegin(Mesh.PrimitiveType.Triangles, mat);
        foreach (var f in faces)
        {
            Vector3 a = c[f[0]], b = c[f[1]], cc = c[f[2]], d = c[f[3]];
            var n = (cc - a).Cross(d - b);
            var mid = (a + b + cc + d) * 0.25f;
            if (n.Dot(mid - center) < 0)
            {
                n = -n;
                (b, d) = (d, b);
            }
            n = n.Normalized();
            // Godot 正面是顺时针（从外面看）
            foreach (var p in new[] { a, d, cc, a, cc, b })
            {
                im.SurfaceSetNormal(n);
                im.SurfaceAddVertex(p);
            }
        }
        im.SurfaceEnd();
    }
}
