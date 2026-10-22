using System.Collections.Generic;
using Godot;

namespace PingPong;

/// <summary>场地：地板网格、球台（台面、白线、台腿）、球网、灯光，以及球、拖尾和接发位置提示圈。</summary>
public partial class Arena : Node3D
{
    public static readonly Color BG = Gfx.C(18, 22, 30);
    public static readonly Color BALL_COL = Gfx.C(255, 250, 235);

    MeshInstance3D _ball;
    StandardMaterial3D _ballMat;
    readonly ImmediateMesh _trail = new(), _cue = new();
    StandardMaterial3D _trailMat, _cueMat;

    const string FloorShader = @"
shader_type spatial;
uniform vec3 base_col : source_color;
uniform vec3 line_col : source_color;
varying vec3 wpos;
void vertex() { wpos = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz; }
void fragment() {
    vec2 g = abs(fract(wpos.xz + 0.5) - 0.5) / fwidth(wpos.xz);
    float l = 1.0 - clamp(min(g.x, g.y), 0.0, 1.0);
    ALBEDO = mix(base_col, line_col, l * 0.9);
    ROUGHNESS = 0.9;
}";

    public override void _Ready()
    {
        var env = new Godot.Environment
        {
            BackgroundMode = Godot.Environment.BGMode.Color,
            BackgroundColor = BG,
            AmbientLightSource = Godot.Environment.AmbientSource.Color,
            AmbientLightColor = Gfx.C(150, 160, 185),
            AmbientLightEnergy = 0.55f,
            TonemapMode = Godot.Environment.ToneMapper.Filmic,
            SsaoEnabled = true,
            SsaoRadius = 0.6f,
            SsaoIntensity = 1.2f,
        };
        AddChild(new WorldEnvironment { Environment = env });
        var sun = new DirectionalLight3D
        {
            ShadowEnabled = true,
            LightEnergy = 1.15f,
            DirectionalShadowMaxDistance = 14f,
            ShadowBlur = 1.5f,
        };
        AddChild(sun);
        sun.LookAtFromPosition(new Vector3(2.5f, 9f, 3.5f), Vector3.Zero, Vector3.Up);
        var fill = new OmniLight3D { Position = new Vector3(0, 4.5f, 0), OmniRange = 12f, LightEnergy = 0.5f };
        AddChild(fill);

        // 地板
        var sh = new Shader { Code = FloorShader };
        var fm = new ShaderMaterial { Shader = sh };
        fm.SetShaderParameter("base_col", Gfx.C(34, 40, 52));
        fm.SetShaderParameter("line_col", Gfx.C(44, 51, 66));
        AddChild(new MeshInstance3D { Mesh = new PlaneMesh { Size = new Vector2(10f, 7f) }, MaterialOverride = fm });

        BuildTable();
        BuildNet();

        _ballMat = new StandardMaterial3D { AlbedoColor = BALL_COL, Roughness = 0.4f, EmissionEnabled = true };
        _ball = new MeshInstance3D
        {
            Mesh = new SphereMesh { Radius = (float)(Sim.BALL_R * 1.4), Height = (float)(Sim.BALL_R * 2.8) },
            MaterialOverride = _ballMat,
        };
        AddChild(_ball);
        _trailMat = new StandardMaterial3D
        {
            ShadingMode = BaseMaterial3D.ShadingModeEnum.Unshaded, VertexColorUseAsAlbedo = true,
            Transparency = BaseMaterial3D.TransparencyEnum.Alpha, CullMode = BaseMaterial3D.CullModeEnum.Disabled,
        };
        AddChild(new MeshInstance3D { Mesh = _trail, CastShadow = GeometryInstance3D.ShadowCastingSetting.Off });
        _cueMat = new StandardMaterial3D
        {
            ShadingMode = BaseMaterial3D.ShadingModeEnum.Unshaded, VertexColorUseAsAlbedo = true,
            CullMode = BaseMaterial3D.CullModeEnum.Disabled,
        };
        AddChild(new MeshInstance3D { Mesh = _cue, CastShadow = GeometryInstance3D.ShadowCastingSetting.Off });
    }

    void BuildTable()
    {
        float L = (float)Sim.HALF_L, W = (float)Sim.HALF_W, Z = (float)Sim.TABLE_H;
        var top = Gfx.Mat(Gfx.C(24, 86, 140));
        var under = Gfx.Mat(Gfx.C(16, 52, 82));
        var white = Gfx.Mat(Gfx.C(235, 240, 245));
        var leg = Gfx.Mat(Gfx.C(60, 64, 72));
        Gfx.Box(this, new Vector3(2 * L, 0.025f, 2 * W), new Vector3(0, Z - 0.0125f, 0), top);
        Gfx.Box(this, new Vector3(2 * L - 0.1f, 0.03f, 2 * W - 0.1f), new Vector3(0, Z - 0.04f, 0), under);
        const float lw = 0.02f, lh = 0.002f;
        Gfx.Box(this, new Vector3(2 * L, lh, lw), new Vector3(0, Z + lh / 2, W - lw / 2), white);
        Gfx.Box(this, new Vector3(2 * L, lh, lw), new Vector3(0, Z + lh / 2, -W + lw / 2), white);
        Gfx.Box(this, new Vector3(lw, lh, 2 * W), new Vector3(L - lw / 2, Z + lh / 2, 0), white);
        Gfx.Box(this, new Vector3(lw, lh, 2 * W), new Vector3(-L + lw / 2, Z + lh / 2, 0), white);
        Gfx.Box(this, new Vector3(2 * L, lh, 0.003f), new Vector3(0, Z + lh / 2, 0), white);
        foreach (var sx in new[] { -1f, 1f })
            foreach (var sy in new[] { -1f, 1f })
            {
                var mi = Gfx.Cylinder(this, 0.03f, 0.03f, leg);
                Gfx.Span(mi, new Vector3(sx * (L - 0.25f), 0, sy * (W - 0.12f)),
                    new Vector3(sx * (L - 0.25f), Z - 0.05f, sy * (W - 0.12f)));
            }
    }

    void BuildNet()
    {
        float z0 = (float)Sim.TABLE_H, z1 = (float)(Sim.TABLE_H + Sim.NET_H), nw = (float)Sim.NET_HALF_W;
        var mesh = Gfx.Mat(Gfx.C(220, 225, 235, 80), false, true);
        var net = Gfx.Box(this, new Vector3(0.004f, z1 - z0, 2 * nw), new Vector3(0, (z0 + z1) / 2, 0), mesh);
        net.CastShadow = GeometryInstance3D.ShadowCastingSetting.Off;
        Gfx.Box(this, new Vector3(0.008f, 0.015f, 2 * nw), new Vector3(0, z1 - 0.0075f, 0), Gfx.Mat(Gfx.C(245, 245, 250)));
        foreach (var y in new[] { -nw, nw })
        {
            var p = Gfx.Cylinder(this, 0.012f, 0.012f, Gfx.Mat(Gfx.C(90, 95, 105)));
            Gfx.Span(p, new Vector3(0, z0 - 0.03f, y), new Vector3(0, z1, y));
        }
    }

    public void SetBall(V3 p, Color spin)
    {
        _ball.Position = p.G;
        _ballMat.Emission = spin;
        _ballMat.EmissionEnergyMultiplier = 0.25f;
    }

    /// <summary>拖尾：朝着镜头的带子，越新越宽越亮，颜色表示旋转。</summary>
    public void SetTrail(List<(V3 P, Color C)> pts, Vector3 camPos)
    {
        _trail.ClearSurfaces();
        if (pts.Count < 2) return;
        _trail.SurfaceBegin(Mesh.PrimitiveType.TriangleStrip, _trailMat);
        int n = pts.Count;
        for (int k = 0; k < n; k++)
        {
            float t = (k + 1) / (float)n;
            var p = pts[k].P.G;
            var q = pts[k == n - 1 ? k : k + 1].P.G;
            var r = pts[k == n - 1 ? k - 1 : k].P.G;
            var dir = (k == n - 1 ? p - r : q - p);
            var side = dir.Cross(camPos - p);
            side = side.LengthSquared() > 1e-12f ? side.Normalized() : Vector3.Up;
            float w = 0.002f + 0.009f * t;
            var c = pts[k].C;
            c.A = 0.15f + 0.75f * t;
            _trail.SurfaceSetColor(c);
            _trail.SurfaceAddVertex(p - side * w);
            _trail.SurfaceSetColor(c);
            _trail.SurfaceAddVertex(p + side * w);
        }
        _trail.SurfaceEnd();
    }

    /// <summary>接发球方站位提示：台面上一圈虚线。h &lt; 0 不显示。</summary>
    public void SetCue(int h, double y, Color col)
    {
        _cue.ClearSurfaces();
        if (h < 0) return;
        double m = h == 0 ? 1.0 : -1.0;
        var c = new V3(m * 0.9, m * y, Sim.TABLE_H + 0.003);
        _cue.SurfaceBegin(Mesh.PrimitiveType.Triangles, _cueMat);
        const double r0 = 0.11, r1 = 0.125;
        for (int k = 0; k < 360; k += 40)
        {
            for (int s = 0; s < 4; s++)
            {
                double a0 = Mathf.DegToRad(k + s * 5.5), a1 = Mathf.DegToRad(k + (s + 1) * 5.5);
                var p0 = c + new V3(System.Math.Cos(a0), System.Math.Sin(a0), 0) * r0;
                var p1 = c + new V3(System.Math.Cos(a0), System.Math.Sin(a0), 0) * r1;
                var p2 = c + new V3(System.Math.Cos(a1), System.Math.Sin(a1), 0) * r1;
                var p3 = c + new V3(System.Math.Cos(a1), System.Math.Sin(a1), 0) * r0;
                foreach (var p in new[] { p0, p1, p2, p0, p2, p3 })
                {
                    _cue.SurfaceSetColor(col);
                    _cue.SurfaceAddVertex(p.G);
                }
            }
        }
        _cue.SurfaceEnd();
    }
}
