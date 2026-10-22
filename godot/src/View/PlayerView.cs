using System.Collections.Generic;
using Godot;

namespace PingPong;

/// <summary>立体人物（play.py 里 draw_player 的移植）：四肢是锥台，关节和头是球，躯干、短裤、鞋是多面体，球拍是两面贴胶皮的拍板。</summary>
public partial class PlayerView : Node3D
{
    public static readonly Color SKIN = Gfx.C(236, 196, 162), HAIR = Gfx.C(45, 34, 28);
    public static readonly Color SHORTS = Gfx.C(38, 42, 54), SHOE = Gfx.C(236, 236, 240);
    public static readonly Color WOOD = Gfx.C(196, 160, 112), SOCK = Gfx.C(238, 238, 242);
    public static readonly Color RED_RUBBER = Gfx.C(190, 30, 35), BLACK_RUBBER = Gfx.C(25, 25, 30);

    readonly Color _col;
    readonly ImmediateMesh _torso = new(), _shorts = new(), _shoes = new();
    readonly Dictionary<string, MeshInstance3D> _p = new();
    MeshInstance3D _rubOut, _rubIn, _blade;

    public PlayerView(Color col)
    {
        _col = col;
    }

    public override void _Ready()
    {
        var skin = Gfx.Mat(SKIN);
        var shirt = Gfx.Mat(_col);
        var shorts = Gfx.Mat(SHORTS);
        AddChild(new MeshInstance3D { Mesh = _torso, MaterialOverride = Gfx.Mat(_col, false, true) });
        AddChild(new MeshInstance3D { Mesh = _shorts, MaterialOverride = Gfx.Mat(SHORTS, false, true) });
        AddChild(new MeshInstance3D { Mesh = _shoes, MaterialOverride = Gfx.Mat(SHOE, false, true) });

        void Cyl(string k, float r0, float r1, Material m) => _p[k] = Gfx.Cylinder(this, r0, r1, m);
        void Sph(string k, float r, Material m) => _p[k] = Gfx.Sphere(this, r, m);

        Cyl("chest", 0.07f, 0.07f, shirt);
        Sph("chestL", 0.07f, shirt);
        Sph("chestR", 0.07f, shirt);
        Cyl("number", 0.018f, 0.018f, Gfx.Mat(Gfx.Mix(_col, Colors.White, 0.55f)));
        foreach (var s in new[] { "l", "r" })
        {
            Cyl(s + "pants", 0.088f, 0.075f, shorts);
            Cyl(s + "thigh", 0.07f, 0.052f, skin);
            Sph(s + "knee", 0.052f, skin);
            Cyl(s + "shin", 0.05f, 0.04f, skin);
            Cyl(s + "sock", 0.04f, 0.036f, Gfx.Mat(SOCK));
            Cyl(s + "sleeve", 0.055f, 0.05f, shirt);
            Cyl(s + "upper", 0.045f, 0.04f, skin);
            Sph(s + "elbow", 0.042f, skin);
            Cyl(s + "fore", 0.04f, s == "l" ? 0.03f : 0.032f, skin);
            Sph(s + "hand", 0.036f, skin);
            Sph(s + "sh", 0.052f, shirt);
        }
        Cyl("neck", 0.045f, 0.042f, skin);
        Sph("head", 0.105f, skin);
        _p["hair"] = Gfx.Sphere(this, 0.112f, Gfx.Mat(HAIR), true);
        Sph("eyeL", 0.013f, Gfx.Mat(Gfx.C(30, 26, 24)));
        Sph("eyeR", 0.013f, Gfx.Mat(Gfx.C(30, 26, 24)));
        Cyl("handle", 0.016f, 0.014f, Gfx.Mat(WOOD));
        _blade = Gfx.Cylinder(this, 1f, 1f, Gfx.Mat(WOOD), 28);
        _rubOut = Gfx.Cylinder(this, 1f, 1f, Gfx.Mat(BLACK_RUBBER), 28);
        _rubIn = Gfx.Cylinder(this, 1f, 1f, Gfx.Mat(RED_RUBBER), 28);
    }

    static Vector3 U(Vector3 v) => v.Normalized();

    static Color RubberTint(Color c, int r)
    {
        if (r == Sim.RUB_ANTI) return c.Lerp(Gfx.C(90, 90, 90), 0.35f);
        if (r == Sim.RUB_LONG) return c.Lerp(Colors.White, 0.12f);
        return c;
    }

    /// <summary>b：骨架（模拟坐标）；face：拍面法向（模拟坐标）；pen：直板；fhOut：击球面是正手面；rubOut/rubIn：两面胶皮。</summary>
    public void UpdateBody(Body b, int j, V3 faceSim, bool pen, bool bhPen, bool fhOut, int rf, int rb)
    {
        float sj = j == 0 ? -1f : 1f;
        var up = Vector3.Up;
        var face = new V3(-sj, 0, 0).G;
        Vector3 lsh = b.LSh.G, rsh = b.RSh.G, lh = b.LHip.G, rh = b.RHip.G, neck = b.Neck.G, hip = b.Hip.G;
        var tUp = U(neck - hip);
        var across = U(rsh - lsh);
        var fw = U(tUp.Cross(across));
        if (fw.Dot(face) < 0) fw = -fw;
        var hipAcross = U(rh - lh);
        var hfw = U(up.Cross(hipAcross));
        if (hfw.Dot(face) < 0) hfw = -hfw;

        // 躯干：肩宽、胸厚，向下收到腰
        const float shIn = 0.035f;
        var lsT = lsh + across * shIn - tUp * 0.02f;
        var rsT = rsh - across * shIn - tUp * 0.02f;
        var wl = lh + tUp * 0.14f - hipAcross * 0.02f;
        var wr = rh + tUp * 0.14f + hipAcross * 0.02f;
        const float dt = 0.105f, db = 0.085f;
        _torso.ClearSurfaces();
        Gfx.Hexa(_torso, new[]
        {
            wl - hfw * db, wr - hfw * db, wr + hfw * db, wl + hfw * db,
            lsT - fw * dt, rsT - fw * dt, rsT + fw * dt, lsT + fw * dt,
        }, null);
        Gfx.Span(_p["chest"], lsT - tUp * 0.01f, rsT - tUp * 0.01f);
        _p["chestL"].Position = lsT - tUp * 0.01f;
        _p["chestR"].Position = rsT - tUp * 0.01f;
        var midT = (lsT + rsT) / 2 * 0.55f + (wl + wr) / 2 * 0.45f + fw * (dt * 0.95f);
        Gfx.Span(_p["number"], midT - across * 0.07f, midT + across * 0.07f);
        // 短裤：腰臀一块随骨盆走，裤腿跟着各自的大腿走
        const float lo = 0.03f;
        var ll = lh - hipAcross * 0.03f - tUp * lo;
        var rl = rh + hipAcross * 0.03f - tUp * lo;
        _shorts.ClearSurfaces();
        Gfx.Hexa(_shorts, new[]
        {
            ll - hfw * 0.09f, rl - hfw * 0.09f, rl + hfw * 0.09f, ll + hfw * 0.09f,
            wl - hfw * db, wr - hfw * db, wr + hfw * db, wl + hfw * db,
        }, null);

        _shoes.ClearSurfaces();
        foreach (var (s, hp, kn, ft) in new[] { ("l", b.LHip, b.LKnee, b.LFoot), ("r", b.RHip, b.RKnee, b.RFoot) })
        {
            Vector3 h = hp.G, knee = kn.G, foot = ft.G;
            var ankle = foot + up * 0.07f;
            var th0 = h + (knee - h) * 0.35f;
            var sk0 = ankle + (knee - ankle) * 0.25f;
            Gfx.Span(_p[s + "pants"], h + tUp * 0.04f, h + (knee - h) * 0.4f);
            Gfx.Span(_p[s + "thigh"], th0, knee);
            _p[s + "knee"].Position = knee;
            Gfx.Span(_p[s + "shin"], knee, sk0);
            Gfx.Span(_p[s + "sock"], sk0, ankle);
            var toe = foot + face * 0.13f + up * 0.035f;
            var heel = foot - face * 0.07f + up * 0.04f;
            var sw = new V3(0, 1, 0).G * 0.045f;
            Gfx.Hexa(_shoes, new[]
            {
                heel - sw - up * 0.035f, toe - sw * 0.9f - up * 0.03f, toe + sw * 0.9f - up * 0.03f,
                heel + sw - up * 0.035f, heel - sw + up * 0.04f, toe - sw * 0.8f + up * 0.01f,
                toe + sw * 0.8f + up * 0.01f, heel + sw + up * 0.04f,
            }, null);
        }

        var head = b.Head.G;
        Gfx.Span(_p["neck"], neck - tUp * 0.03f, head - tUp * 0.06f);
        _p["head"].Position = head;
        // 头发：半球盖在后脑和头顶；眼睛朝向身前，让朝向一眼看得出来
        Gfx.Orient(_p["hair"], head - face * 0.005f, U(up * 0.75f - face * 0.65f), Vector3.One);
        var side = U(up.Cross(face));
        _p["eyeL"].Position = head + face * 0.096f + up * 0.022f + side * 0.036f;
        _p["eyeR"].Position = head + face * 0.096f + up * 0.022f - side * 0.036f;
        _p["lsh"].Position = lsh;
        _p["rsh"].Position = rsh;

        void Arm(string s, Vector3 sh, Vector3 el, Vector3 hd)
        {
            Gfx.Span(_p[s + "sleeve"], sh, sh + (el - sh) * 0.5f);
            Gfx.Span(_p[s + "upper"], sh + (el - sh) * 0.45f, el);
            _p[s + "elbow"].Position = el;
            Gfx.Span(_p[s + "fore"], el, hd);
            _p[s + "hand"].Position = hd;
        }

        Arm("l", lsh, b.LElbow.G, b.LHand.G);

        // 持拍臂和球拍：拍面法向来自动画层（触球瞬间就是物理里的真实拍面），拍柄沿前臂方向
        var c = b.Paddle.G;
        var elbow = b.RElbow.G;
        var n = U(faceSim.G);
        var u = c - elbow;
        if (bhPen) u = u * 0.55f - up * 0.45f * u.Length();
        u -= n * u.Dot(n);
        if (u.Length() < 1e-3f) u = up - n * n.Y;
        u = U(u);
        var e2 = n.Cross(u);
        var grip = c - u * (pen ? 0.13f : 0.16f);
        Arm("r", rsh, elbow, grip);
        Gfx.Span(_p["handle"], grip, c - u * 0.07f);
        const float ru = 0.085f, re = 0.078f;
        _blade.Transform = new Transform3D(new Basis(u * ru, n * 0.008f, e2 * re), c);
        var (cOut, cIn) = fhOut ? (BLACK_RUBBER, RED_RUBBER) : (RED_RUBBER, BLACK_RUBBER);
        var (rOut, rIn) = fhOut ? (rf, rb) : (rb, rf);
        _rubOut.MaterialOverride = Gfx.Mat(RubberTint(cOut, rOut));
        _rubIn.MaterialOverride = Gfx.Mat(RubberTint(cIn, rIn));
        _rubOut.Transform = new Transform3D(new Basis(u * (ru * 0.97f), n * 0.004f, e2 * (re * 0.97f)), c + n * 0.006f);
        _rubIn.Transform = new Transform3D(new Basis(u * (ru * 0.97f), n * 0.004f, e2 * (re * 0.97f)), c - n * 0.006f);
    }
}
