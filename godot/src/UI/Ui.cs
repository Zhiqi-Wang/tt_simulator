using System;
using System.Collections.Generic;
using System.Linq;
using Godot;

namespace PingPong;

/// <summary>画面上的 2D 层（play.py 的 HUD、旋转小窗和右侧面板）：比分、发球/接发说明、上一拍信息、属性/风格/器材/练习页。
/// 左边视图区域的鼠标拖动、滚轮用来转镜头。</summary>
public partial class Ui : Control
{
    public const int FONT_S = 14, FONT = 17, FONT_L = 25, FONT_XL = 38;
    public static readonly Color TEXT = Gfx.C(225, 230, 240), DIM = Gfx.C(140, 148, 165);
    public static readonly Color PANEL_BG = Gfx.C(28, 32, 42), BOX_BG = Gfx.C(10, 12, 18), BOX_EDGE = Gfx.C(60, 66, 82);
    public static readonly Color TOP_COL = Gfx.C(255, 120, 50), BACK_COL = Gfx.C(70, 170, 255);
    public static readonly Color SIDE_COL = Gfx.C(120, 230, 110);
    const int ROW = 40, STAT_ROWS = 4, STATUS_EXTRA = 40;

    Main _g;
    Font _font;
    public readonly List<Widget> Widgets = new();
    public readonly List<UiSlider> StatSliders = new(), StyleSliders = new();
    readonly List<(string, float)> _pracLabels = new();
    readonly List<(string, float, Color, bool)> _equipLabels = new();
    readonly List<float> _blockY = new();
    float _helpY;
    Vector2? _dragCam;

    public void Init(Main g)
    {
        _g = g;
        _font = new SystemFont
        {
            FontNames = new[] { "Microsoft YaHei UI", "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "SimHei" },
            Antialiasing = TextServer.FontAntialiasing.Gray,
        };
        MouseFilter = MouseFilterEnum.Stop;
        SetAnchorsPreset(LayoutPreset.FullRect);
        BuildUi();
    }

    // ------------------------------------------------------------------ 绘图小工具
    public void Text(string s, Vector2 topLeft, int size, Color c) =>
        DrawString(_font, topLeft + new Vector2(0, _font.GetAscent(size)), s, HorizontalAlignment.Left, -1, size, c);

    public Vector2 Measure(string s, int size) =>
        new(_font.GetStringSize(s, HorizontalAlignment.Left, -1, size).X, _font.GetHeight(size));

    public void TextRight(string s, Vector2 topRight, int size, Color c) =>
        Text(s, topRight - new Vector2(Measure(s, size).X, 0), size, c);

    public void TextCenter(string s, Vector2 center, int size, Color c) => Text(s, center - Measure(s, size) / 2, size, c);

    public Rect2 TextRect(string s, Vector2 center, int size) =>
        new(center - Measure(s, size) / 2, Measure(s, size));

    readonly Dictionary<(Color, int, Color?), StyleBoxFlat> _sb = new();

    public void RoundRect(Rect2 r, Color c, int radius, Color? border = null)
    {
        if (!_sb.TryGetValue((c, radius, border), out var sb))
        {
            sb = new StyleBoxFlat { BgColor = c };
            sb.SetCornerRadiusAll(radius);
            if (border != null)
            {
                sb.BorderColor = border.Value;
                sb.SetBorderWidthAll(1);
            }
            _sb[(c, radius, border)] = sb;
        }
        DrawStyleBox(sb, r);
    }

    /// <summary>带深色底的一行文字（居中）。</summary>
    void Banner(string s, Vector2 center, int size, Color c, Vector2 pad, int radius)
    {
        var r = TextRect(s, center, size).Grow(0).GrowIndividual(pad.X / 2, pad.Y / 2, pad.X / 2, pad.Y / 2);
        RoundRect(r, BOX_BG, radius);
        TextCenter(s, center, size, c);
    }

    // ------------------------------------------------------------------ 面板
    bool IsShown(Widget w) => w.Pages == null || w.Pages.Contains(_g.Page);

    void Add(Widget w, params string[] pages)
    {
        w.Pages = pages.Length > 0 ? pages : null;
        Widgets.Add(w);
    }

    void BuildUi()
    {
        float x0 = Main.VIEW_W + 22, w = Main.PANEL_W - 44;
        Add(new UiButton(new Rect2(x0 + w - 120, 14, 120, 30),
            () => _g.Page switch
            {
                "stat" => "下一页：风格", "style" => "下一页：器材", "equip" => "下一页：练习", _ => "下一页：属性",
            }, _g.NextPage));
        BuildPracticeUi(x0, w);
        BuildEquipUi(x0, w);
        float y = 96;
        for (int j = 0; j < 2; j++)
        {
            int jj = j;
            _blockY.Add(y);
            const float bw = 96;
            Add(new UiButton(new Rect2(x0 + w - 2 * bw - 6, y - 2, bw, 28), "RL 模型", () => _g.SetMode(jj, "rl"),
                () => _g.Modes[jj] == "rl", Main.COLORS[j]), "stat", "style");
            Add(new UiButton(new Rect2(x0 + w - bw, y - 2, bw, 28), "脚本 AI", () => _g.SetMode(jj, "bot"),
                () => _g.Modes[jj] == "bot", Main.COLORS[j]), "stat", "style");
            y += 38;
            for (int k = 0; k < Sim.N_STYLE; k++)
            {
                int kk = k;
                var sl = new UiSlider(x0, y + k * ROW, w, Sim.STYLE_NAMES[k], -100, 100, 0, Main.COLORS[j],
                    v => _g.SetStat(jj, Sim.N_STATS + kk, v));
                Add(sl, "style");
                StyleSliders.Add(sl);
            }
            // 属性两列排：左列力量/敏捷/技术/发球，右列反应/体力/心态/适应
            float cw = (w - 20) / 2;
            for (int k = 0; k < Sim.STAT_ORDER.Length; k++)
            {
                int idx = Sim.STAT_ORDER[k];
                var sl = new UiSlider(x0 + k / STAT_ROWS * (cw + 20), y + k % STAT_ROWS * ROW, cw,
                    Sim.STAT_ORDER_NAMES[k], 0, 100, Main.DEFAULT_STATS, Main.COLORS[j], v => _g.SetStat(jj, idx, v));
                Add(sl, "stat");
                StatSliders.Add(sl);
            }
            // 疲劳、压力、手感三行状态条
            y += Math.Max(STAT_ROWS, Sim.N_STYLE) * ROW + 40 + STATUS_EXTRA;
        }
        Add(new UiSlider(x0, y, w, "时间倍速", 0.1, 3.0, 1.0, Gfx.C(230, 190, 90), v => _g.Speed = v, v => $"{v:F2}x"));
        y += 52;
        float b3 = (w - 20) / 3;
        Add(new UiButton(new Rect2(x0, y, b3, 34), () => _g.Paused ? "继续" : "暂停", () => _g.Paused = !_g.Paused));
        Add(new UiButton(new Rect2(x0 + b3 + 10, y, b3, 34), "重置比赛", _g.Restart));
        Add(new UiButton(new Rect2(x0 + 2 * (b3 + 10), y, b3, 34), () => _g.Sfx.On ? "静音" : "开声音",
            () => _g.Sfx.On = !_g.Sfx.On, () => !_g.Sfx.On));
        float bw2 = (w - 10) / 2;
        y += 44;
        Add(new UiButton(new Rect2(x0, y, bw2, 34), "随机属性", _g.Randomize));
        Add(new UiButton(new Rect2(x0 + bw2 + 10, y, bw2, 34), "重载模型", _g.Reload));
        _helpY = y + 38;
    }

    void BuildEquipUi(float x0, float w)
    {
        float y = 92;
        for (int j = 0; j < 2; j++)
        {
            int jj = j;
            _equipLabels.Add((Main.NAMES[j], y, Main.COLORS[j], true));
            y += 28;
            var rows = new (string, int, (string, double)[])[]
            {
                ("持拍手", Sim.HAND, new[] { ("右手", 1.0), ("左手", -1.0) }),
                ("握法", Sim.GRIP, Sim.GRIP_NAMES.Select((s, k) => (s, (double)k)).ToArray()),
                ("正手胶皮", Sim.RFH, Sim.RUB_NAMES.Select((s, k) => (s, (double)k)).ToArray()),
                ("反手胶皮", Sim.RBH, Sim.RUB_NAMES.Select((s, k) => (s, (double)k)).ToArray()),
            };
            foreach (var (label, key, opts) in rows)
            {
                _equipLabels.Add((label, y, DIM, false));
                float bw = (w - 76 - 4 * (opts.Length - 1)) / opts.Length;
                for (int n = 0; n < opts.Length; n++)
                {
                    var (text, val) = opts[n];
                    Add(new UiButton(new Rect2(x0 + 76 + n * (bw + 4), y - 3, bw, 26), text,
                        () => _g.SetEquip(jj, key, val),
                        () => key != Sim.HAND
                            ? Math.Abs(_g.Env.Stat(jj, key) - val) < 0.5
                            : (_g.Env.Stat(jj, key) < 0) == (val < 0), Main.COLORS[j]), "equip");
                }
                y += 32;
            }
            y += 12;
        }
        Add(new UiButton(new Rect2(x0, y, w, 30), "随机器材（双方）", _g.RandomEquip), "equip");
        y += 42;
        foreach (var s in new[]
                 {
                     "反胶：旋转最强，也最吃对方的转", "正胶：出球快、不太吃转；生胶：更平更沉", "长胶：几乎不造转，来球上旋回成下旋",
                     "防弧：出球慢，能把来球旋转卸掉", "直板：发球、台内小球细腻，推挡稳；", "　　　反手横打发力弱、覆盖面小",
                     "左手：动作整套镜像，侧旋方向相反", "换持拍手/握法会重开这一分",
                 })
        {
            _equipLabels.Add((s, y, DIM, false));
            y += 20;
        }
    }

    void BuildPracticeUi(float x0, float w)
    {
        float y = 96;

        void Row(string label, string key, (string, int?)[] options, int ncol = 0)
        {
            _pracLabels.Add((label, y));
            y += 24;
            if (ncol == 0) ncol = options.Length;
            float bw = (w - 6 * (ncol - 1)) / ncol;
            for (int n = 0; n < options.Length; n++)
            {
                var (text, val) = options[n];
                int r = n / ncol, c = n % ncol;
                Add(new UiButton(new Rect2(x0 + c * (bw + 6), y + r * 34, bw, 28), text, () => _g.SetPrac(key, val),
                    () => _g.Prac[key] == val), "practice");
            }
            y += 34 * ((options.Length + ncol - 1) / ncol) + 8;
        }

        _pracLabels.Add(("练习模式（不计分、无疲劳、对手不熟悉发球）", y));
        y += 24;
        float bw2 = (w - 6) / 2;
        Add(new UiButton(new Rect2(x0, y, bw2, 28), "开启", () => _g.SetPractice(true), () => _g.Practice), "practice");
        Add(new UiButton(new Rect2(x0 + bw2 + 6, y, bw2, 28), "关闭（回到比赛）", () => _g.SetPractice(false),
            () => !_g.Practice), "practice");
        y += 42;
        Row("发球方", "server", new (string, int?)[] { ("蓝方", 0), ("红方", 1) });
        var shortNames = new[] { "正手侧旋", "勾手", "逆旋转", "下蹲砍式", "砍式", "正手奔球", "反手侧旋", "反手急球" };
        Row("发球方式", "kind", shortNames.Select((s, k) => (s, (int?)k)).ToArray(), 4);
        Row("抛球", "toss", new (string, int?)[] { ("AI 决定", null), ("低抛", 0), ("中抛", 1), ("高抛", 2) });
        Row("旋转（侧旋手法选“不转”为纯侧旋）", "var",
            new (string, int?)[] { ("AI 决定", null), ("侧/下旋", 0), ("不转", 1), ("侧/上旋", 2) });
        Row("落点", "depth", new (string, int?)[] { ("AI 决定", null), ("短球", 0), ("半出台", 1), ("长球", 2) });
        Add(new UiButton(new Rect2(x0, y + 6, w, 44),
            () => _g.Practice
                ? (_g.Env.Phase == Sim.PH_SERVE && !_g.ServeGo ? "发球！" : "这一分进行中…")
                : "开启练习并发球",
            _g.PressServe, () => _g.Practice && _g.Env.Phase == Sim.PH_SERVE && !_g.ServeGo, Gfx.C(90, 170, 110)),
            "practice");
        _pracLabels.Add(("奔球、急球固定低抛；砍式发不出强上旋", y + 54));
        y += 78;
        var rally = new List<(string, int?)> { ("AI 决定", null) };
        for (int n = 0; n < Main.PRAC_TECH.Length; n++)
            if (n != Main.PRAC_WIDE)
                rally.Add((Main.PRAC_TECH[n].Name, n));
        Row("相持技术（接发方用，发球方陪练）", "rally", rally.ToArray(), 5);
    }

    // ------------------------------------------------------------------ 输入
    public override void _GuiInput(InputEvent e)
    {
        foreach (var wdg in Widgets)
            if (IsShown(wdg) && wdg.Handle(e))
            {
                AcceptEvent();
                return;
            }
        var cam = _g.Cam;
        if (e is InputEventMouseButton mb)
        {
            if (mb.ButtonIndex == MouseButton.Left)
                _dragCam = mb.Pressed && mb.Position.X < Main.VIEW_W ? mb.Position : null;
            else if (mb.Pressed && mb.Position.X < Main.VIEW_W &&
                     (mb.ButtonIndex == MouseButton.WheelUp || mb.ButtonIndex == MouseButton.WheelDown))
            {
                cam.Dist *= Math.Pow(0.9, mb.ButtonIndex == MouseButton.WheelUp ? 1 : -1);
                cam.Update();
            }
        }
        else if (e is InputEventMouseMotion mm && _dragCam != null)
        {
            var d = mm.Position - _dragCam.Value;
            cam.Az -= d.X * 0.006;
            cam.El += d.Y * 0.004;
            cam.Update();
            _dragCam = mm.Position;
        }
    }

    public override void _Process(double delta) => QueueRedraw();

    // ------------------------------------------------------------------ 绘制
    public override void _Draw()
    {
        if (_g?.Env == null) return;
        DrawRescueLabels();
        DrawHud();
        DrawSpinInset();
        DrawPanel();
    }

    void DrawRescueLabels()
    {
        if (_g.Poses == null || _g.Bodies == null) return;
        for (int j = 0; j < 2; j++)
        {
            var pose = _g.Poses[j];
            var b = _g.Bodies[j];
            if (pose == null || b == null) continue;
            var (name, t0) = pose.Gait;
            double age = pose.T - t0;
            if ((name == "救球" || name == "倒地爆冲") && 0.0 <= age && age < 0.8)
            {
                var mid = (b.LFoot + b.RFoot) / 2;
                var p = _g.Cam.Screen(new V3(mid.X, mid.Y, 0.0));
                Banner(name + "！", p + new Vector2(0, 16), FONT, Gfx.C(255, 220, 120), new Vector2(14, 6), 6);
            }
        }
    }

    void DrawHud()
    {
        var I = _g.Env.I;
        var T = _g.Env.T;
        if (_g.FakeShown is int fh && _g.Env.Phase == Sim.PH_DEAD)
            Banner($"{Main.NAMES[fh]} 假动作得手！骗到重心", new Vector2(Main.VIEW_W / 2f, 170), FONT_L, Main.COLORS[fh],
                new Vector2(26, 12), 10);
        var box = new Rect2(Main.VIEW_W / 2f - 220, 14, 440, 74);
        RoundRect(box, BOX_BG, 12, BOX_EDGE);
        float cx = box.GetCenter().X;
        TextCenter(":", new Vector2(cx, box.Position.Y + 30), FONT_XL, TEXT);
        TextRight(I[Sim.SC0].ToString(), new Vector2(cx - 22, box.Position.Y + 30 - Measure("0", FONT_XL).Y / 2), FONT_XL,
            Main.COLORS[0]);
        Text(I[Sim.SC1].ToString(), new Vector2(cx + 22, box.Position.Y + 30 - Measure("0", FONT_XL).Y / 2), FONT_XL,
            Main.COLORS[1]);
        for (int j = 0; j < 2; j++)
        {
            var sz = Measure(Main.NAMES[j], FONT);
            float ty = box.Position.Y + 30 - sz.Y / 2;
            float tx = j == 0 ? box.Position.X + 18 : box.End.X - 18 - sz.X;
            Text(Main.NAMES[j], new Vector2(tx, ty), FONT, Main.COLORS[j]);
            if (I[Sim.SERVER] == j)
            {
                float dx = j == 0 ? tx + sz.X + 10 : tx - 10;
                DrawCircle(new Vector2(dx, box.Position.Y + 30), 5, Gfx.C(255, 230, 120));
            }
        }
        string g = _g.Practice
            ? "练习模式 · 不计分 · 在右侧“练习”页选择发球后点“发球”"
            : $"局分 {I[Sim.G0]} : {I[Sim.G1]}   ·   场分 {I[Sim.M0]} : {I[Sim.M1]}   ·   {Sim.GAME_PTS} 分制 {2 * Sim.MATCH_WIN - 1} 局 {Sim.MATCH_WIN} 胜";
        TextCenter(g, new Vector2(cx, box.Position.Y + 60), FONT_S, _g.Practice ? Gfx.C(140, 210, 150) : DIM);
        if (_g.Env.Phase == Sim.PH_DEAD && _g.Msg != "")
            Banner(_g.Msg, new Vector2(Main.VIEW_W / 2f, 160), FONT_L, _g.MsgColor, new Vector2(30, 14), 10);

        var bv = _g.Env.BallV;
        var bw = _g.Env.BallW;
        double speed = bv.Norm() * 3.6;
        double rpm = bw.Norm() * 60 / (2 * Math.PI);
        if (_g.Env.Phase != Sim.PH_SERVE)
        {
            int h = (int)I[Sim.LHIT];
            string kind;
            if (I[Sim.HITS] == 0)
            {
                double toss = T[Sim.TOSS];
                kind = Sim.SERVE_NAMES[Math.Max((int)I[Sim.LSRV], 0)] +
                       $"（{Txt.ServeSpin(bv, bw)}·{Txt.Toss(toss)}" + (I[Sim.LHID] != 0 ? "·回抛" : "") +
                       $"·对手熟悉度 {T[Sim.SFAM] * 100:F0}%）";
            }
            else
            {
                kind = Txt.Tech((int)I[Sim.LTECH], I[Sim.LSIDE] != 0, (int)I[Sim.LSTK]);
                if (I[Sim.HITS] == 1) kind = "接发球 · " + kind;
            }
            var last = $"上一拍：{Main.NAMES[h]} {kind} · 挥拍 {T[Sim.LSW],4:F1} m/s";
            if (I[Sim.HITS] > 0) last += $" · 身体发力 {T[Sim.LBODY] * 100,3:F0}%";
            Text(last, new Vector2(16, Main.H - 74), FONT_S, Main.COLORS[h]);
        }
        // 左上角：这一分的发球（抛球 · 手法 · 旋转）和接发球技术
        var info = new List<(int, string)>();
        if (_g.Env.Phase == Sim.PH_SERVE)
        {
            int srv = (int)I[Sim.SERVER];
            int k = (int)T[Sim.SPLAN0 + srv];
            double tz = T[Sim.TOSS];
            var wait = _g.Practice && !_g.ServeGo ? "等待点击发球" : "准备中";
            info.Add((srv, $"发球  {(tz >= 0 ? Txt.Toss(tz) : wait)} · {Sim.SERVE_NAMES[k]}"));
        }
        else if (_g.ServeInfo != null)
        {
            info.Add((_g.ServeInfo.Value.Item1, "发球  " + _g.ServeInfo.Value.Item2));
            if (_g.RecvInfo != null) info.Add((_g.RecvInfo.Value.Item1, "接发  " + _g.RecvInfo.Value.Item2));
        }
        for (int n = 0; n < info.Count; n++)
        {
            var (j, s) = info[n];
            var str = $"{Main.NAMES[j]} {s}";
            var sz = Measure(str, FONT);
            var r = new Rect2(16 - 9, 100 + 36 * n - 5, sz.X + 18, sz.Y + 10);
            RoundRect(r, BOX_BG, 8);
            Text(str, new Vector2(16, 100 + 36 * n), FONT, Main.COLORS[j]);
        }
        string ModeName(int j) => _g.Modes[j] == "rl" && _g.Pol != null ? "RL 模型" : "脚本 AI";
        var lines = new[]
        {
            $"本回合击球 {I[Sim.HITS]} 拍    球速 {speed,5:F1} km/h    转速 {rpm,5:F0} rpm",
            $"蓝方：{ModeName(0)}    红方：{ModeName(1)}    倍速 {_g.Speed:F2}x{(_g.Paused ? "    [已暂停]" : "")}",
        };
        for (int k = 0; k < lines.Length; k++) Text(lines[k], new Vector2(16, Main.H - 52 + k * 22), FONT_S, DIM);
    }

    /// <summary>放大的球：两条正交的“缝线”随旋转转动（画面转速放慢），箭头为旋转轴（右手定则）。</summary>
    void DrawSpinInset()
    {
        float cx = Main.VIEW_W - 100, cy = Main.H - 150, r = 46;
        var box = new Rect2(cx - 90, cy - 70, 180, 175);
        RoundRect(box, BOX_BG, 10, BOX_EDGE);
        var v = _g.Env.BallV;
        var w = _g.Env.BallW;
        var cam = _g.Cam;

        (Vector2, double) Proj(V3 p) =>
            (new Vector2(cx + r * (float)p.Dot(cam.Right), cy - r * (float)p.Dot(cam.Up)), p.Dot(cam.Fwd));

        DrawCircle(new Vector2(cx, cy), r, Gfx.C(238, 234, 222));
        var rot = _g.BallRot;
        for (int c = 0; c < 2; c++)
        {
            var col = c == 0 ? Gfx.C(230, 120, 40) : Gfx.C(60, 60, 70);
            (Vector2, double)? prev = null;
            for (int k = 0; k <= 48; k++)
            {
                double t = 2 * Math.PI * k / 48;
                var p = c == 0 ? new V3(Math.Cos(t), Math.Sin(t), 0) : new V3(0, Math.Cos(t), Math.Sin(t));
                var q = Txt.Mul(rot, p);
                var pr = Proj(q);
                if (prev != null && prev.Value.Item2 < 0 && pr.Item2 < 0) DrawLine(prev.Value.Item1, pr.Item1, col, 3);
                prev = pr;
            }
        }
        DrawArc(new Vector2(cx, cy), r, 0, Mathf.Tau, 48, Gfx.C(120, 125, 135), 1);
        double wn = w.Norm();
        if (wn > 5 && _g.Env.Phase != Sim.PH_SERVE)
        {
            var col = Txt.SpinColor(v, w);
            var ax = w / wn;
            double ln = 0.9 + 0.3 * Math.Min(wn / 500.0, 1.0);
            var a0 = Proj(-ax * ln).Item1;
            var a1 = Proj(ax * ln).Item1;
            DrawLine(a0, a1, col, 3);
            var d = a1 - new Vector2(cx, cy);
            var u = d / (d.Length() + 1e-9f);
            var nrm = new Vector2(-u.Y, u.X);
            DrawColoredPolygon(new[] { a1 + u * 8, a1 + nrm * 6, a1 - nrm * 6 }, col);
            var (top, side) = Txt.SpinParts(v, w);
            double rpmk = 60 / (2 * Math.PI);
            Text($"{(top >= 0 ? "上旋" : "下旋")} {Math.Abs(top) * rpmk,4:F0} rpm", new Vector2(box.Position.X + 12, cy + r + 6),
                FONT_S, top >= 0 ? TOP_COL : BACK_COL);
            Text($"侧旋 {Math.Abs(side) * rpmk,4:F0} rpm", new Vector2(box.Position.X + 12, cy + r + 26), FONT_S, SIDE_COL);
        }
        else Text("无旋转", new Vector2(box.Position.X + 12, cy + r + 6), FONT_S, DIM);
        Text($"球的旋转（慢放 {(int)(1 / Main.VIS_SPIN)}×）", new Vector2(box.Position.X + 12, box.Position.Y + 4), FONT_S, DIM);
    }

    void Bar(float x, float y, float w, float frac, Color c)
    {
        RoundRect(new Rect2(x, y, w, 8), Gfx.C(55, 60, 75), 4);
        if (frac > 0) RoundRect(new Rect2(x, y, w * Math.Clamp(frac, 0, 1), 8), c, 4);
    }

    void DrawPanel()
    {
        float x0 = Main.VIEW_W;
        DrawRect(new Rect2(x0, 0, Main.PANEL_W, Main.H), PANEL_BG);
        Text("乒乓模拟器", new Vector2(x0 + 22, 16), FONT_L, TEXT);
        Text(_g.ModelMsg, new Vector2(x0 + 22, 56), FONT_S, _g.Pol != null ? Gfx.C(140, 210, 150) : Gfx.C(230, 160, 110));
        if (_g.Page == "practice")
            foreach (var (s, y) in _pracLabels)
                Text(s, new Vector2(x0 + 22, y), FONT_S, DIM);
        if (_g.Page == "equip")
            foreach (var (s, y, c, big) in _equipLabels)
            {
                if (big) RoundRect(new Rect2(x0 + 22, y, 6, 22), c, 3);
                Text(s, new Vector2(x0 + (big ? 36 : 22), y), big ? FONT : FONT_S, big ? TEXT : c);
            }
        if (_g.Page == "stat" || _g.Page == "style")
            for (int j = 0; j < 2; j++)
            {
                float y = _blockY[j];
                RoundRect(new Rect2(x0 + 22, y, 6, 24), Main.COLORS[j], 3);
                Text(Main.NAMES[j], new Vector2(x0 + 36, y), FONT, TEXT);
                float fy = y + 38 + ROW * Math.Max(STAT_ROWS, Sim.N_STYLE) + 2;
                double fat = _g.Env.F[j];
                float bw = Main.PANEL_W - 132;
                Text($"疲劳 {fat * 100,4:F0}%", new Vector2(x0 + 22, fy), FONT_S, DIM);
                Bar(x0 + 110, fy + 6, bw, (float)fat, Gfx.C(230, 150, 80));
                // 比赛中的心理状态：压力（局点/赛点 + 连丢分 + 被怪胶打烦）× 心态越低越受影响；手感：对对手怪胶的适应程度
                double men = _g.Env.Stat(j, Sim.MEN);
                double p = _g.Env.Pressure(j);
                double adp = _g.Env.T[Sim.ADP0 + j];
                double eff = Sim.MEN_SIG * p * (1.0 - men);
                float y2 = fy + 20;
                Text($"压力 +{eff * 100,3:F0}%", new Vector2(x0 + 22, y2), FONT_S, eff > 0.08 ? Gfx.C(225, 140, 120) : DIM);
                Bar(x0 + 110, y2 + 6, bw, (float)Math.Min(p / 1.5, 1.0), Gfx.C(225, 110, 110));
                float y3 = y2 + 20;
                Text($"手感 {adp * 100,4:F0}%", new Vector2(x0 + 22, y3), FONT_S, DIM);
                Bar(x0 + 110, y3 + 6, bw, (float)(adp / Sim.ADAPT_MAX), Gfx.C(110, 180, 225));
            }
        foreach (var wdg in Widgets)
            if (IsShown(wdg))
                wdg.Draw(this);
        var helps = new[] { "拖动转视角 · 滚轮缩放 · 空格暂停 · C 复位 · M 静音", "拍面黑=正手面 红=反手面 · 轨迹颜色=旋转" };
        for (int k = 0; k < helps.Length; k++)
        {
            float y = _helpY + k * 16;
            if (y + 16 <= Main.H) Text(helps[k], new Vector2(x0 + 22, y), FONT_S, DIM);
        }
    }
}

/// <summary>显示用的文字和颜色规则（与 Python 版相同）。</summary>
public static class Txt
{
    public static readonly Color BALL_COL = Gfx.C(255, 250, 235);

    /// <summary>旋转分解为相对飞行方向的上旋分量（负为下旋）和侧旋分量，单位 rad/s。</summary>
    public static (double Top, double Side) SpinParts(V3 v, V3 w)
    {
        double hn = Math.Sqrt(v.X * v.X + v.Y * v.Y) + 1e-9;
        return ((w.X * -v.Y + w.Y * v.X) / hn, w.Z);
    }

    public static Color SpinColor(V3 v, V3 w)
    {
        var (top, side) = SpinParts(v, w);
        float mag = (float)Math.Min(Math.Sqrt(top * top + side * side) / 400.0, 1.0);
        var tc = top >= 0 ? Ui.TOP_COL : Ui.BACK_COL;
        float f = (float)(Math.Abs(top) / (Math.Abs(top) + Math.Abs(side) + 1e-9));
        return BALL_COL.Lerp(Ui.SIDE_COL.Lerp(tc, f), mag);
    }

    public static string ServeSpin(V3 v, V3 w)
    {
        var (top, side) = SpinParts(v, w);
        string ud = top > 40 ? "上旋" : top < -40 ? "下旋" : "";
        if (Math.Abs(side) > 40) return "侧" + (ud == "" ? "旋" : ud);
        return ud == "" ? "不转" : ud;
    }

    public static string Toss(double toss) => toss > 0.66 ? "高抛" : toss > 0.33 ? "中抛" : "低抛";

    public static string Tech(int tech, bool bh, int stroke)
    {
        if (tech < 0) return (bh ? "反手" : "正手") + Sim.STROKE_NAMES[Math.Max(stroke, 0)];
        var name = Sim.TECH_NAMES[tech];
        return Sim.TechNoSide(tech) ? name : (bh ? "反手" : "正手") + name;
    }

    /// <summary>3×3 旋转矩阵（行优先）乘向量。</summary>
    public static V3 Mul(double[] m, V3 p) =>
        new(m[0] * p.X + m[1] * p.Y + m[2] * p.Z, m[3] * p.X + m[4] * p.Y + m[5] * p.Z,
            m[6] * p.X + m[7] * p.Y + m[8] * p.Z);

    /// <summary>绕单位轴转 ang 的旋转矩阵左乘 m（Rodrigues）。</summary>
    public static double[] Rotate(double[] m, V3 axis, double ang)
    {
        double x = axis.X, y = axis.Y, z = axis.Z, c = Math.Cos(ang), s = Math.Sin(ang), t = 1 - c;
        double[] r =
        {
            c + x * x * t, x * y * t - z * s, x * z * t + y * s,
            y * x * t + z * s, c + y * y * t, y * z * t - x * s,
            z * x * t - y * s, z * y * t + x * s, c + z * z * t,
        };
        var o = new double[9];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                o[3 * i + j] = r[3 * i] * m[j] + r[3 * i + 1] * m[3 + j] + r[3 * i + 2] * m[6 + j];
        return o;
    }
}
