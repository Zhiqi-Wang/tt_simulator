using System;
using System.Collections.Generic;
using System.Linq;
using Godot;

namespace PingPong;

/// <summary>乒乓模拟器（play.py 的 Godot 版）：模拟按 1/30 秒定步推进（物理在 C++ 核心里），
/// 双方由 RL 模型或脚本 AI 控制；画面层按子步插值显示，人物动作由 Animator + Skeleton 生成。</summary>
public partial class Main : Node
{
    public const int W = 1440, H = 900, PANEL_W = 380, VIEW_W = W - PANEL_W;
    public const int DEFAULT_STATS = 60;
    public const double VIS_SPIN = 0.1;
    const double SERVE_SETTLE = 0.3;
    public static readonly string[] NAMES = { "蓝方", "红方" };
    public static readonly Color[] COLORS = { Gfx.C(70, 150, 255), Gfx.C(255, 90, 90) };
    public static readonly int[] PUSHBLOCK_TECH = { Sim.TQ_BLOCK, Sim.TQ_SHORT, Sim.TQ_LONG, Sim.TQ_PUSH, Sim.TQ_SIDECUT };
    // 练习页强制的相持技术：(名称, 力度动作 a5, 旋转动作 a6, 离台目标距离 m / null 不管站位, 陪练方用的技术)
    public static readonly (string Name, double A5, double A6, double? Dist, int Partner)[] PRAC_TECH =
    {
        ("拉球", 0.3, 0.85, 0.9, 0), ("快攻", 0.2, 0.15, 0.4, 1), ("削球", -0.2, -0.75, 2.0, 0),
        ("搓球", -0.6, -0.6, 0.15, 3), ("挡球", -0.6, 0.1, 0.2, 0), ("放高球", -0.3, 0.45, 2.6, 6),
        ("扣杀", 0.9, 0.1, 0.5, 5), ("救球爆冲", 0.6, 0.85, 0.9, 7), ("大角调动", 0.3, 0.85, 0.9, 0),
    };
    // 陪练调动时落点横向：左右两大角交替
    public const int PRAC_WIDE = 7;

    public World Env;
    public Policy Pol;
    public string ModelMsg = "";
    public string[] Modes = { "bot", "bot" };
    public bool Paused;
    public double Speed = 1.0;
    public string Page = "stat";
    public bool Practice, ServeGo;
    public readonly Dictionary<string, int?> Prac = new()
    {
        ["server"] = 0, ["kind"] = Sim.SV_HOOK, ["toss"] = null, ["var"] = null, ["depth"] = null, ["rally"] = null,
    };
    public string Msg = "";
    public Color MsgColor = Ui.TEXT;
    public (int, string)? ServeInfo, RecvInfo;
    public int? FakeShown;
    public Pose[] Poses;
    public Body[] Bodies = new Body[2];
    public double[] BallRot = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
    public CameraRig Cam;
    public Sfx Sfx;

    readonly Animator _anim = new();
    readonly Skeleton _skel = new();
    readonly Random _rng = new();
    double[] _obs;
    readonly double[] _act = new double[2 * Sim.ACT_DIM];
    double _acc, _settle;
    readonly List<(V3, Color)> _trail = new();
    readonly V3?[] _ballTrail = new V3?[3];
    readonly (V3 Ob, V3 Oh, double T, double Dur)?[] _glide = new (V3, V3, double, double)?[2];
    readonly (V3 Body, V3 Hand)?[] _glidePrev = new (V3, V3)?[2], _glideDisp = new (V3, V3)?[2];
    readonly V3?[] _handShown = new V3?[2];
    (int, int)? _fakePending;
    int _pointRally;
    bool _pointWow;
    double[] _foot = new double[7];
    readonly double[] _blends = new double[2];

    Arena _arena;
    PlayerView[] _pv;
    Ui _ui;
    string _modelPath = "res://data/policy_v41.bin";

    public override void _Ready()
    {
        DisplayServer.WindowSetTitle("乒乓模拟器");
        var args = OS.GetCmdlineUserArgs();
        for (int k = 0; k + 1 < args.Length; k++)
        {
            if (args[k] == "--model") _modelPath = args[k + 1];
            if (args[k] == "--speed") Speed = double.Parse(args[k + 1]);
            // 调试：--cam 方位角度,仰角度,距离
            if (args[k] == "--cam") _camArg = args[k + 1].Split(',').Select(double.Parse).ToArray();
            // 调试：--shot 文件.png 秒数[,秒数...]：到时间截图，最后一张截完退出
            if (args[k] == "--shot")
            {
                _shotPath = args[k + 1];
                _shotTimes = k + 2 < args.Length
                    ? args[k + 2].Split(',').Select(double.Parse).ToList()
                    : new List<double> { 6.0 };
            }
        }

        // 3D 视图在左边，右边是面板
        var vc = new SubViewportContainer { Stretch = true, Position = Vector2.Zero, Size = new Vector2(VIEW_W, H) };
        AddChild(vc);
        var vp = new SubViewport
        {
            Size = new Vector2I(VIEW_W, H), Msaa3D = Viewport.Msaa.Msaa4X, AudioListenerEnable3D = true,
            HandleInputLocally = false,
        };
        vc.AddChild(vp);
        _arena = new Arena();
        vp.AddChild(_arena);
        _pv = new[] { new PlayerView(COLORS[0]), new PlayerView(COLORS[1]) };
        foreach (var p in _pv) _arena.AddChild(p);
        Sfx = new Sfx();
        _arena.AddChild(Sfx);
        var cam = new Camera3D { Current = true };
        _arena.AddChild(cam);
        Cam = new CameraRig(cam);
        if (_camArg != null)
        {
            Cam.Az = Mathf.DegToRad(_camArg[0]);
            Cam.El = Mathf.DegToRad(_camArg[1]);
            Cam.Dist = _camArg[2];
            Cam.Update();
        }

        Env = new World((ulong)(DateTime.Now.Ticks % 100000));
        for (int j = 0; j < 2; j++)
            foreach (var idx in Sim.STAT_ORDER)
                Env.SetStat(j, idx, DEFAULT_STATS / 100.0);
        _obs = Env.Reset();
        LoadModel();
        Modes[0] = Modes[1] = Pol != null ? "rl" : "bot";

        _ui = new Ui();
        AddChild(_ui);
        _ui.Init(this);
        for (int k = 0; k + 1 < args.Length; k++)
            if (args[k] == "--animtest")
            {
                AnimTest(args[k + 1]);
                GetTree().Quit();
            }
    }

    // ------------------------------------------------------------------ 模型与设置
    void LoadModel()
    {
        try
        {
            if (!FileAccess.FileExists(_modelPath))
            {
                ModelMsg = "未找到模型（data/policy_*.bin）";
                Pol = null;
                return;
            }
            var pol = Policy.Load(FileAccess.GetFileAsBytes(_modelPath));
            if (pol.ObsDim > Sim.OBS_DIM || pol.ActDim != Sim.ACT_DIM)
            {
                ModelMsg = "模型版本不匹配，需要重新导出";
                Pol = null;
                return;
            }
            Pol = pol;
            ModelMsg = $"已加载 RL 模型（训练 {pol.Iteration} 轮）" + (pol.ObsDim < Sim.OBS_DIM ? "·旧版模型" : "");
        }
        catch (Exception ex)
        {
            ModelMsg = $"模型加载失败：{ex.Message}";
            Pol = null;
        }
    }

    public void Reload()
    {
        LoadModel();
        if (Pol == null) Modes[0] = Modes[1] = "bot";
    }

    public void SetMode(int j, string mode) => Modes[j] = mode;
    public void SetStat(int j, int k, double v) => Env.SetStat(j, k, v / 100.0);

    public void NextPage()
    {
        var pages = new[] { "stat", "style", "equip", "practice" };
        Page = pages[(Array.IndexOf(pages, Page) + 1) % pages.Length];
    }

    /// <summary>改器材。换持拍手、握法从下一分开始生效（这一分站位、动作已经按原来的手在做）。</summary>
    public void SetEquip(int j, int key, double val)
    {
        Env.SetStat(j, key, val);
        if (key == Sim.HAND || key == Sim.GRIP) Restart();
    }

    public string EquipText(int j) =>
        $"{(Env.Stat(j, Sim.HAND) < 0 ? "左手" : "右手")}{Sim.GRIP_NAMES[Env.Stat(j, Sim.GRIP) > 0.5 ? 1 : 0]} · " +
        $"正手{Sim.RUB_NAMES[(int)Env.Stat(j, Sim.RFH)]} 反手{Sim.RUB_NAMES[(int)Env.Stat(j, Sim.RBH)]}";

    public void RandomEquip()
    {
        for (int j = 0; j < 2; j++) Env.RandomEquipment(j);
        Restart();
    }

    public void Randomize()
    {
        foreach (var sl in _ui.StatSliders) sl.SetValue(_rng.Next(10, 101));
        foreach (var sl in _ui.StyleSliders) sl.SetValue(_rng.Next(-100, 101));
    }

    public void SetPractice(bool on)
    {
        if (on == Practice) return;
        Practice = on;
        ServeGo = false;
        if (on)
        {
            Env.F[0] = Env.F[1] = 0.0;
            NewPracticePoint();
        }
        else
        {
            _anim.RescueTh = (0.72, 0.6, 9.0);
            Restart();
        }
    }

    void NewPracticePoint()
    {
        _obs = Env.RestartPoint(Prac["server"] ?? 0);
        _trail.Clear();
        _anim.Reset();
        ServeGo = false;
        ServeInfo = RecvInfo = null;
        Msg = "";
    }

    public void SetPrac(string key, int? value)
    {
        Prac[key] = value;
        if (key == "server" && Practice && Env.Phase == Sim.PH_SERVE && !ServeGo) NewPracticePoint();
    }

    public void PressServe()
    {
        if (!Practice) SetPractice(true);
        if (Env.Phase == Sim.PH_SERVE) ServeGo = true;
    }

    public void Restart()
    {
        if (Practice)
        {
            NewPracticePoint();
            return;
        }
        _obs = Env.Reset();
        _trail.Clear();
        _anim.Reset();
        _fakePending = null;
        FakeShown = null;
        ServeInfo = RecvInfo = null;
        Msg = "";
    }

    /// <summary>练习模式下改写这一步的动作与发球计划。</summary>
    void PracticeOverride(double[] a)
    {
        var I = Env.I;
        var T = Env.T;
        int? tech = Prac["rally"];
        _anim.RescueTh = tech != null && PRAC_TECH[tech.Value].Partner == PRAC_WIDE ? (0.25, 0.4, 7.5) : (0.72, 0.6, 9.0);
        if (I[Sim.PHASE] == Sim.PH_FLY && tech != null)
        {
            // 接发方用选中的技术，发球方用陪练技术；离台距离决定能不能用这项技术
            int srv = (int)I[Sim.SERVER];
            for (int j = 0; j < 2; j++)
            {
                int k = j == srv ? PRAC_TECH[tech.Value].Partner : tech.Value;
                var (_, a5, a6, dist, _) = PRAC_TECH[k];
                int o = j * Sim.ACT_DIM;
                a[o + 5] = a5;
                a[o + 6] = a6;
                if (j != srv && PRAC_TECH[tech.Value].Partner == PRAC_WIDE)
                {
                    // 救球练习：接球方横向起动慢半拍，只能伸到极限去够
                    a[o + 1] *= 0.4;
                    if (I[Sim.PVALID] == 1 && T[Sim.LM0 + j] * (T[Sim.PY] - Env.Pos(j).Y) < -0.1) a[o + 8] = 1.0;
                }
                if (k == PRAC_WIDE)
                {
                    // 正手大角与中路交替（对方正手在自己视角里的方向用两人的横向换算系数求）
                    double fh = T[Sim.LM0 + j] * T[Sim.LM0 + 1 - j];
                    a[o + 4] = (I[Sim.HITS] / 2) % 2 == 0 ? 0.8 * fh : -0.15 * fh;
                    a[o + 3] = 0.6;
                }
                if (dist != null)
                {
                    double d = Math.Abs(Env.Pos(j).X) - Sim.HALF_L;
                    a[o + 0] = Math.Clamp(2.5 * (d - dist.Value), -1.0, 1.0);
                }
            }
        }
        if (I[Sim.PHASE] != Sim.PH_SERVE) return;
        int s = (int)I[Sim.SERVER];
        T[Sim.SPLAN0 + s] = Prac["kind"] ?? 0;
        if (Prac["var"] != null) T[Sim.VPLAN0 + s] = Prac["var"].Value;
        if (Prac["toss"] != null) a[s * Sim.ACT_DIM + 9] = new[] { -0.8, 0.0, 0.8 }[Prac["toss"].Value];
        if (Prac["depth"] != null) a[s * Sim.ACT_DIM + 3] = new[] { -0.9, 0.1, 0.9 }[Prac["depth"].Value];
        // 每一球都当作对手第一次见：熟悉度清零
        for (int k = Sim.FAM0; k < Sim.FAM0 + 2 * Sim.N_SERVE; k++) T[k] = 0.0;
        if (!ServeGo) T[Sim.TIMER] = Math.Max(T[Sim.TIMER], Sim.SERVE_WAIT);
    }

    double[] Actions()
    {
        var a = new double[2 * Sim.ACT_DIM];
        for (int j = 0; j < 2; j++)
        {
            var o = new ReadOnlySpan<double>(_obs, j * Sim.OBS_DIM, Sim.OBS_DIM);
            var outp = new Span<double>(a, j * Sim.ACT_DIM, Sim.ACT_DIM);
            // 新增的观察维度都追加在末尾，旧模型截掉即可
            if (Modes[j] == "rl" && Pol != null) Pol.Act(o[..Pol.ObsDim], outp);
            else Bot.Act(o, outp, _rng);
        }
        return a;
    }

    // ------------------------------------------------------------------ 输入
    public override void _UnhandledInput(InputEvent e)
    {
        if (e is not InputEventKey { Pressed: true, Echo: false } k) return;
        switch (k.Keycode)
        {
            case Key.Space: Paused = !Paused; break;
            case Key.M: Sfx.On = !Sfx.On; break;
            case Key.R: Reload(); break;
            case Key.C: Cam.Reset(); break;
            case Key.Key1:
            case Key.Key2:
                int j = k.Keycode == Key.Key1 ? 0 : 1;
                Modes[j] = Modes[j] == "rl" ? "bot" : "rl";
                break;
            case Key.Escape: GetTree().Quit(); break;
        }
    }

    // ------------------------------------------------------------------ 推进模拟
    public override void _Process(double delta)
    {
        long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
        UpdateSim(delta);
        Render(delta);
        if (_shotPath != null)
        {
            double ms = System.Diagnostics.Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
            _msSum += ms;
            _msMax = Math.Max(_msMax, ms);
            _frames++;
            _clock += delta;
            if (_shotTimes.Count > 0 && _clock >= _shotTimes[0])
            {
                _shotTimes.RemoveAt(0);
                var path = _shotPath.Replace(".png", $"_{_shotN++}.png");
                GetViewport().GetTexture().GetImage().SavePng(path);
                System.IO.File.AppendAllText(_shotPath.Replace(".png", ".log"),
                    $"shot {path} t={_clock:F1}s fps={Engine.GetFramesPerSecond()} score {Env.I[Sim.SC0]}:{Env.I[Sim.SC1]} " +
                    $"games {Env.I[Sim.G0]}:{Env.I[Sim.G1]} matches {Env.I[Sim.M0]}:{Env.I[Sim.M1]} " +
                    $"logic avg {_msSum / Math.Max(_frames, 1):F2} ms max {_msMax:F1} ms sound {(Sfx.Ok ? "ok" : "MISSING")}\n");
                _msSum = _msMax = 0;
                _frames = 0;
                if (_shotTimes.Count == 0) GetTree().Quit();
            }
        }
    }

    string _shotPath;
    List<double> _shotTimes = new();
    double _clock, _msSum, _msMax;
    int _shotN, _frames;
    double[] _camArg;

    void UpdateSim(double dt)
    {
        if (Paused) return;
        // 一分开始时球员还在走回站位：先不推进模拟，站定一会儿再开始发球
        bool gliding = _glide.Any(g => g != null);
        if (Env.Phase == Sim.PH_SERVE && Env.T[Sim.TOSS] < 0.0 && (gliding || _settle > 0.0))
        {
            _settle = !gliding ? Math.Max(_settle - Math.Min(dt, 0.1) * Speed, 0.0) : SERVE_SETTLE;
            _acc = Math.Min(_acc, Env.Dt * 0.999);
            return;
        }
        _acc += Math.Min(dt, 0.1) * Speed;
        while (_acc >= Env.Dt)
        {
            var I = Env.I;
            int prevPhase = (int)I[Sim.PHASE], prevHits = (int)I[Sim.HITS];
            var a = Actions();
            if (Practice) PracticeOverride(a);
            Array.Copy(a, _act, a.Length);
            var prevM = (I[Sim.M0], I[Sim.M1]);
            int prevNet = (int)I[Sim.NETT];
            _obs = Env.Step(a);
            PlaySounds(prevPhase, prevHits, prevNet, Env.Done);
            if (Practice)
            {
                I[Sim.SC0] = I[Sim.SC1] = I[Sim.G0] = I[Sim.G1] = I[Sim.M0] = I[Sim.M1] = 0;
                I[Sim.FIRST] = Prac["server"] ?? 0;
                Env.F[0] = Env.F[1] = 0.0;
                if (prevPhase == Sim.PH_SERVE && I[Sim.PHASE] != Sim.PH_SERVE) ServeGo = false;
            }
            _anim.OnStep(Env, prevPhase, prevHits, Env.Dt);
            if (prevPhase == Sim.PH_SERVE && I[Sim.PHASE] != Sim.PH_SERVE && I[Sim.LSRV] >= 0)
            {
                int srv = (int)I[Sim.SERVER];
                int k = (int)I[Sim.LSRV];
                double rpm = Env.BallW.Norm() * 60 / (2 * Math.PI);
                ServeInfo = (srv, $"{Txt.Toss(Env.T[Sim.TOSS])} · {Sim.SERVE_NAMES[k]} · " +
                                  $"{Txt.ServeSpin(Env.BallV, Env.BallW)} {rpm:F0} rpm" + (I[Sim.LHID] != 0 ? " · 回抛" : ""));
            }
            if (prevHits == 0 && I[Sim.HITS] == 1)
                RecvInfo = ((int)I[Sim.LHIT], Txt.Tech((int)I[Sim.LTECH], I[Sim.LSIDE] != 0, (int)I[Sim.LSTK]));
            if (I[Sim.PHASE] == Sim.PH_SERVE && prevPhase != Sim.PH_SERVE) ServeInfo = RecvInfo = null;
            if (I[Sim.HITS] > prevHits)
            {
                int lf = (int)I[Sim.LFAKE];
                _fakePending = lf > 0 ? (lf - 1, (int)I[Sim.HITS]) : null;
            }
            if (prevPhase == Sim.PH_SERVE) FakeShown = null;
            _acc -= Env.Dt;
            if (Env.Done)
            {
                int w = (int)Env.Winner;
                int r = (int)Env.Reason;
                // 只有被骗的一方真的没接住这一拍，才算假动作得手
                if (_fakePending is { } fp && w == fp.Item1 && Env.Hits == fp.Item2 && r == Sim.R_MISS)
                    FakeShown = fp.Item1;
                _fakePending = null;
                if (w >= 0)
                {
                    Msg = $"{NAMES[w]}得分！ {NAMES[1 - w]}{Sim.REASON_TEXT[r]}";
                    if (I[Sim.M0 + w] > (w == 0 ? prevM.Item1 : prevM.Item2))
                        Msg = $"{NAMES[w]}赢下比赛！（疲劳清零，开始新一场）";
                    MsgColor = COLORS[w];
                }
                else
                {
                    Msg = r == Sim.R_LET ? "发球擦网，重发" : "回合超时，重发球";
                    MsgColor = Ui.TEXT;
                }
            }
        }
    }

    /// <summary>按这一步发生的事件放音效：击球（按挥拍速度和胶皮）、球落台/落地、擦网、得分后的掌声。</summary>
    void PlaySounds(int prevPhase, int prevHits, int prevNet, bool done)
    {
        var I = Env.I;
        var T = Env.T;
        var ball = Env.Ball;
        bool hit = (prevPhase == Sim.PH_SERVE && I[Sim.PHASE] == Sim.PH_FLY) || I[Sim.HITS] > prevHits;
        if (hit)
        {
            Sfx.Racket(T[Sim.LSW], (int)T[Sim.LRUB], ball);
            _pointRally = (int)I[Sim.HITS];
        }
        if (I[Sim.NETT] != 0 && prevNet == 0) Sfx.Net(0.8, ball);
        // 落台/落地按子步轨迹找最低点：快球一步飞出半米多，只看步末会漏掉端线附近落台、或落台后同一步就被击回的球
        var path = new List<V3>();
        foreach (var p in _ballTrail)
            if (p != null)
                path.Add(p.Value);
        for (int s = 0; s < Env.NSub; s++) path.Add(Env.HV(s, 0));
        double pdt = Env.Dt / Env.NSub;
        for (int k = 1; k < path.Count - 1; k++)
        {
            V3 a = path[k - 1], b = path[k], c = path[k + 1];
            if ((b - a).Norm() > 0.5 || (c - b).Norm() > 0.5) continue;
            double vin = Math.Max(a.Z - b.Z, k >= 2 ? path[k - 2].Z - a.Z : 0.0) / pdt;
            if (!(vin > 0.6 && c.Z > b.Z - 0.002) || k < path.Count - 1 - Env.NSub) continue;
            if (Math.Abs(b.X) <= Sim.HALF_L + 0.03 && Math.Abs(b.Y) <= Sim.HALF_W + 0.03 &&
                Math.Abs(b.Z - Sim.TABLE_H) < 0.07)
                Sfx.Table((0.45 + 0.55 * Math.Min(vin / 5.0, 1.0)) * Math.Min(vin / 1.5, 1.0), b);
            else if (b.Z < 0.3)
                Sfx.Floor(Math.Min(vin / 6.0, 1.0), b);
        }
        for (int k = 0; k < 3; k++) _ballTrail[k] = path[path.Count - 3 + k];
        for (int j = 0; j < 2; j++)
            if (_anim.GaitOf[j].Name == "倒地爆冲" && _anim.Tm - _anim.GaitOf[j].T0 < 0.05)
                _pointWow = true;
        if (done)
        {
            if (Env.Winner >= 0 && (_pointRally >= 9 || _pointWow)) Sfx.Applause(0.7, new V3(0, 0, 2));
            _pointRally = 0;
            _pointWow = false;
        }
    }

    // ------------------------------------------------------------------ 画面
    /// <summary>一分打完模拟里直接把球员放回起始位置；画面上让人从原地走回去，而不是瞬移。返回显示用的球。</summary>
    V3 GlideBack(V3[] bodies, V3[] hands, V3 ball, double adt)
    {
        var raw = bodies.ToArray();
        for (int j = 0; j < 2; j++)
        {
            var prev = _glidePrev[j];
            // 模拟有时先把手放到发球位置、下一帧才挪身体，所以手一帧跳得比最快的挥拍还远也算瞬移
            if (prev != null && _glideDisp[j] != null &&
                ((raw[j] - prev.Value.Body).NormXY() > 0.35 || (hands[j] - prev.Value.Hand).Norm() > 0.6))
            {
                var ob = _glideDisp[j].Value.Body - raw[j];
                var oh = _glideDisp[j].Value.Hand - hands[j];
                double dur = Math.Clamp(ob.NormXY() / 2.0, 0.5, 1.2);
                _glide[j] = (ob, oh, 0.0, dur);
            }
            _glidePrev[j] = (raw[j], hands[j]);
            if (_glide[j] is { } g)
            {
                var (gob, goh, gt, gdur) = g;
                gt += adt;
                double u = Math.Min(gt / gdur, 1.0);
                double w = 1.0 - u * u * (3.0 - 2.0 * u);
                if (w <= 0.0) _glide[j] = null;
                else
                {
                    _glide[j] = (gob, goh, gt, gdur);
                    bodies[j] += gob * w;
                    hands[j] += goh * w;
                    if (Env.Phase == Sim.PH_SERVE && Env.I[Sim.SERVER] == j) ball += goh * w;
                }
            }
            _glideDisp[j] = (bodies[j], hands[j]);
        }
        return ball;
    }

    bool FhFaceOut(int j, double bl)
    {
        if (bl >= 0) return true;
        if (Env.Stat(j, Sim.GRIP) > 0.5)
            return _anim.HitKind[j].Kind == Sim.ST_BLOCK || _anim.HitKind[j].Kind == Sim.ST_PUSH ||
                   PUSHBLOCK_TECH.Contains(_anim.PredTech[j]);
        return false;
    }

    V3 _ballShown;

    void Render(double dt)
    {
        Compute(dt, Math.Min(dt, 0.1));
        for (int j = 0; j < 2; j++)
        {
            var b = Bodies[j];
            bool pen = Env.Stat(j, Sim.GRIP) > 0.5;
            var face = Poses[j].Face ?? new V3(j == 0 ? 1 : -1, 0, 0.15);
            _pv[j].UpdateBody(b, j, face, pen, pen && _blends[j] < 0, FhFaceOut(j, _blends[j]),
                (int)Env.Stat(j, Sim.RFH), (int)Env.Stat(j, Sim.RBH));
        }
        _arena.SetBall(_ballShown, Txt.SpinColor(Env.BallV, Env.BallW));
        _arena.SetTrail(_trail, Cam.Pos.G);
        int h = (int)Math.Round(_foot[6]);
        _arena.SetCue(h, h >= 0 ? _foot[4 + h] : 0.0, h >= 0 ? COLORS[h] : Colors.White);
    }

    /// <summary>这一帧显示用的球、骨架（不碰画面节点）。skelDt：骨架避让记忆按多长时间衰减。</summary>
    void Compute(double dt, double skelDt)
    {
        int nsub = Env.NSub;
        int s = Math.Min((int)(_acc / Env.Dt * nsub), nsub - 1);
        var ball = Env.HV(s, 0);
        var bodies = new[] { Env.HV(s, 3), Env.HV(s, 6) };
        var hands = new[] { Env.HV(s, 9), Env.HV(s, 12) };
        _blends[0] = Env.H(s, 15);
        _blends[1] = Env.H(s, 16);
        for (int k = 0; k < 7; k++) _foot[k] = Env.H(s, 17 + k);
        double lag = (nsub - 1 - s) * Env.Dt / nsub;
        double fdt = Math.Min(dt, 0.1);
        double adt = Paused ? 0.0 : fdt * Speed;
        ball = GlideBack(bodies, hands, ball, adt);
        var foot = _foot;
        var (b2, h2, poses) = _anim.Frame(Env, _act, ball, hands, _blends,
            (j, pose, hand) => _skel.Home(Env, foot, bodies[j], hand, j, pose), adt, lag);
        ball = b2;
        hands = h2;
        Poses = poses;
        // 一分刚开始动画会直接换到发球准备姿势；抛球前球拍限速移过去，不要一帧跳过去
        if (Env.Phase == Sim.PH_SERVE)
        {
            int srv = (int)Env.I[Sim.SERVER];
            if (_handShown[srv] is V3 prev && _anim.ServePhase(Env, _act, lag) < 0.25)
            {
                var d = hands[srv] - prev;
                double dn = d.Norm(), lim = 3.0 * adt;
                if (dn > lim) hands[srv] = prev + d * (lim / dn);
            }
        }
        _handShown[0] = hands[0];
        _handShown[1] = hands[1];
        if (Env.Phase == Sim.PH_SERVE) _trail.Clear();
        else if (!Paused)
        {
            _trail.Add((ball, Txt.SpinColor(Env.BallV, Env.BallW)));
            if (_trail.Count > 28) _trail.RemoveRange(0, _trail.Count - 28);
            var w = Env.BallW;
            double wn = w.Norm();
            if (wn > 1e-6) BallRot = Txt.Rotate(BallRot, w / wn, wn * fdt * Speed * VIS_SPIN);
        }

        for (int j = 0; j < 2; j++) Bodies[j] = _skel.Build(Env, _foot, bodies[j], hands[j], j, Poses[j], skelDt);
        _ballShown = ball;
    }
}
