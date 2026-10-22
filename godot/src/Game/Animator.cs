using System;
using System.Collections.Generic;
using System.Linq;

namespace PingPong;

/// <summary>一名球员这一帧的显示姿态（anim.py 里 frame() 返回的 pose 字典）。</summary>
public class Pose
{
    public double Twist, Stance, Crouch, Bob, T;
    public V3? Face, Epole, LHand, LFoot, RFoot;
    public V2 Tilt = V2.Zero, Lunge = V2.Zero;
    public (string Name, double T0) Gait = ("", -9.0);
}

/// <summary>技术动作：引拍位置、随挥位置、引拍转体、引拍下沉、肘尖朝向、引拍时拍面朝向（右手持拍、自己视角）。</summary>
public record TechSpec(V3 Back, V3 Follow, double Twist, double Sink, V3? Elbow, V3? Face);

/// <summary>纯画面层的人物动画（anim.py 的移植）：迈步、发球抛球、引拍/随挥与转体。只读模拟状态，不改物理结果。</summary>
public class Animator
{
    const double BACK_T = 0.5, SWING_T = 0.18;
    const double PRED_LOCK_T = 0.03, PRED_TAU = 0.05;
    const double FOLLOW_UP = 0.12, FOLLOW_END = 0.5;
    const double SERVE_FOLLOW_UP = 0.15, SERVE_FOLLOW_HOLD = 0.32, SERVE_FOLLOW_END = 0.65;
    const double STEP_ERR = 0.1;
    const double READY_CROUCH = 0.06;
    const double RECV_READY_Z = Sim.TABLE_H + 0.22;
    const double RECV_BLEND_T = 0.35;
    const double READY_MAX_Z = Sim.TABLE_H + 0.40;
    const double DRIVE_RISE = 0.07;
    const double DIVE_BACK_AMP = 1.2;
    static readonly (double, double) DIVE_PRE = (0.06, 0.15);
    static readonly (double, double) DIVE_DOWN = (0.42, 0.65);
    const double DIVE_UP = 0.8;
    const double DIVE_REACH = 0.6, DIVE_LUNGE = 0.7;

    static V3 v(double x, double y, double z) => new(x, y, z);

    static readonly V3 BH_PUSH_ELBOW = v(0.2, 0.8, -0.5);
    static readonly V3 TWIST_ELBOW = v(0.5, 0.4, 0.8), TWIST_FACE = v(0.55, 0.45, 0.5);
    public static readonly V3 ELBOW_POLE = v(-0.3, 0.6, -1.0);

    static readonly Dictionary<int, TechSpec> FH_TECH = new()
    {
        [Sim.TQ_LOOP] = new(v(-0.45, 0.3, -0.3), v(0.25, -0.45, 0.6), 0.8, 0.07, null, null),
        [Sim.TQ_LIFT] = new(v(-0.35, 0.25, -0.42), v(0.15, -0.35, 0.65), 0.7, 0.1, null, null),
        [Sim.TQ_DRIVELOOP] = new(v(-0.5, 0.32, -0.1), v(0.4, -0.5, 0.35), 0.9, 0.05, null, null),
        [Sim.TQ_DRIVE] = new(v(-0.22, 0.2, -0.04), v(0.28, -0.48, 0.22), 0.45, 0.03, null, null),
        [Sim.TQ_BLOCK] = new(v(-0.05, 0.05, 0.0), v(0.1, -0.05, 0.02), 0.1, 0.0, null, null),
        [Sim.TQ_SHORT] = new(v(-0.04, 0.03, 0.05), v(0.1, 0.0, -0.03), 0.1, 0.04, null, null),
        [Sim.TQ_LONG] = new(v(-0.1, 0.06, 0.16), v(0.28, -0.06, -0.14), 0.25, 0.04, null, null),
        [Sim.TQ_PUSH] = new(v(-0.1, 0.05, 0.1), v(0.2, -0.05, -0.08), 0.2, 0.03, null, null),
        [Sim.TQ_FLICK] = new(v(-0.08, 0.06, -0.1), v(0.22, -0.12, 0.16), 0.25, 0.04, null, null),
        [Sim.TQ_WIPE] = new(v(-0.05, 0.12, 0.0), v(0.14, -0.22, 0.02), 0.2, 0.03, null, null),
        [Sim.TQ_CHOP] = new(v(-0.05, 0.15, 0.55), v(0.25, -0.25, -0.45), 0.4, 0.05, null, null),
        [Sim.TQ_SMASH] = new(v(-0.35, 0.3, 0.28), v(0.38, -0.5, -0.15), 1.0, 0.02, null, null),
        [Sim.TQ_LOB] = new(v(-0.1, 0.2, -0.4), v(0.25, -0.2, 0.6), 0.5, 0.08, null, null),
    };

    static readonly Dictionary<int, TechSpec> BH_TECH = new()
    {
        [Sim.TQ_LOOP] = new(v(-0.12, -0.08, -0.35), v(0.3, 0.32, 0.62), -0.3, 0.08, v(0.6, 0.6, -0.2), null),
        [Sim.TQ_LIFT] = new(v(-0.1, -0.05, -0.42), v(0.2, 0.25, 0.68), -0.3, 0.1, v(0.6, 0.6, -0.2), null),
        [Sim.TQ_DRIVELOOP] = new(v(-0.15, -0.15, -0.2), v(0.35, 0.3, 0.3), -0.35, 0.06, v(0.6, 0.6, -0.1), null),
        [Sim.TQ_RIP] = new(v(-0.1, -0.18, -0.08), v(0.3, 0.3, 0.12), -0.25, 0.03, v(0.5, 0.7, 0.1), null),
        [Sim.TQ_DRIVE] = new(v(-0.1, -0.15, -0.03), v(0.25, 0.25, 0.12), -0.25, 0.03, null, null),
        [Sim.TQ_BLOCK] = new(v(-0.03, -0.03, 0.0), v(0.08, 0.03, 0.0), -0.05, 0.0, null, null),
        [Sim.TQ_SIDECUT] = new(v(-0.05, -0.12, 0.12), v(0.1, -0.2, -0.12), -0.15, 0.02, v(-0.2, 0.9, -0.6),
            v(0.6, 0.6, 0.15)),
        [Sim.TQ_SHORT] = new(v(-0.04, -0.02, 0.05), v(0.1, 0.0, -0.03), -0.05, 0.04, BH_PUSH_ELBOW, null),
        [Sim.TQ_LONG] = new(v(-0.1, -0.06, 0.16), v(0.28, 0.06, -0.14), -0.15, 0.04, BH_PUSH_ELBOW, null),
        [Sim.TQ_PUSH] = new(v(-0.1, -0.06, 0.1), v(0.2, 0.05, -0.08), -0.1, 0.03, BH_PUSH_ELBOW, null),
        [Sim.TQ_TWIST] = new(v(-0.15, -0.1, -0.06), v(0.25, 0.35, 0.25), -0.2, 0.06, TWIST_ELBOW, TWIST_FACE),
        [Sim.TQ_BANANA] = new(v(-0.18, -0.12, -0.08), v(0.3, 0.42, 0.32), -0.3, 0.07, TWIST_ELBOW, TWIST_FACE),
        [Sim.TQ_WIPE] = new(v(-0.06, -0.12, 0.0), v(0.14, -0.1, 0.02), -0.1, 0.03, null, v(0.8, -0.3, 0.1)),
        [Sim.TQ_FLICK] = new(v(-0.06, -0.08, -0.1), v(0.2, 0.2, 0.15), -0.15, 0.04, null, null),
        [Sim.TQ_CHOP] = new(v(-0.05, -0.25, 0.55), v(0.25, 0.25, -0.45), -0.3, 0.05, null, null),
        [Sim.TQ_SMASH] = new(v(-0.2, -0.25, 0.15), v(0.35, 0.35, -0.05), -0.4, 0.02, null, null),
        [Sim.TQ_LOB] = new(v(-0.1, -0.1, -0.35), v(0.25, 0.15, 0.5), -0.2, 0.08, null, null),
    };

    static readonly int[] ATTACK_TECH = { Sim.TQ_DRIVE, Sim.TQ_LOOP, Sim.TQ_DRIVELOOP, Sim.TQ_RIP, Sim.TQ_LIFT };

    static Animator()
    {
        // 拉球类动作的放大倍数：(引拍与随挥位置, 转体, 下蹲)。基础表按“刚够打到”的幅度定，看起来不够猛
        foreach (var (bh, tbl) in new[] { (false, FH_TECH), (true, BH_TECH) })
        {
            var (p, t, s) = bh ? (1.5, 1.4, 1.6) : (1.4, 1.35, 1.6);
            foreach (var k in new[] { Sim.TQ_LOOP, Sim.TQ_LIFT, Sim.TQ_DRIVELOOP, Sim.TQ_RIP })
                if (tbl.TryGetValue(k, out var e))
                    tbl[k] = e with { Back = e.Back * p, Follow = e.Follow * p, Twist = e.Twist * t, Sink = e.Sink * s };
        }
    }

    public static TechSpec Spec(int tech, bool bh)
    {
        var tbl = bh ? BH_TECH : FH_TECH;
        return tbl.TryGetValue(tech, out var s) ? s : tbl[Sim.TQ_DRIVE];
    }

    /// <summary>快攻、加转、前冲（反手撕）、提拉在意图间连续过渡，不会在相近的意图间来回跳。</summary>
    static TechSpec AttackSpec(bool bh, double pw, double top, double topIn)
    {
        var tbl = bh ? BH_TECH : FH_TECH;
        var hard = bh ? tbl[Sim.TQ_RIP] : tbl[Sim.TQ_DRIVELOOP];
        double uTop = M.Clamp(top / 0.7, 0.0, 1.0);
        double uPw = M.Clamp((pw - 0.45) / 0.35, 0.0, 1.0) * uTop;
        double uBack = M.Clamp((-topIn - 40.0) / 100.0, 0.0, 1.0) * uTop;
        var (a, b, c, d) = (tbl[Sim.TQ_DRIVE], tbl[Sim.TQ_LOOP], tbl[Sim.TQ_LIFT], hard);

        V3 Mix3(Func<TechSpec, V3> f)
        {
            var x = f(a) + (f(b) - f(a)) * uTop;
            x += (f(d) - x) * uPw;
            return x + (f(c) - x) * uBack;
        }

        double Mix1(Func<TechSpec, double> f)
        {
            double x = f(a) + (f(b) - f(a)) * uTop;
            x += (f(d) - x) * uPw;
            return x + (f(c) - x) * uBack;
        }

        return new TechSpec(Mix3(s => s.Back), Mix3(s => s.Follow), Mix1(s => s.Twist), Mix1(s => s.Sink),
            Mix3(s => s.Elbow ?? ELBOW_POLE), null);
    }

    // 发球动作（右手持拍，自己视角：前、右、上；球拍相对击球点的偏移）
    static readonly V3[] SERVE_READY =
    {
        v(-0.05, 0.12, -0.08), v(0.0, -0.05, 0.05), v(0.05, 0.05, -0.03), v(-0.05, 0.1, 0.25),
        v(-0.05, 0.15, 0.25), v(-0.1, 0.15, -0.05), v(-0.03, -0.06, -0.08), v(-0.05, -0.05, -0.08),
    };
    static readonly V3[] SERVE_BACK =
    {
        v(-0.2, 0.4, 0.15), v(-0.05, -0.25, 0.3), v(0.05, -0.4, 0.06), v(-0.15, -0.05, 0.5),
        v(-0.1, 0.25, 0.45), v(-0.35, 0.3, 0.02), v(-0.08, -0.3, 0.16), v(-0.25, -0.22, 0.04),
    };
    static readonly V3[] SERVE_FOLLOW =
    {
        v(0.1, -0.3, -0.02), v(0.12, 0.35, 0.22), v(0.15, 0.45, 0.06), v(0.15, 0.35, -0.2),
        v(0.25, -0.1, -0.25), v(0.3, -0.25, 0.2), v(0.15, 0.32, -0.04), v(0.35, 0.15, 0.06),
    };
    static readonly V3[] SERVE_STANCE =
    {
        v(1.3, 1.0, 0.05), v(1.55, 1.0, 0.04), v(1.0, 0.8, 0.05), v(1.1, 0.9, 0.03),
        v(0.6, 0.6, 0.03), v(1.2, 1.0, 0.06), v(-0.35, -0.5, 0.06), v(-0.25, -0.45, 0.05),
    };
    static readonly Dictionary<int, V3> SERVE_ELBOW = new()
    {
        [Sim.SV_SHOVEL] = v(0.2, 1.0, 0.35), [Sim.SV_BH] = v(0.2, 0.9, -0.35), [Sim.SV_BHFAST] = v(0.2, 0.9, -0.45),
    };
    static readonly double[] SERVE_COIL = { 0.35, -0.25, -0.4, 0.2, 0.25, 0.45, -0.35, -0.3 };
    static double SquatExtra(int k) => k == Sim.SV_SQUAT ? 0.3 : 0.0;
    static readonly V3 TOSS_CLEAR_DEFAULT = v(-0.12, -0.3, -0.08);
    static readonly Dictionary<int, V3> TOSS_CLEAR = new()
    {
        [Sim.SV_BH] = v(-0.38, -0.42, -0.25), [Sim.SV_BHFAST] = v(-0.38, -0.42, -0.25),
        [Sim.SV_SHOVEL] = v(-0.15, -0.3, -0.1),
    };
    static readonly V3[] SERVE_FACE =
    {
        v(0.4, -0.9, 0.2), v(0.4, 0.9, 0.2), v(0.35, 0.9, 0.2), v(0.5, 0.7, 0.3),
        v(0.6, 0.0, 0.8), v(0.9, -0.3, -0.1), v(0.4, 0.9, 0.2), v(0.9, 0.2, 0.0),
    };
    static readonly V3 FACE_READY = v(1.0, 0.0, 0.15);

    class Step
    {
        public int[] Ks;
        public V2[] A;
        public Dictionary<int, V2> Bt;
        public double U, Dur, Lift;
    }

    record struct Rescue(string Kind, double T0, V2 Side);

    // 判为救球需要的够球程度（练习“救球爆冲”时调低，方便观察）
    public (double, double, double) RescueTh = (0.72, 0.6, 9.0);

    public double Tm, Wall;
    public double[] HitT = new double[2];
    public (int Kind, bool Bh, double Sw, int Tech)[] HitKind = new (int, bool, double, int)[2];
    public int[] PredTech = new int[2];
    public (string Name, double T0)[] GaitOf = new (string, double)[2];
    (double[] Cur, double T)?[] _backS = new (double[], double)?[2];
    V3?[] _contact = new V3?[2];
    (V3 P, double W)?[] _anchor = new (V3, double)?[2];
    double[] _hs = new double[2];
    (V3 Pole, double W)?[] _arm = new (V3, double)?[2];
    V2[][] _feet = new V2[2][];
    Step[] _steps = new Step[2];
    List<(int[] Ks, double Dur, double Lift)>[] _queue = { new(), new() };
    Rescue?[] _rescue = new Rescue?[2];
    (V2 Side, double S, double Carry)?[] _divePrep = new (V2, double, double)?[2];
    double[] _diveSink = new double[2];
    V2[] _diveHold = new V2[2];
    double _serveT0, _serveWall0;
    double[] _recvW = new double[2], _capW = new double[2];
    int[] _serveKind = new int[2];
    V3?[] _serveHand = new V3?[2];

    public Animator() => Reset();

    public void Reset()
    {
        Tm = 0.0;
        HitT[0] = HitT[1] = -9.0;
        HitKind[0] = HitKind[1] = (-1, false, 0.0, -1);
        PredTech[0] = PredTech[1] = -1;
        GaitOf[0] = GaitOf[1] = ("", -9.0);
        for (int j = 0; j < 2; j++)
        {
            _backS[j] = null;
            _contact[j] = null;
            _anchor[j] = null;
            _hs[j] = 1.0;
            _arm[j] = null;
            _feet[j] = null;
            _steps[j] = null;
            _queue[j].Clear();
            _rescue[j] = null;
            _divePrep[j] = null;
            _diveSink[j] = 0.0;
            _diveHold[j] = V2.Zero;
            _recvW[j] = _capW[j] = 0.0;
            _serveKind[j] = Sim.SV_PEND;
            _serveHand[j] = null;
        }
        _serveT0 = -9.0;
        Wall = 0.0;
        _serveWall0 = -9.0;
    }

    static double Act(double[] act, int j, int k) => act[j * Sim.ACT_DIM + k];
    static double ActC(double[] act, int j, int k) => M.Clamp(act[j * Sim.ACT_DIM + k], -1.0, 1.0);

    /// <summary>每个模拟步之后调用：记录谁刚击球、什么技术，用来播随挥。</summary>
    public void OnStep(World w, int prevPhase, int prevHits, double dt)
    {
        Tm += dt;
        var I = w.I;
        var T = w.T;
        int h = -1, tech = -1, kind = -1;
        if (prevPhase != Sim.PH_SERVE && I[Sim.PHASE] == Sim.PH_SERVE)
        {
            _serveT0 = Tm;
            _serveWall0 = Wall;
        }
        if (prevPhase == Sim.PH_SERVE && I[Sim.PHASE] == Sim.PH_FLY)
        {
            h = (int)I[Sim.SERVER];
            kind = -1;
        }
        else if (I[Sim.HITS] > prevHits && I[Sim.PHASE] == Sim.PH_FLY)
        {
            h = (int)I[Sim.LHIT];
            kind = (int)I[Sim.LSTK];
            tech = (int)I[Sim.LTECH];
        }
        if (h < 0) return;
        HitT[h] = Tm;
        HitKind[h] = (kind, I[Sim.LSIDE] != 0, T[Sim.LSW], tech);
        PredTech[h] = -1;
        if (kind < 0) return;
        // 够球程度：模拟里伸手够球让挥拍打折 1 - 0.4·min(stretch/0.2, 1)
        double sk = (1.0 - T[Sim.DBG0 + 6]) / 0.4;
        var pos = w.Pos(h);
        V2 pd = _contact[h] is V3 cp ? new V2(cp.X - pos.X, cp.Y - pos.Y) : V2.Zero;
        double pn = pd.Norm();
        // 倒地爆冲由模拟判定；救球只是画面效果，只算横向够球
        if (T[Sim.DV0 + h] > 0.0)
        {
            _rescue[h] = new Rescue("倒地爆冲", Tm, OffSide(h));
            GaitOf[h] = ("倒地爆冲", Tm);
        }
        else if (sk > RescueTh.Item1 && pn > RescueTh.Item2 && Math.Abs(pd.Y) > 0.7 * pn)
        {
            _rescue[h] = new Rescue("救球", Tm, new V2(0.0, Math.CopySign(1.0, pd.Y)));
            GaitOf[h] = ("救球", Tm);
        }
    }

    /// <summary>来球击到之前预判这一板用什么技术：用预测击球点和当前的击球意图套模拟里同样的分类。</summary>
    static int PredictTech(World w, double[] act, int j, bool bh)
    {
        var T = w.T;
        double pw = (ActC(act, j, 5) + 1.0) * 0.5;
        int k = World.ClassifyStroke(new V3(T[Sim.PX], T[Sim.PY], T[Sim.PZ]), pw, ActC(act, j, 6));
        double depth = Sim.DEPTH_MIN + (ActC(act, j, 3) + 1.0) * 0.5 * (1.2 - Sim.DEPTH_MIN);
        double topIn = World.IncomingTop(w.BallV, w.BallW);
        return World.TechOf(k, bh, pw, ActC(act, j, 6), ActC(act, j, 7), depth, topIn);
    }

    /// <summary>球员自己视角（前、持拍手一侧、上）换到世界坐标；左手球员横向镜像。</summary>
    public V3 OwnToWorld(int j, V3 o)
    {
        double sj = j == 0 ? -1.0 : 1.0;
        return new V3(-sj * o.X, sj * _hs[j] * o.Y, o.Z);
    }

    V2 OffSide(int j)
    {
        var w = OwnToWorld(j, new V3(0.0, -1.0, 0.0));
        return new V2(0.0, Math.CopySign(1.0, w.Y));
    }

    /// <summary>返回 (球拍位移, 转体角, 下沉量)：来球前引拍并屈膝蓄力，击球瞬间回到击球点，击球后随挥、蹬腿伸膝，再还原。</summary>
    (V3 Off, double Twist, double Sink) Swing(World w, double[] act, int j, double bl, double lag)
    {
        var I = w.I;
        var T = w.T;
        var off = V3.Zero;
        double twist = 0.0, sink = 0.0;
        bool bh = bl < 0.0;
        _arm[j] = null;
        _anchor[j] = null;
        _divePrep[j] = null;
        if (I[Sim.PHASE] == Sim.PH_FLY && I[Sim.TURN] == j && I[Sim.PVALID] == 1)
        {
            double ttc = T[Sim.PRT] - T[Sim.RT] + lag;
            if (0.0 < ttc && ttc < BACK_T)
            {
                if (ttc > PRED_LOCK_T || PredTech[j] < 0) PredTech[j] = PredictTech(w, act, j, bh);
                double pw = (ActC(act, j, 5) + 1.0) * 0.5;
                TechSpec spec;
                double amp;
                if (ATTACK_TECH.Contains(PredTech[j]))
                {
                    spec = AttackSpec(bh, pw, ActC(act, j, 6), World.IncomingTop(w.BallV, w.BallW));
                    amp = 0.6 + 0.4 * pw;
                }
                else
                {
                    spec = Spec(PredTech[j], bh);
                    amp = 1.0;
                }
                var el = spec.Elbow ?? ELBOW_POLE;
                var tgt = new[]
                {
                    spec.Back.X * amp, spec.Back.Y * amp, spec.Back.Z * amp, spec.Twist * amp, spec.Sink,
                    el.X, el.Y, el.Z,
                };
                // 要倒地爆冲的球：引拍拉得特别大、特别低，重心压低并提前往来球一侧倒
                var Pj = w.Pos(j);
                bool dive = !bh && pw > Sim.DIVE_PW && ATTACK_TECH.Contains(PredTech[j]) &&
                            T[Sim.LM0 + j] * (T[Sim.PY] - Pj.Y) < Sim.FH_CLO - Sim.DIVE_ST;
                if (dive)
                {
                    tgt[0] *= DIVE_BACK_AMP;
                    tgt[2] = tgt[2] * DIVE_BACK_AMP - 0.12;
                    tgt[3] *= 1.3;
                    tgt[4] += 0.05;
                }
                // 引拍以预测击球点为基准：模拟里的持拍手此时还在往击球点移动，跟着它引拍位置会被抵消
                _contact[j] = new V3(T[Sim.PX], T[Sim.PY], T[Sim.PZ]);
                double sb = M.Smooth((BACK_T - ttc) / (BACK_T - SWING_T));
                _anchor[j] = (_contact[j].Value, sb);
                double[] cur;
                var st = _backS[j];
                if (st == null || Tm - st.Value.T > 0.3) cur = tgt;
                else
                {
                    cur = new double[8];
                    double e = 1.0 - Math.Exp(-(Tm - st.Value.T) / PRED_TAU);
                    for (int k = 0; k < 8; k++) cur[k] = st.Value.Cur[k] + (tgt[k] - st.Value.Cur[k]) * e;
                }
                _backS[j] = (cur, Tm);
                double s = ttc > SWING_T ? sb : ttc / SWING_T;
                off += s * OwnToWorld(j, new V3(cur[0], cur[1], cur[2]));
                twist += s * cur[3];
                sink += s * cur[4];
                if (dive)
                {
                    // 倒地爆冲向前挥拍时不蹬腿起身：引拍时压下的重心一直保持到击球，接着扑出去
                    _divePrep[j] = (OffSide(j), sb, ttc <= SWING_T ? (1.0 - s) * cur[4] : 0.0);
                    _diveSink[j] = cur[4];
                }
                else _diveSink[j] = 0.0;
                _arm[j] = (new V3(cur[5], cur[6], cur[7]), sb);
            }
        }
        else if (I[Sim.PHASE] == Sim.PH_SERVE && I[Sim.SERVER] == j)
        {
            int k = CurrentServe(w, act, j);
            double p = ServePhase(w, act, lag);
            V3 ready = SERVE_READY[k], back = SERVE_BACK[k], pos;
            double coil;
            if (p < 0.25)
            {
                pos = ready;
                coil = 0.0;
            }
            else if (p < 0.8)
            {
                double u = M.Smooth((p - 0.25) / 0.55);
                pos = V3.Lerp(ready, back, u);
                coil = u;
            }
            else
            {
                // 下摆加速：越接近击球越快
                double u = Math.Pow((p - 0.8) / 0.2, 2);
                pos = V3.Lerp(back, V3.Zero, u);
                coil = 1.0 - u;
            }
            off += OwnToWorld(j, pos);
            twist += SERVE_COIL[k] * coil;
        }
        double tsh = Tm - lag - HitT[j];
        var (kind, hbh, sw, tech) = HitKind[j];
        // 上一分的随挥不带进新一分的发球准备
        double fade = I[Sim.PHASE] == Sim.PH_SERVE ? 1.0 - M.Smooth((Wall - _serveWall0) / 0.3) : 1.0;
        if (kind == -1 && 0.0 <= tsh && tsh < SERVE_FOLLOW_END)
        {
            double f = tsh < SERVE_FOLLOW_UP
                ? M.Smooth(tsh / SERVE_FOLLOW_UP)
                : 1.0 - M.Smooth((tsh - SERVE_FOLLOW_HOLD) / (SERVE_FOLLOW_END - SERVE_FOLLOW_HOLD));
            f *= fade;
            int k = _serveKind[j];
            off += f * OwnToWorld(j, SERVE_FOLLOW[k]);
            twist -= f * SERVE_COIL[k];
            sink -= DRIVE_RISE * f * Math.Min(0.3 + 2.0 * T[Sim.LBODY], 1.0);
        }
        else if (kind != -1 && -0.1 <= tsh && tsh < FOLLOW_END)
        {
            // 击球后第一帧 tsh 可能略小于 0，这一帧也锚在击球点，否则球拍会闪到模拟里的手的位置
            tsh = Math.Max(tsh, 0.0);
            double f = tsh < FOLLOW_UP
                ? M.Smooth(tsh / FOLLOW_UP)
                : 1.0 - M.Smooth((tsh - FOLLOW_UP) / (FOLLOW_END - FOLLOW_UP));
            if (_anchor[j] == null && _contact[j] != null)
                _anchor[j] = (_contact[j].Value, (tsh < FOLLOW_UP ? 1.0 : f) * fade);
            f *= fade;
            var spec = Spec(tech, hbh);
            double amp = M.Clamp(sw / 12.0, 0.4, 1.3) * f;
            off += amp * OwnToWorld(j, spec.Follow);
            twist -= amp * 0.8 * spec.Twist;
            if (_rescue[j] == null || _rescue[j].Value.Kind != "倒地爆冲")
                sink -= DRIVE_RISE * f * Math.Min(0.3 + 2.0 * T[Sim.LBODY], 1.0);
            if (_arm[j] == null) _arm[j] = (spec.Elbow ?? ELBOW_POLE, tsh < FOLLOW_UP ? 1.0 : f);
        }
        return (off, twist, sink);
    }

    /// <summary>画面上的拍面法向（世界坐标）：触球瞬间用模拟里真实的拍面；引拍时按击球意图预先调好角度。</summary>
    V3 RacketFace(World w, double[] act, int j, double lag)
    {
        var I = w.I;
        var T = w.T;
        var ready = OwnToWorld(j, FACE_READY).Unit();
        var face = ready;
        if (I[Sim.PHASE] == Sim.PH_FLY && I[Sim.TURN] == j && I[Sim.PVALID] == 1)
        {
            double ttc = T[Sim.PRT] - T[Sim.RT] + lag;
            if (0.0 < ttc && ttc < BACK_T)
            {
                var tf = Spec(PredTech[j], T[Sim.BL0 + j] < 0.0).Face;
                var target = OwnToWorld(j, tf ?? new V3(1.0, 0.3 * ActC(act, j, 7), -0.75 * ActC(act, j, 6))).Unit();
                face = (ready + (target - ready) * M.Smooth((BACK_T - ttc) / (BACK_T - SWING_T))).Unit();
            }
        }
        else if (I[Sim.PHASE] == Sim.PH_SERVE && I[Sim.SERVER] == j)
        {
            int k = CurrentServe(w, act, j);
            double top = Sim.VAR_TOP[(int)T[Sim.VPLAN0 + j]];
            var f = SERVE_FACE[k];
            var target = OwnToWorld(j, new V3(f.X, f.Y, f.Z - 0.6 * top)).Unit();
            double p = ServePhase(w, act, lag);
            face = (ready + (target - ready) * M.Smooth((p - 0.25) / 0.6)).Unit();
        }
        double tsh = Tm - lag - HitT[j];
        var raw = new V3(T[Sim.PN0 + 3 * j], T[Sim.PN0 + 3 * j + 1], T[Sim.PN0 + 3 * j + 2]);
        if (0.0 <= tsh && tsh < FOLLOW_END && raw.Norm() > 0.5)
        {
            var hit = raw.Unit();
            double u = tsh < FOLLOW_UP ? 0.0 : M.Smooth((tsh - FOLLOW_UP) / (FOLLOW_END - FOLLOW_UP));
            face = (hit + (ready - hit) * u).Unit();
        }
        return face;
    }

    (double Ttc, bool SideOn, bool Cross, double Vmax) GaitContext(World w, int j)
    {
        var I = w.I;
        var T = w.T;
        bool incoming = I[Sim.PHASE] == Sim.PH_FLY && I[Sim.TURN] == j && I[Sim.PVALID] == 1;
        double ttc = incoming ? T[Sim.PRT] - T[Sim.RT] : 9.0;
        bool sideOn = incoming && T[Sim.BT0 + j] > 0.0 && T[Sim.LM0 + j] * (T[Sim.PY] - w.Pos(j).Y) > 0.02;
        bool cross = T[Sim.CX0 + j] > 0.0;
        double vmax = World.VmaxOf(w.Stat(j, Sim.AGI), w.F[j]);
        return (ttc, sideOn, cross, vmax);
    }

    /// <summary>按步法分类选这一步怎么走。脚的编号：0 = 非持拍手一侧，1 = 持拍手一侧。</summary>
    (string, List<(int[], double, double)>) PlanGait(World w, int j, double[] err)
    {
        var (ttc, sideOn, cross, vmax) = GaitContext(w, j);
        double spd = w.Vel(j).Norm();
        int lead = err[1] > err[0] ? 1 : 0;
        int other = 1 - lead;
        double d = err[lead];
        var L = new List<(int[], double, double)>();
        if (d < STEP_ERR)
        {
            // 小碎步：快到击球时高频小垫步微调击球点
            if (ttc < 0.5 && d > 0.04)
            {
                L.Add((new[] { lead }, 0.08, 0.02));
                return ("小碎步", L);
            }
            return (null, L);
        }
        if (cross)
        {
            L.Add((new[] { lead }, 0.2, 0.08));
            L.Add((new[] { other }, 0.15, 0.05));
            return ("交叉步", L);
        }
        if (sideOn && d > 0.12)
        {
            L.Add((new[] { 0 }, 0.13, 0.05));
            L.Add((new[] { 1 }, 0.15, 0.05));
            return ("侧身步", L);
        }
        if (d < 0.3)
        {
            L.Add((new[] { lead }, 0.16, 0.06));
            return ("单步", L);
        }
        if (ttc < 0.35)
        {
            L.Add((new[] { lead }, 0.14, 0.07));
            L.Add((new[] { other }, 0.14, 0.04));
            return ("跨步", L);
        }
        if (spd > 0.7 * vmax && d > 0.45)
        {
            L.Add((new[] { 0, 1 }, 0.2, 0.09));
            return ("跳步", L);
        }
        L.Add((new[] { other }, 0.11, 0.03));
        L.Add((new[] { lead }, 0.13, 0.04));
        return ("并步", L);
    }

    /// <summary>脚踩实不动，离理想站位太远才按选好的步法迈步。返回 (两只脚的位置, 抬脚高度；跳步为负)。</summary>
    (V3[] Feet, double Lift) StepFeet(World w, int j, V2[] home, double dt)
    {
        if (_feet[j] == null || Math.Max((home[0] - _feet[j][0]).Norm(), (home[1] - _feet[j][1]).Norm()) > 0.9)
        {
            _feet[j] = new[] { home[0], home[1] };
            _steps[j] = null;
            _queue[j].Clear();
        }
        var fp = _feet[j];
        var lift = new double[2];
        var vel = w.Vel(j);
        if (_steps[j] == null && dt > 0.0)
        {
            if (_queue[j].Count == 0)
            {
                var err = new[] { (home[0] - fp[0]).Norm(), (home[1] - fp[1]).Norm() };
                var (name, plan) = PlanGait(w, j, err);
                if (name != null && _rescue[j] == null && (GaitOf[j].Name != name || Tm - GaitOf[j].T0 > 0.35))
                    GaitOf[j] = (name, Tm);
                _queue[j].AddRange(plan);
            }
            if (_queue[j].Count > 0)
            {
                var (ks, dur, lh) = _queue[j][0];
                _queue[j].RemoveAt(0);
                var bt = new Dictionary<int, V2>();
                foreach (int k in ks) bt[k] = home[k] + vel * dur;
                _steps[j] = new Step { Ks = ks, A = new[] { fp[0], fp[1] }, Bt = bt, U = 0.0, Dur = dur, Lift = lh };
            }
        }
        var st = _steps[j];
        if (st != null)
        {
            st.U += dt / st.Dur;
            double u = M.Smooth(st.U);
            for (int n = 0; n < st.Ks.Length; n++)
            {
                int k = st.Ks[n];
                st.Bt[k] = 0.7 * st.Bt[k] + 0.3 * (home[k] + vel * (st.Dur * (1.0 - Math.Min(st.U, 1.0))));
                // 跳步：离来球远的脚先落地
                double uk = st.Ks.Length == 1 ? u : M.Smooth(Math.Min(st.U * (n == 0 ? 1.15 : 0.95), 1.0));
                fp[k] = st.A[k] + (st.Bt[k] - st.A[k]) * uk;
                lift[k] = st.Lift * Math.Sin(Math.PI * Math.Min(st.U, 1.0));
            }
            if (st.U >= 1.0) _steps[j] = null;
        }
        bool hop = st != null && st.Ks.Length == 2;
        var feet = new[] { new V3(fp[0].X, fp[0].Y, 0.02 + lift[0]), new V3(fp[1].X, fp[1].Y, 0.02 + lift[1]) };
        return (feet, Math.Max(lift[0], lift[1]) * (hop ? -1.0 : 1.0));
    }

    /// <summary>救球/倒地爆冲：返回 (额外下蹲, 身体外倾方向与幅度（世界 xy）)。</summary>
    (double, V2) RescuePose(int j, double lag)
    {
        var r = _rescue[j];
        if (r == null)
        {
            var prep = _divePrep[j];
            if (prep == null) return (0.0, V2.Zero);
            var (side0, s, carry) = prep.Value;
            // 引拍时重心压在持拍手一侧的腿上（与之后倒下的方向相反）
            return (DIVE_PRE.Item1 * s + carry, -side0 * (DIVE_PRE.Item2 * s));
        }
        var (kind, t0, side) = r.Value;
        double tsh = Tm - lag - t0;
        if (kind == "倒地爆冲")
        {
            tsh = Math.Max(tsh, 0.0);
            if (tsh >= Sim.DIVE_T + DIVE_UP)
            {
                _rescue[j] = null;
                return (0.0, V2.Zero);
            }
            double rise = M.Smooth(tsh / 0.3);
            double fall = 1.0 - M.Smooth((tsh - Sim.DIVE_T) / DIVE_UP);
            double d0 = DIVE_PRE.Item1 + _diveSink[j];
            double depth = (d0 + (DIVE_DOWN.Item1 - d0) * rise) * fall;
            double lean = (-DIVE_PRE.Item2 + (DIVE_DOWN.Item2 + DIVE_PRE.Item2) * rise) * fall;
            return (depth, side * lean);
        }
        const double dur = 0.75;
        if (!(0.0 <= tsh && tsh < dur))
        {
            if (tsh >= dur) _rescue[j] = null;
            return (0.0, V2.Zero);
        }
        double rs = M.Smooth(tsh / 0.2);
        double fl = 1.0 - M.Smooth((tsh - 0.45 * dur) / (0.55 * dur));
        double wgt = rs * fl;
        return (0.13 * wgt, side * (0.2 * wgt));
    }

    /// <summary>倒地爆冲时整个人朝球拍跨出去的位移（世界 xy）。</summary>
    V2 DiveStep(World w, int j, V3 hand, double lag)
    {
        var r = _rescue[j];
        if (r != null && r.Value.Kind == "倒地爆冲")
        {
            double tsh = Math.Max(Tm - lag - r.Value.T0, 0.0);
            return _diveHold[j] * (1.0 - M.Smooth((tsh - Sim.DIVE_T) / DIVE_UP));
        }
        var prep = _divePrep[j];
        if (prep == null)
        {
            _diveHold[j] = V2.Zero;
            return _diveHold[j];
        }
        // 画面上的身体中心在模拟位置之后 0.32 米（与骨架一致）
        var p = w.Pos(j);
        var c = new V2(p.X + (j == 0 ? -1.0 : 1.0) * 0.32, p.Y);
        var d = new V2(hand.X, hand.Y) - c;
        double dn = d.Norm() + 1e-9;
        _diveHold[j] = d / dn * (Math.Min(Math.Max(dn - DIVE_REACH, 0.0), DIVE_LUNGE) * prep.Value.S);
        return _diveHold[j];
    }

    /// <summary>来球离得远、时间又紧：最后一步跨出去时重心压低。</summary>
    static double ApproachLunge(World w, int j, double lag)
    {
        var I = w.I;
        var T = w.T;
        if (!(I[Sim.PHASE] == Sim.PH_FLY && I[Sim.TURN] == j && I[Sim.PVALID] == 1)) return 0.0;
        double ttc = T[Sim.PRT] - T[Sim.RT] + lag;
        var p = w.Pos(j);
        double reach = Math.Sqrt(Math.Pow(T[Sim.PX] - p.X, 2) + Math.Pow(T[Sim.PY] - p.Y, 2));
        if (ttc > 0.35 || reach < 0.75) return 0.0;
        return 0.12 * Math.Min((reach - 0.75) / 0.35, 1.0) * M.Smooth((0.35 - ttc) / 0.25);
    }

    /// <summary>发球手法由模拟里的发球战术决定（USE_TACTIC）。</summary>
    public static int CurrentServe(World w, double[] act, int j) => (int)w.T[Sim.SPLAN0 + j];

    static double ServeTossValue(World w, double[] act)
    {
        var T = w.T;
        int srv = (int)w.I[Sim.SERVER];
        if (T[Sim.TOSS] >= 0.0) return T[Sim.TOSS];
        return World.ServeToss((int)T[Sim.SPLAN0 + srv], ActC(act, srv, 9));
    }

    /// <summary>发球进度：抛球前为 0，抛球离手到击球从 0 到 1。</summary>
    public double ServePhase(World w, double[] act, double lag)
    {
        double d = World.TossTime(ServeTossValue(w, act));
        double tr = w.T[Sim.TIMER] + lag;
        return M.Clamp((d - tr) / d, 0.0, 1.0);
    }

    /// <summary>发球：球先托在掌心，按抛球高度抛起，下落到原击球高度时出手。返回 (显示的球, 抛球手, 发球方) 或 null。</summary>
    (V3 Ball, V3 Hand, int Srv)? Toss(World w, double[] act, V3 ball, double lag)
    {
        var I = w.I;
        var T = w.T;
        if (I[Sim.PHASE] != Sim.PH_SERVE) return null;
        int srv = (int)I[Sim.SERVER];
        double d = World.TossTime(ServeTossValue(w, act));
        double tr = T[Sim.TIMER] + lag;
        var b = ball;
        if (tr > d) return (b, new V3(b.X, b.Y, b.Z - 0.03), srv);
        double tau = d - tr;
        double vup = Sim.G * d / 2.0;
        b.Z = ball.Z + vup * tau - 0.5 * Sim.G * tau * tau;
        double rel = Math.Min(tau / 0.12, 1.0);
        // 手只跟着球抬到出手高度（约 +16cm），球继续往上升时手停在原处
        double lift = M.Clamp(b.Z - ball.Z, 0.0, 0.16);
        var hand = new V3(ball.X, ball.Y, ball.Z - 0.03 + lift * rel);
        // 抛球出手后左手迅速离开击球点，不挡球、不挡挥拍路线
        double clear = M.Smooth((tau - 0.1) / 0.25);
        var away = TOSS_CLEAR.TryGetValue(CurrentServe(w, act, srv), out var a) ? a : TOSS_CLEAR_DEFAULT;
        hand += clear * OwnToWorld(srv, away);
        return (b, hand, srv);
    }

    /// <summary>发球站位：返回 (转体, 站位侧向, 下蹲)。发完球约 0.4 秒内转回正面。</summary>
    (double, double, double) ServeStance(World w, double[] act, int j, double lag)
    {
        var I = w.I;
        bool serving = I[Sim.PHASE] == Sim.PH_SERVE && I[Sim.SERVER] == j;
        int k;
        double wt, extra;
        if (serving)
        {
            k = CurrentServe(w, act, j);
            wt = M.Smooth((Tm - lag - _serveT0) / 0.5);
            extra = SquatExtra(k) * M.Smooth((ServePhase(w, act, lag) - 0.6) / 0.4);
        }
        else
        {
            double tsh = Tm - lag - HitT[j];
            k = _serveKind[j];
            double rec = k == Sim.SV_SQUAT ? 0.7 : 0.4;
            if (HitKind[j].Kind != -1 || !(0.0 <= tsh && tsh < rec)) return (0.0, 0.0, 0.0);
            wt = 1.0 - M.Smooth(tsh / rec);
            extra = SquatExtra(k) * wt;
        }
        _serveKind[j] = k;
        var s = SERVE_STANCE[k];
        return (s.X * wt, s.Y * wt, s.Z * wt + extra);
    }

    /// <summary>持拍臂肘尖朝向（世界坐标）：发球时按手法，随挥收回时渐渐回到平时的朝向。</summary>
    V3 ElbowPole(World w, double[] act, int j, double lag)
    {
        var I = w.I;
        int k;
        double wt;
        if (I[Sim.PHASE] == Sim.PH_SERVE && I[Sim.SERVER] == j)
        {
            k = CurrentServe(w, act, j);
            wt = Math.Max(M.Smooth(ServePhase(w, act, lag) / 0.4), 0.5);
        }
        else
        {
            double tsh = Tm - lag - HitT[j];
            if (HitKind[j].Kind != -1 || !(0.0 <= tsh && tsh < SERVE_FOLLOW_END))
            {
                var arm = _arm[j];
                if (arm != null) return OwnToWorld(j, V3.Lerp(ELBOW_POLE, arm.Value.Pole, arm.Value.W));
                return OwnToWorld(j, ELBOW_POLE);
            }
            k = _serveKind[j];
            wt = 1.0 - M.Smooth((tsh - SERVE_FOLLOW_HOLD) / (SERVE_FOLLOW_END - SERVE_FOLLOW_HOLD));
        }
        var se = SERVE_ELBOW.TryGetValue(k, out var e) ? e : ELBOW_POLE;
        return OwnToWorld(j, V3.Lerp(ELBOW_POLE, se, wt));
    }

    /// <summary>每帧调用：返回显示用的球、球拍位置和每名球员的姿态。homeFn(j, pose, hand) 给出该姿态下两只脚的理想落点。</summary>
    public (V3 Ball, V3[] Hands, Pose[] Poses) Frame(World w, double[] act, V3 ball, V3[] hands, double[] blends,
        Func<int, Pose, V3, V2[]> homeFn, double dt, double lag)
    {
        Wall += dt;
        var ballDisp = ball;
        var poses = new Pose[2];
        var I = w.I;
        for (int j = 0; j < 2; j++)
            _hs[j] = w.T[Sim.LM0 + j] * (j == 0 ? 1.0 : -1.0) < 0.0 ? -1.0 : 1.0;
        var tossed = Toss(w, act, ball, lag);
        // 新一分刚开始的一两帧，画面用的那一帧球还停在上一分的落点：球离持拍手太远就先不摆托球
        if (tossed != null)
        {
            var tv = tossed.Value;
            var hd = hands[tv.Srv];
            if (Math.Sqrt(Math.Pow(tv.Hand.X - hd.X, 2) + Math.Pow(tv.Hand.Y - hd.Y, 2)) > 0.8) tossed = null;
        }
        if (tossed != null) ballDisp = tossed.Value.Ball;
        var handsDisp = new V3[2];
        for (int j = 0; j < 2; j++)
        {
            var (off, twist, sink) = Swing(w, act, j, blends[j], lag);
            var bas = hands[j];
            bool serving = I[Sim.PHASE] == Sim.PH_SERVE && I[Sim.SERVER] == j;
            if (serving) _serveHand[j] = bas;
            // 模型给的等球手位常举到头边：除了自己发球，持拍手不高过胸口；接发球一方端在准备高度
            bool recv = (I[Sim.PHASE] == Sim.PH_SERVE && I[Sim.SERVER] != j) ||
                        (I[Sim.PHASE] == Sim.PH_FLY && I[Sim.TURN] == j && HitKind[1 - j].Kind == -1);
            double kb = Math.Min(dt / RECV_BLEND_T * 3.0, 1.0);
            _capW[j] += ((serving ? 0.0 : 1.0) - _capW[j]) * kb;
            _recvW[j] += ((recv ? 1.0 : 0.0) - _recvW[j]) * kb;
            double wc = M.Smooth(_capW[j]);
            if (wc > 0.0 && bas.Z > READY_MAX_Z) bas.Z += (READY_MAX_Z - bas.Z) * wc;
            double wr = M.Smooth(_recvW[j]);
            if (wr > 0.0) bas.Z += (RECV_READY_Z - bas.Z) * wr;
            double tsh = Tm - lag - HitT[j];
            if (HitKind[j].Kind == -1 && _serveHand[j] != null && 0.0 <= tsh && tsh < SERVE_FOLLOW_END)
            {
                double u = M.Smooth((tsh - SERVE_FOLLOW_HOLD) / (SERVE_FOLLOW_END - SERVE_FOLLOW_HOLD));
                bas = _serveHand[j].Value + (bas - _serveHand[j].Value) * u;
            }
            else if (_anchor[j] != null)
            {
                var (p, wt) = _anchor[j].Value;
                bas += (p - bas) * wt;
            }
            var hand = bas + off;
            handsDisp[j] = hand;
            var (yaw, stance, crouch) = ServeStance(w, act, j, lag);
            double spd = w.Vel(j).Norm();
            double ready = READY_CROUCH * (1.0 - 0.5 * Math.Min(spd / 1.5, 1.0));
            var (dive, tilt) = RescuePose(j, lag);
            sink += dive + ApproachLunge(w, j, lag);
            var pose = new Pose
            {
                Twist = twist + yaw, Stance = stance, Crouch = Math.Max(ready + crouch + sink, 0.0), Bob = 0.0,
                Face = RacketFace(w, act, j, lag), Epole = ElbowPole(w, act, j, lag), Tilt = tilt,
                Lunge = DiveStep(w, j, hand, lag), Gait = GaitOf[j], T = Tm - lag,
            };
            if (tossed != null && tossed.Value.Srv == j) pose.LHand = tossed.Value.Hand;
            var (feet, lift) = StepFeet(w, j, homeFn(j, pose, hand), dt);
            // 迈步时身体随抬脚略降；跳步双脚离地，身体整体腾起
            pose.LFoot = feet[0];
            pose.RFoot = feet[1];
            pose.Bob = lift < 0.0 ? 0.5 * -lift : -0.25 * lift;
            poses[j] = pose;
        }
        return (ballDisp, handsDisp, poses);
    }
}
