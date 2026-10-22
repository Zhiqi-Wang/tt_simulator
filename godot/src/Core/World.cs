using System;

namespace PingPong;

/// <summary>一张球台的模拟状态（数组布局与 Python 版 PongEnv 的第 0 张台相同），物理全部交给 ppcore.dll。</summary>
public unsafe class World : IDisposable
{
    public readonly int NSub;
    public readonly double Dt;
    public readonly bool Train;

    public readonly double[] B = new double[9];
    public readonly double[] P = new double[2 * 6];
    public readonly long[] I = new long[Sim.NI];
    public readonly double[] T = new double[Sim.NT];
    public readonly double[] S = new double[2 * Sim.N_S];
    public readonly double[] F = new double[2];
    public readonly double[] O = new double[2 * Sim.OBS_DIM];
    public readonly double[] Rew = new double[2];
    public readonly double[] Hist;
    public bool Done;
    public long Winner = -1, Reason, Hits;

    IntPtr _h;
    readonly byte[] _done = new byte[1];
    readonly long[] _winner = new long[1], _reason = new long[1], _hits = new long[1];

    public World(ulong seed, bool train = false, int nsub = 8, double dt = 1.0 / 30.0)
    {
        Native.Init();
        CheckDims();
        NSub = nsub;
        Dt = dt;
        Train = train;
        Hist = new double[nsub * Sim.HIST_DIM];
        _h = Native.pp_create(1, seed);
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < Sim.N_S; k++) S[j * Sim.N_S + k] = 0.6;
            for (int k = Sim.N_STATS; k < Sim.N_STATS + Sim.N_STYLE; k++) S[j * Sim.N_S + k] = 0.0;
            S[j * Sim.N_S + Sim.HAND] = 1.0;
            S[j * Sim.N_S + Sim.GRIP] = 0.0;
            S[j * Sim.N_S + Sim.RFH] = Sim.RUB_INV;
            S[j * Sim.N_S + Sim.RBH] = Sim.RUB_INV;
        }
    }

    static void CheckDims()
    {
        var d = new long[6];
        fixed (long* p = d) Native.pp_dims(p);
        if (d[0] != Sim.NI || d[1] != Sim.NT || d[2] != Sim.N_S || d[3] != Sim.OBS_DIM || d[4] != Sim.ACT_DIM ||
            d[5] != Sim.HIST_DIM)
            throw new Exception($"ppcore.dll 与 Sim.cs 的数组布局不一致: {string.Join(",", d)}");
    }

    public void Dispose()
    {
        if (_h != IntPtr.Zero) Native.pp_destroy(_h);
        _h = IntPtr.Zero;
    }

    ~World() => Dispose();

    public double Stat(int j, int k) => S[j * Sim.N_S + k];
    public void SetStat(int j, int k, double v) => S[j * Sim.N_S + k] = v;
    public V3 Ball => new(B[0], B[1], B[2]);
    public V3 BallV => new(B[3], B[4], B[5]);
    public V3 BallW => new(B[6], B[7], B[8]);
    public V3 Pos(int j) => new(P[6 * j], P[6 * j + 1], P[6 * j + 2]);
    public V2 Vel(int j) => new(P[6 * j + 3], P[6 * j + 4]);
    public int Phase => (int)I[Sim.PHASE];
    public double H(int s, int k) => Hist[s * Sim.HIST_DIM + k];
    public V3 HV(int s, int k) => new(Hist[s * Sim.HIST_DIM + k], Hist[s * Sim.HIST_DIM + k + 1],
        Hist[s * Sim.HIST_DIM + k + 2]);

    public double[] Reset()
    {
        fixed (double* b = B, p = P, t = T, s = S, f = F)
        fixed (long* i = I)
            Native.pp_reset(_h, b, p, i, t, s, f, Train ? 1 : 0);
        return FillHist();
    }

    /// <summary>比分清零、指定发球方，重新开始这一分（练习模式用）。</summary>
    public double[] RestartPoint(int server)
    {
        I[Sim.SC0] = 0;
        I[Sim.SC1] = 0;
        I[Sim.FIRST] = server;
        fixed (double* b = B, p = P, t = T, s = S, f = F)
        fixed (long* i = I)
            Native.pp_start_point(_h, 0, b, p, i, t, s, f, Train ? 1 : 0);
        return FillHist();
    }

    public void RandomEquipment(int j, bool balanced = false)
    {
        fixed (double* s = S) Native.pp_random_equipment(_h, 0, j, s, balanced ? 1 : 0);
    }

    double[] FillHist()
    {
        for (int s = 0; s < NSub; s++)
        {
            int o = s * Sim.HIST_DIM;
            for (int k = 0; k < 3; k++)
            {
                Hist[o + k] = B[k];
                Hist[o + 3 + k] = P[k];
                Hist[o + 6 + k] = P[6 + k];
            }
            for (int j = 0; j < 2; j++)
            {
                for (int k = 0; k < 3; k++) Hist[o + 9 + 3 * j + k] = P[6 * j + k];
                Hist[o + 10 + 3 * j] += T[Sim.LM0 + j] * HandOff(T[Sim.BL0 + j]);
                Hist[o + 15 + j] = T[Sim.BL0 + j];
            }
            Hist[o + 23] = -1.0;
        }
        return Observe();
    }

    public double[] Observe()
    {
        fixed (double* b = B, p = P, t = T, s = S, f = F, o = O)
        fixed (long* i = I)
            Native.pp_observe(_h, b, p, i, t, s, f, o);
        return O;
    }

    /// <summary>A：2×ACT_DIM 的动作。走一个模拟步（1/30 秒，内部 NSub 个子步，子步轨迹写进 Hist）。</summary>
    public double[] Step(double[] A)
    {
        fixed (double* b = B, p = P, t = T, s = S, f = F, a = A, r = Rew, h = Hist)
        fixed (long* i = I, w = _winner, rs = _reason, ht = _hits)
        fixed (byte* d = _done)
            Native.pp_step(_h, b, p, i, t, s, f, a, r, d, w, rs, ht, h, Train ? 1 : 0, NSub, Dt);
        Done = _done[0] != 0;
        Winner = _winner[0];
        Reason = _reason[0];
        Hits = _hits[0];
        return Observe();
    }

    public double Pressure(int who)
    {
        fixed (long* i = I)
        fixed (double* t = T)
            return Native.pp_pressure(_h, 0, who, i, t);
    }

    // 模拟里的几条规则（画面层预判动作用）
    public static double HandOff(double blend) => Native.pp_hand_off(blend);
    public static double TossTime(double toss) => Native.pp_toss_time(toss);
    public static double ServeToss(int k, double a9) => Native.pp_serve_toss(k, a9);
    public static double VmaxOf(double agi, double fat) => Native.pp_vmax_of(agi, fat);
    public static int ServeTypeOf(double a8, double a10) => Native.pp_serve_type_of(a8, a10);

    public static int ClassifyStroke(V3 b, double pw, double top)
    {
        double* p = stackalloc double[3] { b.X, b.Y, b.Z };
        return Native.pp_classify_stroke(p, pw, top);
    }

    public static int TechOf(int k, bool bh, double pw, double top, double side, double depth, double topIn) =>
        Native.pp_tech_of(k, bh ? 1 : 0, pw, top, side, depth, topIn);

    public static double IncomingTop(V3 v, V3 w)
    {
        double* a = stackalloc double[3] { v.X, v.Y, v.Z };
        double* b = stackalloc double[3] { w.X, w.Y, w.Z };
        return Native.pp_incoming_top(a, b);
    }
}
