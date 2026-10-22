using System;

namespace PingPong;

/// <summary>脚本 AI（pingpong/bot.py 的移植）：来球在身体右侧用正手、左侧到身前用反手，移动让击球点落进舒适区，
/// 随机选落点和打法。没有模型时的默认控制器。</summary>
public static class Bot
{
    const double MARGIN = 0.04;
    static readonly double[] READY = { -(Sim.HALF_L + 0.6), 0.0, 1.0 };

    static double U(Random r, double lo, double hi) => lo + (hi - lo) * r.NextDouble();

    public static void Act(ReadOnlySpan<double> o, Span<double> a, Random rng)
    {
        a.Clear();
        bool vis = o[21] > 0.5;
        double ty = o[41];
        bool fh = ty < -0.07;
        a[8] = vis && !fh ? -1.0 : 1.0;
        double lo = (fh ? Sim.FH_CLO : Sim.BH_CLO) + MARGIN;
        double hi = (fh ? Sim.FH_CHI : Sim.BH_CHI) - MARGIN;
        double tx = o[42];
        Span<double> rel = stackalloc double[3] { o[22], o[23], o[24] };
        rel[0] = tx - Math.Clamp(tx, Sim.FWD_CLO + MARGIN, Sim.FWD_CHI - MARGIN);
        rel[1] = ty - Math.Clamp(ty, lo, hi);
        if (!vis)
            for (int k = 0; k < 3; k++) rel[k] = READY[k] - o[k];
        for (int k = 0; k < 3; k++) a[k] = Math.Clamp(rel[k] * 2.5, -1.0, 1.0);

        double strength = o[26], tech = o[28];
        a[3] = U(rng, -0.7, 0.5);
        a[4] = U(rng, -0.8, 0.8);
        double bas = -0.6 + 0.6 * tech + 0.2 * strength;
        double style = rng.NextDouble();
        bool loop = style < 0.45, drive = style >= 0.45 && style < 0.75, chop = style >= 0.75 && style < 0.92;
        a[5] = drive ? bas + 0.25 : chop ? bas - 0.4 : bas;
        a[6] = loop ? U(rng, 0.5, 1.0) : drive ? U(rng, 0.1, 0.4) : chop ? U(rng, -1.0, -0.4) : U(rng, 0.0, 0.4);
        a[7] = rng.NextDouble() < 0.35 ? U(rng, -0.8, 0.8) : 0.0;

        bool oppCue = o[47] > 0.5;
        a[9] = oppCue ? Math.Clamp((o[48] - o[1]) * 3.0, -1.0, 1.0) : 0.0;
        bool fake = rng.NextDouble() < 0.25;
        a[10] = fake ? -a[4] : a[4];
        a[11] = vis && Math.Abs(rel[1]) > 0.55 ? 1.0 : -1.0;

        bool serving = o[20] > 0.5 && o[38] > 0.5;
        if (serving)
            foreach (int k in new[] { 1, 3, 4, 5, 6, 7, 8, 9, 10, 11 })
                a[k] = U(rng, -1.0, 1.0);
    }
}
