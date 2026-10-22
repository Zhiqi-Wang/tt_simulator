using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;

namespace PingPong;

/// <summary>动作/骨架移植的逐帧对照：回放 core/anim_dump.py 录下的模拟状态，比较每帧关节位置与 Python 版的差。
/// 用法：godot --path . -- --animtest _animtest.json</summary>
public partial class Main
{
    void AnimTest(string path)
    {
        var doc = JsonDocument.Parse(File.ReadAllText(path));
        var root = doc.RootElement;
        double dt = root.GetProperty("dt").GetDouble();
        var keys = root.GetProperty("keys").EnumerateArray().Select(e => e.GetString()).ToArray();
        LoadState(root.GetProperty("init"));
        var sb = new StringBuilder();
        double worst = 0.0, worstBall = 0.0;
        var perKey = new double[keys.Length];
        int firstBad = -1, nf = 0;
        foreach (var fr in root.GetProperty("frames").EnumerateArray())
        {
            foreach (var st in fr.GetProperty("steps").EnumerateArray())
            {
                int prevPhase = (int)Env.I[Sim.PHASE], prevHits = (int)Env.I[Sim.HITS];
                LoadState(st);
                Fill(st.GetProperty("act"), _act);
                _anim.OnStep(Env, prevPhase, prevHits, Env.Dt);
            }
            _acc = fr.GetProperty("acc").GetDouble();
            // Python 对照脚本每帧调一次 body_points：避让记忆按一次更新换算
            Compute(dt, 1.0 / 180.0);
            var ball = fr.GetProperty("ball").EnumerateArray().Select(e => e.GetDouble()).ToArray();
            worstBall = Math.Max(worstBall, (new V3(ball[0], ball[1], ball[2]) - _ballShown).Norm());
            var bodies = fr.GetProperty("bodies");
            for (int j = 0; j < 2; j++)
            {
                var b = Bodies[j];
                V3[] mine =
                {
                    b.Hip, b.LHip, b.RHip, b.Neck, b.Head, b.RSh, b.LSh, b.LHand, b.RElbow, b.LElbow, b.LKnee, b.RKnee,
                    b.LFoot, b.RFoot, b.Paddle,
                };
                int k = 0;
                foreach (var pt in bodies[j].EnumerateArray())
                {
                    var a = pt.EnumerateArray().Select(e => e.GetDouble()).ToArray();
                    double err = (new V3(a[0], a[1], a[2]) - mine[k]).Norm();
                    perKey[k] = Math.Max(perKey[k], err);
                    worst = Math.Max(worst, err);
                    if (err > 1e-6 && firstBad < 0)
                    {
                        firstBad = nf;
                        sb.AppendLine($"first mismatch frame {nf} player {j} joint {keys[k]}: py ({a[0]:F5},{a[1]:F5},{a[2]:F5}) cs {mine[k]} err {err:E2}");
                    }
                    k++;
                }
            }
            nf++;
        }
        sb.AppendLine($"frames {nf} worst joint err {worst:E2} worst ball err {worstBall:E2}");
        for (int k = 0; k < keys.Length; k++) sb.AppendLine($"  {keys[k],-7} {perKey[k]:E2}");
        File.WriteAllText(path.Replace(".json", ".result.txt"), sb.ToString());
    }

    static void Fill(JsonElement arr, double[] dst)
    {
        int k = 0;
        foreach (var e in arr.EnumerateArray()) dst[k++] = e.GetDouble();
    }

    void LoadState(JsonElement st)
    {
        Fill(st.GetProperty("B"), Env.B);
        Fill(st.GetProperty("P"), Env.P);
        Fill(st.GetProperty("T"), Env.T);
        Fill(st.GetProperty("S"), Env.S);
        Fill(st.GetProperty("F"), Env.F);
        Fill(st.GetProperty("hist"), Env.Hist);
        int k = 0;
        foreach (var e in st.GetProperty("I").EnumerateArray()) Env.I[k++] = e.GetInt64();
    }
}
