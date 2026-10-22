using System;
using System.IO;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace PingPong;

/// <summary>训练好的决策网络（core/export_policy.py 导出的二进制）：观测归一化 + tanh 隐层 MLP，输出确定性动作。</summary>
public class Policy
{
    public int ObsDim, ActDim, Iteration;
    readonly int[] _sizes;
    readonly double[] _mean, _std;
    readonly float[][] _w, _b;
    readonly float[][] _buf;

    Policy(int[] sizes, double[] mean, double[] var, float[][] w, float[][] b, int iteration)
    {
        _sizes = sizes;
        ObsDim = sizes[0];
        ActDim = sizes[^1];
        Iteration = iteration;
        _mean = mean;
        _std = new double[mean.Length];
        for (int k = 0; k < mean.Length; k++) _std[k] = Math.Sqrt(var[k] + 1e-8);
        _w = w;
        _b = b;
        _buf = new float[sizes.Length][];
        for (int k = 0; k < sizes.Length; k++) _buf[k] = new float[sizes[k]];
    }

    public static Policy Load(byte[] data)
    {
        using var r = new BinaryReader(new MemoryStream(data));
        if (Encoding.ASCII.GetString(r.ReadBytes(4)) != "PPOL") throw new Exception("不是策略文件");
        int ver = r.ReadInt32();
        if (ver != 2) throw new Exception($"策略文件版本 {ver} 不支持");
        int obs = r.ReadInt32(), act = r.ReadInt32(), L = r.ReadInt32(), iter = r.ReadInt32();
        var sizes = new int[L + 1];
        for (int k = 0; k <= L; k++) sizes[k] = r.ReadInt32();
        if (sizes[0] != obs || sizes[L] != act) throw new Exception("策略文件层宽不一致");
        var mean = new double[obs];
        var var = new double[obs];
        for (int k = 0; k < obs; k++) mean[k] = r.ReadDouble();
        for (int k = 0; k < obs; k++) var[k] = r.ReadDouble();
        var w = new float[L][];
        var b = new float[L][];
        for (int l = 0; l < L; l++)
        {
            w[l] = ReadFloats(r, sizes[l] * sizes[l + 1]);
            b[l] = ReadFloats(r, sizes[l + 1]);
        }
        return new Policy(sizes, mean, var, w, b, iter);
    }

    static float[] ReadFloats(BinaryReader r, int n)
    {
        var bytes = r.ReadBytes(n * 4);
        var a = new float[n];
        Buffer.BlockCopy(bytes, 0, a, 0, bytes.Length);
        return a;
    }

    /// <summary>obs 是一名球员的观测（可以比模型的输入长：新加的观测维度都在末尾，旧模型截掉）；结果写进 act。</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void Act(ReadOnlySpan<double> obs, Span<double> act)
    {
        var x = _buf[0];
        for (int k = 0; k < ObsDim; k++)
            x[k] = (float)Math.Clamp((obs[k] - _mean[k]) / _std[k], -10.0, 10.0);
        int L = _w.Length;
        for (int l = 0; l < L; l++)
        {
            var inp = _buf[l];
            var outp = _buf[l + 1];
            int nin = _sizes[l], nout = _sizes[l + 1];
            Array.Copy(_b[l], outp, nout);
            var W = _w[l];
            int vw = Vector<float>.Count;
            int nv = nout / vw;
            var ov = MemoryMarshal.Cast<float, Vector<float>>(outp.AsSpan(0, nv * vw));
            // 行优先 [in×out]：逐个输入把那一行权重乘上去累加
            for (int i = 0; i < nin; i++)
            {
                float xi = inp[i];
                if (xi == 0f) continue;
                int row = i * nout;
                var vx = new Vector<float>(xi);
                if (row % vw == 0)
                {
                    var wv = MemoryMarshal.Cast<float, Vector<float>>(W.AsSpan(row, nv * vw));
                    for (int o = 0; o < nv; o++) ov[o] += vx * wv[o];
                }
                else
                    for (int o = 0; o < nv * vw; o++) outp[o] += xi * W[row + o];
                for (int o = nv * vw; o < nout; o++) outp[o] += xi * W[row + o];
            }
            if (l < L - 1)
                for (int o = 0; o < nout; o++)
                    outp[o] = MathF.Tanh(outp[o]);
        }
        var y = _buf[L];
        for (int k = 0; k < ActDim; k++) act[k] = y[k];
    }
}
