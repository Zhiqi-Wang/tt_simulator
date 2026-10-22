using System;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using Godot;

namespace PingPong;

/// <summary>C++ 物理核心 ppcore.dll（core/ppcore.cpp，与训练用的同一份源码）的接口。</summary>
public static unsafe class Native
{
    const string Lib = "ppcore";
    static bool _init;

    /// <summary>编辑器里从 res://bin 加载；导出后 DLL 放在可执行文件旁边。</summary>
    public static void Init()
    {
        if (_init) return;
        _init = true;
        NativeLibrary.SetDllImportResolver(typeof(Native).Assembly, Resolve);
    }

    static IntPtr Resolve(string name, Assembly asm, DllImportSearchPath? path)
    {
        if (name != Lib) return IntPtr.Zero;
        string[] dirs =
        {
            Path.GetDirectoryName(OS.GetExecutablePath()) ?? "",
            ProjectSettings.GlobalizePath("res://bin"),
            AppContext.BaseDirectory,
        };
        foreach (var d in dirs)
        {
            if (string.IsNullOrEmpty(d)) continue;
            var f = Path.Combine(d, "ppcore.dll");
            if (File.Exists(f) && NativeLibrary.TryLoad(f, out var h)) return h;
        }
        // 导出的游戏里 DLL 打在 pck 里：解到用户目录再加载
        const string packed = "res://bin/ppcore.dll";
        if (Godot.FileAccess.FileExists(packed))
        {
            var bytes = Godot.FileAccess.GetFileAsBytes(packed);
            var f = Path.Combine(OS.GetUserDataDir(), "ppcore.dll");
            if (!File.Exists(f) || new FileInfo(f).Length != bytes.Length) File.WriteAllBytes(f, bytes);
            if (NativeLibrary.TryLoad(f, out var h)) return h;
        }
        return IntPtr.Zero;
    }

    [DllImport(Lib)] public static extern int pp_version();
    [DllImport(Lib)] public static extern void pp_dims(long* outDims);
    [DllImport(Lib)] public static extern IntPtr pp_create(int n, ulong seed);
    [DllImport(Lib)] public static extern void pp_destroy(IntPtr h);
    [DllImport(Lib)] public static extern void pp_seed(IntPtr h, ulong seed);
    [DllImport(Lib)] public static extern void pp_set_param(int which, double* v);

    [DllImport(Lib)]
    public static extern void pp_step(IntPtr h, double* B, double* P, long* I, double* T, double* S, double* F,
        double* A, double* rew, byte* done, long* winner, long* reason, long* hits, double* hist,
        int train, int nsub, double dt);

    [DllImport(Lib)]
    public static extern void pp_reset(IntPtr h, double* B, double* P, long* I, double* T, double* S, double* F,
        int train);

    [DllImport(Lib)]
    public static extern void pp_start_point(IntPtr h, int i, double* B, double* P, long* I, double* T, double* S,
        double* F, int train);

    [DllImport(Lib)] public static extern void pp_random_equipment(IntPtr h, int i, int j, double* S, int balanced);

    [DllImport(Lib)]
    public static extern void pp_observe(IntPtr h, double* B, double* P, long* I, double* T, double* S, double* F,
        double* O);

    [DllImport(Lib)] public static extern int pp_serve_type_of(double a8, double a10);
    [DllImport(Lib)] public static extern int pp_classify_stroke(double* b, double pw, double top);

    [DllImport(Lib)]
    public static extern int pp_tech_of(int k, int bh, double pw, double top, double side, double depth,
        double topIn);

    [DllImport(Lib)] public static extern double pp_incoming_top(double* v, double* w);
    [DllImport(Lib)] public static extern double pp_serve_toss(int k, double a9);
    [DllImport(Lib)] public static extern double pp_toss_time(double toss);
    [DllImport(Lib)] public static extern double pp_vmax_of(double agi, double fat);
    [DllImport(Lib)] public static extern double pp_hand_off(double blend);
    [DllImport(Lib)] public static extern double pp_pressure(IntPtr h, int i, int who, long* I, double* T);
}
