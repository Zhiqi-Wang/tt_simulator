using System;
using System.Collections.Generic;
using Godot;

namespace PingPong;

/// <summary>音效（core/bake_sounds.py 预先渲染好的变体）：击球按胶皮和挥拍力度、落台、落地、擦网、掌声。
/// 每种声音随机挑一个变体；用 3D 声源放在发声位置，左右声道跟着镜头。</summary>
public partial class Sfx : Node3D
{
    public bool On = true;
    public bool Ok;
    readonly Dictionary<(int, int), List<AudioStream>> _hit = new();
    List<AudioStream> _table = new(), _floor = new(), _net = new();
    AudioStream _applause;
    readonly List<AudioStreamPlayer3D> _pool = new();
    readonly Random _rng = new();
    int _next;

    static List<AudioStream> LoadAll(string prefix)
    {
        var l = new List<AudioStream>();
        for (int k = 0; k < 16; k++)
        {
            var p = $"res://assets/sounds/{prefix}{k}.wav";
            if (!ResourceLoader.Exists(p)) break;
            l.Add(GD.Load<AudioStream>(p));
        }
        return l;
    }

    public override void _Ready()
    {
        for (int rub = 0; rub < Sim.N_RUB; rub++)
            for (int lvl = 0; lvl < 3; lvl++)
                _hit[(rub, lvl)] = LoadAll($"hit_{rub}_{lvl}_");
        _table = LoadAll("table_");
        _floor = LoadAll("floor_");
        _net = LoadAll("net_");
        if (ResourceLoader.Exists("res://assets/sounds/applause.wav"))
            _applause = GD.Load<AudioStream>("res://assets/sounds/applause.wav");
        for (int k = 0; k < 24; k++)
        {
            var p = new AudioStreamPlayer3D
            {
                AttenuationModel = AudioStreamPlayer3D.AttenuationModelEnum.Disabled,
                PanningStrength = 0.7f,
                MaxPolyphony = 1,
            };
            AddChild(p);
            _pool.Add(p);
        }
        Ok = _hit[(0, 0)].Count > 0 && _table.Count > 0;
    }

    public void Play(List<AudioStream> set, double vol, V3 at)
    {
        if (set == null || set.Count == 0) return;
        Play(set[_rng.Next(set.Count)], vol, at);
    }

    public void Play(AudioStream s, double vol, V3 at)
    {
        if (!(Ok && On) || s == null) return;
        vol = Math.Clamp(vol, 0.0, 1.0);
        if (vol <= 1e-3) return;
        var p = _pool[_next];
        _next = (_next + 1) % _pool.Count;
        p.Stream = s;
        p.Position = at.G;
        p.VolumeDb = Mathf.LinearToDb((float)vol);
        p.Play();
    }

    /// <summary>swing：挥拍速度 m/s；rubber：胶皮编号。</summary>
    public void Racket(double swing, int rubber, V3 at)
    {
        int lvl = swing < 8 ? 0 : swing < 15 ? 1 : 2;
        double loud = 0.45 + 0.55 * Math.Min(swing / 22.0, 1.0);
        if (_hit.TryGetValue((rubber, lvl), out var set)) Play(set, loud, at);
    }

    public void Table(double vol, V3 at) => Play(_table, vol, at);
    public void Floor(double vol, V3 at) => Play(_floor, vol, at);
    public void Net(double vol, V3 at) => Play(_net, vol, at);
    public void Applause(double vol, V3 at) => Play(_applause, vol, at);
}
