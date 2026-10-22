using System;
using System.Collections.Generic;

namespace PingPong;

/// <summary>一名球员这一帧的关节位置（模拟坐标）。</summary>
public class Body
{
    public V3 Hip, LHip, RHip, Neck, Head, RSh, LSh, LHand, RElbow, LElbow, LKnee, RKnee, LFoot, RFoot, Paddle;
}

/// <summary>骨架（play.py 里 body_points 的移植）：手伸得比胳膊长时上身跟着前倾去够；球拍探进台内时持拍侧的脚上步；
/// 手臂两段 IK，肘和手不穿进躯干、不碰头、不穿台面，并记住上一帧的避让结果，避免在两种解之间来回闪。</summary>
public class Skeleton
{
    public const double ARM_LEN = 0.6, TORSO_BEND = 0.32, HIP_W = 0.12, SEG_LEN = 0.46, FOOT_R = 0.5;
    public const double UPPER_ARM = 0.3, FORE_ARM = 0.34;
    // 避让的帧间记忆（按 60 帧/秒、每帧更新 3 次标定）：肘向偏离每次保留的比例；手的避让偏移每次追上新值的比例
    const double POLE_KEEP = 0.85, HAND_EASE = 0.3, HOLD_MARGIN = 0.015, PUSH_STICK = 0.04;
    const double MEM_RATE = 180.0;

    static readonly double[] US_R = { 0.0, 0.33, 0.66 }, PADS_R = { 0.025, 0.03, 0.03, 0.03, 0.03 };
    static readonly double[] US_L = { 0.0, 0.5 }, PADS_L = { 0.025, 0.03, 0.03, 0.03 };

    class Mem
    {
        public readonly Dictionary<string, char> Push = new();
        public readonly Dictionary<string, V3> Pole = new();
        public readonly Dictionary<string, (V3 Raw, V3 Corr)> Ease = new();
    }

    readonly Mem[] _mem = { new(), new() };

    public void ClearMemory()
    {
        foreach (var m in _mem)
        {
            m.Push.Clear();
            m.Pole.Clear();
            m.Ease.Clear();
        }
    }

    static V3 Unit(V3 v) => v.Unit();

    /// <summary>两段手臂的肘部位置：肘尖朝 pole 方向弯；够不着时手臂伸直。</summary>
    public static V3 ArmElbow(V3 s, V3 h, V3 pole, double a, double b)
    {
        var d = h - s;
        double dn = d.Norm() + 1e-9;
        var u = d / dn;
        if (dn >= a + b) return s + u * (a * dn / (a + b));
        var p = pole - u * pole.Dot(u);
        p /= p.Norm() + 1e-9;
        // 手几乎贴到肩上时，肩→手的方向随手的微小移动乱转：这时肘直接朝 pole 方向折起
        double w = M.Clamp((0.28 - dn) / 0.12, 0.0, 1.0);
        dn = Math.Max(dn, 0.18);
        double x = (a * a - b * b + dn * dn) / (2.0 * dn);
        double y = Math.Sqrt(Math.Max(a * a - x * x, 0.0));
        var ik = s + u * x + p * y;
        var poleU = pole / (pole.Norm() + 1e-9);
        return ik + (s + poleU * a - ik) * w;
    }

    /// <summary>持拍手：右手 +1，左手 -1（以模拟里这一分实际生效的设置为准）。</summary>
    public static double HandSign(World w, int j)
    {
        double m = j == 0 ? 1.0 : -1.0;
        return w.T[Sim.LM0 + j] * m < 0.0 ? -1.0 : 1.0;
    }

    /// <summary>两只脚的理想落点（迈步动画的目标）。</summary>
    public V2[] Home(World w, double[] foot, V3 body, V3 hand, int j, Pose pose) =>
        Compute(w, foot, body, hand, j, pose, true, 0.0, out _);

    public Body Build(World w, double[] foot, V3 body, V3 hand, int j, Pose pose, double dt)
    {
        Compute(w, foot, body, hand, j, pose, false, dt, out var b);
        return b;
    }

    V2[] Compute(World w, double[] foot, V3 body, V3 hand, int j, Pose pose, bool raw, double dt, out Body result)
    {
        result = null;
        pose ??= new Pose();
        double sj = j == 0 ? -1.0 : 1.0;
        // 横向分量乘持拍手方向：左手球员整副骨架左右镜像
        double sl = sj * HandSign(w, j);
        double bx = body.X + sj * 0.32;
        bx = sj * Math.Max(sj * bx, Sim.HALF_L + 0.25);
        double by = body.Y;
        bx += pose.Lunge.X;
        by += pose.Lunge.Y;
        double bob = pose.Bob, cr = pose.Crouch, tw = pose.Twist, stc = pose.Stance;
        var paddle = hand;
        double sz = M.Clamp(1.25 + 0.35 * (hand.Z - 0.9), 1.1, 1.55) - 1.2 * cr + bob;
        double lean = -sl * foot[j];
        double cross = -sl * foot[2 + j];
        double hy = by + 0.07 * lean, ny = by + 0.2 * lean;

        double sx = sj * 0.2 * Math.Sin(tw), sy = sl * 0.2 * Math.Cos(tw);
        double fx = -sj;
        // 往侧面扑倒时上身是侧倒，不是往前弯腰
        double tn = pose.Tilt.Norm();
        double bend = TORSO_BEND + 0.6 * cr * (1.0 - 0.7 * Math.Min(tn / 0.5, 1.0));
        double cx = bx + fx * bend;
        sz -= 0.25 * bend;
        var shift = V2.Zero;
        void Reach(V3 sh, V3? tgt)
        {
            if (tgt == null) return;
            var t = tgt.Value;
            var d = new V2(t.X - sh.X, t.Y - sh.Y);
            double dist = Math.Sqrt(d.Dot(d) + Math.Pow(t.Z - sh.Z, 2));
            if (dist > ARM_LEN) shift += d / (d.Norm() + 1e-9) * (dist - ARM_LEN);
        }
        Reach(new V3(cx + sx, ny + sy, sz), paddle);
        Reach(new V3(cx - sx, ny - sy, sz), pose.LHand);
        shift.X = fx * Math.Max(fx * shift.X, 0.0);
        // 救球、倒地爆冲：击球后上身继续朝球的方向扑出去
        shift += pose.Tilt;
        double sn = shift.Norm();
        // 上限跟着救球外倾一起渐变，突然放开/收紧会让身体一帧内被拉出去或弹回来
        double smax = 0.5 + 0.25 * Math.Min(tn / 0.15, 1.0);
        if (sn > smax)
        {
            shift *= smax / sn;
            sn = smax;
        }
        double drop = 0.6 * sn;
        double nx = sj * Math.Max(sj * (cx + shift.X), Sim.HALF_L - 0.4);
        double nz = sz - drop;
        // 探身够台内短球时上身压到台面以下会陷进台面里：在台面范围内就抬到台面之上
        if (Math.Abs(nx) < Sim.HALF_L + 0.05 && Math.Abs(ny + shift.Y) < Sim.HALF_W + 0.05)
            nz = Math.Max(nz, Sim.TABLE_H + 0.09);
        var neck = new V3(nx, ny + shift.Y, nz);
        var rsh = new V3(nx + sx, neck.Y + sy, neck.Z);
        var lsh = new V3(nx - sx, neck.Y - sy, neck.Z);
        var hip = new V3(bx + 0.6 * shift.X, hy + 0.6 * shift.Y, 0.92 - 1.2 * cr - 0.3 * drop + bob);
        // 头顺着躯干往前探一半
        var tor = Unit(neck - hip);
        var hd = Unit(0.5 * tor + new V3(0.5 * fx * 0.3, 0.0, 0.5));
        var head = new V3(nx + fx * 0.05 + 0.25 * shift.X + 0.17 * hd.X,
            neck.Y + 0.25 * shift.Y + 0.04 * lean + 0.17 * hd.Y, neck.Z + 0.17 * hd.Z - 0.05 * sn);

        double wdt = 0.22 * (1.0 - 0.3 * Math.Abs(stc));
        double lfy = by - sl * wdt, rfy = by + sl * wdt;
        if (cross != 0.0)
        {
            lfy = by + cross * 0.08;
            rfy = by - cross * 0.04;
        }
        double lfx = bx + sj * (0.1 - 0.25 * Math.Max(stc, 0.0) + 0.15 * Math.Max(-stc, 0.0));
        double rfx = bx + sj * (0.0 - 0.25 * Math.Max(-stc, 0.0) + 0.15 * Math.Max(stc, 0.0));
        double reach = -sj * (paddle.X - bx);
        if (reach > 0.5)
        {
            double u = Math.Min((reach - 0.5) / 0.3, 1.0);
            u = u * u * (3.0 - 2.0 * u);
            double tx = sj * Math.Max(sj * (paddle.X + sj * 0.25), Sim.HALF_L - 0.3);
            rfx += (tx - rfx) * u;
            rfy += (paddle.Y + sl * 0.1 - rfy) * u;
        }
        var feet = new[] { new V2(lfx, lfy), new V2(rfx, rfy) };
        var hipXY = new V2(hip.X, hip.Y);
        for (int k = 0; k < 2; k++)
        {
            var d = feet[k] - hipXY;
            double dn = d.Norm();
            if (dn > FOOT_R) feet[k] = hipXY + d * (FOOT_R / dn);
        }
        if (raw) return feet;

        var shown = new V3[2];
        var dflt = new[] { new V3(feet[0].X, feet[0].Y, 0.02), new V3(feet[1].X, feet[1].Y, 0.02) };
        var given = new[] { pose.LFoot ?? dflt[0], pose.RFoot ?? dflt[1] };
        for (int k = 0; k < 2; k++)
        {
            var f = given[k];
            var d = new V2(f.X, f.Y) - hipXY;
            double dn = d.Norm();
            if (dn > FOOT_R + 0.12)
            {
                var xy = hipXY + d * ((FOOT_R + 0.12) / dn);
                f = new V3(xy.X, xy.Y, f.Z);
            }
            shown[k] = f;
        }
        var lfoot = shown[0];
        var rfoot = shown[1];
        double fwdX = -sj;
        var free = new V3(lsh.X + fwdX * (0.28 + 0.15 * tw), lsh.Y + sl * 0.12, lsh.Z - 0.38);
        var lhand = pose.LHand ?? free;
        // 持拍手、肘不能进到躯干里
        var upT = Unit(neck - hip);
        var acT = Unit(rsh - lsh);
        acT = Unit(acT - upT * acT.Dot(upT));
        var fwT = upT.Cross(acT);
        if (fwT.X * fwdX < 0) fwT = -fwT;
        double hzT = (neck - hip).Norm();

        // 上身扑出去的距离有上限，够不着的球拍跟着手臂收回来，否则前臂会被拉长；要在躯干/头的避让之前做
        {
            var d = paddle - rsh;
            double dn = d.Norm();
            if (dn > UPPER_ARM + FORE_ARM) paddle = rsh + d * ((UPPER_ARM + FORE_ARM) / dn);
        }

        var pmem = _mem[j];
        // 每帧的记忆衰减按时间换算（Python 版 60 帧/秒、每帧 3 次）
        double nUpd = Math.Max(dt * MEM_RATE, 0.0);
        double keep = Math.Pow(POLE_KEEP, nUpd);
        double ease = 1.0 - Math.Pow(1.0 - HAND_EASE, nUpd);

        // 点穿进躯干盒就往两侧或身前推出去；绝不往后推。key 记住上一帧往哪边推，避免在胸口正中来回换边
        V3 PushOut(V3 p, double pad, string key)
        {
            var q = p - hip;
            double x = q.Dot(acT), y = q.Dot(fwT), z = q.Dot(upT);
            double fy = 0.105 + pad - y;
            double byy = y + 0.085 + pad;
            if (!(-0.06 < z && z < hzT + 0.03) || Math.Min(Math.Min(fy, byy), 0.17 + pad - Math.Abs(x)) <= 0.0)
            {
                pmem.Push.Remove(key);
                return p;
            }
            double cR = 0.17 + pad - x, cL = 0.17 + pad + x, cF = fy;
            double Cost(char c) => c == 'R' ? cR : c == 'L' ? cL : cF;
            char c = 'R';
            if (cL < Cost(c)) c = 'L';
            if (cF < Cost(c)) c = 'F';
            if (pmem.Push.TryGetValue(key, out var prev) && Cost(prev) <= Cost(c) + PUSH_STICK) c = prev;
            pmem.Push[key] = c;
            if (c == 'F') return p + fwT * fy;
            return p + acT * (c == 'R' ? cR : -cL);
        }

        V3 AboveTable(V3 p, double lift = 0.05, double m = 0.03)
        {
            if (Math.Abs(p.X) < Sim.HALF_L + m && Math.Abs(p.Y) < Sim.HALF_W + m && p.Z < Sim.TABLE_H + lift)
                return new V3(p.X, p.Y, Sim.TABLE_H + lift);
            return p;
        }

        V3 HeadClear(V3 p)
        {
            var q = p - head;
            double f = q.Dot(fwT);
            var sideD = q - fwT * f;
            if (sideD.Norm() < 0.24 && -0.15 < f && f < 0.24) return p + fwT * (0.24 - f);
            return p;
        }

        var rawPd = paddle;
        paddle = AboveTable(PushOut(paddle, 0.09, "po_p0"));
        paddle = HeadClear(paddle);

        double PtPen(V3 p, double pad)
        {
            var q = p - hip;
            double x = q.Dot(acT), y = q.Dot(fwT), z = q.Dot(upT);
            double t;
            if (z < -0.06 || z > hzT + 0.03) t = 0.0;
            else
            {
                double ax = Math.Abs(x);
                t = Math.Min(Math.Min(0.105 + pad - y, y + 0.085 + pad), 0.17 + pad - ax);
                if (t < 0.0) t = 0.0;
            }
            double h = 0.105 + pad + 0.03 - (p - head).Norm();
            return t > h ? t : (h > 0.0 ? h : 0.0);
        }

        // 上臂、前臂采样点穿进躯干或碰头的最大深度
        double ArmBad(V3 sh, V3 el, V3 hd2, double[] us, double[] pads, double extra = 0.0)
        {
            double d = 0.0;
            double[] uu = { 0.5, 0.8 };
            for (int k = 0; k < 2; k++) d = Math.Max(d, PtPen(sh + (el - sh) * uu[k], pads[k] + extra));
            for (int k = 0; k < us.Length; k++) d = Math.Max(d, PtPen(el + (hd2 - el) * us[k], pads[2 + k] + extra));
            return d;
        }

        // 把肘的极向量往能减小穿透的方向推：先试几个方向挑最好的，再沿选中的方向细调一次
        (V3 El, V3 Pole) Settle(V3 sh, V3 hd2, V3 pole, double blen, double[] us, double[] pads, V3 outDir, V3? hold)
        {
            var el = ArmElbow(sh, hd2, pole, UPPER_ARM, blen);
            double d;
            if (hold == null)
            {
                d = ArmBad(sh, el, hd2, us, pads);
                if (d <= 1e-3) return (el, pole);
            }
            else
            {
                // 往自然朝向回一点要留够余量才回，否则停在上一帧（hold），不然下一帧又穿、又跳开
                if (ArmBad(sh, el, hd2, us, pads, HOLD_MARGIN) <= 1e-3) return (el, pole);
                var e1 = ArmElbow(sh, hd2, hold.Value, UPPER_ARM, blen);
                double d1 = ArmBad(sh, e1, hd2, us, pads);
                if (d1 <= 1e-3) return (e1, hold.Value);
                pole = hold.Value;
                el = e1;
                d = d1;
            }
            var best = (D: d, El: el, Pole: pole);
            foreach (var (dirn, gain) in new[] { (outDir, 120.0), (Unit(fwT), 120.0), (Unit(outDir - fwT), 160.0) })
            {
                var pl = pole + dirn * (gain * d);
                var e2 = ArmElbow(sh, hd2, pl, UPPER_ARM, blen);
                double d2 = ArmBad(sh, e2, hd2, us, pads);
                if (d2 < best.D) best = (d2, e2, pl);
            }
            if (best.D > 1e-3)
            {
                (d, el, pole) = best;
                foreach (var gain in new[] { 30.0, 400.0 })
                {
                    var pl = pole + outDir * (gain * d);
                    var e2 = ArmElbow(sh, hd2, pl, UPPER_ARM, blen);
                    double d2 = ArmBad(sh, e2, hd2, us, pads);
                    if (d2 < best.D) best = (d2, e2, pl);
                }
            }
            return (best.El, best.Pole);
        }

        // 先把肘挪开；挪不开（手贴在躯干正前方）再把手一点点往身侧外推
        (V3 El, V3 Hd) FitArm(V3 sh, V3 hd2, V3 pole, double blen, double[] us, double[] pads, V3 outDir,
            V3 hdDir, V3? hold)
        {
            var hd0 = hd2;
            var (el, _) = Settle(sh, hd2, pole, blen, us, pads, outDir, hold);
            for (int it = 0; it < 2; it++)
            {
                double dmax = ArmBad(sh, el, hd2, us, pads);
                if (dmax <= 1e-3 || (hd2 - hd0).Norm() > 0.16) break;
                hd2 += hdDir * (dmax * 1.5);
                (el, _) = Settle(sh, hd2, pole, blen, us, pads, outDir, hold);
            }
            return (el, hd2);
        }

        (V3, V3?) RecallPole(string key, V3 natural)
        {
            if (!pmem.Pole.TryGetValue(key, out var dev)) return (natural, null);
            return (natural + dev * keep, natural + dev);
        }

        V3 EaseHand(string key, V3 rawP, V3 fixd)
        {
            var corr = fixd - rawP;
            if (pmem.Ease.TryGetValue(key, out var prev) && (rawP - prev.Raw).Norm() < 0.4)
                corr = prev.Corr + (corr - prev.Corr) * ease;
            pmem.Ease[key] = (rawP, corr);
            return rawP + corr;
        }

        var epole = pose.Epole ?? new V3(sj * 0.3, sl * 0.6, -1.0);
        paddle = HeadClear(AboveTable(PushOut(paddle, 0.09, "po_p1")));
        var (rp0, rhold) = RecallPole("rpole", epole);
        V3 relbow;
        (relbow, paddle) = FitArm(rsh, paddle, rp0, FORE_ARM, US_R, PADS_R, Unit(acT + 0.8 * fwT),
            Unit(acT + 0.5 * fwT), rhold);
        // 推完手再用最终手位收一遍躯干/头/台面的边
        paddle = HeadClear(AboveTable(PushOut(paddle, 0.09, "po_p2")));
        paddle = AboveTable(EaseHand("pad", rawPd, paddle));
        V3 rp;
        (relbow, rp) = Settle(rsh, paddle, rp0, FORE_ARM, US_R, PADS_R, Unit(acT + 0.8 * fwT), rhold);
        pmem.Pole["rpole"] = rp - epole;
        // 抛球手：肘尖朝向跟着躯干转
        var lpole = -0.3 * fwT - 0.6 * acT - 1.0 * upT;
        var rawLh = lhand;
        lhand = AboveTable(PushOut(lhand, 0.03, "po_l0"));
        var (lp0, lhold) = RecallPole("lpole", lpole);
        V3 lelbow;
        (lelbow, lhand) = FitArm(lsh, lhand, lp0, FORE_ARM - 0.04, US_L, PADS_L, Unit(-acT + 0.8 * fwT),
            Unit(-acT + 0.5 * fwT), lhold);
        lhand = HeadClear(AboveTable(PushOut(lhand, 0.03, "po_l1")));
        lhand = AboveTable(EaseHand("lh", rawLh, lhand));
        V3 lp;
        (lelbow, lp) = Settle(lsh, lhand, lp0, FORE_ARM - 0.04, US_L, PADS_L, Unit(-acT + 0.8 * fwT), lhold);
        pmem.Pole["lpole"] = lp - lpole;

        double hyaw = 0.5 * tw;
        double hxo = sj * HIP_W * Math.Sin(hyaw), hyo = sl * HIP_W * Math.Cos(hyaw);
        var rhip = new V3(hip.X + hxo, hip.Y + hyo, hip.Z);
        var lhip = new V3(hip.X - hxo, hip.Y - hyo, hip.Z);
        var fwd = new V3(-sj, 0.0, 0.0);

        V3 Knee(V3 h, V3 f)
        {
            var d = f - h;
            double dn = d.Norm() + 1e-9;
            var ax = d / dn;
            var outv = new V3(0.0, Math.Sign(h.Y - hip.Y), 0.0);
            var p = fwd + 0.12 * outv;
            p -= ax * p.Dot(ax);
            p /= p.Norm() + 1e-9;
            double bendK = Math.Sqrt(Math.Max(SEG_LEN * SEG_LEN - Math.Pow(Math.Min(dn, 2 * SEG_LEN) / 2, 2), 0.0));
            return (h + f) / 2 + p * bendK;
        }

        result = new Body
        {
            Hip = hip, LHip = lhip, RHip = rhip, Neck = neck, Head = head, RSh = rsh, LSh = lsh, LHand = lhand,
            RElbow = relbow, LElbow = lelbow, LKnee = Knee(lhip, lfoot), RKnee = Knee(rhip, rfoot), LFoot = lfoot,
            RFoot = rfoot, Paddle = paddle,
        };
        return feet;
    }
}
