"""单拍命中率测试：固定来球，让 0 号球员按给定意图击球，统计上台率、出界/下网和出球速度。

    python shot_test.py
"""
import numpy as np

from pingpong import sim
from pingpong.env import PongEnv


def trial(pw, top, lat, tec=0.6, n=400, v_in=(-6.0, 0.0, -1.0), w_in=(0.0, 0.0, 0.0), bh=False, seed=0,
          lean=0.0, settle=1.0, vel=(0.0, 0.0), z=0.98, avail=1.0):
    env = PongEnv(n, train=True, seed=seed)
    env.reset()
    B, P, I, T, S, F = env.B, env.P, env.I, env.T, env.S, env.F
    S[:] = 0.6
    S[:, 0, sim.TEC] = tec
    F[:] = 0.0
    a = np.zeros(sim.ACT_DIM)
    a[3], a[4], a[5], a[6] = 0.3, lat, pw * 2 - 1, top
    res = np.zeros(4, int)
    speed = []
    for i in range(n):
        B[i, 0:3] = (-1.75, 0.0, z)
        T[i, sim.EVT] = 0.0
        T[i, sim.RT] = avail + sim.react_delay(S[i, 0, sim.REA])
        B[i, 3:6] = v_in
        B[i, 6:9] = w_in
        P[i, 0, 0:3] = (-2.0, 0.13, 0.98)
        P[i, 0, 3:6] = (vel[0], vel[1], 0.0)
        T[i, sim.SET0] = settle * sim.SETTLE_T
        T[i, sim.LN0] = lean * sim.LEAN_MAX
        T[i, sim.BL0] = -1.0 if bh else 1.0
        T[i, sim.CU0] = lat * sim.LAT_MAX
        I[i, sim.PHASE] = sim.PH_FLY
        I[i, sim.TURN] = 0
        sim.do_hit(i, 0, a, 0.0, B, P, I, T, S, F)
        speed.append(np.linalg.norm(B[i, 3:6]) * 3.6)
        b = B[i].copy()
        net = False
        for _ in range(2000):
            ev = sim.ball_substep(b, 1.0 / 480)
            if ev == 2:
                net = True
            elif ev == 1:
                res[0 if b[0] > 0 else 2] += 1
                break
            elif ev == 3 or b[2] < sim.TABLE_H - 0.3:
                res[2 if net else 1] += 1
                break
        else:
            res[3] += 1
    r = res / n
    return r[0], r[1], r[2], float(np.mean(speed))


def body_test():
    print("身体发力：正手全力拉球（力度 1.0，上旋 0.8，技术 0.6）| 上台 出界 下网 | 球速km/h")
    for name, kw in (("站稳 + 重心压右脚蓄力 + 向前迎球", dict(lean=-1.0, vel=(0.8, 0.0))),
                     ("站稳 + 重心压右脚蓄力", dict(lean=-1.0)),
                     ("站稳，重心居中", dict()),
                     ("站稳，但重心压在反方向", dict(lean=1.0)),
                     ("站稳，边后退边打", dict(vel=(-0.8, 0.0))),
                     ("没站稳（刚跑过来）", dict(settle=0.0)),
                     ("横向跑动中击球", dict(settle=0.0, vel=(0.0, 1.0)))):
        ok, out, net, sp = trial(1.0, 0.8, 0.5, 0.6, **kw)
        print(f"  {name:22s} | {ok:4.0%} {out:4.0%} {net:4.0%} | {sp:5.0f}")
    print()


def chance_test():
    print("机会球与准备时间：正手全力（力度 1.0，上旋 0.6，技术 0.6）| 上台 出界 下网 | 球速km/h")
    for name, kw in (("高球 4m/s，离台 35cm，准备充分", dict(z=sim.TABLE_H + 0.35, v_in=(-4.0, 0, -0.5))),
                     ("中等 6m/s，离台 22cm，准备充分", dict()),
                     ("中等，准备时间 0.25s", dict(avail=0.25)),
                     ("中等，准备时间 0.1s", dict(avail=0.1)),
                     ("低球 6m/s，离台 10cm，准备充分", dict(z=sim.TABLE_H + 0.10))):
        ok, out, net, sp = trial(1.0, 0.6, 0.5, 0.6, w_in=(0, 100.0, 0), **kw)
        print(f"  {name:26s} | {ok:4.0%} {out:4.0%} {net:4.0%} | {sp:5.0f}")
    print()


def main():
    chance_test()
    body_test()
    print("来球: 6m/s 上旋 150rad/s / 10m/s 上旋 / 6m/s 下旋 / 反手 6m/s 上旋（技术 0.6）")
    print("力度 上旋 | 上台 出界 下网 | 球速")
    for v_in, w_in, bh in (((-6.0, 0, -1.0), (0, 150.0, 0), False), ((-10.0, 0, -1.5), (0, 150.0, 0), False),
                           ((-6.0, 0, -1.0), (0, -150.0, 0), False), ((-6.0, 0, -1.0), (0, 150.0, 0), True)):
        for pw, top in ((0.3, 0.3), (0.6, 0.6), (1.0, 0.8), (1.0, 1.0)):
            ok, out, net, sp = trial(pw, top, 0.5, 0.6, v_in=v_in, w_in=w_in, bh=bh)
            print(f"  {pw:4.1f} {top:4.1f} | {ok:4.0%} {out:4.0%} {net:4.0%} | {sp:5.0f}")
        print()
    print("力度 上旋  横向 技术 | 上台 出界 下网 | 球速km/h")
    for tec in (0.3, 0.6, 0.9):
        for pw, top, lat in ((0.2, 0.3, 0.0), (0.5, 0.5, 0.0), (1.0, 0.0, 0.0), (1.0, 0.8, 0.0),
                             (1.0, 0.8, 0.8), (0.5, 0.5, 0.8)):
            ok, out, net, sp = trial(pw, top, lat, tec)
            print(f"{pw:4.1f} {top:4.1f} {lat:5.1f} {tec:4.1f} | {ok:4.0%} {out:4.0%} {net:4.0%} | {sp:5.0f}")


if __name__ == "__main__":
    main()
