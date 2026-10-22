"""实战中发力的收益：相持中蓝方力度随机，记录每拍的击球条件和结果
（自己失误 / 对方回不过来 / 对方回过来），按力度和条件分组。

    python attack_test.py [--model models/policy.npz]
"""
import argparse

import numpy as np

from pingpong import sim
from pingpong.env import PongEnv
from pingpong.nn import Policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/policy.npz")
    ap.add_argument("--steps", type=int, default=3000)
    args = ap.parse_args()
    pol = Policy.load(args.model)
    env = PongEnv(512, train=True, seed=1)
    obs = env.reset()
    n = env.n
    rng = np.random.default_rng(0)
    pw_draw = rng.uniform(-1, 1, n)
    pend = np.zeros(n, bool)
    cur = np.zeros((n, 6))
    rows = []
    extra = []
    cur_x = np.zeros((n, 3))
    for _ in range(args.steps):
        a0 = pol.act(obs[:, 0])
        a1 = pol.act(obs[:, 1])
        rally = env.I[:, sim.PHASE] != sim.PH_SERVE
        a0[:, 5] = np.where(rally, pw_draw, a0[:, 5])
        prev_h = env.I[:, sim.HITS].copy()
        prev_turn = env.I[:, sim.TURN].copy()
        prev_b = env.B.copy()
        prev_v = np.linalg.norm(env.B[:, 3:6], axis=1)
        prev_awk = np.maximum(0.0, 0.9 - np.abs(env.T[:, sim.BL0])) / 0.9
        prev_P = env.P.copy()
        prev_T = env.T.copy()
        obs, _, done = env.step(np.stack([a0, a1], 1))
        I, T = env.I, env.T
        hit = I[:, sim.HITS] > prev_h
        blue = hit & (prev_turn == 0) & rally
        red = hit & (prev_turn == 1)
        for i in np.where(red & pend)[0]:
            rows.append((*cur[i], 2))
            extra.append(cur_x[i].copy())
            pend[i] = False
        for i in np.where(blue)[0]:
            if pend[i]:
                pend[i] = False
            m = 1.0
            run = np.hypot(min(m * env.P[i, 0, 3], 0), env.P[i, 0, 4])
            cur[i] = ((pw_draw[i] + 1) / 2, T[i, sim.LPREP], prev_b[i, 2] - sim.TABLE_H, prev_v[i], run,
                      abs(prev_b[i, 0]) - sim.HALF_L)
            pend[i] = True
            _, _, stretch = sim.reach_hand(prev_P, prev_T, i, 0, prev_b[i, 0], prev_b[i, 1])
            cur_x[i] = (prev_awk[i], prev_v[i], stretch)
            pw_draw[i] = rng.uniform(-1, 1)
        for i in np.where(done & pend)[0]:
            w = env.winner[i]
            rows.append((*cur[i], 0 if w == 1 else 1))
            extra.append(cur_x[i].copy())
            pend[i] = False
    r = np.array(rows)
    out = r[:, 6]

    def show(title, col, bins):
        print(title)
        for lo, hi in bins:
            msk = (r[:, col] >= lo) & (r[:, col] < hi)
            if msk.sum() < 80:
                continue
            o = out[msk]
            print(f"  {lo:5.2f}-{hi:5.2f}  n={msk.sum():6d}  自己失误 {np.mean(o == 0):5.1%}  "
                  f"对方回不过来 {np.mean(o == 1):5.1%}  净 {np.mean(o == 1) - np.mean(o == 0):+6.1%}")

    pb = ((0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01))
    show("按力度（全部相持球）：", 0, pb)
    for name, msk in (("准备充分(lack=0)", r[:, 1] == 0), ("来不及(lack>0.3)", r[:, 1] > 0.3),
                      ("高球(>0.25m)", r[:, 2] > 0.25), ("低球(<0.12m)", r[:, 2] < 0.12),
                      ("跑动快(>1m/s)", r[:, 4] > 1.0), ("站定(<0.3m/s)", r[:, 4] < 0.3),
                      ("台内(<0)", r[:, 5] < 0), ("中远台(>0.6)", r[:, 5] > 0.6)):
        sub = r[msk]
        print(f"{name}: 占 {msk.mean():.1%}")
        for lo, hi in ((0, 0.4), (0.7, 1.01)):
            m2 = (sub[:, 0] >= lo) & (sub[:, 0] < hi)
            if m2.sum() < 60:
                continue
            o = sub[m2, 6]
            print(f"    力度 {lo:.1f}-{hi:.1f}  n={m2.sum():5d}  自己失误 {np.mean(o == 0):5.1%}  "
                  f"对方回不过来 {np.mean(o == 1):5.1%}  净 {np.mean(o == 1) - np.mean(o == 0):+6.1%}")
    print("lack 分布：", np.percentile(r[:, 1], [25, 50, 75, 90]).round(2))
    x = np.array(extra)
    rushed = r[:, 1] > 0.3
    print("来不及的球里：")
    for name, col, lo, hi in (("正反手没换好(awk>0.3)", 0, 0.3, 9), ("换好了(awk<0.05)", 0, -1, 0.05),
                              ("来球快(>9m/s)", 1, 9, 99), ("来球慢(<6m/s)", 1, -1, 6),
                              ("伸手够(>0.15m)", 2, 0.15, 9), ("不用够(<0.03m)", 2, -1, 0.03)):
        msk = rushed & (x[:, col] >= lo) & (x[:, col] < hi)
        if msk.sum() > 50:
            print(f"  {name:22s} 占 {msk.sum() / rushed.sum():5.1%}  自己失误 {np.mean(out[msk] == 0):5.1%}")
    clean = rushed & (x[:, 0] < 0.05) & (x[:, 2] < 0.03) & (x[:, 1] < 7)
    print(f"来不及但其他条件都好的球 n={clean.sum()}，按击球点高度：")
    for lo, hi in ((-1, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 9)):
        msk = clean & (r[:, 2] >= lo) & (r[:, 2] < hi)
        if msk.sum() > 30:
            print(f"  {lo:4.1f}-{hi:4.1f}m n={msk.sum():5d} 自己失误 {np.mean(out[msk] == 0):5.1%}")
    for lo, hi in ((-9, 0.0), (0.0, 0.5), (0.5, 9)):
        msk = clean & (r[:, 5] >= lo) & (r[:, 5] < hi)
        if msk.sum() > 30:
            print(f"  离端线 {lo:4.1f}-{hi:4.1f}m n={msk.sum():5d} 自己失误 {np.mean(out[msk] == 0):5.1%}")
    for lo, hi in ((0, 0.3), (0.3, 0.6), (0.6, 1.01)):
        msk = clean & (r[:, 1] >= lo) & (r[:, 1] < hi)
        if msk.sum() > 30:
            print(f"  lack {lo:.1f}-{hi:.1f} n={msk.sum():5d} 自己失误 {np.mean(out[msk] == 0):5.1%}")


if __name__ == "__main__":
    main()
