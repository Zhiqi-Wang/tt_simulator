"""台内球处理：击球点在台面上方（离端线内）的那一拍，用了什么技术、来球是什么旋转、自己失误率多少；
以及对方回过来的球离网多远、多快，再看下一拍对手的处理结果。

    python short_test.py models/xxx.npz
"""
import sys

import numpy as np

from pingpong import sim
from pingpong.env import PongEnv
from pingpong.nn import Policy


def main():
    pol = Policy.load(sys.argv[1] if len(sys.argv) > 1 else "models/policy.npz")
    env = PongEnv(512, train=False, seed=0)
    env.S[:, :, :sim.N_STATS] = 0.6
    obs = env.reset()
    env.S[:, :, :sim.N_STATS] = 0.6
    n = env.n
    pend = np.full(n, -1)
    info = np.zeros((n, 5))
    rows = []
    miss_short = 0
    short_land = 0
    for _ in range(3000):
        prev_h = env.I[:, sim.HITS].copy()
        prev_b = env.B.copy()
        prev_bounced = env.I[:, sim.BOUNCED].copy()
        obs, _, done = env.step(np.stack([pol.act(obs[:, 0]), pol.act(obs[:, 1])], 1))
        hit = (env.I[:, sim.HITS] > prev_h) & ~done
        for i in np.where(hit & (pend >= 0))[0]:
            rows.append((*info[i], 1))
            pend[i] = -1
        for i in np.where(hit)[0]:
            dxt = abs(prev_b[i, 0]) - sim.HALF_L
            v, w = prev_b[i, 3:6], prev_b[i, 6:9]
            top = sim.incoming_top(v, w)
            info[i] = (int(env.I[i, sim.LSTK]), dxt, top, prev_b[i, 2] - sim.TABLE_H, env.I[i, sim.HITS])
            pend[i] = int(env.I[i, sim.LHIT])
        land = (env.I[:, sim.BOUNCED] == 1) & (prev_bounced == 0) & (env.I[:, sim.HITS] > 0) & ~done
        short_land += int((land & (np.abs(env.B[:, 0]) < 0.6)).sum())
        for i in np.where(done)[0]:
            if pend[i] >= 0:
                rows.append((*info[i], 0 if env.winner[i] != pend[i] else 2))
                pend[i] = -1
            if env.reason[i] == sim.R_MISS and abs(prev_b[i, 0]) < sim.HALF_L:
                miss_short += 1
    r = np.array(rows)
    inside = r[:, 1] < 0.0
    print(f"击球 {len(r)} 拍，其中台内击球 {inside.mean():.0%}；短球落台 {short_land} 次，"
          f"在台面上空没接到的球 {miss_short} 次")
    print("台内击球（结果：0=自己失误，1=对方接住，2=对方没接住）：")
    for k, name in enumerate(sim.STROKE_NAMES):
        m = inside & (r[:, 0] == k)
        if m.sum() < 30:
            continue
        o = r[m, 5]
        print(f"  {name:4s} 占 {m.sum() / inside.sum():5.1%}  自己失误 {np.mean(o == 0):5.1%}  "
              f"直接得分 {np.mean(o == 2):5.1%}  来球下旋占 {np.mean(r[m, 2] < -50):5.1%}  "
              f"击球点高 {np.median(r[m, 3]):.2f}m")
    for name, m in (("来球下旋(<-50)", inside & (r[:, 2] < -50)), ("来球上旋(>50)", inside & (r[:, 2] > 50)),
                    ("接发球(第2拍)", inside & (r[:, 4] == 2))):
        o = r[m, 5]
        print(f"  {name}: 占台内 {m.sum() / inside.sum():5.1%} 自己失误 {np.mean(o == 0):5.1%}")


if __name__ == "__main__":
    main()
