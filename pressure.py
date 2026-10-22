"""施压分析：只看打上对方台面的好球，按球速、落点（离对方身体多远）分组，
统计对方这一拍回不过来（失误或没接到）的比例，以及对方回球的球速。
用来检验“快球、大角度、调动”在当前物理下是否真的给对手造成压力。

    python pressure.py [--model models/policy.npz]
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
    env = PongEnv(512, train=True, seed=0)
    obs = env.reset()
    n = env.n
    pend = np.full(n, -1)
    speed = np.zeros(n)
    dist = np.zeros(n)
    rows = []
    for _ in range(args.steps):
        prev_b = env.I[:, sim.BOUNCED].copy()
        prev_h = env.I[:, sim.HITS].copy()
        obs, _, done = env.step(np.stack([pol.act(obs[:, 0]), pol.act(obs[:, 1])], 1))
        I = env.I
        legal = (I[:, sim.BOUNCED] == 1) & (prev_b == 0) & (I[:, sim.PHASE] == sim.PH_FLY) & (I[:, sim.HITS] > 0)
        for i in np.where(legal & (pend < 0))[0]:
            rec = I[i, sim.TURN]
            pend[i] = rec
            speed[i] = np.linalg.norm(env.B[i, 3:5])
            dist[i] = abs(env.B[i, 1] - env.P[i, rec, 1])
        returned = (I[:, sim.HITS] > prev_h) & (pend >= 0)
        for i in np.where(returned)[0]:
            rows.append((speed[i], dist[i], 0))
            pend[i] = -1
        for i in np.where(done & (pend >= 0))[0]:
            failed = env.winner[i] != pend[i]
            rows.append((speed[i], dist[i], 1 if failed else 0))
            pend[i] = -1
    r = np.array(rows)
    print(f"好球 {len(r)} 个，对方回不过来 {r[:, 2].mean():.1%}")
    print("按球速（落台后水平速度）：")
    for lo, hi in ((0, 4), (4, 6), (6, 8), (8, 11), (11, 30)):
        m = (r[:, 0] >= lo) & (r[:, 0] < hi)
        if m.sum() > 50:
            print(f"  {lo:2d}-{hi:2d} m/s  占 {m.mean():5.1%}  对方回不过来 {r[m, 2].mean():5.1%}")
    print("按落点离对方身体的横向距离：")
    for lo, hi in ((0, 0.2), (0.2, 0.4), (0.4, 0.7), (0.7, 3)):
        m = (r[:, 1] >= lo) & (r[:, 1] < hi)
        if m.sum() > 50:
            print(f"  {lo:.1f}-{hi:.1f} m  占 {m.mean():5.1%}  对方回不过来 {r[m, 2].mean():5.1%}")


if __name__ == "__main__":
    main()
