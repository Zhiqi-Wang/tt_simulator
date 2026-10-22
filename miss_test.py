"""“没接到”的球落在哪里：记录每分最后一次合法落台的位置（离网距离、离中线距离）和接球方的位置。

    python miss_test.py models/xxx.npz
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
    last = np.zeros((env.n, 7))
    rows = []
    for _ in range(3000):
        prev_b = env.I[:, sim.BOUNCED].copy()
        obs, _, done = env.step(np.stack([pol.act(obs[:, 0]), pol.act(obs[:, 1])], 1))
        nb = (env.I[:, sim.BOUNCED] == 1) & (prev_b == 0)
        for i in np.where(nb)[0]:
            rec = env.I[i, sim.TURN]
            last[i] = (abs(env.B[i, 0]), abs(env.B[i, 1]), abs(env.P[i, rec, 0]) - sim.HALF_L,
                       abs(env.B[i, 1] - env.P[i, rec, 1]), np.linalg.norm(env.B[i, 3:5]),
                       env.I[i, sim.HITS], env.I[i, sim.NETT])
        for i in np.where(done)[0]:
            if env.reason[i] == sim.R_MISS:
                rows.append(last[i].copy())
    r = np.array(rows)
    print(f"没接到 {len(r)} 个")
    print("落点离网距离分布(m)：", np.percentile(r[:, 0], [10, 25, 50, 75, 90]).round(2))
    print("落点离中线距离分布(m)：", np.percentile(r[:, 1], [10, 25, 50, 75, 90]).round(2))
    print("落台时接球方离端线(m)：", np.percentile(r[:, 2], [10, 25, 50, 75, 90]).round(2))
    print("落台时球与接球方横向距离(m)：", np.percentile(r[:, 3], [10, 25, 50, 75, 90]).round(2))
    print("落台后水平速度(m/s)：", np.percentile(r[:, 4], [10, 25, 50, 75, 90]).round(2))
    print(f"短球(离网<0.5m) 占 {np.mean(r[:, 0] < 0.5):.0%}")
    s = r[:, 0] < 0.5
    print(f"短球里：发球 {np.mean(r[s, 5] == 0):.0%}  擦网 {np.mean(r[s, 6] > 0):.0%}  "
          f"第2拍(接发球) {np.mean(r[s, 5] == 1):.0%}  之后 {np.mean(r[s, 5] >= 2):.0%}")


if __name__ == "__main__":
    main()
