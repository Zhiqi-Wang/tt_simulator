"""接发球探针：红方发球时，蓝方接发球那一拍强制成某种打法，统计蓝方接发球局胜率。

    python recv_probe.py [models/xxx.npz]
"""
import sys

import numpy as np

from pingpong import sim
from pingpong.env import PongEnv
from pingpong.nn import Policy

MODES = {
    "base": None,
    "loop": dict(pw=1.0, top=0.7),
    "drive": dict(pw=0.6, top=0.3),
    "push_short": dict(pw=-0.8, top=-0.8, depth=-1.0),
    "push_long": dict(pw=-0.4, top=-0.8, depth=1.0),
    "block": dict(pw=-1.0, top=0.0),
}


def run(pol, mode, steps=2500):
    env = PongEnv(512, train=False, seed=0)
    env.S[:, :, :sim.N_STATS] = 0.6
    obs = env.reset()
    env.S[:, :, :sim.N_STATS] = 0.6
    won = pts = 0
    for _ in range(steps):
        a0, a1 = pol.act(obs[:, 0]), pol.act(obs[:, 1])
        if mode is not None:
            recv = (env.I[:, sim.SERVER] == 1) & (env.I[:, sim.HITS] == 0) & (env.I[:, sim.PHASE] == sim.PH_FLY)
            a0[recv, 5] = mode["pw"]
            a0[recv, 6] = mode["top"]
            if "depth" in mode:
                a0[recv, 3] = mode["depth"]
        server = env.I[:, sim.SERVER].copy()
        obs, _, done = env.step(np.stack([a0, a1], 1))
        m = done & (server == 1) & (env.winner >= 0)
        won += int((env.winner[m] == 0).sum())
        pts += int(m.sum())
    return won / max(pts, 1), pts


def main():
    pol = Policy.load(sys.argv[1] if len(sys.argv) > 1 else "models/policy.npz")
    for name, mode in MODES.items():
        wr, n = run(pol, mode)
        print(f"{name:11s} 接发球方胜率 {wr:5.1%}  ({n} 分)", flush=True)


if __name__ == "__main__":
    main()
