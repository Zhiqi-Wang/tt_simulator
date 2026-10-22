"""发球平衡：RL 对 RL，蓝方每次发球都强制成某种发球（其余意图由模型决定），统计蓝方发球局胜率。

    python serve_probe.py [models/xxx.npz]
"""
import sys

import numpy as np

from pingpong import sim
from pingpong.env import PongEnv
from pingpong.nn import Policy
from serve_test import serve_action


def run(pol, k, steps=2500):
    env = PongEnv(512, train=False, seed=0)
    env.S[:, :, :sim.N_STATS] = 0.6
    obs = env.reset()
    env.S[:, :, :sim.N_STATS] = 0.6
    won = pts = 0
    hits = []
    for _ in range(steps):
        a0, a1 = pol.act(obs[:, 0]), pol.act(obs[:, 1])
        if k >= 0:
            serving = (env.I[:, sim.PHASE] == sim.PH_SERVE) & (env.I[:, sim.SERVER] == 0)
            forced = a0.copy()
            serve_action(k, forced, 0.0, 0.0, 0.0)
            for col in (8, 10):
                a0[serving, col] = forced[serving, col]
        server = env.I[:, sim.SERVER].copy()
        obs, _, done = env.step(np.stack([a0, a1], 1))
        m = done & (server == 0) & (env.winner >= 0)
        won += int((env.winner[m] == 0).sum())
        pts += int(m.sum())
        hits += list(env.hits[m])
    return won / max(pts, 1), float(np.mean(hits)), pts


def main():
    pol = Policy.load(sys.argv[1] if len(sys.argv) > 1 else "models/policy.npz")
    wr, h, n = run(pol, -1)
    print(f"{'模型自选':10s} 发球方胜率 {wr:5.1%}  每分 {h:.1f} 拍  ({n} 分)")
    for k, name in enumerate(sim.SERVE_NAMES):
        wr, h, n = run(pol, k)
        print(f"{name:10s} 发球方胜率 {wr:5.1%}  每分 {h:.1f} 拍  ({n} 分)")


if __name__ == "__main__":
    main()
