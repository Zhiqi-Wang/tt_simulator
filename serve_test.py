"""发球强度测试：强制某一方用指定发球（全力、指定落点），另一方用脚本 AI 接发，统计发球直接得分率。

    python serve_test.py
"""
import numpy as np

from pingpong import sim
from pingpong.bot import bot_actions
from pingpong.env import PongEnv
from pingpong.sim import ACT_DIM, N_STATS, SERVE_NAMES

N_ENVS = 512
STEPS = 2000

FH_KEYS = (sim.SV_PEND, sim.SV_HOOK, sim.SV_SHOVEL, sim.SV_SQUAT, sim.SV_CHOP, sim.SV_FAST)


def serve_action(k, a, lat, depth, toss):
    if k in (sim.SV_BH, sim.SV_BHFAST):
        a[:, 8] = -1.0
        a[:, 10] = -0.5 if k == sim.SV_BH else 0.5
    else:
        a[:, 8] = 1.0
        a[:, 10] = FH_KEYS.index(k) / 3.0 - 1.0 + 1.0 / 6.0
    a[:, 3] = depth
    a[:, 4] = lat
    a[:, 5] = 1.0
    a[:, 9] = toss
    a[:, 11] = -1.0


def run(k, lat=1.0, depth=1.0, toss=1.0, seed=0):
    env = PongEnv(N_ENVS, train=False, seed=seed)
    env.S[:, :, :N_STATS] = 0.6
    obs = env.reset()
    rng = np.random.default_rng(seed)
    aces = pts = 0
    for _ in range(STEPS):
        act = np.stack([bot_actions(obs[:, 0], rng), bot_actions(obs[:, 1], rng)], 1)
        serving = env.I[:, sim.PHASE] == sim.PH_SERVE
        for j in range(2):
            m = serving & (env.I[:, sim.SERVER] == j)
            if m.any():
                sub = act[m, j]
                serve_action(k, sub, lat * rng.choice([-1, 1], m.sum()), depth, toss)
                act[m, j] = sub
        srv = env.I[:, sim.SERVER].copy()
        obs, _, done = env.step(act)
        if done.any():
            srv_won = env.winner[done] == srv[done]
            ace = (env.hits[done] == 0) & srv_won
            aces += int(ace.sum())
            pts += int(done.sum())
    return aces / max(pts, 1), pts


def main():
    for k in range(sim.N_SERVE):
        for lat, depth in ((0.0, 1.0), (1.0, 1.0), (1.0, -1.0)):
            r, n = run(k, lat, depth)
            print(f"{SERVE_NAMES[k]:10s} 横向 {lat:+.0f} 长度 {depth:+.0f} | 直接得分 {r:5.1%}  ({n} 分)")


if __name__ == "__main__":
    main()
