"""战术探针：蓝方移动仍由模型控制，但击球意图被强制成某种打法，看它对模型的胜率。
胜率明显高于 50% 说明这种打法在当前物理下“值得”，RL 才有动力学会它。

    python probe.py                       # 默认 models/policy.npz
    python probe.py --model xxx.npz --steps 3000
"""
import argparse

import numpy as np

from pingpong import sim
from pingpong.env import PongEnv
from pingpong.nn import Policy


def override(name, a, env, obs):
    """改写蓝方（0 号，自己视角 = 世界坐标）的击球意图。"""
    opp_y = env.P[:, 1, 1]
    open_side = -np.sign(opp_y + 1e-6)
    if name == "base":
        return a
    if name == "use_bh":
        vis = obs[:, 41] != 0
        a[:, 8] = np.where(vis, np.where(obs[:, 41] > -0.07, -1.0, 1.0), a[:, 8])
        return a
    if name in ("fast", "fast_wide", "fake_wide", "fast_fake"):
        a[:, 5] = 1.0
    if name in ("wide", "fast_wide", "fake_wide", "fast_fake"):
        a[:, 4] = open_side * 0.8
        a[:, 10] = a[:, 4]
    if name in ("fake_wide", "fast_fake"):
        a[:, 10] = -a[:, 4]
    if name == "slow":
        a[:, 5] = -1.0
    if name == "middle":
        a[:, 4] = 0.0
        a[:, 10] = 0.0
    if name == "elbow":
        a[:, 4] = np.clip((opp_y + 0.15) / sim.LAT_MAX, -1.0, 1.0)
        a[:, 10] = a[:, 4]
    if name == "switch":
        global LAST_SIDE
        a[:, 4] = -LAST_SIDE * 0.8
        a[:, 10] = a[:, 4]
        hitting = (env.I[:, sim.TURN] == 0) & (obs[:, 21] > 0.5)
        LAST_SIDE = np.where(hitting & (env.I[:, sim.BOUNCED] == 1), np.sign(a[:, 4]), LAST_SIDE)
    if name == "pivot":
        a[:, 8] = 1.0
    serving = (obs[:, 20] > 0.5) & (obs[:, 38] > 0.5)
    rally = env.I[:, sim.PHASE] != sim.PH_SERVE
    high = rally & (env.B[:, 2] - sim.TABLE_H > 0.2)
    if name == "fast_rally":
        a[:, 5] = np.where(rally, 1.0, a[:, 5])
    if name in ("chance", "chance_wide"):
        a[:, 5] = np.where(high, 1.0, a[:, 5])
        a[:, 6] = np.where(high, 0.6, a[:, 6])
    if name == "chance_wide":
        a[:, 4] = np.where(high, open_side * 0.8, a[:, 4])
        a[:, 10] = a[:, 4]
    if name in ("push_back", "push_short"):
        v, w = env.B[:, 3:6], env.B[:, 6:9]
        hn = np.hypot(v[:, 0], v[:, 1]) + 1e-9
        top_in = w[:, 0] * (-v[:, 1] / hn) + w[:, 1] * (v[:, 0] / hn)
        mine = (env.I[:, sim.TURN] == 0) & rally
        over = np.abs(env.T[:, sim.PX]) < sim.HALF_L + 0.1
        cond = mine & (top_in < -50) & over
        a[:, 5] = np.where(cond, -0.6, a[:, 5])
        a[:, 6] = np.where(cond, -0.8, a[:, 6])
        if name == "push_short":
            a[:, 3] = np.where(cond, -1.0, a[:, 3])
    if name == "short_serve":
        a[:, 3] = np.where(serving, -1.0, a[:, 3])
    if name == "long_serve":
        a[:, 3] = np.where(serving, 1.0, a[:, 3])
    if name == "third_attack":
        third = (env.I[:, sim.SERVER] == 0) & (env.I[:, sim.HITS] == 1)
        a[:, 5] = np.where(third, 1.0, a[:, 5])
        a[:, 6] = np.where(third, 0.8, a[:, 6])
    return a


LAST_SIDE = np.ones(512)


def run(pol, name, steps, seed):
    env = PongEnv(512, train=True, seed=seed)
    obs = env.reset()
    rng = np.random.default_rng(seed)
    wins = pts = 0
    reasons = np.zeros((2, 7), int)
    hits = []
    for _ in range(steps):
        a0 = pol.act(obs[:, 0])
        a1 = pol.act(obs[:, 1])
        a0 = override(name, a0.copy(), env, obs[:, 0])
        obs, _, done = env.step(np.stack([a0, a1], 1))
        if done.any():
            w = env.winner[done]
            ok = w >= 0
            wins += int((w[ok] == 0).sum())
            pts += int(ok.sum())
            for w in (0, 1):
                np.add.at(reasons[w], env.reason[done & (env.winner == w)], 1)
            hits += list(env.hits[done])
    del rng
    r = reasons / max(reasons.sum(), 1)
    err = lambda w: r[w, sim.R_OUT] + r[w, sim.R_NET] + r[w, sim.R_OWN]  # noqa: E731
    print(f"{name:10s} 胜率 {wins / max(pts, 1):6.1%}  每分 {np.mean(hits):4.1f} 拍 | 全部分数中："
          f"蓝失误 {err(1):.0%} 蓝没接到 {r[1, sim.R_MISS]:.0%} | 红失误 {err(0):.0%} 红没接到 {r[0, sim.R_MISS]:.0%}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/policy.npz")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    pol = Policy.load(args.model)
    names = args.only.split(",") if args.only else ["base", "slow", "middle", "fast", "wide", "fast_wide",
                                                    "fake_wide", "fast_fake"]
    for name in names:
        run(pol, name, args.steps, args.seed)


if __name__ == "__main__":
    main()
