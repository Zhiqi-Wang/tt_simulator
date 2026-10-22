"""批量对局评估：不同属性 / 控制器组合下的胜率和回合长度。

    python evaluate.py [models/xxx.npz]
"""
import sys

import numpy as np

from pingpong import sim
from pingpong.bot import bot_actions
from pingpong.env import PongEnv
from pingpong.nn import Policy
from pingpong.tactics import TacticTracker
from pingpong.sim import ACT_DIM, STAT_ORDER, STAT_ORDER_NAMES

N_ENVS = 512
STEPS = 3000


def act_of(policy, obs):
    """新增的观察维度都追加在末尾，旧模型截掉即可。"""
    return policy.act(obs[..., :policy.norm.mean.shape[0]])


def apply_stats(env, j, stats):
    """stats：按界面顺序 力量/敏捷/技术/发球/反应/体力（0-100）。"""
    for idx, v in zip(STAT_ORDER, stats):
        env.S[:, j, idx] = v / 100.0


def run(policy, stats0, stats1, ctrl=("rl", "rl"), seed=0):
    env = PongEnv(N_ENVS, train=False, seed=seed)
    apply_stats(env, 0, stats0)
    apply_stats(env, 1, stats1)
    obs = env.reset()
    rng = np.random.default_rng(seed)
    wins = np.zeros(2)
    hits = []
    for _ in range(STEPS):
        act = np.zeros((N_ENVS, 2, ACT_DIM))
        for j in range(2):
            if ctrl[j] == "rl":
                act[:, j] = act_of(policy, obs[:, j])
            else:
                act[:, j] = bot_actions(obs[:, j], rng)
        obs, _, done = env.step(act)
        if done.any():
            w = env.winner[done]
            wins += [(w == 0).sum(), (w == 1).sum()]
            hits += list(env.hits[done])
    return wins[0] / max(wins.sum(), 1), float(np.mean(hits)), int(wins.sum())


def tactics(policy, stats=(60,) * 6, seed=0, steps=3000):
    """RL 对 RL：统计每一拍的正反手、球速、旋转类型、落点，以及每分的结束方式。"""
    env = PongEnv(N_ENVS, train=False, seed=seed)
    apply_stats(env, 0, stats)
    apply_stats(env, 1, stats)
    obs = env.reset()
    last_hits = env.I[:, sim.HITS].copy()
    last_bounced = env.I[:, sim.BOUNCED].copy()
    shots, lands, reasons, hits = [], [], np.zeros(len(sim.REASON_TEXT), int), []
    fakes, fake_wins, cross_t, move_t, leans = 0, 0, 0, 0, []
    serves = []
    tracker = TacticTracker(N_ENVS)
    for _ in range(steps):
        obs, _, done = env.step(np.stack([act_of(policy, obs[:, 0]), act_of(policy, obs[:, 1])], 1))
        fly = env.I[:, sim.PHASE] == sim.PH_FLY
        for j in range(2):
            moving = np.abs(env.P[:, j, 4]) > 0.5
            move_t += int(moving.sum())
            cross_t += int((moving & (env.T[:, sim.CX0 + j] > 0)).sum())
            cueing = obs[:, j, 47] > 0.5
            leans += list(np.abs(obs[cueing, j, 43]))
        fakes += int((fly & (env.I[:, sim.LFAKE] > 0) & (env.I[:, sim.HITS] != last_hits)).sum())
        if done.any():
            fake_wins += int((done & (env.I[:, sim.LFAKE] > 0) & (env.reason == sim.R_MISS)).sum())
        new = (env.I[:, sim.HITS] != last_hits) & fly
        for e in np.nonzero(new)[0]:
            v, w = env.B[e, 3:6], env.B[e, 6:9]
            hn = np.hypot(v[0], v[1]) + 1e-9
            top = (w[0] * -v[1] + w[1] * v[0]) / hn * 60 / (2 * np.pi)
            shots.append((env.I[e, sim.LSIDE], np.linalg.norm(v) * 3.6, top, abs(w[2]) * 60 / (2 * np.pi),
                          env.I[e, sim.LSTK], env.I[e, sim.LTECH], env.I[e, sim.HITS]))
        tracker.on_step(env, last_hits, last_bounced, done)
        bounced = (env.I[:, sim.BOUNCED] == 1) & (last_bounced == 0) & fly & (env.I[:, sim.HITS] > 0)
        for e in np.nonzero(bounced)[0]:
            lands.append((abs(env.B[e, 0]), abs(env.B[e, 1])))
        last_hits = env.I[:, sim.HITS].copy()
        last_bounced = env.I[:, sim.BOUNCED].copy()
        if done.any():
            for e in np.nonzero(done)[0]:
                serves.append((env.I[e, sim.LSRV], env.winner[e] == env.I[e, sim.SERVER], env.hits[e]))
            for r in env.reason[done]:
                reasons[r] += 1
            hits += list(env.hits[done])
    s, ld = np.array(shots), np.array(lands)
    n_pts = max(reasons.sum(), 1)
    print(f"  每分平均击球 {np.mean(hits):.1f} 拍；得分方式：对方没接到（主动得分）{reasons[sim.R_MISS] / n_pts:.0%}，"
          f"下网 {reasons[sim.R_NET] / n_pts:.0%}，出界 {(reasons[sim.R_OUT] + reasons[sim.R_OWN]) / n_pts:.0%}，"
          f"超时 {reasons[sim.R_TIMEOUT] / n_pts:.0%}，发球违例 {(reasons[sim.R_FAULT] + reasons[sim.R_SFAULT]) / n_pts:.0%}，"
          f"擦网重发 {reasons[sim.R_LET] / n_pts:.0%}")
    print(f"  反手 {s[:, 0].mean():.0%}   球速 平均 {s[:, 1].mean():.0f} km/h，"
          f"慢球(<40) {(s[:, 1] < 40).mean():.0%}，快球(>60) {(s[:, 1] > 60).mean():.0%}")
    print(f"  上旋 {(s[:, 2] > 300).mean():.0%}  下旋 {(s[:, 2] < -300).mean():.0%}  "
          f"不转 {(np.abs(s[:, 2]) <= 300).mean():.0%}  带明显侧旋 {(s[:, 3] > 400).mean():.0%}")
    print(f"  落点：短球(离网<0.5m) {(ld[:, 0] < 0.5).mean():.0%}  底线(>1.0m) {(ld[:, 0] > 1.0).mean():.0%}  "
          f"大角度(离中线>0.5m) {(ld[:, 1] > 0.5).mean():.0%}  中路(<0.2m) {(ld[:, 1] < 0.2).mean():.0%}")
    kinds = "  ".join(f"{name} {(s[:, 4] == k).mean():.0%}" for k, name in enumerate(sim.STROKE_NAMES))
    print(f"  技术：{kinds}")
    for title, m in (("接发球", s[:, 6] == 1), ("相持", s[:, 6] > 1)):
        sub = s[m]
        parts = [(f"{'' if t in sim.TECH_NOSIDE else ('反手' if b else '正手')}{sim.TECH_NAMES[t]}",
                  ((sub[:, 5] == t) & (sub[:, 0] == b)).mean())
                 for t in range(len(sim.TECH_NAMES)) for b in (0, 1)
                 if not (t in sim.TECH_NOSIDE and b == 0)]
        merged = {}
        for name, f in parts:
            merged[name] = merged.get(name, 0.0) + f
        print(f"  {title}技术细分：" + "  ".join(f"{n} {f:.0%}" for n, f in sorted(merged.items(), key=lambda x: -x[1])
                                          if f >= 0.01))
    sv = np.array(serves, dtype=float).reshape(-1, 3)
    parts = []
    for k, name in enumerate(sim.SERVE_NAMES):
        m = sv[:, 0] == k
        if m.any():
            parts.append(f"{name} {m.mean():.0%}(发球方得分 {sv[m, 1].mean():.0%})")
    print(f"  发球：{'  '.join(parts)}；发球直接得分 {((sv[:, 2] == 0) & (sv[:, 1] == 1)).mean():.0%}")
    print(f"  假动作骗到重心 {fakes / max(len(s), 1):.1%} 拍，其中直接得分 {fake_wins / max(fakes, 1):.0%}；"
          f"移动中交叉步占 {cross_t / max(move_t, 1):.0%}；对手引拍时压重心程度 {np.mean(leans) if leans else 0:.2f}")
    print("  战术：")
    for line in tracker.report(int(n_pts)):
        print("    " + line)


def main():
    pol = Policy.load(sys.argv[1] if len(sys.argv) > 1 else "models/policy.npz")
    print(f"model iteration {pol.iteration}\n")
    print("技战术统计（RL 对 RL，属性全 60）")
    tactics(pol)
    print()
    mid = [60] * 6
    cases = [
        ("RL vs 脚本AI（同属性60）", mid, mid, ("rl", "bot")),
        ("RL vs RL（同属性60）", mid, mid, ("rl", "rl")),
        ("RL 全80 vs RL 全40", [80] * 6, [40] * 6, ("rl", "rl")),
    ]
    for k, name in enumerate(STAT_ORDER_NAMES):
        hi = list(mid)
        hi[k] = 95
        lo = list(mid)
        lo[k] = 25
        cases.append((f"只比{name}：95 vs 25", hi, lo, ("rl", "rl")))
    for name, s0, s1, ctrl in cases:
        wr, h, n = run(pol, s0, s1, ctrl)
        print(f"{name:<22s}  左方胜率 {wr:6.1%}   平均每分击球 {h:5.2f}   ({n} 分)")


if __name__ == "__main__":
    main()
