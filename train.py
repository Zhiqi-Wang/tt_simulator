"""PPO 自我对弈训练（PyTorch 网络，有显卡用显卡；numba 物理多线程并行）。

    python train.py                 # 从头训练
    python train.py --resume        # 接着 models/policy.npz 继续训练
    python train.py --iters 600 --envs 2048 --threads 8

训练中随时 Ctrl+C，会保存当前模型。模型仍存成 numpy 格式，游戏 play.py 不依赖 PyTorch。
"""
import argparse
import csv
import os
import time

import numba
import numpy as np
import torch
from torch import nn

from pingpong.bot import bot_actions
from pingpong.env import PongEnv
from pingpong.nn import Policy
from pingpong.sim import ACT_DIM, OBS_DIM

LOG2PI = float(np.log(2.0 * np.pi))


def warm_start(path, seed, var_floor_dims=(), hidden=None):
    """用旧维度的模型初始化新维度的网络：重合的输入/输出维度照搬权重，新增维度从零开始。
    hidden 比旧网络宽时，新增神经元的输入权重随机、通往旧神经元和输出的权重为零，
    所以加宽后的网络输出与旧网络完全相同，已学到的打法不丢，多出的容量在训练中慢慢用起来。"""
    old = Policy.load(path)
    old_hidden = tuple(old.pi.params[2 * k].shape[1] for k in range(old.pi.n_layers - 1))
    hidden = tuple(hidden) if hidden else old_hidden
    assert len(hidden) == len(old_hidden) and all(h >= o for h, o in zip(hidden, old_hidden))
    pol = Policy(OBS_DIM, ACT_DIM, hidden=hidden, seed=seed)
    for new_net, old_net in ((pol.pi, old.pi), (pol.vf, old.vf)):
        for k, (p, q) in enumerate(zip(new_net.params, old_net.params)):
            if p.ndim == 2:
                if k == 0:
                    p[q.shape[0]:] = 0.0
                else:
                    p[q.shape[0]:, :q.shape[1]] = 0.0
                if k == len(new_net.params) - 2:
                    p[q.shape[0]:] = 0.0
                else:
                    p[:, q.shape[1]:] *= 0.5
            else:
                p[q.shape[0]:] = 0.0
            sl = tuple(slice(0, s) for s in q.shape)
            p[sl] = q
    pol.logstd[:old.logstd.shape[0]] = old.logstd
    d = old.norm.mean.shape[0]
    pol.norm.mean[:d], pol.norm.var[:d], pol.norm.count = old.norm.mean, old.norm.var, old.norm.count
    # 新追加在末尾的维度：按 0..1 的均匀分布先给一套标准化的均值/方差，别让它们一上来是 0.6/1.0 的怪尺度
    pol.norm.mean[d:] = 0.5
    pol.norm.var[d:] = 1.0 / 12.0
    for k in var_floor_dims:
        pol.norm.var[k] = max(pol.norm.var[k], 0.1)
    return pol


def to_torch(mlp, dev):
    layers = []
    for k in range(mlp.n_layers):
        w, b = mlp.params[2 * k], mlp.params[2 * k + 1]
        lin = nn.Linear(w.shape[0], w.shape[1])
        with torch.no_grad():
            lin.weight.copy_(torch.from_numpy(w.T.copy()))
            lin.bias.copy_(torch.from_numpy(b))
        layers.append(lin)
        if k < mlp.n_layers - 1:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers).to(dev)


def to_numpy(net, mlp):
    lins = [m for m in net if isinstance(m, nn.Linear)]
    for k, lin in enumerate(lins):
        mlp.params[2 * k][...] = lin.weight.detach().cpu().numpy().T
        mlp.params[2 * k + 1][...] = lin.bias.detach().cpu().numpy()


class Adam:
    """torch.optim 会连带导入 torch._dynamo → onnx，本机装的 onnx 版本与 torch 不兼容会崩溃，所以手写。"""

    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-5):
        self.params = list(params)
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self):
        self.t += 1
        c1 = 1.0 - self.b1 ** self.t
        c2 = 1.0 - self.b2 ** self.t
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            m.mul_(self.b1).add_(p.grad, alpha=1.0 - self.b1)
            v.mul_(self.b2).addcmul_(p.grad, p.grad, value=1.0 - self.b2)
            p.addcdiv_(m, (v / c2).sqrt_().add_(self.eps), value=-self.lr / c1)


def clip_grad(params, max_norm):
    gs = [p.grad for p in params if p.grad is not None]
    norm = torch.sqrt(sum((g * g).sum() for g in gs))
    scale = torch.clamp(max_norm / (norm + 1e-6), max=1.0)
    for g in gs:
        g.mul_(scale)


def normalize(x, mean, var):
    return np.clip((x - mean) / np.sqrt(var + 1e-8), -10, 10).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--bound", type=float, default=0.1)
    ap.add_argument("--var-floor", default="", help="热启动时给这些观察维度（如 57:65）的标准化方差设下限")
    ap.add_argument("--ent", type=float, default=0.0)
    ap.add_argument("--max-std", type=float, default=0.35, help="探索噪声上限，太大时策略会只敢打保险球")
    ap.add_argument("--min-std", type=float, default=0.15, help="探索噪声下限，防止过早收敛到单一打法")
    ap.add_argument("--bot-frac", type=float, default=0.15)
    ap.add_argument("--pool-frac", type=float, default=0.3, help="与历史版本对打的球台比例")
    ap.add_argument("--pool-every", type=int, default=25)
    ap.add_argument("--pool-size", type=int, default=12)
    ap.add_argument("--pool-add", default="", help="常驻对手池的外部模型，逗号分隔")
    ap.add_argument("--fixed-frac", type=float, default=0.5, help="每轮对手池抽到常驻外部模型的概率")
    ap.add_argument("--threads", type=int, default=16, help="物理模拟用的 CPU 线程数（物理是唯一的 CPU 大头）")
    ap.add_argument("--tthreads", type=int, default=2,
                    help="torch 的 CPU 线程数。网络跑在 GPU 上，torch 的 CPU 线程池空转时"
                         "会和 numba 抢物理线程，实测设成 1~2 明显更快")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models/policy.npz")
    ap.add_argument("--serve-explore", type=float, default=0.0,
                    help="每分以此概率强制随机换一种发球手法（离策略样本，大了会让训练不稳定）")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--prof", action="store_true", help="打印每轮各阶段耗时")
    ap.add_argument("--warm", default=None, help="用旧版本（观测/动作维度不同）的模型初始化")
    ap.add_argument("--hidden", default="", help="隐藏层宽度，如 1024,1024,1024,1024；热启动时只能加宽（输出不变）")
    ap.add_argument("--fp32", action="store_true", help="关掉 bf16 混合精度（默认网络矩阵乘法用 bf16，权重与损失仍是 fp32）")
    args = ap.parse_args()

    numba.set_num_threads(min(args.threads, numba.config.NUMBA_NUM_THREADS))
    torch.set_num_threads(max(1, min(args.tthreads, numba.config.NUMBA_NUM_THREADS)))
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    use_amp = dev.type == "cuda" and not args.fp32

    def amp():
        return torch.autocast("cuda", torch.bfloat16, enabled=use_amp)

    print(f"device {dev}, physics threads {numba.get_num_threads()}, bf16 {use_amp}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rng = np.random.default_rng(args.seed)
    if args.resume and os.path.exists(args.out):
        pol = Policy.load(args.out)
        print(f"resumed from {args.out} (iteration {pol.iteration})")
    elif args.warm:
        floor = range(*map(int, args.var_floor.split(":"))) if args.var_floor else ()
        grow = tuple(map(int, args.hidden.split(","))) if args.hidden else None
        pol = warm_start(args.warm, args.seed, floor, grow)
        print(f"warm-started from {args.warm}")
    else:
        hidden = tuple(map(int, args.hidden.split(","))) if args.hidden else (128, 128)
        pol = Policy(OBS_DIM, ACT_DIM, hidden=hidden, seed=args.seed)

    max_logstd = float(np.log(args.max_std))
    min_logstd = float(np.log(args.min_std))
    pi_net = to_torch(pol.pi, dev)
    vf_net = to_torch(pol.vf, dev)
    opp_net = to_torch(pol.pi, dev)
    logstd = nn.Parameter(torch.from_numpy(np.clip(pol.logstd, min_logstd, max_logstd)).to(dev))
    pi_params = list(pi_net.parameters()) + [logstd]
    opt_pi = Adam(pi_params, args.lr)
    opt_vf = Adam(vf_net.parameters(), args.lr)

    def sync_numpy():
        to_numpy(pi_net, pol.pi)
        to_numpy(vf_net, pol.vf)
        pol.logstd[...] = logstd.detach().cpu().numpy()

    n, T = args.envs, args.steps
    M = 2 * n
    env = PongEnv(n, train=True, seed=args.seed)
    obs = env.reset()

    # 观测归一化放到 GPU 上做：4096x87 的 numpy 归一化每步约 1.8 ms，一轮 128 步就是 0.23 s，
    # 而搬上 GPU 后几乎免费。运行均值/方差每次更新后同步一次（只有 87 个数）。
    mean_t = torch.from_numpy(pol.norm.mean).float().to(dev)
    inv_t = torch.from_numpy(1.0 / np.sqrt(pol.norm.var + 1e-8)).float().to(dev)

    def norm_obs(x):
        """numpy 观测 → 归一化后的 GPU 张量（等价于 pol.norm）。"""
        return torch.clamp((torch.from_numpy(x).to(dev).float() - mean_t) * inv_t, -10.0, 10.0)

    def refresh_norm():
        mean_t.copy_(torch.from_numpy(pol.norm.mean).float())
        inv_t.copy_(torch.from_numpy(1.0 / np.sqrt(pol.norm.var + 1e-8)).float())

    n_bot = int(n * args.bot_frac)
    bot_env = np.arange(n_bot)
    bot_side = bot_env % 2
    n_pool = int(n * args.pool_frac)
    pool_env = np.arange(n_bot, n_bot + n_pool)
    pool_side = pool_env % 2
    learn_mask = np.ones((n, 2), bool)
    learn_mask[bot_env, bot_side] = False
    learn_mask[pool_env, pool_side] = False
    learn_mask = learn_mask.reshape(M)
    sel = np.repeat(learn_mask[None], T, 0).reshape(-1)
    sel_t = torch.from_numpy(sel).to(dev)

    pool = []

    def snapshot():
        pool.append(({k: v.clone() for k, v in pi_net.state_dict().items()},
                     pol.norm.mean.copy(), pol.norm.var.copy()))
        del pool[:-args.pool_size]

    snapshot()
    # 固定陪练：外部模型常驻对手池（旧观测维度按热启动方式补零），防止自我对弈只学会克制自己
    fixed = []
    for path in filter(None, args.pool_add.split(",")):
        old = warm_start(path, args.seed)
        fixed.append((to_torch(old.pi, dev), old.norm.mean.copy(), old.norm.var.copy()))
        print(f"fixed pool opponent: {path}")

    log_path = os.path.splitext(args.out)[0] + "_log.csv"
    new_log = not (args.resume and os.path.exists(log_path))
    log_f = open(log_path, "w" if new_log else "a", newline="")
    log = csv.writer(log_f)
    if new_log:
        log.writerow(["iter", "samples", "reward_per_step", "hits_per_point", "win_vs_bot",
                      "points", "pi_loss", "v_loss", "kl", "std", "sec"])

    ob_buf = torch.zeros((T, M, OBS_DIM), device=dev)
    ac_buf = torch.zeros((T, M, ACT_DIM), device=dev)
    lp_buf = torch.zeros((T, M), device=dev)
    v_buf = np.zeros((T, M), np.float32)
    r_buf = np.zeros((T, M), np.float32)
    d_buf = np.zeros((T, M), np.float32)
    raw_buf = np.zeros((T // 4 + 1, M, OBS_DIM))
    sv_on = rng.random(M) < args.serve_explore
    sv_val = rng.uniform(-1.0, 1.0, (M, 3))

    def logp(a, mu):
        z = (a - mu) / torch.exp(logstd)
        return (-0.5 * z * z - logstd - 0.5 * LOG2PI).sum(-1)

    start_iter = pol.iteration
    t_start = time.time()
    try:
        for it in range(start_iter, start_iter + args.iters):
            t0 = time.perf_counter()
            frac = 1.0 - (it - start_iter) / args.iters
            lr = args.lr * max(frac, 0.1)
            opt_pi.lr = opt_vf.lr = lr
            hits, wins_vs_bot, games_vs_bot, points = [], 0, 0, 0
            tp_phys = [0.0]
            if fixed and rng.random() < args.fixed_frac:
                opp, opp_mean, opp_var = fixed[rng.integers(len(fixed))]
            else:
                sd, opp_mean, opp_var = pool[rng.integers(len(pool))]
                opp_net.load_state_dict(sd)
                opp = opp_net

            with torch.no_grad(), amp():
                for t in range(T):
                    raw = obs.reshape(M, OBS_DIM)
                    if t % 4 == 0:
                        raw_buf[t // 4] = raw
                    on = norm_obs(raw)
                    mu = pi_net(on).float()
                    a = mu + torch.exp(logstd) * torch.randn_like(mu)
                    serving = (raw[:, 20] > 0.5) & (raw[:, 38] > 0.5) & sv_on
                    if serving.any():
                        idx = torch.from_numpy(np.nonzero(serving)[0]).to(dev)
                        a[idx, 8] = torch.from_numpy(sv_val[serving, 0]).float().to(dev)
                        a[idx, 10] = torch.from_numpy(sv_val[serving, 1]).float().to(dev)
                        a[idx, 6] = torch.from_numpy(sv_val[serving, 2]).float().to(dev)
                    v = vf_net(on)[:, 0].float()
                    ob_buf[t], ac_buf[t] = on, a
                    lp_buf[t] = logp(a, mu)
                    act = a.cpu().numpy().reshape(n, 2, ACT_DIM).astype(np.float64)
                    v_buf[t] = v.cpu().numpy()
                    if n_bot:
                        act[bot_env, bot_side] = bot_actions(obs[bot_env, bot_side], rng)
                    if n_pool:
                        po = torch.from_numpy(normalize(obs[pool_env, pool_side], opp_mean, opp_var)).to(dev)
                        act[pool_env, pool_side] = opp(po).float().cpu().numpy()
                    q0 = time.perf_counter()
                    obs, rew, done = env.step(act)
                    if args.prof:
                        tp_phys[0] += time.perf_counter() - q0
                    r_buf[t] = rew.reshape(M)
                    d_buf[t] = np.repeat(done, 2)
                    redraw = np.repeat(done, 2)
                    if redraw.any():
                        k = int(redraw.sum())
                        sv_on[redraw] = rng.random(k) < args.serve_explore
                        sv_val[redraw] = rng.uniform(-1.0, 1.0, (k, 3))
                    if done.any():
                        hits += list(env.hits[done])
                        points += int(done.sum())
                        bd = done[:n_bot] & (env.winner[:n_bot] >= 0)
                        games_vs_bot += int(bd.sum())
                        wins_vs_bot += int((env.winner[:n_bot][bd] != bot_side[bd]).sum())
                last_v = vf_net(norm_obs(obs.reshape(M, OBS_DIM)))[:, 0].float().cpu().numpy()
            tp1 = time.perf_counter()

            pol.norm.update(raw_buf.reshape(-1, M, OBS_DIM)[:, learn_mask].reshape(-1, OBS_DIM))
            refresh_norm()
            tp2 = time.perf_counter()
            adv = np.zeros((T, M), np.float32)
            gae = np.zeros(M, np.float32)
            for t in reversed(range(T)):
                nv = last_v if t == T - 1 else v_buf[t + 1]
                nonterm = 1.0 - d_buf[t]
                delta = r_buf[t] + args.gamma * nv * nonterm - v_buf[t]
                gae = delta + args.gamma * args.lam * nonterm * gae
                adv[t] = gae
            ret = adv + v_buf
            tp3 = time.perf_counter()

            b_ob = ob_buf.reshape(-1, OBS_DIM)[sel_t]
            b_ac = ac_buf.reshape(-1, ACT_DIM)[sel_t]
            b_lp = lp_buf.reshape(-1)[sel_t]
            b_adv = torch.from_numpy(adv.reshape(-1)[sel]).to(dev)
            b_ret = torch.from_numpy(ret.reshape(-1)[sel]).to(dev)
            N = b_ob.shape[0]
            tp4 = time.perf_counter()

            pi_losses, v_losses, kls = [], [], []
            stop = False
            for ep in range(args.epochs):
                perm = torch.randperm(N, device=dev)
                # 每个 epoch 只做一次大重排，之后按连续切片取 minibatch。
                # 逐个 minibatch 用花式索引会多出 3×108 次 16384x87 的随机 gather，实测每轮约 0.2 s。
                bo, ba, bl = b_ob[perm], b_ac[perm], b_lp[perm]
                bA, bR = b_adv[perm], b_ret[perm]
                for s in range(0, N - args.minibatch + 1, args.minibatch):
                    e = s + args.minibatch
                    o, ac, lp_old = bo[s:e], ba[s:e], bl[s:e]
                    A = bA[s:e]
                    A = (A - A.mean()) / (A.std() + 1e-8)

                    with amp():
                        mu_b = pi_net(o).float()
                    lp = logp(ac, mu_b)
                    ratio = torch.exp((lp - lp_old).clamp(-20.0, 20.0))
                    surr = torch.min(ratio * A, torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * A)
                    over = (mu_b.abs() - 1.0).clamp(min=0.0)
                    loss_pi = (-surr.mean() - args.ent * logstd.sum()
                               + args.bound * (over * over).sum(-1).mean())
                    opt_pi.zero_grad()
                    loss_pi.backward()
                    clip_grad(pi_params, 0.5)
                    opt_pi.step()
                    with torch.no_grad():
                        logstd.clamp_(min_logstd, max_logstd)

                    with amp():
                        vo = vf_net(o)[:, 0].float()
                    err = vo - bR[s:e]
                    loss_v = 0.5 * (err * err).mean()
                    opt_vf.zero_grad()
                    loss_v.backward()
                    clip_grad(opt_vf.params, 0.5)
                    opt_vf.step()

                    kl = float((lp_old - lp).mean())
                    pi_losses.append(float(loss_pi))
                    v_losses.append(2.0 * float(loss_v))
                    kls.append(kl)
                    if kl > 0.03:
                        stop = True
                        break
                if stop:
                    break

            pol.iteration = it + 1
            if (it + 1) % args.pool_every == 0:
                snapshot()
            tp5 = time.perf_counter()
            sec = time.perf_counter() - t0
            hpp = float(np.mean(hits)) if hits else 0.0
            wvb = wins_vs_bot / max(games_vs_bot, 1)
            rps = float(r_buf.reshape(-1)[sel].mean())
            std = float(torch.exp(logstd).mean())
            row = [it + 1, (it + 1 - start_iter) * T * M, round(rps, 5), round(hpp, 3), round(wvb, 3), points,
                   round(float(np.mean(pi_losses)), 4), round(float(np.mean(v_losses)), 4),
                   round(float(np.mean(kls)), 4), round(std, 3), round(sec, 2)]
            log.writerow(row)
            log_f.flush()
            elapsed = time.time() - t_start
            eta = elapsed / (it + 1 - start_iter) * (start_iter + args.iters - it - 1)
            print(f"iter {it + 1:4d} | hits/point {hpp:5.2f} | win vs bot {wvb:5.1%} | r/step {rps:+.4f} | "
                  f"vloss {row[7]:.3f} kl {row[8]:.4f} std {row[9]:.2f} | {sec:4.1f}s | eta {eta / 60:5.1f} min",
                  flush=True)
            if args.prof:
                parts = (("采样(物理+前向)", tp1 - t0), ("归一化统计", tp2 - tp1), ("GAE", tp3 - tp2),
                         ("拼 batch", tp4 - tp3), ("PPO 更新", tp5 - tp4), ("其它", sec - (tp5 - t0)))
                print("        " + "  ".join(f"{k} {v * 1000:5.0f}ms" for k, v in parts)
                      + f"  [其中物理 {tp_phys[0] * 1000:5.0f}ms]", flush=True)
            if (it + 1) % 5 == 0:
                sync_numpy()
                pol.save(args.out)
    except KeyboardInterrupt:
        print("interrupted, saving...")
    sync_numpy()
    pol.save(args.out)
    log_f.close()
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
