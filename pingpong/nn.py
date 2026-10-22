"""纯 numpy 的 MLP / Adam / 观测归一化 / 高斯策略（不依赖 PyTorch）。"""
import numpy as np


def orthogonal(shape, gain, rng):
    a = rng.standard_normal(shape)
    u, _, vt = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == shape else vt
    return (gain * q).astype(np.float32)


class MLP:
    def __init__(self, sizes, rng, out_gain=1.0):
        self.n_layers = len(sizes) - 1
        self.params = []
        for k, (i, o) in enumerate(zip(sizes[:-1], sizes[1:])):
            g = out_gain if k == self.n_layers - 1 else np.sqrt(2.0)
            # rng 为 None 时不做正交初始化（大网络的 SVD 要好几秒），由调用方填入权重
            w = orthogonal((i, o), g, rng) if rng is not None else np.zeros((i, o), np.float32)
            self.params += [w, np.zeros(o, np.float32)]
        self._hs = None

    def forward(self, x, keep=False):
        hs = [x]
        h = x
        for k in range(self.n_layers):
            h = h @ self.params[2 * k] + self.params[2 * k + 1]
            if k < self.n_layers - 1:
                h = np.tanh(h)
            hs.append(h)
        if keep:
            self._hs = hs
        return h

    def backward(self, d):
        hs = self._hs
        grads = [None] * len(self.params)
        for k in reversed(range(self.n_layers)):
            hin = hs[k]
            grads[2 * k] = hin.T @ d
            grads[2 * k + 1] = d.sum(0)
            if k > 0:
                d = (d @ self.params[2 * k].T) * (1.0 - hin * hin)
        return grads


class Adam:
    def __init__(self, params, b1=0.9, b2=0.999, eps=1e-5):
        self.params = params
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.b1, self.b2, self.eps, self.t = b1, b2, eps, 0

    def step(self, grads, lr):
        self.t += 1
        c1 = 1.0 - self.b1 ** self.t
        c2 = 1.0 - self.b2 ** self.t
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.b1
            m += (1.0 - self.b1) * g
            v *= self.b2
            v += (1.0 - self.b2) * g * g
            p -= (lr * (m / c1) / (np.sqrt(v / c2) + self.eps)).astype(p.dtype)


class RunningNorm:
    def __init__(self, dim):
        self.mean = np.zeros(dim)
        self.var = np.ones(dim)
        self.count = 1e-4

    def update(self, x):
        bm, bv, bn = x.mean(0), x.var(0), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bn
        self.mean = self.mean + delta * bn / tot
        m2 = self.var * self.count + bv * bn + delta ** 2 * self.count * bn / tot
        self.var = m2 / tot
        self.count = tot

    def __call__(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -10, 10).astype(np.float32)


class Policy:
    def __init__(self, obs_dim, act_dim, hidden=(128, 128), seed=0):
        rng = np.random.default_rng(seed) if seed is not None else None
        self.pi = MLP([obs_dim, *hidden, act_dim], rng, out_gain=0.01)
        self.vf = MLP([obs_dim, *hidden, 1], rng, out_gain=1.0)
        self.logstd = np.full(act_dim, -0.5, np.float32)
        self.norm = RunningNorm(obs_dim)
        self.iteration = 0

    def act(self, obs, deterministic=True, rng=None):
        mu = self.pi.forward(self.norm(obs))
        if deterministic:
            return mu
        return mu + np.exp(self.logstd) * rng.standard_normal(mu.shape).astype(np.float32)

    def save(self, path):
        d = {f"pi{k}": p for k, p in enumerate(self.pi.params)}
        d.update({f"vf{k}": p for k, p in enumerate(self.vf.params)})
        np.savez(path, logstd=self.logstd, mean=self.norm.mean, var=self.norm.var,
                 count=self.norm.count, iteration=self.iteration, **d)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        obs_dim = z["pi0"].shape[0]
        act_dim = z["logstd"].shape[0]
        n_pi = sum(1 for k in z.files if k.startswith("pi"))
        hidden = tuple(z[f"pi{2 * k}"].shape[1] for k in range(n_pi // 2 - 1))
        p = cls(obs_dim, act_dim, hidden, seed=None)
        for k in range(len(p.pi.params)):
            p.pi.params[k][...] = z[f"pi{k}"]
            p.vf.params[k][...] = z[f"vf{k}"]
        p.logstd[...] = z["logstd"]
        p.norm.mean, p.norm.var, p.norm.count = z["mean"], z["var"], float(z["count"])
        p.iteration = int(z["iteration"])
        return p
