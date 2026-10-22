import os

import numpy as np

from . import sim
from .sim import ACT_DIM, HIST_DIM, N_S, N_STATS, N_STYLE, NI, NT, OBS_DIM

# PP_BACKEND=c：物理用 C++ 核心（core/ppcore.dll，与游戏共用、与 numba 版逐位一致），否则用 numba 版
_USE_C = os.environ.get("PP_BACKEND", "").lower() == "c"
if _USE_C:
    from . import core as _core


class PongEnv:
    """n 张球台并行对打。每个球员的观测和动作都在“自己视角”下（自己总在 x<0 一侧）。"""

    def __init__(self, n, train=True, seed=0, nsub=8, dt=1.0 / 30.0):
        self.n = n
        self.train = train
        self.nsub = nsub
        self.dt = dt
        self.core = _core.Core(n, seed) if _USE_C else None
        if self.core is None:
            sim.seed_rng(seed)
        rng = np.random.default_rng(seed)
        self.B = np.zeros((n, 9))
        self.P = np.zeros((n, 2, 6))
        self.I = np.zeros((n, NI), dtype=np.int64)
        self.T = np.zeros((n, NT))
        self.S = rng.random((n, 2, N_S)) if train else np.full((n, 2, N_S), 0.6)
        # 只有“球风”维度是 -1..1；属性（含发球/接发）都是 0..1
        self.S[:, :, N_STATS:N_STATS + N_STYLE] = (
            self.S[:, :, N_STATS:N_STATS + N_STYLE] * 2.0 - 1.0 if train else 0.0)
        # 器材：训练时随机；对战默认右手横板、两面反胶
        self.S[:, :, sim.HAND] = 1.0
        self.S[:, :, sim.GRIP] = 0.0
        self.S[:, :, sim.RFH] = sim.RUB_INV
        self.S[:, :, sim.RBH] = sim.RUB_INV
        if train:
            for i in range(n):
                for j in range(2):
                    self.random_equipment(i, j, np.random.random() < sim.BAL_FRAC)
        self.F = np.zeros((n, 2))
        self.O = np.zeros((n, 2, OBS_DIM))
        self.rew = np.zeros((n, 2))
        self.done = np.zeros(n, dtype=np.bool_)
        self.winner = np.full(n, -1, dtype=np.int64)
        self.reason = np.zeros(n, dtype=np.int64)
        self.hits = np.zeros(n, dtype=np.int64)
        self.hist = np.zeros((n, nsub, HIST_DIM))

    def random_equipment(self, i, j, balanced=False):
        if self.core is not None:
            self.core.random_equipment(self.S, i, j, balanced)
        else:
            sim.random_equipment(self.S, i, j, balanced)

    def reset(self):
        if self.core is not None:
            self.core.reset(self.B, self.P, self.I, self.T, self.S, self.F, self.train)
        else:
            sim.reset_kernel(self.B, self.P, self.I, self.T, self.S, self.F, self.train)
        return self._fill_hist()

    def restart_point(self, server):
        """比分清零、指定发球方，重新开始这一分（练习模式用）。"""
        for i in range(self.n):
            self.I[i, sim.SC0] = 0
            self.I[i, sim.SC1] = 0
            self.I[i, sim.FIRST] = server
            if self.core is not None:
                self.core.start_point(i, self.train, self.B, self.P, self.I, self.T, self.S, self.F)
            else:
                sim.start_point(i, self.train, self.B, self.P, self.I, self.T, self.S, self.F)
        return self._fill_hist()

    def _fill_hist(self):
        for s in range(self.nsub):
            self.hist[:, s, 0:3] = self.B[:, 0:3]
            self.hist[:, s, 3:6] = self.P[:, 0, 0:3]
            self.hist[:, s, 6:9] = self.P[:, 1, 0:3]
            for j in range(2):
                self.hist[:, s, 9 + 3 * j:12 + 3 * j] = self.P[:, j, 0:3]
                self.hist[:, s, 10 + 3 * j] += self.T[:, sim.LM0 + j] * sim.hand_off(self.T[:, sim.BL0 + j])
                self.hist[:, s, 15 + j] = self.T[:, sim.BL0 + j]
            self.hist[:, s, 23] = -1.0
        return self.observe()

    def observe(self):
        if self.core is not None:
            self.core.observe(self.B, self.P, self.I, self.T, self.S, self.F, self.O)
        else:
            sim.obs_kernel(self.B, self.P, self.I, self.T, self.S, self.F, self.O)
        return self.O.copy()

    def step(self, actions):
        actions = np.ascontiguousarray(actions, dtype=np.float64).reshape(self.n, 2, ACT_DIM)
        if self.core is not None:
            self.core.step(self.B, self.P, self.I, self.T, self.S, self.F, actions,
                           self.rew, self.done, self.winner, self.reason, self.hits, self.hist,
                           self.train, self.nsub, self.dt)
        else:
            sim.step_kernel(self.B, self.P, self.I, self.T, self.S, self.F, actions,
                            self.rew, self.done, self.winner, self.reason, self.hits, self.hist,
                            self.train, self.nsub, self.dt)
        return self.observe(), self.rew.copy(), self.done.copy()
