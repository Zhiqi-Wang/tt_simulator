"""战术识别（只读状态，不影响模拟）：每拍击球/落台时判断用了哪种战术，并统计这些战术的得分率。

    发球抢攻    发球方第三板主动发力
    接发球抢攻  接发球（第二板）直接发力上手
    侧身抢攻    球在反手位，侧身用正手发力
    变线        相持中斜线 / 直线互换，打到大角度
    打追身      落点压在对手正反手交界处（肘部）
"""
import numpy as np

from . import sim

TACTICS = ("发球抢攻", "接发球抢攻", "侧身抢攻", "变线", "打追身")
ATTACK = (sim.ST_LOOP, sim.ST_DRIVE, sim.ST_SMASH, sim.ST_FLICK)
ATTACK_KMH = 35.0
WIDE_Y = 0.35
ELBOW_TOL = 0.12


class TacticTracker:
    def __init__(self, n):
        self.n = n
        self.hit_y = np.zeros(n)
        self.prev_dir = np.zeros(n)
        self.label = np.full(n, -1)
        self.label_by = np.full(n, -1)
        self.label_hit = np.zeros(n, int)
        self.count = np.zeros((2, len(TACTICS)), int)
        self.won = np.zeros((2, len(TACTICS)), int)
        self.pending = [[] for _ in range(n)]
        self.base_n = 0
        self.base_w = 0

    def reset(self, i):
        self.prev_dir[i] = 0.0
        self.label[i] = -1
        self.pending[i] = []

    def _tag(self, i, h, k, hits):
        self.label[i], self.label_by[i], self.label_hit[i] = k, h, hits
        self.count[h, k] += 1
        self.pending[i].append((h, k, hits))

    def on_step(self, env, prev_hits, prev_bounced, done):
        """在 env.step 之后调用；返回本步新识别出的 (球台, 球员, 战术编号) 列表。"""
        I, B, P = env.I, env.B, env.P
        out = []
        hit = np.where((I[:, sim.HITS] > prev_hits) & ~done)[0]
        for i in hit:
            h = int(I[i, sim.LHIT])
            m = env.T[i, sim.LM0 + h]
            hits = int(I[i, sim.HITS])
            self.hit_y[i] = B[i, 1]
            self.base_n += 1
            self.pending[i].append((h, -1, hits))
            kmh = float(np.linalg.norm(B[i, 3:6])) * 3.6
            attack = int(I[i, sim.LSTK]) in ATTACK and kmh > ATTACK_KMH
            if not attack:
                continue
            server = int(I[i, sim.SERVER])
            if hits == 3 and h == server:
                self._tag(i, h, 0, hits)
                out.append((i, h, 0))
            elif hits == 2 and h != server:
                self._tag(i, h, 1, hits)
                out.append((i, h, 1))
            fh = I[i, sim.LSIDE] == 0
            if fh and m * (B[i, 1] - P[i, h, 1]) > 0.0 and m * B[i, 1] > 0.15:
                self._tag(i, h, 2, hits)
                out.append((i, h, 2))
        land = np.where((I[:, sim.BOUNCED] == 1) & (prev_bounced == 0) & (I[:, sim.HITS] >= 2)
                        & (I[:, sim.PHASE] == sim.PH_FLY) & ~done)[0]
        for i in land:
            h = int(I[i, sim.LHIT])
            q = 1 - h
            hits = int(I[i, sim.HITS])
            y = B[i, 1]
            straight = 1.0 if np.sign(y) == np.sign(self.hit_y[i]) else -1.0
            if abs(y) > WIDE_Y and abs(self.hit_y[i]) > 0.1:
                if hits >= 4 and self.prev_dir[i] != 0.0 and straight != self.prev_dir[i]:
                    self._tag(i, h, 3, hits)
                    out.append((i, h, 3))
                self.prev_dir[i] = straight
            mq = env.T[i, sim.LM0 + q]
            kmh = float(np.linalg.norm(B[i, 3:5])) * 3.6
            if hits >= 3 and kmh > 25.0 and abs(mq * (y - P[i, q, 1]) - sim.HAND_MID) < ELBOW_TOL:
                self._tag(i, h, 4, hits)
                out.append((i, h, 4))
        for i in np.where(done)[0]:
            w = int(env.winner[i])
            last = int(prev_hits[i])
            for h, k, hits in self.pending[i]:
                if w == h and last - hits <= 2:
                    if k < 0:
                        self.base_w += 1
                    else:
                        self.won[h, k] += 1
            self.reset(i)
        return out

    def report(self, points):
        lines = [f"（对照：任意一拍后两拍内得分 {self.base_w / max(self.base_n, 1):5.1%}）"]
        for k, name in enumerate(TACTICS):
            c = self.count[:, k].sum()
            wr = self.won[:, k].sum() / max(c, 1)
            lines.append(f"{name} 每百分 {100 * c / max(points, 1):5.1f} 次，用后两拍内得分 {wr:5.1%}")
        return lines
