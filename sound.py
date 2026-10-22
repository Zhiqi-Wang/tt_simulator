"""音效：全部来自公开录音（CC0，来源见 assets/sounds/CREDITS.txt），由 _cut_sounds.py 切成单次撞击存在
assets/sounds/cut/。

    击球   拍面击球录音。按胶皮处理：正胶、生胶略升调、加一点高频、余音更短（更脆）；
           长胶、防弧降调、低通、余音很短（更闷）。发力越大音调略高、略亮。
    台面   球落硬面的录音，略压最高频（录音落在硬地上，比球台亮一点）
    地面   同一批录音里弹得低、轻的几次
    球网   落台录音低通到 1.2 kHz，音量小，像撞到网绳的闷响
    掌声   精彩长回合后，约 40 人的小观众鼓掌

每种声音有多个变体，播放时随机挑，避免机关枪式的完全重复；左右声道按球在画面上的位置分配。
"""
import glob
import os
import wave

import numpy as np
import pygame

RATE = 44100
CUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "sounds", "cut")
N_VAR = 4
# 每种胶皮：变调倍数、低通截止（None 不滤）、高频增减、余音衰减常数（秒，None 不改）
#           反胶                正胶                 生胶                长胶                防弧
RUB_REC = [(1.0, None, 0.0, None), (1.04, None, 0.25, 0.03), (1.06, None, 0.35, 0.025),
           (0.94, 2600.0, -0.3, 0.015), (0.92, 2200.0, -0.4, 0.012)]


def _load_cut(prefix):
    out = []
    for f in sorted(glob.glob(os.path.join(CUT_DIR, prefix + "*.wav"))):
        with wave.open(f) as w:
            if w.getframerate() != RATE or w.getsampwidth() != 2:
                continue
            out.append(np.frombuffer(w.readframes(w.getnframes()), np.int16) / 32767.0)
    return out


def _resample(y, r):
    """变调：r > 1 音调升高、变短。"""
    n = max(int(len(y) / r), 2)
    return np.interp(np.arange(n) * r, np.arange(len(y)), y)


def _eq(y, lp=None, bright=0.0):
    """lp：四阶低通的截止频率；bright：高频（以 8 kHz 为参照）按比例加/减。"""
    n = len(y)
    spec = np.fft.rfft(y)
    f = np.fft.rfftfreq(n, 1.0 / RATE)
    g = np.ones_like(f)
    if lp:
        g /= np.sqrt(1.0 + (f / lp) ** 8)
    if bright:
        g *= np.clip(1.0 + bright * np.minimum(f / 8000.0, 1.5), 0.05, None)
    return np.fft.irfft(spec * g, n)


def _finish(y, fade=0.01):
    """去直流、裁掉尾部静音、淡出。"""
    y = y - np.mean(y)
    peak = np.max(np.abs(y)) + 1e-12
    keep = np.nonzero(np.abs(y) > peak * 1e-3)[0]
    y = y[:keep[-1] + 1] if len(keep) else y
    k = min(int(fade * RATE), len(y))
    y[-k:] *= np.linspace(1, 0, k)
    return y


def _to_sound(y, vol=0.8):
    y = y / (np.max(np.abs(y)) + 1e-9) * vol
    a = (y * 32767).astype(np.int16)
    init = pygame.mixer.get_init()
    if init and init[2] == 2:
        a = np.column_stack([a, a])
    return pygame.sndarray.make_sound(np.ascontiguousarray(a))


class Sounds:
    def __init__(self):
        self.ok = False
        self.on = True
        self.hit = {}
        self.table = self.floor = self.net = []
        self.applause = None
        self.rng = np.random.default_rng()
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(RATE, -16, 2, 512)
            pygame.mixer.set_num_channels(24)
            self._build()
            self.ok = bool(self.hit and self.table)
        except Exception:  # noqa: BLE001
            self.ok = False

    def _build(self):
        rng = np.random.default_rng(7)
        rec_hit = _load_cut("hit_")
        rec_table = _load_cut("table_")
        if not (rec_hit and rec_table):
            return
        for rub, (r0, lp, br, dec) in enumerate(RUB_REC):
            for lvl in range(3):
                vs = []
                for k in range(N_VAR):
                    y = rec_hit[(k * 3 + lvl + rub) % len(rec_hit)]
                    y = _resample(y, r0 * (0.98, 1.0, 1.03)[lvl] * rng.uniform(0.98, 1.02))
                    y = _eq(y, lp, br + 0.15 * (lvl - 1))
                    if dec:
                        y = y * np.exp(-np.arange(len(y)) / RATE / dec)
                    vs.append(_to_sound(_finish(y), 0.5 + 0.2 * lvl))
                self.hit[(rub, lvl)] = vs
        self.table = [_to_sound(_finish(_eq(y, 9000.0, -0.15)), 0.6) for y in rec_table]
        self.floor = [_to_sound(_finish(_eq(y, 6000.0, -0.2)), 0.4) for y in _load_cut("floor_")]
        self.net = [_to_sound(_finish(_eq(_resample(y, 0.8), 1200.0), 0.03), 0.35) for y in rec_table[::4]]
        ap = _load_cut("applause")
        self.applause = _to_sound(ap[0], 0.35) if ap else None

    def play(self, snd, vol=1.0, pan=0.0):
        """pan：-1 最左 … +1 最右。snd 可以是一组变体，随机挑一个。"""
        if not (self.ok and self.on) or snd is None:
            return
        if isinstance(snd, list):
            if not snd:
                return
            snd = snd[self.rng.integers(len(snd))]
        ch = snd.play()
        if ch is not None:
            vol = min(max(vol, 0.0), 1.0)
            pan = min(max(pan, -1.0), 1.0)
            ch.set_volume(vol * min(1.0, 1.0 - 0.7 * pan), vol * min(1.0, 1.0 + 0.7 * pan))

    def racket(self, swing, rubber, pan=0.0):
        """swing：挥拍速度 m/s；rubber：sim.RUB_* 编号。"""
        if not self.ok:
            return
        lvl = 0 if swing < 8 else (1 if swing < 15 else 2)
        loud = 0.45 + 0.55 * min(swing / 22.0, 1.0)
        self.play(self.hit.get((int(rubber), lvl)), loud, pan)
