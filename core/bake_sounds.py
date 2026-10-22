"""把 sound.py 里按胶皮/力度处理好的音效变体预先渲染成 wav，给 Godot 版直接播放（处理方法与 pygame 版完全相同）。

    python core/bake_sounds.py
"""
import os
import sys
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from sound import N_VAR, RATE, RUB_REC, _eq, _finish, _load_cut, _resample  # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "godot", "assets", "sounds")


def write(name, y, vol):
    y = y / (np.max(np.abs(y)) + 1e-9) * vol
    a = (y * 32767).astype("<i2")
    with wave.open(os.path.join(OUT, name + ".wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(a.tobytes())


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(7)
    rec_hit = _load_cut("hit_")
    rec_table = _load_cut("table_")
    n = 0
    for rub, (r0, lp, br, dec) in enumerate(RUB_REC):
        for lvl in range(3):
            for k in range(N_VAR):
                y = rec_hit[(k * 3 + lvl + rub) % len(rec_hit)]
                y = _resample(y, r0 * (0.98, 1.0, 1.03)[lvl] * rng.uniform(0.98, 1.02))
                y = _eq(y, lp, br + 0.15 * (lvl - 1))
                if dec:
                    y = y * np.exp(-np.arange(len(y)) / RATE / dec)
                write(f"hit_{rub}_{lvl}_{k}", _finish(y), 0.5 + 0.2 * lvl)
                n += 1
    for k, y in enumerate(rec_table):
        write(f"table_{k}", _finish(_eq(y, 9000.0, -0.15)), 0.6)
        n += 1
    for k, y in enumerate(_load_cut("floor_")):
        write(f"floor_{k}", _finish(_eq(y, 6000.0, -0.2)), 0.4)
        n += 1
    for k, y in enumerate(rec_table[::4]):
        write(f"net_{k}", _finish(_eq(_resample(y, 0.8), 1200.0), 0.03), 0.35)
        n += 1
    ap = _load_cut("applause")
    if ap:
        write("applause", ap[0], 0.35)
        n += 1
    print(f"{n} 个音效 -> {OUT}")


if __name__ == "__main__":
    main()
