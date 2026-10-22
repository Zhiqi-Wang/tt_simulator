"""把 npz 策略导出成游戏（C#）读的二进制：只含决策网络和观测归一化。

格式（小端）：b"PPOL" int32 版本=2 int32 obs_dim int32 act_dim int32 层数 L int32 训练轮数 int32×(L+1) 各层宽度
           float64×obs_dim 均值 float64×obs_dim 方差，然后每层 float32 权重 [in×out 行优先] + float32 偏置 [out]

    python core/export_policy.py models/policy_v41_backup.npz godot/data/policy.bin
"""
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from pingpong.nn import Policy  # noqa: E402

src, dst = sys.argv[1], sys.argv[2]
p = Policy.load(src)
ws = p.pi.params
L = len(ws) // 2
sizes = [ws[0].shape[0]] + [ws[2 * k].shape[1] for k in range(L)]
os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
with open(dst, "wb") as f:
    f.write(b"PPOL")
    f.write(struct.pack("<iiiii", 2, sizes[0], sizes[-1], L, int(p.iteration)))
    f.write(struct.pack(f"<{L + 1}i", *sizes))
    f.write(np.asarray(p.norm.mean, "<f8").tobytes())
    f.write(np.asarray(p.norm.var, "<f8").tobytes())
    for k in range(L):
        f.write(np.ascontiguousarray(ws[2 * k], "<f4").tobytes())
        f.write(np.ascontiguousarray(ws[2 * k + 1], "<f4").tobytes())
print(f"{src} -> {dst}  层宽 {sizes}  {os.path.getsize(dst) / 1e6:.1f} MB")
