"""C++ 核心与 numba 版 sim.py 逐步对照。

两边的随机数都从同一条“磁带”里读（均匀数、正态数各一条），每一步之后比较全部状态、观测、奖励、
历史和随机数用量。发现不一致时打印差在哪里，再把 C++ 的状态同步成参考值继续往下比。
参考实现用 NUMBA_DISABLE_JIT 跑纯 Python（慢，但磁带下标可以改）。

    python core/parity.py [步数] [train 0/1] [种子]
"""
import os
import sys
import types

os.environ["NUMBA_DISABLE_JIT"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 600
train = bool(int(sys.argv[2])) if len(sys.argv) > 2 else False
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1

src = open(os.path.join(ROOT, "pingpong", "sim.py"), encoding="utf-8").read().lstrip("\ufeff")
src = (src.replace("np.random.standard_normal(3)", "_rn3()")
       .replace("np.random.standard_normal()", "_rn()")
       .replace("np.random.random()", "_ru()")
       .replace("np.random.randint(", "_ri("))
assert "np.random." not in src.replace("np.random.seed(s)", ""), "sim.py 里还有没换掉的随机数调用"
tape_src = '''
_TU = np.zeros(1)
_TG = np.zeros(1)
_TI = np.zeros(2, np.int64)


def _ru():
    v = _TU[_TI[0] % len(_TU)]
    _TI[0] += 1
    return v


def _rn():
    v = _TG[_TI[1] % len(_TG)]
    _TI[1] += 1
    return v


def _rn3():
    a = _rn()
    b = _rn()
    c = _rn()
    return np.array([a, b, c])


def _ri(lo, hi):
    return min(lo + int(_ru() * (hi - lo)), hi - 1)
'''
src = src.replace("\nG = 9.81\n", tape_src + "\nG = 9.81\n", 1)
ref = types.ModuleType("sim_ref")
ref.__file__ = os.path.join(ROOT, "pingpong", "sim_ref.py")
exec(compile(src, ref.__file__, "exec"), ref.__dict__)

from pingpong import core  # noqa: E402
from pingpong.nn import Policy  # noqa: E402

rng = np.random.default_rng(seed)
TU = rng.random(1 << 20)
TG = rng.standard_normal(1 << 20)
ref._TU, ref._TG = TU, TG
ref._TI[:] = 0
cidx = np.zeros(2, np.int64)
core.set_tape(TU, TG, cidx)

n, nsub, dt = 1, 8, 1.0 / 30.0
NI, NT, N_S = ref.NI, ref.NT, ref.N_S
assert core.dims() == dict(NI=NI, NT=NT, N_S=N_S, OBS_DIM=ref.OBS_DIM, ACT_DIM=ref.ACT_DIM, HIST_DIM=ref.HIST_DIM), \
    (core.dims(), NI, NT, N_S)


def alloc():
    d = dict(B=np.zeros((n, 9)), P=np.zeros((n, 2, 6)), I=np.zeros((n, NI), np.int64), T=np.zeros((n, NT)),
             S=np.full((n, 2, N_S), 0.6), F=np.zeros((n, 2)), rew=np.zeros((n, 2)), done=np.zeros(n, np.bool_),
             winner=np.full(n, -1, np.int64), reason=np.zeros(n, np.int64), hits=np.zeros(n, np.int64),
             hist=np.zeros((n, nsub, ref.HIST_DIM)), O=np.zeros((n, 2, ref.OBS_DIM)))
    d["S"][:, :, ref.N_STATS:ref.N_STATS + ref.N_STYLE] = 0.0
    d["S"][:, :, ref.HAND] = 1.0
    d["S"][:, :, ref.GRIP] = 0.0
    d["S"][:, :, ref.RFH] = ref.RUB_INV
    d["S"][:, :, ref.RBH] = ref.RUB_INV
    return d


R, C = alloc(), alloc()
if train:
    # 训练模式开局随机属性、器材
    for d in (R,):
        d["S"][:] = rng.random(d["S"].shape)
        d["S"][:, :, ref.N_STATS:ref.N_STATS + ref.N_STYLE] = d["S"][:, :, ref.N_STATS:ref.N_STATS + ref.N_STYLE] * 2 - 1
        for j in range(2):
            d["S"][0, j, ref.HAND] = 1.0 if j == 0 else -1.0
            d["S"][0, j, ref.GRIP] = float(j)
            d["S"][0, j, ref.RFH] = float(j)
            d["S"][0, j, ref.RBH] = float(3 + j)
    C["S"][:] = R["S"]
cx = core.Core(n, 0)
ref.reset_kernel(R["B"], R["P"], R["I"], R["T"], R["S"], R["F"], train)
cx.reset(C["B"], C["P"], C["I"], C["T"], C["S"], C["F"], train)

pol_path = os.path.join(ROOT, "models", "policy_v41_backup.npz")
pol = Policy.load(pol_path)
STATE = ("B", "P", "I", "T", "S", "F")
OUTS = ("rew", "done", "winner", "reason", "hits", "hist")


def diff(name, a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = np.abs(a - b)
    tol = 1e-9 * (1.0 + np.abs(a))
    bad = np.argwhere(d > tol)
    return bad, d


def sync():
    for k in STATE:
        C[k][...] = R[k]
    cidx[:] = ref._TI


def check(tag):
    probs = []
    for k in STATE + OUTS:
        bad, d = diff(k, R[k], C[k])
        if len(bad):
            probs.append(f"{k}: {len(bad)} 处，最大差 {d.max():.3g}，首个 {tuple(bad[0])} 参考 "
                         f"{np.asarray(R[k], float)[tuple(bad[0])]!r} C++ {np.asarray(C[k], float)[tuple(bad[0])]!r}")
    if not np.array_equal(ref._TI, cidx):
        probs.append(f"随机数用量不同：参考 {ref._TI.tolist()} C++ {cidx.tolist()}")
    return probs


probs = check("reset")
if probs:
    print("RESET 不一致:", *probs, sep="\n  ")
    sync()

mism = 0
events = {"hits": 0, "points": 0}
arng = np.random.default_rng(seed + 7)
for s in range(steps):
    ref.obs_kernel(R["B"], R["P"], R["I"], R["T"], R["S"], R["F"], R["O"])
    cx.observe(C["B"], C["P"], C["I"], C["T"], C["S"], C["F"], C["O"])
    bad, d = diff("O", R["O"], C["O"])
    if len(bad):
        print(f"步 {s} 观测不一致 {len(bad)} 处，首个 {tuple(bad[0])}")
        mism += 1
    o = R["O"].reshape(2, -1)[:, :pol.norm.mean.shape[0]]
    A = pol.act(o.astype(np.float64)).astype(np.float64).reshape(1, 2, -1)
    A = A + arng.standard_normal(A.shape) * 0.3
    A = np.ascontiguousarray(A)
    h0 = int(R["I"][0, ref.HITS])
    ref.step_kernel(R["B"], R["P"], R["I"], R["T"], R["S"], R["F"], A, R["rew"], R["done"], R["winner"],
                    R["reason"], R["hits"], R["hist"], train, nsub, dt)
    cx.step(C["B"], C["P"], C["I"], C["T"], C["S"], C["F"], A, C["rew"], C["done"], C["winner"], C["reason"],
            C["hits"], C["hist"], train, nsub, dt)
    events["hits"] += int(R["I"][0, ref.HITS] > h0)
    events["points"] += int(R["done"][0])
    probs = check(s)
    if probs:
        mism += 1
        if mism <= 6:
            print(f"步 {s} 相位 {R['I'][0, ref.PHASE]} 不一致：", *probs, sep="\n  ")
        sync()
    if s % 100 == 99:
        print(f"... {s + 1} 步，不一致 {mism} 步，击球 {events['hits']}，得失分 {events['points']}", flush=True)
print(f"SUMMARY steps={steps} train={int(train)} mismatched_steps={mism} hits={events['hits']} "
      f"points={events['points']} rand_used={ref._TI.tolist()}")
