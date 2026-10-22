"""录一段 Python 版画面层的输入输出，给 Godot 版做动作/骨架移植的逐帧对照（--animtest）。

每帧 1/60 秒：记录这一帧里走过的模拟步（步后的全部状态数组 + 动作）、渲染时的 acc，以及 body_points 算出的关节。
body_points 每帧只调一次（原版画面每帧调 3 次，Godot 版按时间换算，对照时用同样的一次）。

    python core/anim_dump.py 900 godot/_animtest.json
"""
import json
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ["PP_BACKEND"] = "c"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
nframes, out = int(sys.argv[1]), sys.argv[2]
sys.argv = ["play.py", os.path.join(ROOT, "models", "policy_v41_backup.npz")]
import play  # noqa: E402

KEYS = ("hip", "lhip", "rhip", "neck", "head", "rsh", "lsh", "lhand", "relbow", "lelbow", "lknee", "rknee",
        "lfoot", "rfoot", "paddle")


def state(env):
    return dict(B=env.B[0].tolist(), P=env.P[0].ravel().tolist(), I=env.I[0].tolist(), T=env.T[0].tolist(),
                S=env.S[0].ravel().tolist(), F=env.F[0].tolist(), hist=env.hist[0].ravel().tolist())


g = play.Game()
env = g.env
frames = []
cur = None
orig_step = env.step


def step(a):
    r = orig_step(a)
    d = state(env)
    d["act"] = a[0].ravel().tolist()
    cur["steps"].append(d)
    return r


env.step = step


def draw_scene(surf, ball, bodies, hands):
    cur["ball"] = [float(v) for v in ball]
    cur["bodies"] = []
    for j in range(2):
        b = g.body_points(bodies[j], hands[j], j)
        cur["bodies"].append([[float(v) for v in b[k]] for k in KEYS])


g.draw_scene = draw_scene
g.draw_hud = g.draw_spin_inset = g.draw_panel = lambda *a: None
if os.environ.get("PP_DUMP_EQUIP"):
    # 左右手、直板、各种胶皮都覆盖到：双方随机器材
    import numpy as np
    np.random.seed(int(os.environ["PP_DUMP_EQUIP"]))
    for j in range(2):
        play.sim.random_equipment(env.S, 0, j, True)
    g.restart()
init = state(env)
dt = 1.0 / 60.0
for f in range(nframes):
    cur = dict(steps=[])
    g.update(dt)
    cur["acc"] = g.acc
    g.render(dt)
    frames.append(cur)
with open(out, "w") as fh:
    json.dump(dict(init=init, dt=dt, keys=KEYS, frames=frames), fh)
print(f"{nframes} 帧 -> {out}  模拟步 {sum(len(f['steps']) for f in frames)}")
