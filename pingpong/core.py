"""C++ 物理核心（core/ppcore.dll）的 Python 接口。数组布局与 sim.py 的 numba 版完全一致，
PongEnv 用它替代 sim.step_kernel / reset_kernel / obs_kernel。"""
import ctypes
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
DLL_PATH = os.environ.get("PP_CORE_DLL", os.path.join(_HERE, "..", "core", "build", "ppcore.dll"))

_lib = ctypes.CDLL(os.path.abspath(DLL_PATH))
_P = ctypes.c_void_p
_lib.pp_create.restype = _P
_lib.pp_create.argtypes = [ctypes.c_int, ctypes.c_uint64]
_lib.pp_destroy.argtypes = [_P]
_lib.pp_seed.argtypes = [_P, ctypes.c_uint64]
_lib.pp_set_tape.argtypes = [_P, ctypes.c_int64, _P, ctypes.c_int64, _P]
_lib.pp_set_param.argtypes = [ctypes.c_int, _P]
_lib.pp_step.argtypes = [_P] * 8 + [_P] * 6 + [ctypes.c_int, ctypes.c_int, ctypes.c_double]
_lib.pp_reset.argtypes = [_P] * 7 + [ctypes.c_int]
_lib.pp_start_point.argtypes = [_P, ctypes.c_int] + [_P] * 6 + [ctypes.c_int]
_lib.pp_random_equipment.argtypes = [_P, ctypes.c_int, ctypes.c_int, _P, ctypes.c_int]
_lib.pp_observe.argtypes = [_P] * 8
_lib.pp_dims.argtypes = [_P]


def _ptr(a, dtype):
    assert a.dtype == dtype and a.flags["C_CONTIGUOUS"], (a.dtype, a.flags)
    return a.ctypes.data_as(_P)


def dims():
    out = np.zeros(6, np.int64)
    _lib.pp_dims(_ptr(out, np.int64))
    return dict(zip(("NI", "NT", "N_S", "OBS_DIM", "ACT_DIM", "HIST_DIM"), (int(v) for v in out)))


def _env_params():
    """把 sim.py 里读环境变量的可调参数同步给 C++ 核心。"""
    f64 = np.float64
    for which, name, default in ((0, "PP_RUB_DEC", None), (1, "PP_RUB_SENS", None)):
        if name in os.environ:
            v = np.array([float(x) for x in os.environ[name].split(",")], f64)
            _lib.pp_set_param(which, _ptr(v, f64))
    for which, name in ((2, "PP_MEN_SIG"), (3, "PP_TCH_READ"), (4, "PP_DIVE_PW"), (5, "PP_ERR_SCALE"),
                        (6, "PP_READ_SCALE")):
        if name in os.environ:
            v = np.array([float(os.environ[name])], f64)
            _lib.pp_set_param(which, _ptr(v, f64))


_env_params()


class Core:
    """n 张球台的一批世界共用的 C++ 句柄（每张台各自的随机数发生器）。"""

    def __init__(self, n, seed=0):
        self.n = n
        self.h = _lib.pp_create(n, seed)

    def __del__(self):
        if getattr(self, "h", None):
            _lib.pp_destroy(self.h)
            self.h = None

    def seed(self, s):
        _lib.pp_seed(self.h, s)

    def step(self, B, P, I, T, S, F, A, rew, done, winner, reason, hits, hist, train, nsub, dt):
        f, i = np.float64, np.int64
        _lib.pp_step(self.h, _ptr(B, f), _ptr(P, f), _ptr(I, i), _ptr(T, f), _ptr(S, f), _ptr(F, f), _ptr(A, f),
                     _ptr(rew, f), _ptr(done, np.bool_), _ptr(winner, i), _ptr(reason, i), _ptr(hits, i),
                     _ptr(hist, f), int(bool(train)), int(nsub), float(dt))

    def reset(self, B, P, I, T, S, F, train):
        f, i = np.float64, np.int64
        _lib.pp_reset(self.h, _ptr(B, f), _ptr(P, f), _ptr(I, i), _ptr(T, f), _ptr(S, f), _ptr(F, f),
                      int(bool(train)))

    def start_point(self, k, train, B, P, I, T, S, F):
        f, i = np.float64, np.int64
        _lib.pp_start_point(self.h, int(k), _ptr(B, f), _ptr(P, f), _ptr(I, i), _ptr(T, f), _ptr(S, f), _ptr(F, f),
                            int(bool(train)))

    def random_equipment(self, S, k, j, balanced=False):
        _lib.pp_random_equipment(self.h, int(k), int(j), _ptr(S, np.float64), int(bool(balanced)))

    def observe(self, B, P, I, T, S, F, O):
        f, i = np.float64, np.int64
        _lib.pp_observe(self.h, _ptr(B, f), _ptr(P, f), _ptr(I, i), _ptr(T, f), _ptr(S, f), _ptr(F, f), _ptr(O, f))


def set_tape(u, g, idx):
    """对照测试用：所有随机数改从 u（均匀）、g（正态）两条序列里按 idx 读；传 None 恢复正常随机数。"""
    if u is None:
        _lib.pp_set_tape(None, 0, None, 0, None)
        return
    _lib.pp_set_tape(_ptr(u, np.float64), len(u), _ptr(g, np.float64), len(g), _ptr(idx, np.int64))
