"""脚本 AI：来球在身体右侧用正手、左侧到身前用反手，移动脚步让击球点落进舒适区；
随机选落点和打法（拉、冲、搓、侧旋）。用作训练陪练和没有模型时的默认控制器。"""
import numpy as np

from .sim import ACT_DIM, BH_CHI, BH_CLO, FH_CHI, FH_CLO, FWD_CHI, FWD_CLO, HALF_L

READY = np.array([-(HALF_L + 0.6), 0.0, 1.0])
MARGIN = 0.04


def bot_actions(obs, rng):
    shape = obs.shape[:-1]
    a = np.zeros(shape + (ACT_DIM,))
    vis = obs[..., 21] > 0.5
    ty = obs[..., 41]
    fh = ty < -0.07
    a[..., 8] = np.where(vis & ~fh, -1.0, 1.0)
    lo = np.where(fh, FH_CLO, BH_CLO) + MARGIN
    hi = np.where(fh, FH_CHI, BH_CHI) - MARGIN
    tx = obs[..., 42]

    rel = obs[..., 22:25].copy()
    rel[..., 0] = tx - np.clip(tx, FWD_CLO + MARGIN, FWD_CHI - MARGIN)
    rel[..., 1] = ty - np.clip(ty, lo, hi)
    rel = np.where(vis[..., None], rel, READY - obs[..., 0:3])
    a[..., 0:3] = np.clip(rel * 2.5, -1.0, 1.0)

    strength, tech = obs[..., 26], obs[..., 28]
    a[..., 3] = rng.uniform(-0.7, 0.5, shape)
    a[..., 4] = rng.uniform(-0.8, 0.8, shape)
    base = -0.6 + 0.6 * tech + 0.2 * strength
    style = rng.random(shape)
    loop, drive, chop = style < 0.45, (style >= 0.45) & (style < 0.75), (style >= 0.75) & (style < 0.92)
    a[..., 5] = np.where(drive, base + 0.25, np.where(chop, base - 0.4, base))
    a[..., 6] = np.where(loop, rng.uniform(0.5, 1.0, shape),
                         np.where(drive, rng.uniform(0.1, 0.4, shape),
                                  np.where(chop, rng.uniform(-1.0, -0.4, shape), rng.uniform(0.0, 0.4, shape))))
    a[..., 7] = np.where(rng.random(shape) < 0.35, rng.uniform(-0.8, 0.8, shape), 0.0)

    opp_cue = obs[..., 47] > 0.5
    a[..., 9] = np.where(opp_cue, np.clip((obs[..., 48] - obs[..., 1]) * 3.0, -1.0, 1.0), 0.0)
    fake = rng.random(shape) < 0.25
    a[..., 10] = np.where(fake, -a[..., 4], a[..., 4])
    a[..., 11] = np.where(vis & (np.abs(rel[..., 1]) > 0.55), 1.0, -1.0)

    serving = (obs[..., 20] > 0.5) & (obs[..., 38] > 0.5)
    for k in (1, 3, 4, 5, 6, 7, 8, 9, 10, 11):
        a[..., k] = np.where(serving, rng.uniform(-1.0, 1.0, shape), a[..., k])
    return a
