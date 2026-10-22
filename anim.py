"""纯画面层的人物动画：迈步（脚踩实后一步一步换位）、发球抛球、引拍/随挥与转体。
只读取模拟状态，不改变物理结果。"""
import math
import os

import numpy as np

from pingpong import sim

G = 9.81
BACK_T, SWING_T = 0.5, 0.18
DEBUG_RESCUE = bool(os.environ.get("DEBUG_RESCUE"))
# 技术预判在击球前多久锁定；引拍姿势追随预判变化的时间常数
PRED_LOCK_T, PRED_TAU = 0.03, 0.05
FOLLOW_UP, FOLLOW_END = 0.12, 0.5
# 发球随挥：挥到位 → 停住 → 收回；收回前球拍以击球点为基准，不被跑位拖走
SERVE_FOLLOW_UP, SERVE_FOLLOW_HOLD, SERVE_FOLLOW_END = 0.15, 0.32, 0.65
STEP_ERR = 0.1
LIFT = 0.07
READY_CROUCH = 0.06
# 对方发球时接发方持拍手的显示高度（胸前）；模型在这段时间不管手的高度，常举到头顶，弯腰前倾时拍子就跑到后脑勺
RECV_READY_Z = sim.TABLE_H + 0.22
RECV_BLEND_T = 0.35
READY_MAX_Z = sim.TABLE_H + 0.40
DRIVE_RISE = 0.07

# 各项击球技术的动作（右手持拍，自己视角：前=朝网，右=持拍手一侧，上；位置是球拍相对击球点的偏移，米）：
# (引拍位置, 随挥位置, 引拍转体（正=向右转）, 引拍下沉, 肘尖朝向或 None, 引拍时拍面朝向或 None)
FH_TECH = {
    # 加转：向右后下方引拍、拍低于球，蹬地转腰向左前上方，收拍到头部附近
    sim.TQ_LOOP: ((-0.45, 0.3, -0.3), (0.25, -0.45, 0.6), 0.8, 0.07, None, None),
    # 提拉（拉下旋）：引拍更低，向上的成分更多
    sim.TQ_LIFT: ((-0.35, 0.25, -0.42), (0.15, -0.35, 0.65), 0.7, 0.1, None, None),
    # 前冲：引拍与球同高、重心偏高，向左前方为主，收拍到额前
    sim.TQ_DRIVELOOP: ((-0.5, 0.32, -0.1), (0.4, -0.5, 0.35), 0.9, 0.05, None, None),
    # 快攻：前臂小幅后引到身体右侧，向左前方挥
    sim.TQ_DRIVE: ((-0.22, 0.2, -0.04), (0.28, -0.48, 0.22), 0.45, 0.03, None, None),
    sim.TQ_BLOCK: ((-0.05, 0.05, 0.0), (0.1, -0.05, 0.02), 0.1, 0.0, None, None),
    # 摆短：上步、手臂伸进台内，几乎不引拍，向前下方轻送
    sim.TQ_SHORT: ((-0.04, 0.03, 0.05), (0.1, 0.0, -0.03), 0.1, 0.04, None, None),
    # 劈长：引拍比搓球高，前臂带手腕向前下方砍
    sim.TQ_LONG: ((-0.1, 0.06, 0.16), (0.28, -0.06, -0.14), 0.25, 0.04, None, None),
    sim.TQ_PUSH: ((-0.1, 0.05, 0.1), (0.2, -0.05, -0.08), 0.2, 0.03, None, None),
    # 挑打：拍插到球下方，向前上方
    sim.TQ_FLICK: ((-0.08, 0.06, -0.1), (0.22, -0.12, 0.16), 0.25, 0.04, None, None),
    # 抹：手腕固定，正面加侧向摩擦，短促
    sim.TQ_WIPE: ((-0.05, 0.12, 0.0), (0.14, -0.22, 0.02), 0.2, 0.03, None, None),
    # 削球：拍举到肩高，向前下方切
    sim.TQ_CHOP: ((-0.05, 0.15, 0.55), (0.25, -0.25, -0.45), 0.4, 0.05, None, None),
    sim.TQ_SMASH: ((-0.35, 0.3, 0.28), (0.38, -0.5, -0.15), 1.0, 0.02, None, None),
    sim.TQ_LOB: ((-0.1, 0.2, -0.4), (0.25, -0.2, 0.6), 0.5, 0.08, None, None),
}
BH_PUSH_ELBOW = (0.2, 0.8, -0.5)
TWIST_ELBOW, TWIST_FACE = (0.5, 0.4, 0.8), (0.55, 0.45, 0.5)
BH_TECH = {
    # 反手加转：拍引到腹前下方、拍头下垂，右肩下沉、肘稍前顶；以肘为轴向前上方甩，收拍到头部偏右、眼前高度
    sim.TQ_LOOP: ((-0.12, -0.08, -0.35), (0.3, 0.32, 0.62), -0.3, 0.08, (0.6, 0.6, -0.2), None),
    sim.TQ_LIFT: ((-0.1, -0.05, -0.42), (0.2, 0.25, 0.68), -0.3, 0.1, (0.6, 0.6, -0.2), None),
    sim.TQ_DRIVELOOP: ((-0.15, -0.15, -0.2), (0.35, 0.3, 0.3), -0.35, 0.06, (0.6, 0.6, -0.1), None),
    # 撕：引拍小而偏高、肘略提、手腕内屈，向前多于向上，收在身体右前方
    sim.TQ_RIP: ((-0.1, -0.18, -0.08), (0.3, 0.3, 0.12), -0.25, 0.03, (0.5, 0.7, 0.1), None),
    # 快拨：引到左腹前，向右前方
    sim.TQ_DRIVE: ((-0.1, -0.15, -0.03), (0.25, 0.25, 0.12), -0.25, 0.03, None, None),
    sim.TQ_BLOCK: ((-0.03, -0.03, 0.0), (0.08, 0.03, 0.0), -0.05, 0.0, None, None),
    # 侧切：大臂贴身、拍面略前倾，触球左侧中部，向左前下方挤切
    sim.TQ_SIDECUT: ((-0.05, -0.12, 0.12), (0.1, -0.2, -0.12), -0.15, 0.02, (-0.2, 0.9, -0.6), (0.6, 0.6, 0.15)),
    sim.TQ_SHORT: ((-0.04, -0.02, 0.05), (0.1, 0.0, -0.03), -0.05, 0.04, BH_PUSH_ELBOW, None),
    sim.TQ_LONG: ((-0.1, -0.06, 0.16), (0.28, 0.06, -0.14), -0.15, 0.04, BH_PUSH_ELBOW, None),
    sim.TQ_PUSH: ((-0.1, -0.06, 0.1), (0.2, 0.05, -0.08), -0.1, 0.03, BH_PUSH_ELBOW, None),
    # 拧拉：上步，抬肘（略低于肩）、前臂下垂、手腕大幅内扣，拍头指向自己小腹（“S”形引拍）；
    # 前臂以肘为轴外旋、手腕外展，向右前上方甩出。霸王拧幅度更大
    sim.TQ_TWIST: ((-0.15, -0.1, -0.06), (0.25, 0.35, 0.25), -0.2, 0.06, TWIST_ELBOW, TWIST_FACE),
    sim.TQ_BANANA: ((-0.18, -0.12, -0.08), (0.3, 0.42, 0.32), -0.3, 0.07, TWIST_ELBOW, TWIST_FACE),
    sim.TQ_WIPE: ((-0.06, -0.12, 0.0), (0.14, -0.1, 0.02), -0.1, 0.03, None, (0.8, -0.3, 0.1)),
    sim.TQ_FLICK: ((-0.06, -0.08, -0.1), (0.2, 0.2, 0.15), -0.15, 0.04, None, None),
    sim.TQ_CHOP: ((-0.05, -0.25, 0.55), (0.25, 0.25, -0.45), -0.3, 0.05, None, None),
    sim.TQ_SMASH: ((-0.2, -0.25, 0.15), (0.35, 0.35, -0.05), -0.4, 0.02, None, None),
    sim.TQ_LOB: ((-0.1, -0.1, -0.35), (0.25, 0.15, 0.5), -0.2, 0.08, None, None),
}


# 拉球类动作的放大倍数：(引拍与随挥位置, 转体, 下蹲)。基础表按“刚够打到”的幅度定，看起来不够猛
LOOP_AMP = {False: (1.4, 1.35, 1.6), True: (1.5, 1.4, 1.6)}
for _bh, _tbl in ((False, FH_TECH), (True, BH_TECH)):
    _p, _t, _s = LOOP_AMP[_bh]
    for _k in (sim.TQ_LOOP, sim.TQ_LIFT, sim.TQ_DRIVELOOP, sim.TQ_RIP):
        if _k in _tbl:
            _b, _f, _tw, _sk, _el, _fc = _tbl[_k]
            _tbl[_k] = (tuple(_p * v for v in _b), tuple(_p * v for v in _f), _tw * _t, _sk * _s, _el, _fc)


def tech_spec(tech, bh):
    return (BH_TECH if bh else FH_TECH).get(tech) or (BH_TECH if bh else FH_TECH)[sim.TQ_DRIVE]


# 快攻、加转、前冲（反手是撕）、提拉在模拟里按力度/上旋意图的阈值划分，意图稍一变化就换类；
# 画面上它们的引拍按意图连续过渡，不会在相近的意图间来回跳
ATTACK_TECH = (sim.TQ_DRIVE, sim.TQ_LOOP, sim.TQ_DRIVELOOP, sim.TQ_RIP, sim.TQ_LIFT)
# 倒地爆冲：引拍幅度倍数；引拍到位时的 (下蹲, 上身外倾)；倒地时的 (下蹲, 外倾)；从地上爬起来要多久
DIVE_BACK_AMP = 1.2
DIVE_PRE = (0.06, 0.15)
DIVE_DOWN = (0.42, 0.65)
DIVE_UP = 0.8
# 倒地爆冲引拍时整个人朝球拍跨出去，跨到身体中心离球拍 DIVE_REACH 米为止，最多跨 DIVE_LUNGE 米；
# 倒下后人就在那里，爬起来时再回到模拟里的位置
DIVE_REACH, DIVE_LUNGE = 0.6, 0.7


def attack_spec(bh, pw, top, top_in):
    tbl = BH_TECH if bh else FH_TECH
    hard = tbl[sim.TQ_RIP] if bh else tbl[sim.TQ_DRIVELOOP]
    u_top = min(max(top / 0.7, 0.0), 1.0)
    u_pw = min(max((pw - 0.45) / 0.35, 0.0), 1.0) * u_top
    u_back = min(max((-top_in - 40.0) / 100.0, 0.0), 1.0) * u_top
    drive, loop, lift = tbl[sim.TQ_DRIVE], tbl[sim.TQ_LOOP], tbl[sim.TQ_LIFT]
    out = []
    for f in range(5):
        a, b, c, d = (np.asarray(s[f] if s[f] is not None else ELBOW_POLE, dtype=float)
                      for s in (drive, loop, lift, hard))
        v = a + (b - a) * u_top
        v = v + (d - v) * u_pw
        v = v + (c - v) * u_back
        out.append(tuple(v) if v.ndim else float(v))
    return (*out, None)


# 发球动作（右手持拍，自己视角：前=朝网，右=持拍手一侧，上）。位置都是球拍相对击球点的偏移（米）。
# 准备：抛球前拍子的位置；引拍：抛球到高点附近拍子引到的位置；随挥：出手后挥到的位置；
# 站位：(转体, 侧身程度, 下蹲)；SQUAT_EXTRA：击球时的额外下蹲（下蹲发球站着抛球引拍，挥拍下砍时才蹲）
SERVE_READY = {
    sim.SV_PEND: (-0.05, 0.12, -0.08),      # 正手侧旋：拍在身体右侧、球的右下方
    sim.SV_HOOK: (0.0, -0.05, 0.05),        # 勾手：拍横在胸前、肘抬起
    sim.SV_SHOVEL: (0.05, 0.05, -0.03),     # 逆旋转：拍在右髋前、向前平伸，拍头略朝下
    sim.SV_SQUAT: (-0.05, 0.1, 0.25),       # 下蹲砍式：拍举到右肩上方、拍头贴近脸侧
    sim.SV_CHOP: (-0.05, 0.15, 0.25),       # 砍式：拍举高在右肩前上方
    sim.SV_FAST: (-0.1, 0.15, -0.05),       # 正手奔球：拍在右侧略低
    sim.SV_BH: (-0.03, -0.06, -0.08),       # 反手侧旋：拍在腹前，肘弯 90°
    sim.SV_BHFAST: (-0.05, -0.05, -0.08),   # 反手急球：拍在腹前
}
SERVE_BACK = {
    sim.SV_PEND: (-0.2, 0.4, 0.15),         # 向右后上方引拍，之后由右向左横摆摩擦
    sim.SV_HOOK: (-0.05, -0.25, 0.3),       # 前臂抬到左上方，先往下砍、再顺势往右上方“勾”
    sim.SV_SHOVEL: (0.05, -0.4, 0.06),      # 肘向外抬起、前臂手腕内收，拍头朝下收到身前左侧，再由左向右甩
    sim.SV_SQUAT: (-0.15, -0.05, 0.5),      # 举到肩上更高，由左后方向右前方半圆形砍下
    sim.SV_CHOP: (-0.1, 0.25, 0.45),        # 高举过肩，向前下方砍
    sim.SV_FAST: (-0.35, 0.3, 0.02),        # 向右后方引拍，引拍短、出手快
    sim.SV_BH: (-0.08, -0.3, 0.16),         # 引到左肩前，再从左上方经身前向右挥
    sim.SV_BHFAST: (-0.25, -0.22, 0.04),    # 向腹前左后方引拍，向右前方快速发力
}
SERVE_FOLLOW = {
    sim.SV_PEND: (0.1, -0.3, -0.02),
    sim.SV_HOOK: (0.12, 0.35, 0.22),
    sim.SV_SHOVEL: (0.15, 0.45, 0.06),
    sim.SV_SQUAT: (0.15, 0.35, -0.2),
    sim.SV_CHOP: (0.25, -0.1, -0.25),
    sim.SV_FAST: (0.3, -0.25, 0.2),
    sim.SV_BH: (0.15, 0.32, -0.04),
    sim.SV_BHFAST: (0.35, 0.15, 0.06),
}
SERVE_STANCE = {
    sim.SV_PEND: (1.3, 1.0, 0.05), sim.SV_HOOK: (1.55, 1.0, 0.04), sim.SV_SHOVEL: (1.0, 0.8, 0.05),
    sim.SV_SQUAT: (1.1, 0.9, 0.03), sim.SV_CHOP: (0.6, 0.6, 0.03), sim.SV_FAST: (1.2, 1.0, 0.06),
    sim.SV_BH: (-0.35, -0.5, 0.06), sim.SV_BHFAST: (-0.25, -0.45, 0.05),
}
# 持拍臂肘尖的朝向（自己视角）。平时肘尖朝后下方、略向外；逆旋转要把肘向外抬高，拍头才能朝下；
# 反手发球肘弯 90° 贴近身体、肘尖朝右前
ELBOW_POLE = (-0.3, 0.6, -1.0)
SERVE_ELBOW = {
    sim.SV_SHOVEL: (0.2, 1.0, 0.35),
    sim.SV_BH: (0.2, 0.9, -0.35),
    sim.SV_BHFAST: (0.2, 0.9, -0.45),
}
# 引拍时额外转体（正手发球向右后转，反手向左）
SERVE_COIL = {
    sim.SV_PEND: 0.35, sim.SV_HOOK: -0.25, sim.SV_SHOVEL: -0.4, sim.SV_SQUAT: 0.2, sim.SV_CHOP: 0.25,
    sim.SV_FAST: 0.45, sim.SV_BH: -0.35, sim.SV_BHFAST: -0.3,
}
SQUAT_EXTRA = {sim.SV_SQUAT: 0.3}
# 抛球后左手相对击球点的让开方向（自己视角）：正手发球收到胸前左侧；反手发球球拍从左往右挥，左手要让到身体左后方
TOSS_CLEAR = {
    -1: (-0.12, -0.3, -0.08),
    sim.SV_BH: (-0.38, -0.42, -0.25),
    sim.SV_BHFAST: (-0.38, -0.42, -0.25),
    sim.SV_SHOVEL: (-0.15, -0.3, -0.1),
}
# 发球触球时的拍面朝向（自己视角，前/右/上）：由右向左摩擦的拍面朝左，由左向右的朝右，砍式拍面后仰；
# 再按旋转变化调整：发下旋拍面后仰（朝上），发上旋拍面前倾
SERVE_FACE = {
    sim.SV_PEND: (0.4, -0.9, 0.2), sim.SV_HOOK: (0.4, 0.9, 0.2), sim.SV_SHOVEL: (0.35, 0.9, 0.2),
    sim.SV_SQUAT: (0.5, 0.7, 0.3), sim.SV_CHOP: (0.6, 0.0, 0.8), sim.SV_FAST: (0.9, -0.3, -0.1),
    sim.SV_BH: (0.4, 0.9, 0.2), sim.SV_BHFAST: (0.9, 0.2, 0.0),
}
# 平时持拍：拍面朝网、略微后仰
FACE_READY = (1.0, 0.0, 0.15)


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / (np.linalg.norm(v) + 1e-9)


def lerp3(a, b, u):
    return tuple(a[k] + (b[k] - a[k]) * u for k in range(3))


def smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


class Animator:
    def __init__(self):
        # 判为救球需要的够球程度（练习“救球爆冲”时调低，方便观察）
        self.rescue_th = (0.72, 0.6, 9.0)
        self.reset()

    def reset(self):
        self.t = 0.0
        self.hit_t = [-9.0, -9.0]
        self.hit_kind = [(-1, False, 0.0, -1), (-1, False, 0.0, -1)]
        self.pred_tech = [-1, -1]
        self.back_s = [None, None]
        self.contact = [None, None]
        self.anchor = [None, None]
        self.hs = [1.0, 1.0]
        self.arm = [None, None]
        self.feet = [None, None]
        self.steps = [None, None]
        self.queue = [[], []]
        self.gait = [("", -9.0), ("", -9.0)]
        self.rescue = [None, None]
        self.dive_prep = [None, None]
        self.dive_sink = [0.0, 0.0]
        self.dive_hold = [np.zeros(2), np.zeros(2)]
        self.serve_t0 = -9.0
        # 画面时钟：每帧都走。self.t 只在模拟走一步时前进，一分打完球员走回站位时模拟暂停，self.t 也停住
        self.wall = 0.0
        self.recv_w = [0.0, 0.0]
        self.cap_w = [0.0, 0.0]
        self.serve_wall0 = -9.0
        self.serve_kind = [sim.SV_PEND, sim.SV_PEND]
        self.serve_hand = [None, None]

    def on_step(self, env, prev_phase, prev_hits, dt):
        """每个模拟步之后调用：记录谁刚击球、什么技术，用来播随挥。"""
        self.t += dt
        I, T = env.I[0], env.T[0]
        h = -1
        tech = -1
        if prev_phase != sim.PH_SERVE and I[sim.PHASE] == sim.PH_SERVE:
            self.serve_t0 = self.t
            self.serve_wall0 = self.wall
        if prev_phase == sim.PH_SERVE and I[sim.PHASE] == sim.PH_FLY:
            h = int(I[sim.SERVER])
            kind = -1
        elif I[sim.HITS] > prev_hits and I[sim.PHASE] == sim.PH_FLY:
            h = int(I[sim.LHIT])
            kind = int(I[sim.LSTK])
            tech = int(I[sim.LTECH])
        if h >= 0:
            self.hit_t[h] = self.t
            self.hit_kind[h] = (kind, bool(I[sim.LSIDE]), float(T[sim.LSW]), tech)
            self.pred_tech[h] = -1
            if kind >= 0:
                # 够球程度：模拟里伸手够球让挥拍打折 1 - 0.4·min(stretch/0.2, 1)
                sk = (1.0 - float(T[sim.DBG0 + 6])) / 0.4
                fh = not I[sim.LSIDE]
                attack = kind in (sim.ST_LOOP, sim.ST_DRIVE, sim.ST_SMASH)
                cp = self.contact[h]
                pd = (cp[:2] - env.P[0, h, 0:2]) if cp is not None else np.zeros(2)
                pn = float(np.linalg.norm(pd))
                if DEBUG_RESCUE:
                    print(f"hit {h} sk {sk:.2f} pn {pn:.2f} lat {abs(pd[1]):.2f} fh {fh} atk {attack} sw {T[sim.LSW]:.1f}")
                # 倒地爆冲由模拟判定（打完要在地上待一阵、接不了下一板）；
                # 救球只是画面效果，只算横向够球（向台内探身够短球不是救球）
                if T[sim.DV0 + h] > 0.0:
                    side = self.off_side(h)
                    self.rescue[h] = ("倒地爆冲", self.t, side)
                    self.gait[h] = ("倒地爆冲", self.t)
                elif sk > self.rescue_th[0] and pn > self.rescue_th[1] and abs(pd[1]) > 0.7 * pn:
                    side = np.array([0.0, math.copysign(1.0, pd[1])])
                    self.rescue[h] = ("救球", self.t, side)
                    self.gait[h] = ("救球", self.t)

    @staticmethod
    def predict_tech(env, act, j, bh):
        """来球击到之前预判这一板用什么技术（引拍要按技术做）：用预测击球点和当前的击球意图套模拟里同样的分类。"""
        T, B = env.T[0], env.B[0]
        a = np.clip(act[j], -1.0, 1.0)
        pw = (a[5] + 1.0) * 0.5
        k = sim.classify_stroke(np.array([T[sim.PX], T[sim.PY], T[sim.PZ]]), pw, a[6])
        depth = sim.DEPTH_MIN + (a[3] + 1.0) * 0.5 * (1.2 - sim.DEPTH_MIN)
        top_in = sim.incoming_top(B[3:6].copy(), B[6:9].copy())
        return int(sim.tech_of(k, bh, pw, a[6], a[7], depth, top_in))

    def own_to_world(self, j, v):
        """球员自己视角（前、持拍手一侧、上）换到世界坐标；左手球员横向镜像。"""
        sj = -1.0 if j == 0 else 1.0
        return np.array([-sj * v[0], sj * self.hs[j] * v[1], v[2]])

    def off_side(self, j):
        """非持拍手一侧（世界坐标 xy 单位向量）：倒地爆冲时右手球员往自己左边倒，左手往右边倒。"""
        w = self.own_to_world(j, (0.0, -1.0, 0.0))
        return np.array([0.0, math.copysign(1.0, w[1])])

    def swing(self, env, act, j, bl, lag):
        """返回 (球拍位移, 转体角, 下沉量)。来球前引拍并屈膝下沉蓄力，击球瞬间回到击球点，
        击球后随挥、蹬腿伸膝（身体发力越多蹬得越明显），再还原。"""
        I, T = env.I[0], env.T[0]
        off = np.zeros(3)
        twist = 0.0
        sink = 0.0
        bh = bl < 0.0
        self.arm[j] = None
        self.anchor[j] = None
        self.dive_prep[j] = None
        if I[sim.PHASE] == sim.PH_FLY and I[sim.TURN] == j and I[sim.PVALID] == 1:
            ttc = T[sim.PRT] - T[sim.RT] + lag
            if 0.0 < ttc < BACK_T:
                # 按预判的技术引拍。击球意图在来球途中还会变，预判一直更新到快击球时，
                # 引拍位置平滑地追向新技术，不会突然跳
                if ttc > PRED_LOCK_T or self.pred_tech[j] < 0:
                    self.pred_tech[j] = self.predict_tech(env, act, j, bh)
                pw = (min(max(act[j, 5], -1.0), 1.0) + 1.0) * 0.5
                if self.pred_tech[j] in ATTACK_TECH:
                    B = env.B[0]
                    spec = attack_spec(bh, pw, min(max(act[j, 6], -1.0), 1.0),
                                       sim.incoming_top(B[3:6].copy(), B[6:9].copy()))
                    amp = 0.6 + 0.4 * pw
                else:
                    spec = tech_spec(self.pred_tech[j], bh)
                    amp = 1.0
                tgt = np.array([*(np.asarray(spec[0]) * amp), spec[2] * amp, spec[3],
                                *(spec[4] if spec[4] is not None else ELBOW_POLE)])
                # 要倒地爆冲的球（正手大力、来球远在身体外侧，判定同模拟）：引拍拉得特别大、特别低，
                # 重心压低并提前往来球一侧倒
                P = env.P[0]
                dive = (not bh and pw > sim.DIVE_PW and self.pred_tech[j] in ATTACK_TECH
                        and T[sim.LM0 + j] * (T[sim.PY] - P[j, 1]) < sim.FH_CLO - sim.DIVE_ST)
                if dive:
                    tgt[0] *= DIVE_BACK_AMP
                    tgt[2] = tgt[2] * DIVE_BACK_AMP - 0.12
                    tgt[3] *= 1.3
                    tgt[4] += 0.05
                # 引拍以预测击球点为基准：模拟里的持拍手此时还在往击球点移动，跟着它引拍位置会被抵消
                self.contact[j] = np.array([T[sim.PX], T[sim.PY], T[sim.PZ]])
                self.anchor[j] = (self.contact[j], smooth((BACK_T - ttc) / (BACK_T - SWING_T)))
                st = self.back_s[j]
                if st is None or self.t - st[1] > 0.3:
                    cur = tgt
                else:
                    cur = st[0] + (tgt - st[0]) * (1.0 - math.exp(-(self.t - st[1]) / PRED_TAU))
                self.back_s[j] = (cur, self.t)
                s = smooth((BACK_T - ttc) / (BACK_T - SWING_T)) if ttc > SWING_T else ttc / SWING_T
                off += s * self.own_to_world(j, cur[0:3])
                twist += s * cur[3]
                sink += s * cur[4]
                if dive:
                    # 倒地爆冲向前挥拍时不蹬腿起身：引拍时压下的重心一直保持到击球，接着扑出去
                    side = self.off_side(j)
                    sb = smooth((BACK_T - ttc) / (BACK_T - SWING_T))
                    self.dive_prep[j] = (side, sb, (1.0 - s) * cur[4] if ttc <= SWING_T else 0.0)
                    self.dive_sink[j] = cur[4]
                else:
                    self.dive_sink[j] = 0.0
                self.arm[j] = (tuple(cur[5:8]), smooth((BACK_T - ttc) / (BACK_T - SWING_T)))
        elif I[sim.PHASE] == sim.PH_SERVE and I[sim.SERVER] == j:
            k = self.current_serve(env, act, j)
            p = self.serve_phase(env, act, lag)
            ready, back = SERVE_READY[k], SERVE_BACK[k]
            if p < 0.25:
                pos, coil = ready, 0.0
            elif p < 0.8:
                u = smooth((p - 0.25) / 0.55)
                pos, coil = lerp3(ready, back, u), u
            else:
                # 下摆加速：越接近击球越快
                u = ((p - 0.8) / 0.2) ** 2
                pos, coil = lerp3(back, (0.0, 0.0, 0.0), u), 1.0 - u
            off += self.own_to_world(j, pos)
            twist += SERVE_COIL[k] * coil
        tsh = self.t - lag - self.hit_t[j]
        kind, hbh, sw, tech = self.hit_kind[j]
        # 上一分的随挥不带进新一分的发球准备：模拟已把手放回准备位，叠着大幅随挥会举到头顶、脑后
        fade = 1.0 - smooth((self.wall - self.serve_wall0) / 0.3) if I[sim.PHASE] == sim.PH_SERVE else 1.0
        if kind == -1 and 0.0 <= tsh < SERVE_FOLLOW_END:
            # 发球随挥：按手法挥向各自的方向
            if tsh < SERVE_FOLLOW_UP:
                f = smooth(tsh / SERVE_FOLLOW_UP)
            else:
                f = 1.0 - smooth((tsh - SERVE_FOLLOW_HOLD) / (SERVE_FOLLOW_END - SERVE_FOLLOW_HOLD))
            f *= fade
            k = self.serve_kind[j]
            off += f * self.own_to_world(j, SERVE_FOLLOW[k])
            twist -= f * SERVE_COIL[k]
            sink -= DRIVE_RISE * f * min(0.3 + 2.0 * float(T[sim.LBODY]), 1.0)
        elif kind != -1 and -0.1 <= tsh < FOLLOW_END:
            # 击球是在模拟步里记下的，比画面显示的时刻（减去 lag）稍早，击球后第一帧 tsh 可能略小于 0；
            # 这一帧也要锚在击球点，否则球拍会闪到模拟里的手的位置（够球时差二三十厘米）
            tsh = max(tsh, 0.0)
            f = smooth(tsh / FOLLOW_UP) if tsh < FOLLOW_UP else 1.0 - smooth((tsh - FOLLOW_UP) / (FOLLOW_END - FOLLOW_UP))
            # 随挥以击球点为基准，挥到位后再跟着身体还原
            if self.anchor[j] is None and self.contact[j] is not None:
                self.anchor[j] = (self.contact[j], (1.0 if tsh < FOLLOW_UP else f) * fade)
            f *= fade
            spec = tech_spec(tech, hbh)
            amp = min(max(sw / 12.0, 0.4), 1.3) * f
            off += amp * self.own_to_world(j, spec[1])
            twist -= amp * 0.8 * spec[2]
            if self.rescue[j] is None or self.rescue[j][0] != "倒地爆冲":
                sink -= DRIVE_RISE * f * min(0.3 + 2.0 * float(T[sim.LBODY]), 1.0)
            if self.arm[j] is None:
                self.arm[j] = (spec[4] if spec[4] is not None else ELBOW_POLE, 1.0 if tsh < FOLLOW_UP else f)
        return off, twist, sink

    def racket_face(self, env, act, j, lag):
        """画面上的拍面法向（世界坐标）。触球瞬间用模拟里真实的拍面；引拍时按击球意图预先调好角度
        （拉上旋前倾、搓下旋后仰、带侧旋时拍面偏转），发球按手法和旋转变化；随挥后慢慢回到平时的持拍角度。"""
        I, T = env.I[0], env.T[0]
        ready = unit(self.own_to_world(j, FACE_READY))
        face = ready
        if I[sim.PHASE] == sim.PH_FLY and I[sim.TURN] == j and I[sim.PVALID] == 1:
            ttc = T[sim.PRT] - T[sim.RT] + lag
            if 0.0 < ttc < BACK_T:
                a = np.clip(act[j], -1.0, 1.0)
                tf = tech_spec(self.pred_tech[j], T[sim.BL0 + j] < 0.0)[5]
                target = unit(self.own_to_world(j, tf if tf is not None else (1.0, 0.3 * a[7], -0.75 * a[6])))
                face = unit(ready + (target - ready) * smooth((BACK_T - ttc) / (BACK_T - SWING_T)))
        elif I[sim.PHASE] == sim.PH_SERVE and I[sim.SERVER] == j:
            k = self.current_serve(env, act, j)
            top = sim.VAR_TOP[int(T[sim.VPLAN0 + j])]
            f = SERVE_FACE[k]
            target = unit(self.own_to_world(j, (f[0], f[1], f[2] - 0.6 * top)))
            p = self.serve_phase(env, act, lag)
            face = unit(ready + (target - ready) * smooth((p - 0.25) / 0.6))
        tsh = self.t - lag - self.hit_t[j]
        raw = T[sim.PN0 + 3 * j:sim.PN0 + 3 * j + 3]
        if 0.0 <= tsh < FOLLOW_END and np.linalg.norm(raw) > 0.5:
            hit = unit(raw)
            u = 0.0 if tsh < FOLLOW_UP else smooth((tsh - FOLLOW_UP) / (FOLLOW_END - FOLLOW_UP))
            face = unit(hit + (ready - hit) * u)
        return face

    def gait_context(self, env, j):
        """选步法用的局面：离击球还有多久、是不是侧身用正手接反手位的球、模拟里是否在走交叉步。"""
        I, T, P = env.I[0], env.T[0], env.P[0]
        incoming = I[sim.PHASE] == sim.PH_FLY and I[sim.TURN] == j and I[sim.PVALID] == 1
        ttc = float(T[sim.PRT] - T[sim.RT]) if incoming else 9.0
        side_on = bool(incoming and T[sim.BT0 + j] > 0.0 and T[sim.LM0 + j] * (T[sim.PY] - P[j, 1]) > 0.02)
        cross = T[sim.CX0 + j] > 0.0
        vmax = sim.vmax_of(env.S[0, j, sim.AGI], env.F[0, j])
        return ttc, side_on, cross, vmax

    def plan_gait(self, env, j, fp, home, err):
        """按资料里的步法分类选这一步怎么走，返回 (步法名, 步子列表)；每一步是 (哪只/哪两只脚, 时长, 抬脚高度)。
        脚的编号：0 = 非持拍手一侧，1 = 持拍手一侧。"""
        ttc, side_on, cross, vmax = self.gait_context(env, j)
        spd = float(np.linalg.norm(env.P[0, j, 3:5]))
        lead = int(np.argmax(err))
        other = 1 - lead
        d = float(err[lead])
        if d < STEP_ERR:
            # 小碎步：快到击球时高频小垫步微调击球点
            if ttc < 0.5 and d > 0.04:
                return "小碎步", [((lead,), 0.08, 0.02)]
            return None, []
        if cross:
            # 交叉步：远离来球的脚从体前交叉跨出一大步，另一只脚再跟上
            return "交叉步", [((lead,), 0.2, 0.08), ((other,), 0.15, 0.05)]
        if side_on and d > 0.12:
            # 侧身步：非持拍侧的脚先向反手位跨，持拍侧的脚随即向后移，让出正手的位置
            return "侧身步", [((0,), 0.13, 0.05), ((1,), 0.15, 0.05)]
        if d < 0.3:
            # 单步：一只脚为轴，另一只脚向来球方向迈一步
            return "单步", [((lead,), 0.16, 0.06)]
        if ttc < 0.35:
            # 跨步：来不及调整，一脚蹬地另一脚跨出一大步，蹬地脚随后跟上半步
            return "跨步", [((lead,), 0.14, 0.07), ((other,), 0.14, 0.04)]
        if spd > 0.7 * vmax and d > 0.45:
            # 跳步：双脚几乎同时离地，离来球较远的脚先落地
            return "跳步", [((0, 1), 0.2, 0.09)]
        # 并步：一脚先向另一脚并半步，另一脚随即向来球方向移动，重心起伏小
        return "并步", [((other,), 0.11, 0.03), ((lead,), 0.13, 0.04)]

    def step_feet(self, env, j, home, dt):
        """脚踩实不动，离理想站位太远才按选好的步法迈步；跑得越快步子越急、越往前落。"""
        if self.feet[j] is None or np.max(np.linalg.norm(home - self.feet[j], axis=1)) > 0.9:
            self.feet[j] = home.copy()
            self.steps[j] = None
            self.queue[j] = []
        fp = self.feet[j]
        lift = np.zeros(2)
        vel = env.P[0, j, 3:5]
        if self.steps[j] is None and dt > 0.0:
            if not self.queue[j]:
                name, plan = self.plan_gait(env, j, fp, home, np.linalg.norm(home - fp, axis=1))
                if name is not None and self.rescue[j] is None and (self.gait[j][0] != name
                                                                    or self.t - self.gait[j][1] > 0.35):
                    self.gait[j] = (name, self.t)
                self.queue[j] = list(plan)
            if self.queue[j]:
                ks, dur, lh = self.queue[j].pop(0)
                self.steps[j] = dict(ks=ks, a=fp.copy(), b={k: home[k] + vel * dur for k in ks},
                                     u=0.0, dur=dur, lift=lh)
        st = self.steps[j]
        if st is not None:
            st["u"] += dt / st["dur"]
            u = smooth(st["u"])
            for n, k in enumerate(st["ks"]):
                st["b"][k] = 0.7 * st["b"][k] + 0.3 * (home[k] + vel * st["dur"] * (1.0 - min(st["u"], 1.0)))
                # 跳步：离来球远的脚先落地
                uk = u if len(st["ks"]) == 1 else smooth(min(st["u"] * (1.15 if n == 0 else 0.95), 1.0))
                fp[k] = st["a"][k] + (st["b"][k] - st["a"][k]) * uk
                lift[k] = st["lift"] * math.sin(math.pi * min(st["u"], 1.0))
            if st["u"] >= 1.0:
                self.steps[j] = None
        hop = st is not None and len(st["ks"]) == 2
        return [(fp[k, 0], fp[k, 1], 0.02 + lift[k]) for k in range(2)], float(lift.max()) * (-1.0 if hop else 1.0)

    def rescue_pose(self, env, j, lag):
        """救球：击球时伸到了够球的极限，先打再调整——击球后重心继续往外冲、压得很低，再碎步还原。
        正手大力拉冲时够到极限就是“倒地爆冲”：身体几乎扑倒、单膝快着地，打完再爬起来。
        返回 (额外下蹲, 身体外倾方向与幅度（世界坐标 xy）)。"""
        r = self.rescue[j]
        if r is None:
            prep = self.dive_prep[j]
            if prep is None:
                return 0.0, np.zeros(2)
            side, s, carry = prep
            # 引拍时重心压在持拍手一侧的腿上（与之后倒下的方向相反）
            return DIVE_PRE[0] * s + carry, -side * DIVE_PRE[1] * s
        kind, t0, side = r
        tsh = self.t - lag - t0
        if kind == "倒地爆冲":
            # 击球后从持拍侧被甩向另一侧扑倒，倒在地上待到模拟里能动为止，再慢慢爬起来
            tsh = max(tsh, 0.0)
            if tsh >= sim.DIVE_T + DIVE_UP:
                self.rescue[j] = None
                return 0.0, np.zeros(2)
            rise = smooth(tsh / 0.3)
            fall = 1.0 - smooth((tsh - sim.DIVE_T) / DIVE_UP)
            d0 = DIVE_PRE[0] + self.dive_sink[j]
            depth = (d0 + (DIVE_DOWN[0] - d0) * rise) * fall
            lean = (-DIVE_PRE[1] + (DIVE_DOWN[1] + DIVE_PRE[1]) * rise) * fall
            return depth, side * lean
        dur = 0.75
        if not 0.0 <= tsh < dur:
            if tsh >= dur:
                self.rescue[j] = None
            return 0.0, np.zeros(2)
        rise = smooth(tsh / 0.2)
        fall = 1.0 - smooth((tsh - 0.45 * dur) / (0.55 * dur))
        w = rise * fall
        return 0.13 * w, side * 0.2 * w

    def dive_step(self, env, j, hand, lag):
        """倒地爆冲时整个人（连脚）朝球拍跨出去的位移（世界坐标 xy）：引拍时跨到够得着球拍的地方，
        倒地时留在那里，爬起来时收回到模拟里的位置。"""
        r = self.rescue[j]
        if r is not None and r[0] == "倒地爆冲":
            tsh = max(self.t - lag - r[1], 0.0)
            return self.dive_hold[j] * (1.0 - smooth((tsh - sim.DIVE_T) / DIVE_UP))
        prep = self.dive_prep[j]
        if prep is None:
            self.dive_hold[j] = np.zeros(2)
            return self.dive_hold[j]
        # 画面上的身体中心在模拟位置之后 0.32 米（与 body_points 一致）
        c = np.array([env.P[0, j, 0] + (-1.0 if j == 0 else 1.0) * 0.32, env.P[0, j, 1]])
        d = np.asarray(hand[:2], dtype=float) - c
        dn = float(np.linalg.norm(d)) + 1e-9
        self.dive_hold[j] = d / dn * min(max(dn - DIVE_REACH, 0.0), DIVE_LUNGE) * prep[1]
        return self.dive_hold[j]

    def approach_lunge(self, env, j, lag):
        """来球离得远、时间又紧：最后一步跨出去时重心压低（跨步够球）。"""
        I, T, P = env.I[0], env.T[0], env.P[0]
        if not (I[sim.PHASE] == sim.PH_FLY and I[sim.TURN] == j and I[sim.PVALID] == 1):
            return 0.0
        ttc = T[sim.PRT] - T[sim.RT] + lag
        reach = math.hypot(T[sim.PX] - P[j, 0], T[sim.PY] - P[j, 1])
        if ttc > 0.35 or reach < 0.75:
            return 0.0
        return 0.12 * min((reach - 0.75) / 0.35, 1.0) * smooth((0.35 - ttc) / 0.25)

    @staticmethod
    def current_serve(env, act, j):
        if sim.USE_TACTIC:
            return int(env.T[0, sim.SPLAN0 + j])
        a = np.clip(act[j], -1.0, 1.0)
        return int(sim.serve_type_of(a[8], a[10]))

    @staticmethod
    def serve_toss_value(env, act):
        T = env.T[0]
        srv = int(env.I[0, sim.SERVER])
        if T[sim.TOSS] >= 0.0:
            return float(T[sim.TOSS])
        return sim.serve_toss(int(T[sim.SPLAN0 + srv]), min(max(act[srv, 9], -1.0), 1.0))

    def serve_phase(self, env, act, lag):
        """发球进度：抛球前为 0，抛球离手到击球从 0 到 1。"""
        d = sim.toss_time(self.serve_toss_value(env, act))
        tr = env.T[0, sim.TIMER] + lag
        return min(max((d - tr) / d, 0.0), 1.0)

    def toss(self, env, act, ball, lag):
        """发球：球先托在掌心，按抛球高度抛起，下落到原击球高度时出手。返回 (显示的球位置, 抛球手位置) 或 None。"""
        I, T = env.I[0], env.T[0]
        if I[sim.PHASE] != sim.PH_SERVE:
            return None
        srv = int(I[sim.SERVER])
        d = sim.toss_time(self.serve_toss_value(env, act))
        tr = T[sim.TIMER] + lag
        b = ball.copy()
        if tr > d:
            return b, (b[0], b[1], b[2] - 0.03), srv
        tau = d - tr
        vup = G * d / 2.0
        b[2] = ball[2] + vup * tau - 0.5 * G * tau * tau
        rel = min(tau / 0.12, 1.0)
        # 手只跟着球抬到出手高度（约 +16cm），球继续往上升时手停在原处——否则整条胳膊会跟着球举到头顶，
        # 抛球臂从脸前、胸前穿过去
        lift = min(max(b[2] - ball[2], 0.0), 0.16)
        hand = np.array([ball[0], ball[1], ball[2] - 0.03 + lift * rel])
        # 抛球出手后左手迅速离开击球点，不挡球、不挡挥拍路线
        clear = smooth((tau - 0.1) / 0.25)
        away = TOSS_CLEAR.get(self.current_serve(env, act, srv), TOSS_CLEAR[-1])
        hand += clear * self.own_to_world(srv, away)
        return b, tuple(hand), srv

    def serve_stance(self, env, act, j, lag):
        """发球站位：正手发球侧身（左脚在前、身体转向侧面），反手发球右脚在前；重心压低，下蹲发球蹲得更深。
        发完球约 0.4 秒内转回正面。返回 (转体, 站位侧向, 下蹲)。"""
        I = env.I[0]
        serving = I[sim.PHASE] == sim.PH_SERVE and I[sim.SERVER] == j
        if serving:
            k = self.current_serve(env, act, j)
            # 一分刚开始时从正面渐渐转成发球站姿，不要一帧转过去
            w = smooth((self.t - lag - self.serve_t0) / 0.5)
            # 站着抛球、引拍；球下落、挥拍下砍时才迅速蹲下，击球瞬间蹲到最低
            extra = SQUAT_EXTRA.get(k, 0.0) * smooth((self.serve_phase(env, act, lag) - 0.6) / 0.4)
        else:
            tsh = self.t - lag - self.hit_t[j]
            k = self.serve_kind[j]
            # 下蹲发球站起来更慢
            rec = 0.7 if k == sim.SV_SQUAT else 0.4
            if self.hit_kind[j][0] != -1 or not (0.0 <= tsh < rec):
                return 0.0, 0.0, 0.0
            w = 1.0 - smooth(tsh / rec)
            extra = SQUAT_EXTRA.get(k, 0.0) * w
        self.serve_kind[j] = k
        yaw, stance, crouch = SERVE_STANCE[k]
        return yaw * w, stance * w, crouch * w + extra

    def elbow_pole(self, env, act, j, lag):
        """持拍臂肘尖朝向（世界坐标）：发球时按手法，随挥收回时渐渐回到平时的朝向。"""
        I = env.I[0]
        if I[sim.PHASE] == sim.PH_SERVE and I[sim.SERVER] == j:
            k, w = self.current_serve(env, act, j), smooth(self.serve_phase(env, act, lag) / 0.4)
            w = max(w, 0.5)
        else:
            tsh = self.t - lag - self.hit_t[j]
            if self.hit_kind[j][0] != -1 or not (0.0 <= tsh < SERVE_FOLLOW_END):
                arm = self.arm[j]
                if arm is not None:
                    return self.own_to_world(j, lerp3(ELBOW_POLE, arm[0], arm[1]))
                return self.own_to_world(j, ELBOW_POLE)
            k = self.serve_kind[j]
            w = 1.0 - smooth((tsh - SERVE_FOLLOW_HOLD) / (SERVE_FOLLOW_END - SERVE_FOLLOW_HOLD))
        return self.own_to_world(j, lerp3(ELBOW_POLE, SERVE_ELBOW.get(k, ELBOW_POLE), w))

    def frame(self, env, act, ball, hands, blends, home_fn, dt, lag):
        """每帧调用：返回显示用的球、球拍位置和每名球员的姿态（脚、转体、站位、下蹲、抛球手、身体起伏）。
        home_fn(j, pose, hand) 给出该姿态下两只脚的理想落点。"""
        self.wall += dt
        ball_disp = ball
        poses = []
        self.hs = [-1.0 if env.T[0, sim.LM0 + j] * (1.0 if j == 0 else -1.0) < 0.0 else 1.0 for j in range(2)]
        tossed = self.toss(env, act, ball, lag)
        # 新一分刚开始的一两帧，模拟已进入发球阶段但画面用的那一帧球还停在上一分的落点，
        # 这时托球手去够那个球会把整个人拽过去；球离持拍手太远就先不摆托球
        if tossed is not None and np.linalg.norm(np.asarray(tossed[1][:2]) - np.asarray(hands[tossed[2]])[:2]) > 0.8:
            tossed = None
        if tossed is not None:
            ball_disp = tossed[0]
        hands_disp = []
        I = env.I[0]
        for j in range(2):
            off, twist, sink = self.swing(env, act, j, blends[j], lag)
            base = np.asarray(hands[j], dtype=float)
            if I[sim.PHASE] == sim.PH_SERVE and I[sim.SERVER] == j:
                self.serve_hand[j] = base.copy()
            # 模型给的等球手位常举到头边：除了自己发球，持拍手不高过胸口；
            # 接发球一方（对手发球时和发球飞来途中）端在准备高度。
            # 都只管等球的时候，引拍后由击球点锚点接管；进出都平滑过渡
            serving = I[sim.PHASE] == sim.PH_SERVE and I[sim.SERVER] == j
            recv = (I[sim.PHASE] == sim.PH_SERVE and I[sim.SERVER] != j) or (
                I[sim.PHASE] == sim.PH_FLY and I[sim.TURN] == j and self.hit_kind[1 - j][0] == -1)
            kb = min(dt / RECV_BLEND_T * 3.0, 1.0)
            self.cap_w[j] += ((0.0 if serving else 1.0) - self.cap_w[j]) * kb
            self.recv_w[j] += ((1.0 if recv else 0.0) - self.recv_w[j]) * kb
            wc = smooth(self.cap_w[j])
            if wc > 0.0 and base[2] > READY_MAX_Z:
                base[2] += (READY_MAX_Z - base[2]) * wc
            wr = smooth(self.recv_w[j])
            if wr > 0.0:
                base[2] += (RECV_READY_Z - base[2]) * wr
            tsh = self.t - lag - self.hit_t[j]
            if self.hit_kind[j][0] == -1 and self.serve_hand[j] is not None and 0.0 <= tsh < SERVE_FOLLOW_END:
                u = smooth((tsh - SERVE_FOLLOW_HOLD) / (SERVE_FOLLOW_END - SERVE_FOLLOW_HOLD))
                base = self.serve_hand[j] + (base - self.serve_hand[j]) * u
            elif self.anchor[j] is not None:
                p, w = self.anchor[j]
                base = base + (p - base) * w
            hand = base + off
            hands_disp.append(hand)
            yaw, stance, crouch = self.serve_stance(env, act, j, lag)
            spd = float(np.linalg.norm(env.P[0, j, 3:5]))
            ready = READY_CROUCH * (1.0 - 0.5 * min(spd / 1.5, 1.0))
            dive, tilt = self.rescue_pose(env, j, lag)
            sink += dive + self.approach_lunge(env, j, lag)
            pose = dict(twist=twist + yaw, stance=stance, crouch=max(ready + crouch + sink, 0.0), bob=0.0,
                        face=self.racket_face(env, act, j, lag), epole=self.elbow_pole(env, act, j, lag),
                        tilt=tilt, lunge=self.dive_step(env, j, hand, lag), gait=self.gait[j], t=self.t - lag)
            if tossed is not None and tossed[2] == j:
                pose["lhand"] = tossed[1]
            feet, lift = self.step_feet(env, j, home_fn(j, pose, hand), dt)
            # 迈步时身体随抬脚略降；跳步双脚离地，身体整体腾起
            pose.update(lfoot=feet[0], rfoot=feet[1], bob=0.5 * -lift if lift < 0.0 else -0.25 * lift)
            poses.append(pose)
        return ball_disp, hands_disp, poses
