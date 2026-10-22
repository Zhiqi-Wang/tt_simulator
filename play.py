"""乒乓球自动对战（可视化）。

    python play.py

鼠标左键拖动旋转视角，滚轮缩放；右侧面板调整双方属性与控制方式。
快捷键：空格 暂停，R 重载模型，C 重置视角，1/2 切换双方控制器，Esc 退出。
"""
import math
import os
import sys
import time

import numpy as np
import pygame

from anim import Animator
from render3d import Painter
from sound import Sounds
from pingpong import sim
from pingpong.bot import bot_actions
from pingpong.env import PongEnv
from pingpong.nn import Policy
from pingpong.sim import (BALL_R, HALF_L, HALF_W, N_STATS, NET_H, NET_HALF_W, REASON_TEXT,
                          TABLE_H)

W, H = 1440, 900
ROW = 40
# 避让的帧间记忆：肘向偏离每帧保留的比例；手的避让偏移每帧追上新值的比例
POLE_KEEP = 0.85
HAND_EASE = 0.3
HOLD_MARGIN = 0.015
PUSH_STICK = 0.04
STAT_ROWS = 4
STATUS_EXTRA = 40
PANEL_W = 380
VIEW_W = W - PANEL_W
MODEL_PATH = (os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "policy.npz"))
NAMES = ["蓝方", "红方"]
# 球员走回站位后再站定多久才开始发球（秒）
SERVE_SETTLE = 0.3
COLORS = [(70, 150, 255), (255, 90, 90)]
BG = (18, 22, 30)
PANEL_BG = (28, 32, 42)
TEXT = (225, 230, 240)
DIM = (140, 148, 165)
DEFAULT_STATS = 60
VIS_SPIN = 0.1
TOP_COL = (255, 120, 50)
BACK_COL = (70, 170, 255)
SIDE_COL = (120, 230, 110)
BALL_COL = (255, 250, 235)
RED_RUBBER = (190, 30, 35)
BLACK_RUBBER = (25, 25, 30)
SKIN = (236, 196, 162)
HAIR_COL = (45, 34, 28)
SHORTS_COL = (38, 42, 54)
SHOE_COL = (236, 236, 240)
WOOD_COL = (196, 160, 112)
PIPS_RUB = (sim.RUB_PIPS, sim.RUB_RAW, sim.RUB_LONG)
PUSHBLOCK_TECH = (sim.TQ_BLOCK, sim.TQ_SHORT, sim.TQ_LONG, sim.TQ_PUSH, sim.TQ_SIDECUT)
# 练习页强制的相持技术：(名称, 力度动作 a5, 旋转动作 a6, 离台目标距离 m / None 不管站位, 陪练方用的技术)
PRAC_TECH = [("拉球", 0.3, 0.85, 0.9, 0), ("快攻", 0.2, 0.15, 0.4, 1), ("削球", -0.2, -0.75, 2.0, 0),
             ("搓球", -0.6, -0.6, 0.15, 3), ("挡球", -0.6, 0.1, 0.2, 0), ("放高球", -0.3, 0.45, 2.6, 6),
             ("扣杀", 0.9, 0.1, 0.5, 5), ("救球爆冲", 0.6, 0.85, 0.9, 7), ("大角调动", 0.3, 0.85, 0.9, 0)]
# 陪练调动时落点横向（自己视角，±LAT_MAX 内）：左右两大角交替
PRAC_WIDE = 7
ARM_LEN = 0.6
TORSO_BEND = 0.32
HIP_W = 0.12
SEG_LEN = 0.46
FOOT_R = 0.5
UPPER_ARM, FORE_ARM = 0.3, 0.34


def arm_elbow(sh, hand, pole, a, b):
    """两段手臂的肘部位置：肩到手的距离给定，肘尖朝 pole 方向弯；够不着时手臂伸直。"""
    s, h = np.asarray(sh, dtype=float), np.asarray(hand, dtype=float)
    d = h - s
    dn = float(np.linalg.norm(d)) + 1e-9
    u = d / dn
    if dn >= a + b:
        return tuple(s + u * a * dn / (a + b))
    p = np.asarray(pole, dtype=float)
    p = p - (p @ u) * u
    p /= np.linalg.norm(p) + 1e-9
    # 手几乎贴到肩上（扑救时上身压过了球拍）时，肩→手的方向随手的微小移动乱转，肘跟着乱跳；
    # 这时肘直接朝 pole 方向折起
    w = min(max((0.28 - dn) / 0.12, 0.0), 1.0)
    dn = max(dn, 0.18)
    x = (a * a - b * b + dn * dn) / (2.0 * dn)
    y = math.sqrt(max(a * a - x * x, 0.0))
    ik = s + u * x + p * y
    pole_u = np.asarray(pole, dtype=float)
    pole_u = pole_u / (np.linalg.norm(pole_u) + 1e-9)
    return tuple(ik + w * (s + pole_u * a - ik))


def lerp_col(a, b, t):
    return tuple(int(a[k] + (b[k] - a[k]) * t) for k in range(3))


mix_col = lerp_col


def unit3(v):
    v = np.asarray(v, dtype=float)
    return v / (float(np.linalg.norm(v)) + 1e-9)


def rubber_tint(col, r):
    """防弧胶表面不黏、发乌；长胶颜色略浅。"""
    if r == sim.RUB_ANTI:
        return lerp_col(col, (90, 90, 90), 0.35)
    if r == sim.RUB_LONG:
        return lerp_col(col, (255, 255, 255), 0.12)
    return col


def spin_parts(v, w):
    """把旋转分解为相对飞行方向的上旋分量（负为下旋）和侧旋分量，单位 rad/s。"""
    hn = math.hypot(v[0], v[1]) + 1e-9
    return (w[0] * -v[1] + w[1] * v[0]) / hn, w[2]


def spin_color(v, w):
    top, side = spin_parts(v, w)
    mag = min(math.hypot(top, side) / 400.0, 1.0)
    tc = TOP_COL if top >= 0 else BACK_COL
    f = abs(top) / (abs(top) + abs(side) + 1e-9)
    return lerp_col(BALL_COL, lerp_col(SIDE_COL, tc, f), mag)


def serve_spin_text(v, w):
    top, side = spin_parts(v, w)
    ud = "上旋" if top > 40 else ("下旋" if top < -40 else "")
    if abs(side) > 40:
        return "侧" + (ud or "旋")
    return ud or "不转"


def toss_text(toss):
    return "高抛" if toss > 0.66 else "中抛" if toss > 0.33 else "低抛"


def tech_text(tech, bh, stroke):
    if tech < 0:
        return ("反手" if bh else "正手") + sim.STROKE_NAMES[max(stroke, 0)]
    name = sim.TECH_NAMES[tech]
    return name if tech in sim.TECH_NOSIDE else ("反手" if bh else "正手") + name


def rotation(axis, ang):
    x, y, z = axis
    c, s = math.cos(ang), math.sin(ang)
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + s * k + (1 - c) * (k @ k)



def load_font(size):
    for path in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/simsun.ttc"]:
        if os.path.exists(path):
            return pygame.font.Font(path, size)
    return pygame.font.SysFont("microsoftyahei,simhei,arialunicode", size)


class Camera:
    def __init__(self):
        self.reset()

    def reset(self):
        self.az = -math.pi / 2
        self.el = math.radians(30)
        self.dist = 7.4
        self.target = np.array([0.3, 0.0, 0.7])
        self.f = 780.0
        self.update()

    def update(self):
        self.el = min(max(self.el, math.radians(3)), math.radians(85))
        self.dist = min(max(self.dist, 2.5), 14.0)
        ce = math.cos(self.el)
        self.pos = self.target + self.dist * np.array(
            [math.sin(self.az) * ce, -math.cos(self.az) * ce, math.sin(self.el)])
        fwd = self.target - self.pos
        self.fwd = fwd / np.linalg.norm(fwd)
        r = np.cross(self.fwd, [0.0, 0.0, 1.0])
        self.right = r / np.linalg.norm(r)
        self.up = np.cross(self.right, self.fwd)

    def project(self, p):
        rel = np.asarray(p, float) - self.pos
        zc = np.maximum(rel @ self.fwd, 0.05)
        sx = VIEW_W / 2 + self.f * (rel @ self.right) / zc
        sy = H / 2 + 30 - self.f * (rel @ self.up) / zc
        return sx, sy, zc

    def pt(self, p):
        sx, sy, _ = self.project(p)
        return float(sx), float(sy)

    def depth(self, p):
        return float((np.asarray(p, float) - self.pos) @ self.fwd)

    def scale(self, p, size):
        return self.f * size / max(self.depth(p), 0.05)


class Slider:
    def __init__(self, x, y, w, label, vmin, vmax, value, color, on_change, fmt="{:.0f}"):
        self.rect = pygame.Rect(x, y, w, 36)
        self.track = pygame.Rect(x, y + 24, w, 6)
        self.label, self.vmin, self.vmax, self.value = label, vmin, vmax, value
        self.color, self.on_change, self.fmt = color, on_change, fmt
        self.drag = False

    def set_from_x(self, x):
        t = min(max((x - self.track.x) / self.track.w, 0.0), 1.0)
        self.value = self.vmin + t * (self.vmax - self.vmin)
        self.on_change(self.value)

    def set_value(self, v):
        self.value = v
        self.on_change(v)

    def handle(self, e):
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and self.rect.inflate(0, 8).collidepoint(e.pos):
            self.drag = True
            self.set_from_x(e.pos[0])
            return True
        if e.type == pygame.MOUSEMOTION and self.drag:
            self.set_from_x(e.pos[0])
            return True
        if e.type == pygame.MOUSEBUTTONUP and e.button == 1 and self.drag:
            self.drag = False
            return True
        return False

    def draw(self, surf, font):
        surf.blit(font.render(self.label, True, TEXT), (self.rect.x, self.rect.y))
        val = font.render(self.fmt.format(self.value), True, self.color)
        surf.blit(val, (self.rect.right - val.get_width(), self.rect.y))
        pygame.draw.rect(surf, (55, 60, 75), self.track, border_radius=3)
        t = (self.value - self.vmin) / (self.vmax - self.vmin)
        fill = self.track.copy()
        fill.w = int(self.track.w * t)
        pygame.draw.rect(surf, self.color, fill, border_radius=3)
        pygame.draw.circle(surf, (240, 240, 245), (self.track.x + fill.w, self.track.centery), 8)


class Button:
    def __init__(self, rect, text, on_click, active=None, color=(80, 110, 170)):
        self.rect = pygame.Rect(rect)
        self.text, self.on_click, self.active, self.color = text, on_click, active, color

    def handle(self, e):
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and self.rect.collidepoint(e.pos):
            self.on_click()
            return True
        return False

    def draw(self, surf, font):
        on = self.active() if self.active else False
        hover = self.rect.collidepoint(pygame.mouse.get_pos())
        col = self.color if on else ((70, 76, 92) if hover else (52, 57, 70))
        pygame.draw.rect(surf, col, self.rect, border_radius=6)
        txt = self.text() if callable(self.text) else self.text
        t = font.render(txt, True, TEXT)
        surf.blit(t, t.get_rect(center=self.rect.center))


class Game:
    def __init__(self):
        pygame.mixer.pre_init(44100, -16, 2, 512)
        pygame.init()
        self.snd = Sounds()
        self.point_rally = 0
        self.point_wow = False
        pygame.display.set_caption("乒乓教练模拟器")
        self.screen = pygame.display.set_mode((W, H))
        self.clock = pygame.time.Clock()
        self.font_s = load_font(15)
        self.font = load_font(18)
        self.font_l = load_font(26)
        self.font_xl = load_font(40)
        self.cam = Camera()
        self.painter = Painter(self.cam, self.screen)
        self.rng = np.random.default_rng()
        self.env = PongEnv(1, train=False, seed=int(time.time()) % 100000)
        for j in range(2):
            for idx in sim.STAT_ORDER:
                self.env.S[0, j, idx] = DEFAULT_STATS / 100.0
        self.obs = self.env.reset()
        self.policy = None
        self.model_msg = ""
        self.load_model()
        self.modes = ["rl" if self.policy else "bot"] * 2
        self.paused = False
        self.speed = 1.0
        self.acc = 0.0
        self.trail = []
        self.ball_rot = np.eye(3)
        self.anim = Animator()
        self.act = np.zeros((2, sim.ACT_DIM))
        self.poses = None
        self.glide = [None, None]
        self.arm_mem = [{}, {}]
        self.ball_trail = [None, None, None]
        self.settle = 0.0
        self.glide_prev = [None, None]
        self.glide_disp = [None, None]
        self.hand_shown = [None, None]
        self.fake_pending = None
        self.fake_shown = None
        self.serve_info = None
        self.recv_info = None
        # 练习模式：不计分、不累积疲劳；手动指定发球，点“发球”才发
        self.practice = False
        self.prac = {"server": 0, "kind": sim.SV_HOOK, "toss": None, "var": None, "depth": None,
                     "rally": None}
        self.serve_go = False
        self.msg = ""
        self.msg_color = TEXT
        self.dragging_cam = None
        self.page = "stat"
        self.build_ui()

    def load_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                pol = Policy.load(MODEL_PATH)
                if pol.norm.mean.shape[0] > sim.OBS_DIM or pol.logstd.shape[0] != sim.ACT_DIM:
                    self.model_msg = "模型版本不匹配，需重新运行 python train.py"
                    self.policy = None
                    return
                self.policy = pol
                self.model_msg = f"已加载 RL 模型（训练 {self.policy.iteration} 轮）"
                if pol.norm.mean.shape[0] < sim.OBS_DIM:
                    self.model_msg += "·旧版模型"
                return
            except Exception as ex:  # noqa: BLE001
                self.model_msg = f"模型加载失败：{ex}"
        else:
            self.model_msg = "未找到模型，先运行 python train.py"
        self.policy = None

    def set_mode(self, j, mode):
        self.modes[j] = mode

    def set_stat(self, j, k, v):
        self.env.S[0, j, k] = v / 100.0

    def visible(self, wdg):
        p = getattr(wdg, "page", None)
        return p is None or self.page in (p if isinstance(p, tuple) else (p,))

    def next_page(self):
        pages = ("stat", "style", "equip", "practice")
        self.page = pages[(pages.index(self.page) + 1) % len(pages)]

    def set_equip(self, j, key, val):
        """改器材。换持拍手、握法从下一分开始生效（这一分站位、动作已经按原来的手在做）。"""
        self.env.S[0, j, key] = val
        if key in (sim.HAND, sim.GRIP):
            self.restart()

    def equip_text(self, j):
        S = self.env.S[0, j]
        return (f"{'左手' if S[sim.HAND] < 0 else '右手'}{sim.GRIP_NAMES[int(S[sim.GRIP] > 0.5)]} · "
                f"正手{sim.RUB_NAMES[int(S[sim.RFH])]} 反手{sim.RUB_NAMES[int(S[sim.RBH])]}")

    def random_equip(self):
        for j in range(2):
            sim.random_equipment(self.env.S, 0, j)
        self.restart()

    def set_practice(self, on):
        if on == self.practice:
            return
        self.practice = on
        self.serve_go = False
        if on:
            self.env.F[0] = 0.0
            self.new_practice_point()
        else:
            self.anim.rescue_th = (0.72, 0.6, 9.0)
            self.restart()

    def new_practice_point(self):
        self.obs = self.env.restart_point(self.prac["server"])
        self.trail.clear()
        self.anim.reset()
        self.serve_go = False
        self.serve_info = None
        self.recv_info = None
        self.msg = ""

    def set_prac(self, key, value):
        self.prac[key] = value
        if key == "server" and self.practice and self.env.I[0, sim.PHASE] == sim.PH_SERVE and not self.serve_go:
            self.new_practice_point()

    def press_serve(self):
        if not self.practice:
            self.set_practice(True)
        if self.env.I[0, sim.PHASE] == sim.PH_SERVE:
            self.serve_go = True

    def practice_override(self, a):
        """练习模式下改写发球方这一步的动作与发球计划。"""
        I, T = self.env.I[0], self.env.T[0]
        tech = self.prac["rally"]
        self.anim.rescue_th = (0.25, 0.4, 7.5) if tech is not None and PRAC_TECH[tech][4] == PRAC_WIDE else (0.72, 0.6, 9.0)
        if I[sim.PHASE] == sim.PH_FLY and tech is not None:
            # 接发方用选中的技术，发球方用陪练技术（削对拉、放高球对扣杀）；离台距离决定能不能用这项技术
            srv = int(I[sim.SERVER])
            for j in range(2):
                k = PRAC_TECH[tech][4] if j == srv else tech
                a5, a6, dist = PRAC_TECH[k][1:4]
                a[0, j, 5], a[0, j, 6] = a5, a6
                if j != srv and PRAC_TECH[tech][4] == PRAC_WIDE:
                    # 救球练习：接球方横向起动慢半拍，只能伸到极限去够
                    a[0, j, 1] *= 0.4
                    if I[sim.PVALID] == 1 and T[sim.LM0 + j] * (T[sim.PY] - self.env.P[0, j, 1]) < -0.1:
                        a[0, j, 8] = 1.0
                if k == PRAC_WIDE:
                    # 正手大角与中路交替（左右手不同时，对方正手在自己视角里的方向用两人的横向换算系数求）
                    fh = T[sim.LM0 + j] * T[sim.LM0 + 1 - j]
                    a[0, j, 4] = 0.8 * fh if (int(I[sim.HITS]) // 2) % 2 == 0 else -0.15 * fh
                    a[0, j, 3] = 0.6
                if dist is not None:
                    d = abs(self.env.P[0, j, 0]) - sim.HALF_L
                    a[0, j, 0] = float(np.clip(2.5 * (d - dist), -1.0, 1.0))
        if I[sim.PHASE] != sim.PH_SERVE:
            return
        srv = int(I[sim.SERVER])
        p = self.prac
        T[sim.SPLAN0 + srv] = p["kind"]
        if p["var"] is not None:
            T[sim.VPLAN0 + srv] = p["var"]
        if p["toss"] is not None:
            a[0, srv, 9] = (-0.8, 0.0, 0.8)[p["toss"]]
        if p["depth"] is not None:
            a[0, srv, 3] = (-0.9, 0.1, 0.9)[p["depth"]]
        # 每一球都当作对手第一次见：熟悉度清零
        T[sim.FAM0:sim.FAM0 + 2 * sim.N_SERVE] = 0.0
        if not self.serve_go:
            T[sim.TIMER] = max(T[sim.TIMER], sim.SERVE_WAIT)

    def randomize(self):
        for sl in self.stat_sliders:
            sl.set_value(float(self.rng.integers(10, 101)))
        for sl in self.style_sliders:
            sl.set_value(float(self.rng.integers(-100, 101)))

    def restart(self):
        if self.practice:
            self.new_practice_point()
            return
        self.obs = self.env.reset()
        self.trail.clear()
        self.anim.reset()
        self.fake_pending = None
        self.fake_shown = None
        self.serve_info = None
        self.recv_info = None
        self.msg = ""

    def build_ui(self):
        x0 = VIEW_W + 22
        w = PANEL_W - 44
        self.widgets = []
        self.stat_sliders = []
        self.style_sliders = []
        self.widgets.append(Button((x0 + w - 120, 14, 120, 30),
                                   lambda: {"stat": "下一页：风格", "style": "下一页：器材", "equip": "下一页：练习",
                                            "practice": "下一页：属性"}[self.page],
                                   self.next_page))
        self.build_practice_ui(x0, w)
        self.build_equip_ui(x0, w)
        y = 96
        self.block_y = []
        for j in range(2):
            self.block_y.append(y)
            bw = 96
            b = Button((x0 + w - 2 * bw - 6, y - 2, bw, 28), "RL 模型",
                       lambda j=j: self.set_mode(j, "rl"),
                       lambda j=j: self.modes[j] == "rl", COLORS[j])
            b.page = ("stat", "style")
            self.widgets.append(b)
            b = Button((x0 + w - bw, y - 2, bw, 28), "脚本 AI",
                       lambda j=j: self.set_mode(j, "bot"),
                       lambda j=j: self.modes[j] == "bot", COLORS[j])
            b.page = ("stat", "style")
            self.widgets.append(b)
            y += 38
            for k in range(sim.N_STYLE):
                sl = Slider(x0, y + k * ROW, w, sim.STYLE_NAMES[k], -100, 100, 0, COLORS[j],
                            lambda v, j=j, k=k: self.set_stat(j, N_STATS + k, v))
                sl.page = "style"
                self.widgets.append(sl)
                self.style_sliders.append(sl)
            # 属性两列排：左列力量/敏捷/技术/发球，右列反应/体力/心态/手感
            cw = (w - 20) // 2
            for k, (idx, name) in enumerate(zip(sim.STAT_ORDER, sim.STAT_ORDER_NAMES)):
                sl = Slider(x0 + (k // STAT_ROWS) * (cw + 20), y + (k % STAT_ROWS) * ROW, cw, name, 0, 100,
                            DEFAULT_STATS, COLORS[j], lambda v, j=j, idx=idx: self.set_stat(j, idx, v))
                sl.page = "stat"
                self.widgets.append(sl)
                self.stat_sliders.append(sl)
            # 疲劳、压力、适应三行状态条
            y += max(STAT_ROWS, sim.N_STYLE) * ROW + 40 + STATUS_EXTRA
        self.speed_y = y
        self.widgets.append(Slider(x0, y, w, "时间倍速", 0.1, 3.0, 1.0, (230, 190, 90),
                                   lambda v: setattr(self, "speed", v), "{:.2f}x"))
        y += 52
        b3 = (w - 20) // 3
        self.widgets.append(Button((x0, y, b3, 34), lambda: "继续" if self.paused else "暂停",
                                   lambda: setattr(self, "paused", not self.paused)))
        self.widgets.append(Button((x0 + b3 + 10, y, b3, 34), "重置比赛", self.restart))
        self.widgets.append(Button((x0 + 2 * (b3 + 10), y, b3, 34),
                                   lambda: "开声音" if not self.snd.on else "静音",
                                   lambda: setattr(self.snd, "on", not self.snd.on),
                                   lambda: not self.snd.on))
        bw = (w - 10) // 2
        y += 44
        self.widgets.append(Button((x0, y, bw, 34), "随机属性", self.randomize))
        self.widgets.append(Button((x0 + bw + 10, y, bw, 34), "重载模型", self.reload))
        self.help_y = y + 38

    def build_equip_ui(self, x0, w):
        """器材页：每名球员的持拍手、握法、正反手胶皮。"""
        self.equip_labels = []
        y = 92

        def add(b):
            b.page = "equip"
            self.widgets.append(b)

        for j in range(2):
            self.equip_labels.append((NAMES[j], y, COLORS[j], True))
            y += 28
            for label, key, opts in (("持拍手", sim.HAND, [("右手", 1.0), ("左手", -1.0)]),
                                     ("握法", sim.GRIP, [(s, float(k)) for k, s in enumerate(sim.GRIP_NAMES)]),
                                     ("正手胶皮", sim.RFH, [(s, float(k)) for k, s in enumerate(sim.RUB_NAMES)]),
                                     ("反手胶皮", sim.RBH, [(s, float(k)) for k, s in enumerate(sim.RUB_NAMES)])):
                self.equip_labels.append((label, y, DIM, False))
                bw = (w - 76 - 4 * (len(opts) - 1)) // len(opts)
                for n, (text, val) in enumerate(opts):
                    add(Button((x0 + 76 + n * (bw + 4), y - 3, bw, 26), text,
                               lambda j=j, key=key, val=val: self.set_equip(j, key, val),
                               lambda j=j, key=key, val=val: abs(self.env.S[0, j, key] - val) < 0.5
                               if key != sim.HAND else (self.env.S[0, j, key] < 0) == (val < 0), COLORS[j]))
                y += 32
            y += 12
        add(Button((x0, y, w, 30), "随机器材（双方）", self.random_equip))
        y += 42
        for s in ("反胶：旋转最强，也最吃对方的转",
                  "正胶：出球快、不太吃转；生胶：更平更沉",
                  "长胶：几乎不造转，来球上旋回成下旋",
                  "防弧：出球慢，能把来球旋转卸掉",
                  "直板：发球、台内小球细腻，推挡稳；",
                  "　　　反手横打发力弱、覆盖面小",
                  "左手：动作整套镜像，侧旋方向相反",
                  "换持拍手/握法会重开这一分"):
            self.equip_labels.append((s, y, DIM, False))
            y += 20

    def build_practice_ui(self, x0, w):
        """练习页：模式开关、发球方、发球方式、抛球、旋转、落点，以及“发球”按钮。"""
        self.prac_labels = []
        y = 96

        def add(b):
            b.page = "practice"
            self.widgets.append(b)

        def row(label, key, options, ncol=None):
            nonlocal y
            self.prac_labels.append((label, y))
            y += 24
            ncol = ncol or len(options)
            bw = (w - 6 * (ncol - 1)) // ncol
            for n, (text, val) in enumerate(options):
                r, c = divmod(n, ncol)
                add(Button((x0 + c * (bw + 6), y + r * 34, bw, 28), text,
                           lambda val=val: self.set_prac(key, val),
                           lambda val=val: self.prac[key] == val))
            y += 34 * ((len(options) + ncol - 1) // ncol) + 8

        self.prac_labels.append(("练习模式（不计分、无疲劳、对手不熟悉发球）", y))
        y += 24
        bw = (w - 6) // 2
        add(Button((x0, y, bw, 28), "开启", lambda: self.set_practice(True), lambda: self.practice))
        add(Button((x0 + bw + 6, y, bw, 28), "关闭（回到比赛）", lambda: self.set_practice(False),
                   lambda: not self.practice))
        y += 42
        row("发球方", "server", [("蓝方", 0), ("红方", 1)])
        short = ["正手侧旋", "勾手", "逆旋转", "下蹲砍式", "砍式", "正手奔球", "反手侧旋", "反手急球"]
        row("发球方式", "kind", [(s, k) for k, s in enumerate(short)], ncol=4)
        row("抛球", "toss", [("AI 决定", None), ("低抛", 0), ("中抛", 1), ("高抛", 2)])
        row("旋转（侧旋手法选“不转”为纯侧旋）", "var",
            [("AI 决定", None), ("侧/下旋", 0), ("不转", 1), ("侧/上旋", 2)])
        row("落点", "depth", [("AI 决定", None), ("短球", 0), ("半出台", 1), ("长球", 2)])
        add(Button((x0, y + 6, w, 44),
                   lambda: ("发球！" if self.env.I[0, sim.PHASE] == sim.PH_SERVE and not self.serve_go
                            else "这一分进行中…") if self.practice else "开启练习并发球",
                   self.press_serve, lambda: self.practice and self.env.I[0, sim.PHASE] == sim.PH_SERVE
                   and not self.serve_go, (90, 170, 110)))
        self.prac_labels.append(("奔球、急球固定低抛；砍式发不出强上旋", y + 54))
        y += 78
        row("相持技术（接发方用，发球方陪练）", "rally",
            [("AI 决定", None)] + [(t[0], n) for n, t in enumerate(PRAC_TECH) if n != PRAC_WIDE], ncol=5)

    def reload(self):
        self.load_model()
        if not self.policy:
            self.modes = ["bot", "bot"]

    def actions(self):
        a = np.zeros((1, 2, sim.ACT_DIM))
        for j in range(2):
            o = self.obs[0, j]
            if self.modes[j] == "rl" and self.policy is not None:
                # 新增的观察维度都追加在末尾，旧模型截掉即可
                o = o[:self.policy.norm.mean.shape[0]]
                a[0, j] = self.policy.act(o[None].astype(np.float64))[0]
            else:
                a[0, j] = bot_actions(o[None], self.rng)[0]
        return a

    def update(self, dt):
        if self.paused:
            return
        # 一分开始时球员还在走回站位：先不推进模拟，站定一会儿再开始发球
        I = self.env.I[0]
        if (I[sim.PHASE] == sim.PH_SERVE and self.env.T[0, sim.TOSS] < 0.0
                and (any(g is not None for g in self.glide) or self.settle > 0.0)):
            self.settle = max(self.settle - min(dt, 0.1) * self.speed, 0.0) if not any(
                g is not None for g in self.glide) else SERVE_SETTLE
            self.acc = min(self.acc, self.env.dt * 0.999)
            return
        self.acc += min(dt, 0.1) * self.speed
        while self.acc >= self.env.dt:
            prev_phase, prev_hits = int(self.env.I[0, sim.PHASE]), int(self.env.I[0, sim.HITS])
            a = self.actions()
            if self.practice:
                self.practice_override(a)
            self.act = a[0]
            prev_m = (int(self.env.I[0, sim.M0]), int(self.env.I[0, sim.M1]))
            prev_net = int(self.env.I[0, sim.NETT])
            self.obs, _, done = self.env.step(a)
            self.play_sounds(prev_phase, prev_hits, prev_net, done[0])
            if self.practice:
                I = self.env.I[0]
                I[sim.SC0] = I[sim.SC1] = I[sim.G0] = I[sim.G1] = I[sim.M0] = I[sim.M1] = 0
                I[sim.FIRST] = self.prac["server"]
                self.env.F[0] = 0.0
                if prev_phase == sim.PH_SERVE and I[sim.PHASE] != sim.PH_SERVE:
                    self.serve_go = False
            self.anim.on_step(self.env, prev_phase, prev_hits, self.env.dt)
            if prev_phase == sim.PH_SERVE and self.env.I[0, sim.PHASE] != sim.PH_SERVE and self.env.I[0, sim.LSRV] >= 0:
                srv = int(self.env.I[0, sim.SERVER])
                k = int(self.env.I[0, sim.LSRV])
                b = self.env.B[0]
                rpm = np.linalg.norm(b[6:9]) * 60 / (2 * math.pi)
                self.serve_info = (srv, f"{toss_text(self.env.T[0, sim.TOSS])} · {sim.SERVE_NAMES[k]} · "
                                        f"{serve_spin_text(b[3:6], b[6:9])} {rpm:.0f} rpm"
                                        + (" · 回抛" if self.env.I[0, sim.LHID] else ""))
            if prev_hits == 0 and self.env.I[0, sim.HITS] == 1:
                h = int(self.env.I[0, sim.LHIT])
                self.recv_info = (h, tech_text(int(self.env.I[0, sim.LTECH]), int(self.env.I[0, sim.LSIDE]),
                                               int(self.env.I[0, sim.LSTK])))
            if self.env.I[0, sim.PHASE] == sim.PH_SERVE and prev_phase != sim.PH_SERVE:
                self.serve_info = None
                self.recv_info = None
            if self.env.I[0, sim.HITS] > prev_hits:
                lf = int(self.env.I[0, sim.LFAKE])
                self.fake_pending = (lf - 1, int(self.env.I[0, sim.HITS])) if lf > 0 else None
            if prev_phase == sim.PH_SERVE:
                self.fake_shown = None
            self.acc -= self.env.dt
            if done[0]:
                w = int(self.env.winner[0])
                r = int(self.env.reason[0])
                # 只有被骗的一方真的没接住这一拍，才算假动作得手
                fp = self.fake_pending
                if fp is not None and w == fp[0] and int(self.env.hits[0]) == fp[1] and r == sim.R_MISS:
                    self.fake_shown = fp[0]
                self.fake_pending = None
                if w >= 0:
                    self.msg = f"{NAMES[w]}得分！ {NAMES[1 - w]}{REASON_TEXT[r]}"
                    if int(self.env.I[0, sim.M0 + w]) > prev_m[w]:
                        self.msg = f"{NAMES[w]}赢下比赛！（疲劳清零，开始新一场）"
                    self.msg_color = COLORS[w]
                else:
                    self.msg = "发球擦网，重发" if r == sim.R_LET else "回合超时，重发球"
                    self.msg_color = TEXT

    def play_sounds(self, prev_phase, prev_hits, prev_net, done):
        """按这一步发生的事件放音效：击球（按挥拍速度和胶皮）、球落台/落地、擦网、得分。"""
        I, T, B = self.env.I[0], self.env.T[0], self.env.B[0]
        if not self.snd.ok:
            return
        sx, _ = self.cam.pt(B[0:3])
        pan = float(np.clip((sx - VIEW_W / 2) / (VIEW_W / 2), -1.0, 1.0))
        hit = (prev_phase == sim.PH_SERVE and I[sim.PHASE] == sim.PH_FLY) or I[sim.HITS] > prev_hits
        if hit:
            self.snd.racket(float(T[sim.LSW]), int(T[sim.LRUB]), pan)
            self.point_rally = int(I[sim.HITS])
        if I[sim.NETT] and not prev_net:
            self.snd.play(self.snd.net, 0.8, pan)
        # 落台/落地按子步轨迹找最低点：一步 1/30 秒，快球一步飞出半米多，只看步末的位置和速度
        # 会漏掉在端线附近落台、或落台后同一步里就被击回的球
        path = [p for p in self.ball_trail if p is not None] + [self.env.hist[0, s, 0:3].copy()
                                                                  for s in range(self.env.nsub)]
        pdt = self.env.dt / self.env.nsub
        for k in range(1, len(path) - 1):
            a, b, c = path[k - 1], path[k], path[k + 1]
            if np.linalg.norm(b - a) > 0.5 or np.linalg.norm(c - b) > 0.5:
                continue
            vin = max(a[2] - b[2], path[k - 2][2] - a[2] if k >= 2 else 0.0) / pdt
            if not (vin > 0.6 and c[2] > b[2] - 0.002) or k < len(path) - 1 - self.env.nsub:
                continue
            bpan = float(np.clip((self.cam.pt(b)[0] - VIEW_W / 2) / (VIEW_W / 2), -1.0, 1.0))
            if abs(b[0]) <= HALF_L + 0.03 and abs(b[1]) <= HALF_W + 0.03 and abs(b[2] - TABLE_H) < 0.07:
                self.snd.play(self.snd.table, (0.45 + 0.55 * min(vin / 5.0, 1.0)) * min(vin / 1.5, 1.0), bpan)
            elif b[2] < 0.3:
                self.snd.play(self.snd.floor, min(vin / 6.0, 1.0), bpan)
        self.ball_trail = path[-3:]
        if any(self.anim.gait[j][0] == "倒地爆冲" and self.anim.t - self.anim.gait[j][1] < 0.05 for j in range(2)):
            self.point_wow = True
        if done:
            if int(self.env.winner[0]) >= 0:
                if self.point_rally >= 9 or self.point_wow:
                    self.snd.play(self.snd.applause, 0.7)
            self.point_rally = 0
            self.point_wow = False

    def handle(self, e):
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_SPACE:
                self.paused = not self.paused
            elif e.key == pygame.K_m:
                self.snd.on = not self.snd.on
            elif e.key == pygame.K_r:
                self.reload()
            elif e.key == pygame.K_c:
                self.cam.reset()
            elif e.key in (pygame.K_1, pygame.K_2):
                j = 0 if e.key == pygame.K_1 else 1
                self.modes[j] = "bot" if self.modes[j] == "rl" else "rl"
            elif e.key == pygame.K_ESCAPE:
                return False
        for wdg in self.widgets:
            if self.visible(wdg) and wdg.handle(e):
                return True
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and e.pos[0] < VIEW_W:
            self.dragging_cam = e.pos
        elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
            self.dragging_cam = None
        elif e.type == pygame.MOUSEMOTION and self.dragging_cam:
            dx = e.pos[0] - self.dragging_cam[0]
            dy = e.pos[1] - self.dragging_cam[1]
            self.cam.az -= dx * 0.006
            self.cam.el += dy * 0.004
            self.cam.update()
            self.dragging_cam = e.pos
        elif e.type == pygame.MOUSEWHEEL and pygame.mouse.get_pos()[0] < VIEW_W:
            self.cam.dist *= 0.9 ** e.y
            self.cam.update()
        return True

    # ------------------------------------------------------------------ 绘制
    def poly(self, surf, color, pts, width=0):
        pygame.draw.polygon(surf, color, [self.cam.pt(p) for p in pts], width)

    def line(self, surf, color, a, b, width=1):
        pygame.draw.line(surf, color, self.cam.pt(a), self.cam.pt(b), max(1, int(width)))

    def draw_floor(self, surf):
        self.poly(surf, (34, 40, 52), [(-5, -3.5, 0), (5, -3.5, 0), (5, 3.5, 0), (-5, 3.5, 0)])
        for x in range(-5, 6):
            self.line(surf, (44, 51, 66), (x, -3.5, 0), (x, 3.5, 0))
        for y in np.arange(-3.5, 3.6, 1.0):
            self.line(surf, (44, 51, 66), (-5, y, 0), (5, y, 0))

    def draw_table(self, surf, ball):
        L, Wd, Z = HALF_L, HALF_W, TABLE_H
        for sx in (-1, 1):
            for sy in (-1, 1):
                a = (sx * (L - 0.25), sy * (Wd - 0.12), Z - 0.03)
                self.line(surf, (60, 64, 72), a, (a[0], a[1], 0), self.cam.scale(a, 0.05))
        self.poly(surf, (16, 52, 82), [(-L, -Wd, Z - 0.035), (L, -Wd, Z - 0.035),
                                        (L, Wd, Z - 0.035), (-L, Wd, Z - 0.035)])
        self.poly(surf, (24, 86, 140), [(-L, -Wd, Z), (L, -Wd, Z), (L, Wd, Z), (-L, Wd, Z)])
        white = (235, 240, 245)
        self.poly(surf, white, [(-L, -Wd, Z), (L, -Wd, Z), (L, Wd, Z), (-L, Wd, Z)], 2)
        self.line(surf, white, (-L, 0, Z), (L, 0, Z), 1)
        if abs(ball[0]) <= L and abs(ball[1]) <= Wd and ball[2] >= Z:
            self.shadow(surf, (ball[0], ball[1], Z), 0.025 * (1 + (ball[2] - Z)))

    def shadow(self, surf, p, r):
        sx, sy = self.cam.pt(p)
        rr = max(2, self.cam.scale(p, r))
        s = pygame.Surface((int(rr * 4), int(rr * 2)), pygame.SRCALPHA)
        pygame.draw.ellipse(s, (0, 0, 0, 90), s.get_rect())
        surf.blit(s, (sx - rr * 2, sy - rr))

    def draw_net(self, surf):
        z0, z1, nw = TABLE_H, TABLE_H + NET_H, NET_HALF_W
        s = pygame.Surface((VIEW_W, H), pygame.SRCALPHA)
        pygame.draw.polygon(s, (220, 225, 235, 70),
                            [self.cam.pt(p) for p in [(0, -nw, z0), (0, nw, z0), (0, nw, z1), (0, -nw, z1)]])
        surf.blit(s, (0, 0))
        self.line(surf, (245, 245, 250), (0, -nw, z1), (0, nw, z1), 2)
        for y in (-nw, nw):
            self.line(surf, (90, 95, 105), (0, y, z0 - 0.03), (0, y, z1), 3)

    def body_points(self, body, hand, j, pose=None, raw=False):
        """火柴人骨架。手伸得比胳膊长时上身跟着前倾压低去够；球拍探进台内时持拍侧的右脚上步插到台下；
        pose 里的转体/站位/下蹲用于引拍转腰和发球侧身。raw=True 只返回两只脚的理想落点。"""
        sj = -1.0 if j == 0 else 1.0
        # 横向分量乘持拍手方向：左手球员整副骨架左右镜像
        sl = sj * self.hand_sign(j)
        bx = body[0] + sj * 0.32
        bx = sj * max(sj * bx, HALF_L + 0.25)
        by = body[1]
        if pose is None and self.poses is not None:
            pose = self.poses[j]
        pose = pose or {}
        lg = pose.get("lunge")
        if lg is not None:
            bx, by = bx + lg[0], by + lg[1]
        bob, cr = pose.get("bob", 0.0), pose.get("crouch", 0.0)
        tw, stc = pose.get("twist", 0.0), pose.get("stance", 0.0)
        paddle = np.asarray(hand, dtype=float)
        sz = min(max(1.25 + 0.35 * (hand[2] - 0.9), 1.1), 1.55) - 1.2 * cr + bob
        lean = -sl * self.foot[j]
        cross = -sl * self.foot[2 + j]
        hy, ny = by + 0.07 * lean, by + 0.2 * lean

        sx, sy = sj * 0.2 * math.sin(tw), sl * 0.2 * math.cos(tw)
        fx = -sj
        # 往侧面扑倒时上身是侧倒，不是往前弯腰
        tn = float(np.linalg.norm(pose.get("tilt", (0.0, 0.0))))
        bend = TORSO_BEND + 0.6 * cr * (1.0 - 0.7 * min(tn / 0.5, 1.0))
        cx = bx + fx * bend
        sz -= 0.25 * bend
        shift = np.zeros(2)
        for sh, tgt in (((cx + sx, ny + sy, sz), paddle), ((cx - sx, ny - sy, sz), pose.get("lhand"))):
            if tgt is None:
                continue
            d = np.asarray(tgt[:2]) - np.asarray(sh[:2])
            dist = math.sqrt(float(d @ d) + (tgt[2] - sh[2]) ** 2)
            if dist > ARM_LEN:
                shift += d / (np.linalg.norm(d) + 1e-9) * (dist - ARM_LEN)
        shift[0] = fx * max(fx * shift[0], 0.0)
        # 救球、倒地爆冲：击球后上身继续朝球的方向扑出去
        shift = shift + np.asarray(pose.get("tilt", (0.0, 0.0)), dtype=float)
        sn = float(np.linalg.norm(shift))
        # 救球时允许上身扑得更远；上限要跟着救球外倾一起渐变，突然放开/收紧会让身体一帧内被拉出去或弹回来
        smax = 0.5 + 0.25 * min(tn / 0.15, 1.0)
        if sn > smax:
            shift *= smax / sn
            sn = smax
        drop = 0.6 * sn
        nx = sj * max(sj * (cx + shift[0]), HALF_L - 0.4)
        nz = sz - drop
        # 探身够台内短球时上身压到台面以下会陷进台面里：在台面范围内就抬到台面之上
        if abs(nx) < HALF_L + 0.05 and abs(ny + shift[1]) < HALF_W + 0.05:
            nz = max(nz, TABLE_H + 0.09)
        neck = (nx, ny + shift[1], nz)
        rsh, lsh = (nx + sx, neck[1] + sy, neck[2]), (nx - sx, neck[1] - sy, neck[2])
        hip = (bx + 0.6 * shift[0], hy + 0.6 * shift[1], 0.92 - 1.2 * cr - 0.3 * drop + bob)
        # 头顺着躯干往前探一半：从背后的镜头看，前倾主要靠头压低到肩上才看得出来
        tor = unit3(np.subtract(neck, hip))
        hd = unit3(0.5 * tor + np.array([0.5 * fx * 0.3, 0.0, 0.5]))
        head = (nx + fx * 0.05 + 0.25 * shift[0] + 0.17 * hd[0], neck[1] + 0.25 * shift[1] + 0.04 * lean + 0.17 * hd[1],
                neck[2] + 0.17 * hd[2] - 0.05 * sn)

        wdt = 0.22 * (1.0 - 0.3 * abs(stc))
        lfy, rfy = by - sl * wdt, by + sl * wdt
        if cross:
            lfy, rfy = by + cross * 0.08, by - cross * 0.04
        lfx = bx + sj * (0.1 - 0.25 * max(stc, 0.0) + 0.15 * max(-stc, 0.0))
        rfx = bx + sj * (0.0 - 0.25 * max(-stc, 0.0) + 0.15 * max(stc, 0.0))
        reach = -sj * (paddle[0] - bx)
        if reach > 0.5:
            u = min((reach - 0.5) / 0.3, 1.0)
            u = u * u * (3.0 - 2.0 * u)
            tx = sj * max(sj * (paddle[0] + sj * 0.25), HALF_L - 0.3)
            rfx, rfy = rfx + (tx - rfx) * u, rfy + (paddle[1] + sl * 0.1 - rfy) * u
        feet = np.array([[lfx, lfy], [rfx, rfy]])
        for k in range(2):
            d = feet[k] - np.asarray(hip[:2])
            dn = float(np.linalg.norm(d))
            if dn > FOOT_R:
                feet[k] = np.asarray(hip[:2]) + d * (FOOT_R / dn)
        if raw:
            return feet
        lfoot, rfoot = (feet[0, 0], feet[0, 1], 0.02), (feet[1, 0], feet[1, 1], 0.02)
        shown = []
        for f in (pose.get("lfoot", lfoot), pose.get("rfoot", rfoot)):
            d = np.asarray(f[:2]) - np.asarray(hip[:2])
            dn = float(np.linalg.norm(d))
            if dn > FOOT_R + 0.12:
                xy = np.asarray(hip[:2]) + d * ((FOOT_R + 0.12) / dn)
                f = (xy[0], xy[1], f[2])
            shown.append(f)
        lfoot, rfoot = shown
        fwd_x = -sj
        free = (lsh[0] + fwd_x * (0.28 + 0.15 * tw), lsh[1] + sl * 0.12, lsh[2] - 0.38)
        lhand = pose.get("lhand", free)
        # 持拍手、肘不能进到躯干里：拍子贴到胸前时推到胸前一拍远，肘进去了就往外推
        up_t = unit3(np.subtract(neck, hip))
        ac_t = unit3(np.subtract(rsh, lsh))
        ac_t = unit3(ac_t - (ac_t @ up_t) * up_t)
        fw_t = np.cross(up_t, ac_t)
        if fw_t[0] * fwd_x < 0:
            fw_t = -fw_t
        hz_t = float(np.linalg.norm(np.subtract(neck, hip)))
        hip_v = np.asarray(hip, dtype=float)

        def torso_xyz(p):
            q = np.asarray(p, dtype=float) - hip_v
            return q, float(q @ ac_t), float(q @ fw_t), float(q @ up_t)

        # 上身扑出去的距离有上限（倒地爆冲时身体倒向另一侧），够不着的球拍跟着手臂收回来，否则前臂会被拉长；
        # 要在下面躯干/头的避让之前做，不然收回来的球拍会穿进身体
        d = np.asarray(paddle, dtype=float) - np.asarray(rsh, dtype=float)
        dn = float(np.linalg.norm(d))
        if dn > UPPER_ARM + FORE_ARM:
            paddle = np.asarray(rsh, dtype=float) + d * ((UPPER_ARM + FORE_ARM) / dn)

        pmem = self.arm_mem[j]

        def push_out(p, pad, key=None):
            """点穿进躯干盒就往两侧或身前推出去；绝不往后推（往后推等于把球拍塞进身体后面、直臂斜穿胸口）。
            背面的深度只用来判断“是否在里面”，推的方向只用侧面和身前。
            key：记住上一帧往哪边推。点在胸口正中时往左、往右、往前推的距离差不多，每帧挑最近的会来回换边（闪）；
            上一帧的方向只要不比最近的远出 PUSH_STICK 就接着用。"""
            q, x, y, z = torso_xyz(p)
            fy = 0.105 + pad - y
            by = y + 0.085 + pad
            if not (-0.06 < z < hz_t + 0.03) or min(fy, by, 0.17 + pad - abs(x)) <= 0.0:
                if key:
                    pmem.pop(key, None)
                return np.asarray(p, dtype=float)
            cost = {"R": 0.17 + pad - x, "L": 0.17 + pad + x, "F": fy}
            c = min(cost, key=cost.get)
            prev = pmem.get(key) if key else None
            if prev is not None and cost[prev] <= cost[c] + PUSH_STICK:
                c = prev
            if key:
                pmem[key] = c
            if c == "F":
                return np.asarray(p, dtype=float) + fw_t * fy
            return np.asarray(p, dtype=float) + ac_t * (cost["R"] if c == "R" else -cost["L"])

        def above_table(p, lift=0.05, m=0.03):
            """手、拍不能穿进台面：在台面范围里（留点余量）就抬到台面之上。"""
            p = np.asarray(p, dtype=float)
            if abs(p[0]) < HALF_L + m and abs(p[1]) < HALF_W + m and p[2] < TABLE_H + lift:
                return np.array([p[0], p[1], TABLE_H + lift])
            return p

        raw_pd = np.asarray(paddle, dtype=float).copy()
        paddle = above_table(push_out(paddle, 0.09, "po_p0"), 0.05)
        # 头往前探了，准备姿势举在胸前的球拍会碰到脸：往身前推开
        q = np.asarray(paddle, dtype=float) - np.asarray(head, dtype=float)
        side_d = q - (q @ fw_t) * fw_t
        if float(np.linalg.norm(side_d)) < 0.24 and -0.15 < float(q @ fw_t) < 0.24:
            paddle = np.asarray(paddle, dtype=float) + fw_t * (0.24 - float(q @ fw_t))

        # 躯干基向量预先拆成标量：arm_bad 每帧要算几百次，用 numpy 建小数组的开销比算式本身还大
        _acx, _acy, _acz = float(ac_t[0]), float(ac_t[1]), float(ac_t[2])
        _fwx, _fwy, _fwz = float(fw_t[0]), float(fw_t[1]), float(fw_t[2])
        _upx, _upy, _upz = float(up_t[0]), float(up_t[1]), float(up_t[2])
        _hx, _hy, _hhz = float(hip_v[0]), float(hip_v[1]), float(hip_v[2])
        _px, _py, _pz = float(head[0]), float(head[1]), float(head[2])

        def pt_pen(px, py, pz, pad):
            qx, qy, qz = px - _hx, py - _hy, pz - _hhz
            x = qx * _acx + qy * _acy + qz * _acz
            y = qx * _fwx + qy * _fwy + qz * _fwz
            z = qx * _upx + qy * _upy + qz * _upz
            if z < -0.06 or z > hz_t + 0.03:
                t = 0.0
            else:
                ax = x if x >= 0.0 else -x
                t = 0.105 + pad - y
                t2 = y + 0.085 + pad
                t3 = 0.17 + pad - ax
                if t2 < t:
                    t = t2
                if t3 < t:
                    t = t3
                if t < 0.0:
                    t = 0.0
            dx, dy, dz = px - _px, py - _py, pz - _pz
            h = 0.105 + pad + 0.03 - math.sqrt(dx * dx + dy * dy + dz * dz)
            return t if t > h else (h if h > 0.0 else 0.0)

        def arm_bad(sh, el, hd, us, pads):
            """上臂、前臂采样点穿进躯干或碰头的最大深度（纯算式，逐点算；逐点 numpy 太慢）。"""
            sh, el, hd = tuple(sh), tuple(el), tuple(hd)
            d = 0.0
            for u, pad in zip((0.5, 0.8), pads[:2]):
                p = pt_pen(sh[0] + (el[0] - sh[0]) * u, sh[1] + (el[1] - sh[1]) * u,
                           sh[2] + (el[2] - sh[2]) * u, pad)
                if p > d:
                    d = p
            for u, pad in zip(us, pads[2:]):
                p = pt_pen(el[0] + (hd[0] - el[0]) * u, el[1] + (hd[1] - el[1]) * u,
                           el[2] + (hd[2] - el[2]) * u, pad)
                if p > d:
                    d = p
            return d

        def settle(sh, hd, pole, blen, us, pads, out_dir, hold=None):
            """把肘的极向量往能减小穿透的方向推：先试几个方向挑最好的，再沿选中的方向细调一次。
            手臂朝向不同时固定往一个方向推推不动（肩→手几乎平行于那个方向时肘只能在垂直方向动）。
            绝大多数帧本来就不穿，只算一次就返回，不拖慢画面。"""
            pole = np.asarray(pole, dtype=float)
            el = arm_elbow(sh, hd, pole, UPPER_ARM, blen)
            if hold is None:
                d = arm_bad(sh, el, hd, us, pads)
                if d <= 1e-3:
                    return el, pole
            else:
                # 有上一帧的朝向时：往自然朝向回一点（pole）要留够余量才回，否则停在上一帧（hold）；
                # 刚好不穿就回的话，下一帧手一动又穿了，又得重新搜、跳开，来回闪
                if arm_bad(sh, el, hd, us, tuple(p + HOLD_MARGIN for p in pads)) <= 1e-3:
                    return el, pole
                e1 = arm_elbow(sh, hd, hold, UPPER_ARM, blen)
                d1 = arm_bad(sh, e1, hd, us, pads)
                if d1 <= 1e-3:
                    return e1, np.asarray(hold, dtype=float)
                # 上一帧的朝向也穿了：从它接着搜，跳得少
                pole, el, d = np.asarray(hold, dtype=float), e1, d1
            best = (d, el, pole)
            for dirn, gain in ((out_dir, 120.0), (unit3(fw_t), 120.0), (unit3(out_dir - fw_t), 160.0)):
                pl = pole + gain * d * dirn
                e2 = arm_elbow(sh, hd, pl, UPPER_ARM, blen)
                d2 = arm_bad(sh, e2, hd, us, pads)
                if d2 < best[0]:
                    best = (d2, e2, pl)
            if best[0] > 1e-3:
                d, el, pole = best
                for gain in (30.0, 400.0):
                    pl = pole + gain * d * out_dir
                    e2 = arm_elbow(sh, hd, pl, UPPER_ARM, blen)
                    d2 = arm_bad(sh, e2, hd, us, pads)
                    if d2 < best[0]:
                        best = (d2, e2, pl)
            return best[1], best[2]

        def fit_arm(sh, hd, pole, blen, us, pads, out_dir, hd_dir, hold=None):
            """先把肘挪开；手贴在躯干正前方时上臂是斜穿胸口的，光挪肘挪不开（肘只能在肩→手这条线的
            垂直方向动），再把手上限一点点往身侧外推。返回 (肘, 手, 总共推了多少)。"""
            hd0 = np.asarray(hd, dtype=float)
            hd = hd0.copy()
            el, _ = settle(sh, hd, pole, blen, us, pads, out_dir, hold)
            for _ in range(2):
                dmax = arm_bad(sh, el, hd, us, pads)
                if dmax <= 1e-3 or float(np.linalg.norm(hd - hd0)) > 0.16:
                    break
                hd = hd + hd_dir * (dmax * 1.5)
                el, _ = settle(sh, hd, pole, blen, us, pads, out_dir, hold)
            return el, hd

        def head_clear(p):
            """球拍、手不碰脸：离头太近就往身前推开。"""
            q = np.asarray(p, dtype=float) - np.asarray(head, dtype=float)
            side_d = q - (q @ fw_t) * fw_t
            f = float(q @ fw_t)
            if float(np.linalg.norm(side_d)) < 0.24 and -0.15 < f < 0.24:
                return np.asarray(p, dtype=float) + fw_t * (0.24 - f)
            return np.asarray(p, dtype=float)

        epole = np.asarray(pose.get("epole", (sj * 0.3, sl * 0.6, -1.0)), dtype=float)
        if os.environ.get("PP_FASTSKIN"):
            relbow = arm_elbow(rsh, paddle, epole, UPPER_ARM, FORE_ARM)
            lelbow = arm_elbow(lsh, lhand, -0.3 * fw_t - 0.6 * ac_t - 1.0 * up_t, UPPER_ARM, FORE_ARM - 0.04)
        else:
            # 避让每帧独立挑“最好”的肘向、推不推手是硬阈值，卡在边界上时两种解逐帧来回切，手臂闪。
            # 所以记住上一帧的结果：肘向从上一帧的偏离出发、逐帧回到自然朝向；手的避让偏移逐帧缓过去
            mem = pmem

            def recall_pole(key, natural):
                """返回 (这一帧先试的朝向：偏离衰减一点, 上一帧的朝向：偏离不衰减)。"""
                dev = mem.get(key)
                if dev is None:
                    return natural, None
                return natural + POLE_KEEP * dev, natural + dev

            def ease_hand(key, raw_p, fixed):
                fixed = np.asarray(fixed, dtype=float)
                prev = mem.get(key)
                corr = fixed - raw_p
                if prev is not None and float(np.linalg.norm(raw_p - prev[0])) < 0.4:
                    corr = prev[1] + (corr - prev[1]) * HAND_EASE
                mem[key] = (raw_p.copy(), corr)
                return raw_p + corr

            paddle = head_clear(above_table(push_out(paddle, 0.09, "po_p1"), 0.05))
            rp0, rhold = recall_pole("rpole", epole)
            relbow, paddle = fit_arm(rsh, paddle, rp0, FORE_ARM, (0.0, 0.33, 0.66),
                                     (0.025, 0.03, 0.03, 0.03, 0.03),
                                     unit3(ac_t + 0.8 * fw_t), unit3(ac_t + 0.5 * fw_t), rhold)
            # 推完手再用最终手位收一遍躯干/头/台面的边
            paddle = head_clear(above_table(push_out(paddle, 0.09, "po_p2"), 0.05))
            paddle = above_table(ease_hand("pad", raw_pd, paddle), 0.05)
            relbow, rp = settle(rsh, paddle, rp0, FORE_ARM, (0.0, 0.33, 0.66),
                                (0.025, 0.03, 0.03, 0.03, 0.03), unit3(ac_t + 0.8 * fw_t), rhold)
            mem["rpole"] = rp - epole
            # 抛球手：肘尖朝向跟着躯干转（侧身发球转体大，按世界坐标定的“后、左”会指进躯干里）
            lpole = -0.3 * fw_t - 0.6 * ac_t - 1.0 * up_t
            raw_lh = np.asarray(lhand, dtype=float).copy()
            lhand = above_table(push_out(lhand, 0.03, "po_l0"), 0.05)
            lp0, lhold = recall_pole("lpole", lpole)
            lelbow, lhand = fit_arm(lsh, lhand, lp0, FORE_ARM - 0.04, (0.0, 0.5), (0.025, 0.03, 0.03, 0.03),
                                    unit3(-ac_t + 0.8 * fw_t), unit3(-ac_t + 0.5 * fw_t), lhold)
            lhand = head_clear(above_table(push_out(lhand, 0.03, "po_l1"), 0.05))
            lhand = above_table(ease_hand("lh", raw_lh, lhand), 0.05)
            lelbow, lp = settle(lsh, lhand, lp0, FORE_ARM - 0.04, (0.0, 0.5), (0.025, 0.03, 0.03, 0.03),
                                unit3(-ac_t + 0.8 * fw_t), lhold)
            mem["lpole"] = lp - lpole
        hyaw = 0.5 * tw
        hxo, hyo = sj * HIP_W * math.sin(hyaw), sl * HIP_W * math.cos(hyaw)
        rhip, lhip = (hip[0] + hxo, hip[1] + hyo, hip[2]), (hip[0] - hxo, hip[1] - hyo, hip[2])
        fwd = np.array([-sj, 0.0, 0.0])

        def knee(h, f):
            h, f = np.asarray(h), np.asarray(f, dtype=float)
            d = f - h
            dn = float(np.linalg.norm(d)) + 1e-9
            ax = d / dn
            out = np.array([0.0, np.sign(h[1] - hip[1]), 0.0])
            p = fwd + 0.12 * out
            p = p - (p @ ax) * ax
            p /= np.linalg.norm(p) + 1e-9
            bend = math.sqrt(max(SEG_LEN ** 2 - (min(dn, 2 * SEG_LEN) / 2) ** 2, 0.0))
            return tuple((h + f) / 2 + p * bend)

        return dict(
            hip=hip, lhip=lhip, rhip=rhip, neck=neck, head=head, rsh=rsh, lsh=lsh, lhand=lhand,
            relbow=relbow, lelbow=lelbow,
            lknee=knee(lhip, lfoot), rknee=knee(rhip, rfoot), lfoot=lfoot, rfoot=rfoot,
            paddle=tuple(paddle),
        )

    def hand_sign(self, j):
        """持拍手：右手 +1，左手 -1（以模拟里这一分实际生效的设置为准）。"""
        m = 1.0 if j == 0 else -1.0
        return -1.0 if self.env.T[0, sim.LM0 + j] * m < 0.0 else 1.0

    def fh_face_out(self, j, bl):
        """拍面法向一侧（击球面）是不是正手面（黑色）。直板反手推挡、搓球用正面，直拍横打用反面。"""
        if bl >= 0:
            return True
        if self.env.S[0, j, sim.GRIP] > 0.5:
            return self.anim.hit_kind[j][0] in (sim.ST_BLOCK, sim.ST_PUSH) or self.anim.pred_tech[j] in PUSHBLOCK_TECH
        return False

    def draw_player(self, surf, body, hand, bl, j):
        """立体人物：四肢是有明暗的圆柱，头和关节是球体，躯干、短裤、鞋是按光照着色的多面体。"""
        b = {k: np.asarray(v, dtype=float) for k, v in self.body_points(body, hand, j).items()}
        col = COLORS[j]
        sj = -1.0 if j == 0 else 1.0
        pt = self.painter
        pt.begin(surf)
        up = np.array([0.0, 0.0, 1.0])
        face = np.array([-sj, 0.0, 0.0])

        lsh, rsh, lh, rh = b["lsh"], b["rsh"], b["lhip"], b["rhip"]
        t_up = unit3(b["neck"] - b["hip"])
        across = unit3(rsh - lsh)
        fw = unit3(np.cross(t_up, across))
        if fw @ face < 0:
            fw = -fw
        hip_across = unit3(rh - lh)
        hfw = unit3(np.cross(up, hip_across))
        if hfw @ face < 0:
            hfw = -hfw

        # 躯干：肩宽、胸厚，向下收到腰
        sh_in = 0.035
        ls_t, rs_t = lsh + across * sh_in - t_up * 0.02, rsh - across * sh_in - t_up * 0.02
        wl = lh + t_up * 0.14 - hip_across * 0.02
        wr = rh + t_up * 0.14 + hip_across * 0.02
        dt_, db_ = 0.105, 0.085
        torso = [wl - hfw * db_, wr - hfw * db_, wr + hfw * db_, wl + hfw * db_,
                 ls_t - fw * dt_, rs_t - fw * dt_, rs_t + fw * dt_, ls_t + fw * dt_]
        pt.hexa(torso, col, edge=False, skip=(0,), per_face=True, subdiv=2)
        # 圆润的胸廓和腰：肩线、腰线各一段粗圆柱，把方块的棱角包起来
        pt.capsule(ls_t - t_up * 0.01, rs_t - t_up * 0.01, 0.07, 0.07, col, depth_bias=0.002, seg=0.1)
        # 胸前号码条，让身体朝向更好认
        mid_t = (ls_t + rs_t) / 2 * 0.55 + (wl + wr) / 2 * 0.45 + fw * (dt_ * 0.95)
        pt.capsule(mid_t - across * 0.07, mid_t + across * 0.07, 0.018, 0.018, mix_col(col, (255, 255, 255), 0.55),
                   depth_bias=-0.01, outline=False)
        # 短裤：只有腰臀一块随骨盆走，裤腿跟着各自的大腿走（整块到大腿中部的盒子不随腿摆，腿一抬就从盒子侧面穿出来）
        lo = 0.03
        ll, rl = lh - hip_across * 0.03 - t_up * lo, rh + hip_across * 0.03 - t_up * lo
        shorts = [ll - hfw * 0.09, rl - hfw * 0.09, rl + hfw * 0.09, ll + hfw * 0.09,
                  wl - hfw * db_, wr - hfw * db_, wr + hfw * db_, wl + hfw * db_]
        pt.hexa(shorts, SHORTS_COL, edge=False, skip=(1,))

        def cover(a0, a1, under):
            """衣物套在皮肤外面：两段圆柱重叠、深度几乎一样，按深度排序会逐帧来回翻（闪），所以让衣物总画在后面。"""
            return min(0.0, self.cam.depth(under) - self.cam.depth((a0 + a1) / 2)) - 1e-4

        for hp, kn, ft in (("lhip", "lknee", "lfoot"), ("rhip", "rknee", "rfoot")):
            hip, knee, foot = b[hp], b[kn], b[ft]
            ankle = foot + up * 0.07
            th0, sk0 = hip + (knee - hip) * 0.35, ankle + (knee - ankle) * 0.25
            p0, p1 = hip + t_up * 0.04, hip + (knee - hip) * 0.4
            pt.capsule(p0, p1, 0.088, 0.075, SHORTS_COL, depth_bias=cover(p0, p1, (th0 + knee) / 2))
            pt.capsule(th0, knee, 0.07, 0.052, SKIN)
            pt.capsule(knee, sk0, 0.05, 0.04, SKIN)
            pt.capsule(sk0, ankle, 0.04, 0.036, (238, 238, 242), depth_bias=cover(sk0, ankle, (knee + sk0) / 2))
            toe = foot + face * 0.13 + up * 0.035
            heel = foot - face * 0.07 + up * 0.04
            sw = np.array([0.0, 1.0, 0.0]) * 0.045
            shoe = [heel - sw - up * 0.035, toe - sw * 0.9 - up * 0.03, toe + sw * 0.9 - up * 0.03, heel + sw - up * 0.035,
                    heel - sw + up * 0.04, toe - sw * 0.8 + up * 0.01, toe + sw * 0.8 + up * 0.01, heel + sw + up * 0.04]
            pt.hexa(shoe, SHOE_COL)

        pt.capsule(b["neck"] - t_up * 0.03, b["head"] - t_up * 0.06, 0.045, 0.042, SKIN)
        pt.sphere(b["head"], 0.105, SKIN, hair=HAIR_COL, face_dir=face)
        pt.sphere(lsh, 0.052, col, depth_bias=0.01)
        pt.sphere(rsh, 0.052, col, depth_bias=0.01)

        def arm(sh, el, hd, wrist_r):
            s1, u0 = sh + (el - sh) * 0.5, sh + (el - sh) * 0.45
            pt.capsule(sh, s1, 0.055, 0.05, col, depth_bias=cover(sh, s1, (u0 + el) / 2))
            pt.capsule(u0, el, 0.045, 0.04, SKIN, seg=0.1)
            pt.capsule(el, hd, 0.04, wrist_r, SKIN, seg=0.1)

        arm(lsh, b["lelbow"], b["lhand"], 0.03)
        pt.sphere(b["lhand"], 0.036, SKIN)

        # 持拍臂和球拍：拍面法向来自动画层（触球瞬间就是物理里的真实拍面），拍柄沿前臂方向
        c = b["paddle"]
        elbow = b["relbow"]
        pose = (self.poses[j] if self.poses is not None else None) or {}
        n = unit3(np.asarray(pose.get("face", (-sj, 0.0, 0.15)), dtype=float))
        u = c - elbow
        pen = self.env.S[0, j, sim.GRIP] > 0.5
        if pen and bl < 0:
            # 直板反手：拍头朝下
            u = u * 0.55 - up * 0.45 * float(np.linalg.norm(u))
        u = u - (u @ n) * n
        if np.linalg.norm(u) < 1e-3:
            u = up - n[2] * n
        u = unit3(u)
        e2 = np.cross(n, u)
        grip = c - u * (0.13 if pen else 0.16)
        arm(rsh, elbow, grip, 0.032)
        pt.sphere(grip, 0.036, SKIN, depth_bias=-0.005)
        pt.capsule(grip, c - u * 0.07, 0.016, 0.014, WOOD_COL, outline=False)
        fh_out = self.fh_face_out(j, bl)
        rf, rb = int(self.env.S[0, j, sim.RFH]), int(self.env.S[0, j, sim.RBH])
        r_out, r_in = (rf, rb) if fh_out else (rb, rf)
        col_out, col_in = (BLACK_RUBBER, RED_RUBBER) if fh_out else (RED_RUBBER, BLACK_RUBBER)
        pt.disc(c, u, e2, n, 0.085, 0.078, rubber_tint(col_out, r_out), rubber_tint(col_in, r_in), WOOD_COL, 0.014,
                pips=(r_out in PIPS_RUB, r_in in PIPS_RUB))
        pt.flush()

        name, t0 = pose.get("gait", ("", -9.0))
        age = pose.get("t", 0.0) - t0
        if name in ("救球", "倒地爆冲") and 0.0 <= age < 0.8:
            t = self.font.render(name + "！", True, (255, 220, 120))
            mid = (b["lfoot"] + b["rfoot"]) / 2
            fx, fy = self.cam.pt((mid[0], mid[1], 0.0))
            r = t.get_rect(center=(fx, fy + 16))
            pygame.draw.rect(surf, (10, 12, 18), r.inflate(14, 6), border_radius=6)
            surf.blit(t, r)

    def draw_ball(self, surf, ball):
        if len(self.trail) > 1:
            n = len(self.trail)
            for k in range(1, n):
                t = k / n
                c = lerp_col(BG, self.trail[k][1], 0.25 + 0.75 * t)
                self.line(surf, c, self.trail[k - 1][0], self.trail[k][0], 1 + 2 * t)
        sx, sy = self.cam.pt(ball)
        r = max(4, int(self.cam.scale(ball, BALL_R * 1.8)))
        b = self.env.B[0]
        pygame.draw.circle(surf, BALL_COL, (sx, sy), r)
        pygame.draw.circle(surf, spin_color(b[3:6], b[6:9]), (sx, sy), r, 2)

    def draw_spin_inset(self, surf):
        """放大的球：画两条正交的“缝线”随旋转转动（画面转速放慢），箭头为旋转轴（右手定则）。"""
        cx, cy, r = VIEW_W - 100, H - 150, 46
        box = pygame.Rect(cx - 90, cy - 70, 180, 175)
        pygame.draw.rect(surf, (10, 12, 18), box, border_radius=10)
        pygame.draw.rect(surf, (60, 66, 82), box, 1, border_radius=10)
        b = self.env.B[0]
        v, w = b[3:6], b[6:9]
        cam = self.cam

        def proj(p):
            return cx + r * float(p @ cam.right), cy - r * float(p @ cam.up), float(p @ cam.fwd)

        pygame.draw.circle(surf, (238, 234, 222), (cx, cy), r)
        ts = np.linspace(0, 2 * math.pi, 49)
        for circ, col in (((np.cos(ts), np.sin(ts), 0 * ts), (230, 120, 40)),
                          ((0 * ts, np.cos(ts), np.sin(ts)), (60, 60, 70))):
            pts = [proj(self.ball_rot @ np.array(p)) for p in zip(*circ)]
            for p0, p1 in zip(pts[:-1], pts[1:]):
                if p0[2] < 0 and p1[2] < 0:
                    pygame.draw.line(surf, col, p0[:2], p1[:2], 3)
        pygame.draw.circle(surf, (120, 125, 135), (cx, cy), r, 1)

        wn = float(np.linalg.norm(w))
        phase = self.env.I[0, sim.PHASE]
        if wn > 5 and phase != sim.PH_SERVE:
            col = spin_color(v, w)
            ax = w / wn
            ln = 0.9 + 0.3 * min(wn / 500.0, 1.0)
            a0, a1 = proj(-ax * ln), proj(ax * ln)
            pygame.draw.line(surf, col, a0[:2], a1[:2], 3)
            d = np.array(a1[:2]) - np.array([cx, cy])
            dn = np.linalg.norm(d) + 1e-9
            u = d / dn
            nrm = np.array([-u[1], u[0]])
            tip = np.array(a1[:2])
            pygame.draw.polygon(surf, col, [tip + u * 8, tip + nrm * 6, tip - nrm * 6])
            top, side = spin_parts(v, w)
            rpm = 60 / (2 * math.pi)
            l1 = f"{'上旋' if top >= 0 else '下旋'} {abs(top) * rpm:4.0f} rpm"
            l2 = f"侧旋 {abs(side) * rpm:4.0f} rpm"
            surf.blit(self.font_s.render(l1, True, TOP_COL if top >= 0 else BACK_COL), (box.x + 12, cy + r + 6))
            surf.blit(self.font_s.render(l2, True, SIDE_COL), (box.x + 12, cy + r + 26))
        else:
            surf.blit(self.font_s.render("无旋转", True, DIM), (box.x + 12, cy + r + 6))
        surf.blit(self.font_s.render(f"球的旋转（慢放 {int(1 / VIS_SPIN)}×）", True, DIM), (box.x + 12, box.y + 4))

    def spin_ball(self, dt):
        w = self.env.B[0, 6:9]
        wn = float(np.linalg.norm(w))
        if wn > 1e-6:
            self.ball_rot = rotation(w / wn, wn * dt * VIS_SPIN) @ self.ball_rot

    def draw_scene(self, surf, ball, bodies, hands):
        surf.fill(BG, pygame.Rect(0, 0, VIEW_W, H))
        self.draw_floor(surf)
        if not (abs(ball[0]) <= HALF_L and abs(ball[1]) <= HALF_W and ball[2] >= TABLE_H):
            self.shadow(surf, (ball[0], ball[1], 0.0), 0.03)
        for j in range(2):
            b = self.body_points(bodies[j], hands[j], j)
            self.shadow(surf, (b["hip"][0], b["hip"][1], 0.0), 0.12)

        table_d = self.cam.depth((0, 0, TABLE_H))
        items = [(table_d, "table", None)]
        for j in range(2):
            items.append((self.cam.depth(self.body_points(bodies[j], hands[j], j)["hip"]), "player", j))
        bd = self.cam.depth(ball)
        if ball[2] >= TABLE_H:
            bd = min(bd, table_d - 1e-3)
        items.append((bd, "ball", None))
        items.append((min(self.cam.depth((0, 0, TABLE_H + 0.07)), table_d - 2e-3), "net", None))
        for _, kind, j in sorted(items, key=lambda t: -t[0]):
            if kind == "table":
                self.draw_table(surf, ball)
                self.draw_cue(surf)
            elif kind == "player":
                self.draw_player(surf, bodies[j], hands[j], self.blends[j], j)
            elif kind == "ball":
                self.draw_ball(surf, ball)
            else:
                self.draw_net(surf)

    def draw_cue(self, surf):
        h = int(round(self.foot[6]))
        if h < 0:
            return
        m = 1.0 if h == 0 else -1.0
        cx, cy = self.cam.pt((m * 0.9, m * self.foot[4 + h], TABLE_H + 0.002))
        r = max(5, int(self.cam.scale((m * 0.9, 0, TABLE_H), 0.12)))
        for k in range(0, 360, 40):
            pygame.draw.arc(surf, COLORS[h], (cx - r, cy - r // 2, 2 * r, r),
                            math.radians(k), math.radians(k + 22), 2)

    def draw_fake(self, surf):
        if self.fake_shown is None or self.env.I[0, sim.PHASE] != sim.PH_DEAD:
            return
        h = self.fake_shown
        t = self.font_l.render(f"{NAMES[h]} 假动作得手！骗到重心", True, COLORS[h])
        bg = t.get_rect(center=(VIEW_W // 2, 170)).inflate(26, 12)
        pygame.draw.rect(surf, (10, 12, 18), bg, border_radius=10)
        surf.blit(t, t.get_rect(center=bg.center))

    def draw_hud(self, surf):
        I = self.env.I[0]
        self.draw_fake(surf)
        box = pygame.Rect(VIEW_W // 2 - 220, 14, 440, 74)
        pygame.draw.rect(surf, (10, 12, 18), box, border_radius=12)
        pygame.draw.rect(surf, (60, 66, 82), box, 1, border_radius=12)
        s0 = self.font_xl.render(str(I[sim.SC0]), True, COLORS[0])
        s1 = self.font_xl.render(str(I[sim.SC1]), True, COLORS[1])
        colon = self.font_xl.render(":", True, TEXT)
        surf.blit(colon, colon.get_rect(center=(box.centerx, box.y + 30)))
        surf.blit(s0, s0.get_rect(midright=(box.centerx - 22, box.y + 30)))
        surf.blit(s1, s1.get_rect(midleft=(box.centerx + 22, box.y + 30)))
        for j in range(2):
            name = self.font.render(NAMES[j], True, COLORS[j])
            if j == 0:
                rect = name.get_rect(midleft=(box.x + 18, box.y + 30))
            else:
                rect = name.get_rect(midright=(box.right - 18, box.y + 30))
            surf.blit(name, rect)
            if I[sim.SERVER] == j:
                cx = rect.right + 10 if j == 0 else rect.left - 10
                pygame.draw.circle(surf, (255, 230, 120), (cx, rect.centery), 5)
        if self.practice:
            g = self.font_s.render("练习模式 · 不计分 · 在右侧“练习”页选择发球后点“发球”", True, (140, 210, 150))
        else:
            g = self.font_s.render(f"局分 {I[sim.G0]} : {I[sim.G1]}   ·   场分 {I[sim.M0]} : {I[sim.M1]}"
                                   f"   ·   {sim.GAME_PTS} 分制 {2 * sim.MATCH_WIN - 1} 局 {sim.MATCH_WIN} 胜",
                                   True, DIM)
        surf.blit(g, g.get_rect(center=(box.centerx, box.y + 60)))
        for j in range(2):
            t = self.font_s.render(self.equip_text(j), True, COLORS[j])
            surf.blit(t, t.get_rect(midtop=(box.centerx, box.bottom + 4 + 20 * j)))

        if I[sim.PHASE] == sim.PH_DEAD and self.msg:
            t = self.font_l.render(self.msg, True, self.msg_color)
            bg = t.get_rect(center=(VIEW_W // 2, 160)).inflate(30, 14)
            pygame.draw.rect(surf, (10, 12, 18), bg, border_radius=10)
            surf.blit(t, t.get_rect(center=bg.center))

        b = self.env.B[0]
        speed = np.linalg.norm(b[3:6]) * 3.6
        rpm = np.linalg.norm(b[6:9]) * 60 / (2 * math.pi)
        if I[sim.PHASE] != sim.PH_SERVE:
            h = int(I[sim.LHIT])
            if I[sim.HITS] == 0:
                toss = self.env.T[0, sim.TOSS]
                kind = (sim.SERVE_NAMES[max(int(I[sim.LSRV]), 0)]
                        + f"（{serve_spin_text(b[3:6], b[6:9])}·{'高抛' if toss > 0.66 else '中抛' if toss > 0.33 else '低抛'}"
                        + ("·回抛" if I[sim.LHID] else "")
                        + f"·对手熟悉度 {self.env.T[0, sim.SFAM] * 100:.0f}%）")
            else:
                kind = tech_text(int(I[sim.LTECH]), int(I[sim.LSIDE]), int(I[sim.LSTK]))
                if I[sim.HITS] == 1:
                    kind = "接发球 · " + kind
            last = f"上一拍：{NAMES[h]} {kind} · 挥拍 {self.env.T[0, sim.LSW]:4.1f} m/s"
            if I[sim.HITS] > 0:
                last += f" · 身体发力 {self.env.T[0, sim.LBODY] * 100:3.0f}%"
            surf.blit(self.font_s.render(last, True, COLORS[h]), (16, H - 74))
        # 左上角：这一分的发球（抛球 · 手法 · 旋转）和接发球技术
        info = []
        if I[sim.PHASE] == sim.PH_SERVE:
            srv = int(I[sim.SERVER])
            k = int(self.env.T[0, sim.SPLAN0 + srv])
            tz = self.env.T[0, sim.TOSS]
            wait = "等待点击发球" if self.practice and not self.serve_go else "准备中"
            info.append((srv, f"发球  {toss_text(tz) if tz >= 0 else wait} · {sim.SERVE_NAMES[k]}"))
        elif self.serve_info is not None:
            info.append((self.serve_info[0], "发球  " + self.serve_info[1]))
            if self.recv_info is not None:
                info.append((self.recv_info[0], "接发  " + self.recv_info[1]))
        for n, (j, s) in enumerate(info):
            t = self.font.render(f"{NAMES[j]} {s}", True, COLORS[j])
            bg = t.get_rect(topleft=(16, 100 + 36 * n)).inflate(18, 10)
            pygame.draw.rect(surf, (10, 12, 18), bg, border_radius=8)
            surf.blit(t, t.get_rect(center=bg.center))
        lines = [
            f"本回合击球 {I[sim.HITS]} 拍    球速 {speed:5.1f} km/h    转速 {rpm:5.0f} rpm",
            f"蓝方：{'RL 模型' if self.modes[0] == 'rl' and self.policy else '脚本 AI'}    "
            f"红方：{'RL 模型' if self.modes[1] == 'rl' and self.policy else '脚本 AI'}    "
            f"倍速 {self.speed:.2f}x{'    [已暂停]' if self.paused else ''}",
        ]
        for k, s in enumerate(lines):
            surf.blit(self.font_s.render(s, True, DIM), (16, H - 52 + k * 22))

    def draw_panel(self, surf):
        x0 = VIEW_W
        pygame.draw.rect(surf, PANEL_BG, (x0, 0, PANEL_W, H))
        surf.blit(self.font_l.render("乒乓教练模拟器", True, TEXT), (x0 + 22, 16))
        ok = self.policy is not None
        surf.blit(self.font_s.render(self.model_msg, True, (140, 210, 150) if ok else (230, 160, 110)),
                  (x0 + 22, 56))
        if self.page == "practice":
            for s, y in self.prac_labels:
                surf.blit(self.font_s.render(s, True, DIM), (x0 + 22, y))
        if self.page == "equip":
            for s, y, c, big in self.equip_labels:
                if big:
                    pygame.draw.rect(surf, c, (x0 + 22, y, 6, 22), border_radius=3)
                surf.blit((self.font if big else self.font_s).render(s, True, TEXT if big else c),
                          (x0 + (36 if big else 22), y))
        for j, y in enumerate(self.block_y if self.page in ("stat", "style") else ()):
            pygame.draw.rect(surf, COLORS[j], (x0 + 22, y, 6, 24), border_radius=3)
            surf.blit(self.font.render(NAMES[j], True, TEXT), (x0 + 36, y))
            fy = y + 38 + ROW * max(STAT_ROWS, sim.N_STYLE) + 2
            fat = float(self.env.F[0, j])
            surf.blit(self.font_s.render(f"疲劳 {fat * 100:4.0f}%", True, DIM), (x0 + 22, fy))
            bar = pygame.Rect(x0 + 110, fy + 6, PANEL_W - 132, 8)
            pygame.draw.rect(surf, (55, 60, 75), bar, border_radius=4)
            pygame.draw.rect(surf, (230, 150, 80), (bar.x, bar.y, int(bar.w * fat), bar.h), border_radius=4)
            # 比赛中的心理状态：压力（局点/赛点 + 连丢分 + 被怪胶打烦）× 心态越低越受影响；
            # 手感：对对手怪胶的适应程度，适应越高读转越准
            T, I = self.env.T[0], self.env.I[0]
            men, tch = self.env.S[0, j, sim.MEN], self.env.S[0, j, sim.TCH]
            p = sim.pressure_py(I, T, j)
            adp = float(T[sim.ADP0 + j])
            eff = sim.MEN_SIG * p * (1.0 - men)
            y2 = fy + 20
            surf.blit(self.font_s.render(f"压力 +{eff * 100:3.0f}%", True,
                                         (225, 140, 120) if eff > 0.08 else DIM), (x0 + 22, y2))
            bar2 = pygame.Rect(x0 + 110, y2 + 6, PANEL_W - 132, 8)
            pygame.draw.rect(surf, (55, 60, 75), bar2, border_radius=4)
            pygame.draw.rect(surf, (225, 110, 110), (bar2.x, bar2.y, int(bar2.w * min(p / 1.5, 1.0)), bar2.h),
                             border_radius=4)
            y3 = y2 + 20
            surf.blit(self.font_s.render(f"适应 {adp * 100:4.0f}%", True, DIM), (x0 + 22, y3))
            bar3 = pygame.Rect(x0 + 110, y3 + 6, PANEL_W - 132, 8)
            pygame.draw.rect(surf, (55, 60, 75), bar3, border_radius=3)
            pygame.draw.rect(surf, (110, 180, 225), (bar3.x, bar3.y, int(bar3.w * adp / sim.ADAPT_MAX), bar3.h),
                             border_radius=3)
        for wdg in self.widgets:
            if self.visible(wdg):
                wdg.draw(surf, self.font_s if isinstance(wdg, Button) else self.font)
        helps = ["拖动转视角 · 滚轮缩放 · 空格暂停 · C 复位 · M 静音",
                 "拍面黑=正手面 红=反手面 · 轨迹颜色=旋转"]
        for k, s in enumerate(helps):
            y = self.help_y + k * 16
            if y + 16 <= H:
                surf.blit(self.font_s.render(s, True, DIM), (x0 + 22, y))

    def glide_back(self, bodies, hands, ball, adt):
        """一分打完模拟里直接把球员放回起始位置；画面上让人从原地走回去，而不是瞬移。
        bodies、hands 原地改成显示位置，返回显示用的球（发球方手里托着的球跟着手走）。"""
        raw = [b.copy() for b in bodies]
        for j in range(2):
            prev = self.glide_prev[j]
            # 模拟有时先把手放到发球位置、下一帧才挪身体，所以手一帧跳得比最快的挥拍还远也算瞬移
            if prev is not None and (float(np.linalg.norm(raw[j][:2] - prev[0][:2])) > 0.35
                                     or float(np.linalg.norm(hands[j] - prev[1])) > 0.6):
                ob = self.glide_disp[j][0] - raw[j]
                oh = self.glide_disp[j][1] - hands[j]
                dur = min(max(float(np.linalg.norm(ob[:2])) / 2.0, 0.5), 1.2)
                self.glide[j] = [ob, oh, 0.0, dur]
            self.glide_prev[j] = (raw[j], hands[j].copy())
            g = self.glide[j]
            if g is not None:
                g[2] += adt
                u = min(g[2] / g[3], 1.0)
                w = 1.0 - u * u * (3.0 - 2.0 * u)
                if w <= 0.0:
                    self.glide[j] = None
                else:
                    bodies[j] += g[0] * w
                    hands[j] += g[1] * w
                    if self.env.I[0, sim.PHASE] == sim.PH_SERVE and int(self.env.I[0, sim.SERVER]) == j:
                        ball = ball + g[1] * w
            self.glide_disp[j] = (bodies[j].copy(), hands[j].copy())
        return ball

    def render(self, dt):
        nsub = self.env.nsub
        s = min(int(self.acc / self.env.dt * nsub), nsub - 1)
        st = self.env.hist[0, s]
        ball = st[0:3].copy()
        bodies = [st[3:6].copy(), st[6:9].copy()]
        hands = [st[9:12].copy(), st[12:15].copy()]
        self.blends = [float(st[15]), float(st[16])]
        self.foot = [float(v) for v in st[17:24]]
        lag = (nsub - 1 - s) * self.env.dt / nsub
        adt = 0.0 if self.paused else min(dt, 0.1) * self.speed
        ball = self.glide_back(bodies, hands, ball, adt)
        home_fn = lambda j, pose, hand: self.body_points(bodies[j], hand, j, pose, raw=True)  # noqa: E731
        ball, hands, self.poses = self.anim.frame(self.env, self.act, ball, hands, self.blends, home_fn, adt, lag)
        # 一分刚开始动画会直接换到发球准备姿势；抛球前球拍限速移过去，不要一帧跳过去
        if self.env.I[0, sim.PHASE] == sim.PH_SERVE:
            srv = int(self.env.I[0, sim.SERVER])
            prev = self.hand_shown[srv]
            if prev is not None and self.anim.serve_phase(self.env, self.act, lag) < 0.25:
                d = np.asarray(hands[srv], dtype=float) - prev
                dn = float(np.linalg.norm(d))
                lim = 3.0 * adt
                if dn > lim:
                    hands[srv] = prev + d * (lim / dn)
        self.hand_shown = [np.asarray(h, dtype=float).copy() for h in hands]
        if self.env.I[0, sim.PHASE] == sim.PH_SERVE:
            self.trail.clear()
        elif not self.paused:
            b = self.env.B[0]
            self.trail.append((tuple(ball), spin_color(b[3:6], b[6:9])))
            self.trail = self.trail[-28:]
            self.spin_ball(min(dt, 0.1) * self.speed)
        self.draw_scene(self.screen, ball, bodies, hands)
        self.draw_hud(self.screen)
        self.draw_spin_inset(self.screen)
        self.draw_panel(self.screen)

    def run(self):
        running = True
        while running:
            dt = self.clock.tick(60) / 1000.0
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif not self.handle(e):
                    running = False
            self.update(dt)
            self.render(dt)
            pygame.display.flip()
        pygame.quit()


if __name__ == "__main__":
    Game().run()
