"""乒乓球物理引擎 + 比赛规则（numba 编译，多环境并行）。

坐标系（米）：x 沿球台长边，球网在 x=0；0 号球员在 x<0 一侧，1 号球员在 x>0 一侧；
y 沿球台宽边；z 竖直向上，地面 z=0。

球的受力：重力、空气阻力（与速度平方成正比）、马格努斯力（旋转 × 速度）。
台面碰撞：法向恢复系数 + 库仑摩擦冲量（摩擦会改变旋转，上旋/下旋落台后表现不同）。
球拍碰撞：同样是法向恢复 + 胶皮摩擦冲量。出球的旋转完全来自拍面角度与挥拍时接触点的切向滑动，
来球自带的旋转也参与这次碰撞，判断错来球旋转就会回高出界或下网。

球员由“身体”（P）和“持拍手”组成：手在身体右侧是正手，身前偏左是反手，正反手切换需要时间。
"""
import math
import os

import numpy as np
from numba import njit, prange

G = 9.81
BALL_R = 0.02
BALL_M = 0.0027
HALF_L = 1.37
HALF_W = 0.7625
TABLE_H = 0.76
NET_H = 0.1525
NET_HALF_W = 0.9144
AIR_RHO = 1.2
KD = 0.5 * AIR_RHO * 0.5 * math.pi * BALL_R ** 2 / BALL_M
KM = 0.5 * AIR_RHO * 0.6 * math.pi * BALL_R ** 3 / BALL_M
# 升力系数随自旋比 S = rω/v 饱和：C_L = 0.6·S / (1 + S/S_SAT)，上限约 0.6·S_SAT。
# 线性的 C_L = 0.6·S 在 100 转/秒以上的弧圈球上会把马格努斯力高估两三倍，弧线下扎/拐弯得离谱
S_SAT = 0.5
E_TABLE = 0.89
MU_TABLE = 0.25
K_INERTIA = 2.0 / 3.0
SPIN_DECAY = 0.05
E_FLOOR = 0.6
E_PAD = 0.86
MU_PAD = 0.9
K_STICK = K_INERTIA / (1.0 + K_INERTIA)
REACH = 0.14
RALLY_TIMEOUT = 20.0
# 比赛模式只防卡死：精彩对拉不应被截断（回合时长不在观察里，训练与比赛用不同上限不影响策略）
RALLY_TIMEOUT_PLAY = 90.0
HAND_MID = -0.13
HAND_HALF = -0.19
FH_LO, FH_HI = -0.68, -0.08
BH_LO, BH_HI = -0.12, 0.45
FH_CLO, FH_CHI = -0.48, -0.18
BH_CLO, BH_CHI = -0.05, 0.28
FWD_LO, FWD_HI = -0.2, 1.1
# 落点目标最近离网 DEPTH_MIN；打得越短落点误差越大（死球难以刻意放出来）
DEPTH_MIN = 0.3
# 网前高球不用压低身体，手臂伸直、上身压到台面上方能探得更远
HIGH_REACH = 0.35
# 向台内探身够短球（跨步 + 前倾）对台内小动作影响较小，按一半计入“够球”变形
FWD_STRETCH_W = 0.5
FWD_CLO, FWD_CHI = -0.15, 0.3
BODY_MIN = HALF_L + 0.1
READY_W = 0.003
# 接发球的等候站位：发球出手前只能站在离端线 RECV_X0–RECV_X1、自己视角横向 RECV_Y ± RECV_DY（正为反手侧）
# 的范围里（和发球方一样，等候时站位是固定动作）；出手后读完发球再前后左右移动不受限制
RECV_X0, RECV_X1, RECV_Y, RECV_DY = 0.3, 0.7, 0.1, 0.25
WINNER_BONUS = 1.0
# 每把球打回对方台面的奖励：太小时大网络学会冒险抢攻，回合很短（多数分三板内结束）
RALLY_W = 0.08
LEAN_MAX = 0.12
# 假动作判定：落点偏离引拍朝向（米）、对手重心倾斜占上限的比例、球离对手身体的横向距离（米）
FAKE_DEV, FAKE_LEAN, FAKE_NEED = 0.3, 0.7, 0.3
CROSS_V = 1.5
CROSS_ACC = 1.15
CUE_RATE = 2.0
LAT_MAX = 0.7
SETTLE_T = 0.25
SPLIT_ACC = 1.8
SPLIT_T = 0.3
HAND_FORCE = 0.5
STEPIN_D, STEPIN_V, STEPIN_ACC = 0.8, 1.4, 1.6
STYLE_W = 0.04

STR, AGI, TEC, REA, STA = 0, 1, 2, 3, 4
N_STATS = 5
STAT_NAMES = ["力量", "敏捷", "技术", "反应", "体力"]
STY_BH, STY_SPIN, STY_POW, STY_WIDE = 5, 6, 7, 8
N_STYLE = 4
# 发球：与技术分开。决定发球旋转上限、控制精度、动作隐蔽程度；接发球读旋转用“反应”。
SRV = 9
# 器材与持拍：HAND 右手 +1 / 左手 -1；GRIP 横板 0 / 直板 1；RFH、RBH 正反手胶皮（RUB_* 编号）
HAND, GRIP, RFH, RBH = 10, 11, 12, 13
# 心态：关键分、连丢分、被怪胶打烦时动作会不会变形；手感：读怪球、吃转、台内小球、适应对手胶皮的快慢。
# 不在 N_STATS 里（观测布局不变，现有模型看不见这两项，只是被动受影响）
MEN, TCH = 14, 15
N_S = N_STATS + N_STYLE + 1 + 4 + 2
STAT_NAMES_ALL = STAT_NAMES + ["发球", "心态", "手感"]
GRIP_NAMES = ["横板", "直板"]
# 胶皮：(恢复系数, 摩擦系数, 咬球系数倍数)。咬球系数决定拍面切向能“带住”球多少，
# 越小越难主动制造旋转、也越不吃来球旋转；摩擦系数限制一板能改变多少旋转。
#   反胶：颗粒朝内，摩擦强，旋转强，最吃转
#   正胶：颗粒朝外，出球快，旋转一般，不太吃转
#   生胶：颗粒朝外、更硬更弹，出球最快最平，击球“沉”，几乎不吃转
#   长胶：颗粒细长易倒，自己几乎造不出旋转，来球旋转大多保留——对方拉上旋，回过去变下旋
#   防弧：表面不黏、海绵低弹，出球慢，抵消来球的冲力和旋转
RUB_INV, RUB_PIPS, RUB_RAW, RUB_LONG, RUB_ANTI = range(5)
RUB_NAMES = ["反胶", "正胶", "生胶", "长胶", "防弧"]
# 用这种胶皮回球后，对手读旋转误差的倍数（长胶倒转、防弧消转，旋转难判断）；实际倍数还按接球方手感、适应程度缩放
RUB_DEC = np.array([float(v) for v in os.environ.get("PP_RUB_DEC", "1,1.3,1.3,3.5,2.8").split(",")])
# 吃转程度：来球旋转、重球带来的失误按胶皮缩放（反胶最吃转，颗粒、长胶、防弧不太吃转）
RUB_SENS = np.array([float(v) for v in os.environ.get("PP_RUB_SENS", "1,0.7,0.55,0.35,0.3").split(",")])
# 胶皮“怪”的程度：决定对手适应它的必要、被它打烦的程度
RUB_ODD = np.array([0.0, 0.3, 0.3, 1.0, 0.8])
# 心态：压力（关键分 + 连丢分 + 被怪胶打烦）× (1 - 心态) 放大击球误差
MEN_SIG = float(os.environ.get("PP_MEN_SIG", "1.2"))
CLUTCH_GP, CLUTCH_MP, TILT_PT, TILT_MAX = 0.5, 1.0, 0.2, 0.8
FRUS_ADD, FRUS_DECAY = 0.35, 0.8
# 手感：读怪球的误差倍数、吃转倍数、台内小球误差倍数都在手感 0.5 时为 1；适应对手怪胶的速度
TCH_READ = float(os.environ.get("PP_TCH_READ", "1.6"))
TCH_SENS, TCH_SMALL = 0.6, 0.4
ADAPT0, ADAPT1, ADAPT_MAX = 0.02, 0.06, 0.7
# 咬球系数 >1：胶皮切向变形在离拍时回弹，球的转速能超过“纯滚动”（切向恢复，Cross 2005），
# 反胶最明显——职业弧圈 100–150 转/秒就靠它
RUB_P = np.array([
    [0.86, 1.20, 1.45],
    [0.88, 0.55, 0.80],
    [0.90, 0.40, 0.55],
    [0.72, 0.14, 0.15],
    [0.70, 0.20, 0.25],
])
# 训练时随机抽器材的概率：正手绝大多数是反胶，反手颗粒、长胶、防弧多一些；左手、直板各约四分之一
N_RUB = len(RUB_NAMES)
BAL_FRAC = 0.5  # 训练时均匀抽器材的比例，其余按真实比例（常见的两面反胶也要练够）
RUB_FH_P = np.array([0.82, 0.10, 0.06, 0.01, 0.01])
RUB_BH_P = np.array([0.60, 0.10, 0.10, 0.13, 0.07])
LEFTY_P, PEN_P = 0.25, 0.25
# 直板：手指手腕灵活，发球、台内小球、正手更细腻；反手（直拍横打）挥拍和覆盖面不如横板，
# 传统推挡（反手挡、搓）用正面胶皮，稳定
PEN_FH_SHORT, PEN_SERVE, PEN_DEC = 0.85, 0.88, 1.1
PEN_BH_SWING, PEN_BH_SIG, PEN_BH_SPIN, PEN_PUSHBLOCK_SIG = 0.88, 1.12, 0.9, 0.92
# 界面展示顺序（S 里的下标，存储顺序不变以兼容观察布局）
STAT_ORDER = (STR, AGI, TEC, SRV, REA, STA, MEN, TCH)
STAT_ORDER_NAMES = ("力量", "敏捷", "技术", "发球", "反应", "体力", "心态", "手感")

# 赛制：GAME_PTS 分一局（GAME_PTS-1 平后需领先 2 分），先赢 MATCH_WIN 局者胜；比赛结束疲劳清零
GAME_PTS = 5
MATCH_WIN = 2
# 疲劳：跑动按 (速度/极速)^2 累积，每拍再按力度扣一笔；系数都乘 (FAT_BASE - 体力)
FAT_BASE = 1.25
FAT_RUN = 0.012
FAT_HIT = 0.0032
FAT_REC_PT = 0.005
FAT_REC_GAME = 0.08
STYLE_NAMES = ["反手偏好", "旋转偏好", "力量偏好", "大角度偏好"]

ST_LOOP, ST_DRIVE, ST_BLOCK, ST_PUSH, ST_FLICK, ST_CHOP, ST_SMASH, ST_LOB = range(8)
STROKE_NAMES = ["拉球", "快攻", "挡球", "搓球", "挑打", "削球", "扣杀", "放高球"]
# 技术细分（只用于显示与统计，物理仍按上面 8 类）：在大类里再按正反手、来球旋转、落点长短、力度、侧旋意图区分
(TQ_LOOP, TQ_LIFT, TQ_DRIVELOOP, TQ_RIP, TQ_DRIVE, TQ_BLOCK, TQ_SIDECUT, TQ_SHORT, TQ_LONG, TQ_PUSH,
 TQ_BANANA, TQ_TWIST, TQ_WIPE, TQ_FLICK, TQ_CHOP, TQ_SMASH, TQ_LOB) = range(17)
TECH_NAMES = ["拉球", "提拉", "前冲弧圈", "撕", "快攻", "挡球", "侧切", "摆短", "劈长", "搓球",
              "霸王拧", "拧拉", "抹", "挑打", "削球", "扣杀", "放高球"]
# 显示时不加“正手/反手”前缀的（本身就是反手技术）
TECH_NOSIDE = (TQ_RIP, TQ_SIDECUT, TQ_BANANA, TQ_TWIST)
# 每种技术：挥拍上限、可用旋转倾角、执行误差、对重球的敏感度、水平球速上限、过网余量、
# 击球前站稳的重要程度、旋转量、击球后还原时间（秒）、还原期间正反手切换时间倍数、
# 挥拍速度中来自身体（蹬地、转髋、躯干、重心转移）的占比
STROKE_P = np.array([
    [1.00, 1.00, 1.00, 1.00, 99.0, 0.00, 1.0, 1.0, 0.30, 1.3, 0.50],
    [0.85, 0.60, 0.90, 0.70, 99.0, 0.00, 0.6, 0.6, 0.18, 1.0, 0.30],
    [0.45, 0.40, 0.70, 0.35, 99.0, 0.00, 0.2, 0.4, 0.08, 0.7, 0.10],
    [0.40, 1.00, 0.45, 0.80, 6.0, 0.00, 0.3, 0.8, 0.12, 0.8, 0.10],
    [0.60, 0.80, 1.20, 1.00, 99.0, 0.00, 0.4, 0.8, 0.15, 0.9, 0.20],
    [0.70, 1.00, 0.85, 0.50, 9.0, 0.03, 0.8, 1.0, 0.30, 1.2, 0.35],
    [1.25, 0.40, 1.10, 1.00, 99.0, 0.00, 1.0, 0.3, 0.35, 1.4, 0.50],
    [0.80, 0.80, 0.80, 0.60, 7.0, 0.60, 0.6, 0.8, 0.30, 1.2, 0.40],
])
# 想要的最大转速（rad/s）= SPIN0 + SPIN1·技术；实际出多少还受拍面倾角、挥拍速度和胶皮限制。
# 实测对照：职业强力弧圈 100–130 转/秒（顶尖 ~150），顶尖下旋发球 60–75 转/秒
STK_SPIN0, STK_SPIN1 = 150.0, 750.0
SRV_SPIN0, SRV_SPIN1 = 150.0, 550.0
# 来球旋转对失误的影响以此为单位（rad/s）
SPIN_REF = 150.0
REC_ACC = 0.3
# 倒地爆冲：正手大力进攻、来球远在身体外侧时整个人扑出去打。正手侧多够 DIVE_EXT，够球不再削弱挥拍；
# 代价是打完倒在地上 DIVE_T 秒几乎动不了、这期间来球接不到，之后还要 DIVE_REC 秒恢复——完全放弃下一板
DIVE_PW, DIVE_ST, DIVE_EXT = float(os.environ.get("PP_DIVE_PW", "0.6")), 0.12, 0.3
DIVE_T, DIVE_REC, DIVE_ACC = 0.9, 0.4, 0.05
BH_BODY = 0.6
# 每种技术完整引拍需要的准备时间（从判断出来球到击球）：不够时挥拍受限、失误增加，快球因此有威胁，挡球最不怕快
PREP_T = np.array([0.38, 0.22, 0.18, 0.20, 0.25, 0.35, 0.40, 0.30])
RUN_SIG = 0.8
# 来不及时本能地打保险球：弧线抬高、球速降低（给对手送出机会球），而不是大多直接失误
LACK_MARGIN = 0.12
LACK_SIG = 0.7
SMASH_V = 1.25
ERR_SCALE = float(os.environ.get("PP_ERR_SCALE", "0.6"))
# 读来球旋转的误差倍数（接发球、相持中判断旋转不准）
READ_SCALE = float(os.environ.get("PP_READ_SCALE", "0.6"))
PLAN_BISECT = 8
EDGE_W = 0.6 * BALL_R
SERVE_WAIT = 1.8
# 抛球高度 TOSS_H0 + TOSS_H1·toss（米）；从抛起到击球这段时间发球者脚下不动
TOSS_H0, TOSS_H1 = 0.3, 1.5
# 急球、奔球是低抛发球
FAST_TOSS_MAX = 0.25
PROBE_DT = 1.0 / 120.0
# 借力：拍面碰撞本身会把来球速度弹回去，想打得比 BORROW×来球水平速度 更慢就得主动卸力（技术好卸得多，但误差变大）
#                 拉    快攻  挡   搓   挑打  削   扣杀  高球
BORROW = np.array([0.25, 0.45, 0.7, 0.2, 0.3, 0.15, 0.0, 0.2])
ABSORB_SIG = 1.0
# 用进攻类技术处理下旋来球（拉/冲/挑/挡下旋容易下网）的误差系数；搓、削本来就是对付下旋的
#                  拉   快攻  挡   搓   挑打  削   扣杀  高球
VS_BACK = np.array([0.7, 1.2, 6.0, 0.0, 1.2, 0.0, 0.8, 0.3])
# 用被动/轻挥拍的技术处理强上旋来球容易“吃转”挡飞；拉球顶着上旋拉回去影响小
#                 拉   快攻  挡   搓   挑打  削   扣杀  高球
VS_TOP = np.array([0.2, 0.4, 1.0, 1.0, 0.6, 0.3, 0.3, 0.2])
# 挡球靠借力：来球太慢（水平速度低于 NOPACE_V）借不上力，挡回去弧线高、偏飘，送出机会球
NOPACE_V = 9.0
NOPACE_MARGIN = 0.12
# 搓球冒高：来球下旋越转、技术越差、连续搓的拍数越多，越容易把球搓高（过网余量随机抬高，变成机会球）
POP_MARGIN = 0.6
POP_CHAIN = 0.5
POP_CAP = 0.45
# 高球好打：击球点高出台面 HI_Z0 起，进攻技术处理下旋的难度和执行误差随高度下降
HI_Z0, HI_Z1 = 0.12, 0.35
HI_BACK = 0.7
HI_SIG = 0.35
SMASH_Z = 0.30

(SV_PEND, SV_HOOK, SV_SHOVEL, SV_SQUAT, SV_CHOP, SV_FAST, SV_BH, SV_BHFAST) = range(8)
N_SERVE = 8
# 名称只描述“动作”，不描述旋转：同样是勾手，可以发侧上、侧下、不转；逆旋转也一样。
SERVE_NAMES = ["正手侧旋发球", "勾手发球", "逆旋转发球", "下蹲砍式发球", "砍式发球", "正手奔球",
               "反手侧旋发球", "反手急球"]
# 每种发球手法（右手持拍）：旋转上限倍数、侧旋基准方向（俯视 -1 顺时针=顺旋 / +1 逆时针=逆旋 / 0 随意）、
# 侧旋占比、执行误差倍数、迷惑性（动作本身能藏住多少旋转差异）、发完后的还原时间（秒）、球速倍数、
# 可发的上旋范围（-1 纯下旋 … +1 纯上旋）
SERVE_P = np.array([
    [1.00, -1.0, 0.80, 1.00, 1.00, 0.25, 1.0, -1.0, 1.0],   # 正手侧旋：左侧上/下旋，最通用
    # 勾手：右侧旋为主；靠重心转换、少用手腕，绝对旋转比逆旋转弱，但弧线稳、上下旋动作一致，最隐蔽
    [0.90, 1.0, 1.00, 0.95, 1.35, 0.30, 1.0, -1.0, 1.0],
    # 逆旋转：右侧旋为主；大幅甩腕，旋转最强的下手发球，弧线难控制，动作幅度大反而好认
    [1.05, 1.0, 1.00, 1.30, 1.15, 0.30, 1.0, -1.0, 1.0],
    # 下蹲砍式：上手类，半圆摩擦，左/右侧旋都能发，侧旋最强；蹲下再站起还原最慢
    [1.25, 0.0, 1.20, 1.30, 1.25, 0.50, 1.0, -1.0, 0.8],
    [1.00, 0.0, 0.25, 1.00, 1.10, 0.22, 1.15, -1.0, 0.25],  # 砍式：以下旋为主，速度快、侧旋少
    # 正手奔球：低抛，击球点近网高，第一跳压端线，右侧上旋，又急又长
    [0.70, 1.0, 0.50, 1.00, 0.80, 0.20, 1.5, 0.3, 1.0],
    [0.80, 1.0, 0.90, 0.90, 0.90, 0.10, 0.9, -1.0, 1.0],    # 反手侧旋：右侧上/下旋
    [0.60, 0.0, 0.20, 1.00, 0.85, 0.10, 1.35, -1.0, 1.0],   # 反手急球：以速度为主，可带急下旋
])
# 发球击球点离台面高度（米）：急球、奔球压得最低（与网同高左右），才能第一跳压端线、又平又快；
# 下蹲砍式击球点稍高
SERVE_Z = np.array([0.18, 0.18, 0.18, 0.20, 0.18, 0.12, 0.17, 0.12])
# 急球/奔球：最浅的落点意图与第一跳离网距离
FAST_DEPTH, FAST_D1 = 1.1, 1.2
FAULT_P = 0.05
# 发球战术脑：先验得分率（强制单一手法实测）、先验权重（相当于几分的经验）、近期加权衰减、
# 对手熟悉度的扣分、softmax 温度
USE_TACTIC = True
SERVE_PRIOR = np.array([0.75, 0.75, 0.75, 0.74, 0.62, 0.60, 0.55, 0.55])
PRIOR_N = 4.0
TAC_DECAY = 0.92
TAC_FAM = 0.25
TAC_TEMP = 0.06
# 旋转变化（下旋 / 不转 / 上旋）：对应的上旋意图、先验得分率；同一种变化用得越多越容易被预判
VAR_TOP = np.array([-1.0, 0.0, 1.0])
VAR_PRIOR = np.array([0.6, 0.6, 0.6])
TAC_VREP = 0.2

(TURN, BOUNCED, NETT, HITS, SERVER, PHASE, SC0, SC1, G0, G1, FIRST, LWIN, LREASON, PVALID,
 LHIT, LSIDE, LFAKE, LSTK, LSRV, LHID, SVB, M0, M1, LTECH) = range(24)
NI = 24
(RT, TIMER, EVT, PRT, PX, PY, PZ, SH0, SH1, BL0, BL1, BT0, BT1, LSW,
 LN0, LN1, CX0, CX1, CD0, CD1, CU0, CU1, LFT, SET0, SET1, STB, REC0, REC1, LK0, LK1, DEC, TOSS, LBODY,
 SPL0, SPL1, LPREP, EX, EY, EZ, ETH, LX, LY, LZ, LTH, SFAM, FAM0) = range(46)
DBG0 = FAM0 + 2 * N_SERVE
SW0 = DBG0 + 9
SN0 = SW0 + 2 * N_SERVE
SPLAN0 = SN0 + 2 * N_SERVE
N_VAR = 3
VW0 = SPLAN0 + 2
VN0 = VW0 + 2 * N_VAR
VPLAN0 = VN0 + 2 * N_VAR
PCHAIN = VPLAN0 + 2
# 每名球员最近一次触球的拍面法向（世界坐标，只给画面用）
PN0 = PCHAIN + 1
# 最近一板用的胶皮（只给画面用）
LRUB = PN0 + 6
# 每名球员世界 y 与自己视角横向的换算系数（含左右手镜像），每分开始时由 S 写入
LM0 = PN0 + 7
# 倒地爆冲后还要在地上待多久（秒）
DV0 = PN0 + 9
# 心态/手感的比赛内状态（每名球员）：连丢分数、被怪胶打烦程度、对对手怪胶的适应程度、自己正要接的这板是什么胶皮打来的
STK0 = PN0 + 11
FRU0 = PN0 + 13
ADP0 = PN0 + 15
ORB0 = PN0 + 17
NT = PN0 + 19
# 发球适应：接发球方对每种发球的熟悉度（指数记忆）。熟悉的发球判断更准、能提前预判；陌生的发球迷惑性更强
FAM_RATE = 0.25
FAM_DEC_NEW, FAM_DEC_OLD = 1.6, 0.3
FAM_READ, FAM_SURPRISE = 0.3, 0.05
SERVE_PICK_SIG = 0.3
# 下降期击球点：球落到台面以下 LATE_ZMIN，或飞到端线后 LATE_XMAX 处为止
LATE_ZMIN = TABLE_H - 0.1
LATE_XMAX = HALF_L + 1.8
CONTACT_FWD = 0.1
LATE_SHORT_Z = 0.08

PH_SERVE, PH_FLY, PH_DEAD = 0, 1, 2
R_NONE, R_OUT, R_NET, R_OWN, R_MISS, R_TIMEOUT, R_FAULT, R_LET, R_SFAULT = 0, 1, 2, 3, 4, 5, 6, 7, 8
REASON_TEXT = ["", "出界", "下网", "落在己方台面", "没接到", "超时", "发球违例（回抛）", "擦网重发",
               "发球违例（没先落本方台面）"]

OBS_DIM = 108
ACT_DIM = 12
HIST_DIM = 24


@njit(cache=True)
def seed_rng(s):
    np.random.seed(s)


@njit(cache=True)
def ball_substep(b, dt):
    """推进球一个物理子步。返回事件：0 无，1 碰台，2 碰网，3 落地。"""
    x, y, z = b[0], b[1], b[2]
    vx, vy, vz = b[3], b[4], b[5]
    wx, wy, wz = b[6], b[7], b[8]
    sp = math.sqrt(vx * vx + vy * vy + vz * vz)
    km = KM / (1.0 + BALL_R * math.sqrt(wx * wx + wy * wy + wz * wz) / (max(sp, 0.5) * S_SAT))
    ax = -KD * sp * vx + km * (wy * vz - wz * vy)
    ay = -KD * sp * vy + km * (wz * vx - wx * vz)
    az = -KD * sp * vz + km * (wx * vy - wy * vx) - G
    vx += ax * dt
    vy += ay * dt
    vz += az * dt
    nx = x + vx * dt
    ny = y + vy * dt
    nz = z + vz * dt
    decay = 1.0 - SPIN_DECAY * dt
    wx *= decay
    wy *= decay
    wz *= decay
    ev = 0

    if (x < 0.0) != (nx < 0.0):
        f = x / (x - nx)
        zc = z + f * (nz - z)
        yc = y + f * (ny - y)
        if zc < TABLE_H + NET_H + BALL_R and zc > TABLE_H - BALL_R and abs(yc) < NET_HALF_W + BALL_R:
            cord = (zc - (TABLE_H + NET_H - BALL_R)) / (2.0 * BALL_R)
            if cord > 0.0 and np.random.random() < cord ** 0.7:
                # 擦网：蹭着网顶翻过去，速度大减、向上弹起、旋转被削弱
                vx *= 0.25 + 0.35 * np.random.random()
                vy = vy * 0.6 + np.random.standard_normal() * 0.25
                vz = abs(vz) * 0.3 + 0.3 + 1.2 * np.random.random() * cord
                nz = max(nz, TABLE_H + NET_H + BALL_R)
                wx *= 0.5
                wy *= 0.5
                wz *= 0.5
            else:
                s = -1.0 if x < 0.0 else 1.0
                nx = s * BALL_R * 1.05
                vx = -0.12 * vx
                vy *= 0.5
                vz *= 0.5
                wx *= 0.3
                wy *= 0.3
                wz *= 0.3
            ev = 2

    if (abs(nx) <= HALF_L and abs(ny) <= HALF_W and nz - BALL_R < TABLE_H
            and vz < 0.0 and z - BALL_R >= TABLE_H - 0.01):
        vin = -vz
        vz = E_TABLE * vin
        ux = vx - BALL_R * wy
        uy = vy + BALL_R * wx
        un = math.sqrt(ux * ux + uy * uy)
        if un > 1e-9:
            jt = min(0.4 * un, MU_TABLE * (1.0 + E_TABLE) * vin)
            hx = ux / un
            hy = uy / un
            vx -= jt * hx
            vy -= jt * hy
            c = jt / (K_INERTIA * BALL_R)
            wx -= c * hy
            wy += c * hx
        nz = TABLE_H + BALL_R
        ev = 1
    elif (nz - BALL_R < TABLE_H and vz < 0.0 and z - BALL_R >= TABLE_H - 0.01
          and abs(nx) <= HALF_L + EDGE_W and abs(ny) <= HALF_W + EDGE_W):
        # 擦边：球心在台外、球体蹭到台沿，算有效落台，但反弹又低又平
        vz = E_TABLE * (-vz) * 0.35 * np.random.random()
        vx *= 0.95
        vy *= 0.95
        nz = TABLE_H + BALL_R
        ev = 1
    elif nz < BALL_R and vz < 0.0:
        vz = -E_FLOOR * vz
        nz = BALL_R
        vx *= 0.85
        vy *= 0.85
        ev = 3

    b[0], b[1], b[2] = nx, ny, nz
    b[3], b[4], b[5] = vx, vy, vz
    b[6], b[7], b[8] = wx, wy, wz
    return ev


@njit(cache=True)
def aim_velocity(px, py, pz, tx, ty, tz, vh, top, side):
    """球员“脑中”的弹道估算：给定水平速度，求打到目标点所需的出球速度（近似考虑阻力与旋转）。
    上旋/下旋改变下坠，侧旋让球横向拐弯，瞄准时把拐弯量反向预留出来。"""
    dx = tx - px
    dy = ty - py
    d = math.sqrt(dx * dx + dy * dy) + 1e-6
    t = math.expm1(KD * d) / (KD * vh)
    km = KM / (1.0 + BALL_R * math.sqrt(top * top + side * side) / (max(vh, 0.5) * S_SAT))
    ge = min(max(G + km * top * vh, 2.0), 60.0)
    vz = (tz - pz + 0.5 * ge * t * t) / t
    shift = 0.35 * km * side * vh * t * t
    ax = tx + shift * dy / d
    ay = ty - shift * dx / d
    ex = ax - px
    ey = ay - py
    e = math.sqrt(ex * ex + ey * ey) + 1e-6
    return ex / e * vh, ey / e * vh, vz


@njit(cache=True)
def plan_shot(px, py, pz, tx, ty, vh, top, margin, side=0.0):
    """球员“脑中”的出球规划：按（自己感知的）旋转在脑中把弹道过一遍，二分出球仰角让落点对准目标；
    速度太快压不到台内、或弧线过不了网时自动降速抬弧线。"""
    vx, vy, vz = aim_velocity(px, py, pz, tx, ty, TABLE_H + BALL_R, vh, top, side)
    if (px < 0.0) == (tx < 0.0):
        return vx, vy, vz
    s = 1.0 if tx > 0.0 else -1.0
    want = tx * s
    ay = ty
    for _ in range(10):
        vz0 = aim_velocity(px, py, pz, tx, ay, TABLE_H + BALL_R, vh, top, side)[2]
        lz = 0.0
        clr = -1.0
        for lat in range(2):
            dx = tx - px
            dy = ay - py
            d = math.sqrt(dx * dx + dy * dy) + 1e-9
            vx = dx / d * vh
            vy = dy / d * vh
            lo = vz0 - 2.0
            hi = vz0 + 3.0
            for _b in range(PLAN_BISECT):
                mid = 0.5 * (lo + hi)
                l, _y, _c = fly_probe(px, py, pz, vx, vy, mid, top, side, s)
                if l < want:
                    lo = mid
                else:
                    hi = mid
            vz = hi
            lz, yl, clr = fly_probe(px, py, pz, vx, vy, vz, top, side, s)
            if lz > 0.0 and lz < 50.0:
                ay = min(max(ay + ty - yl, -HALF_W - 0.6), HALF_W + 0.6)
        if clr >= margin and abs(lz - want) < 0.15:
            return vx, vy, vz
        if vh < 2.2:
            return vx, vy, vz
        vh *= 0.88
    return vx, vy, vz


@njit(cache=True)
def first_bounce(px, py, pz, vx, vy, vz, top, side, s):
    """发球第一跳：返回落在本方台面时离网的距离（没落在本方台面返回 -1）。"""
    b = np.empty(9)
    wx, wy, wz = spin_vector(vx, vy, top, side)
    b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8] = px, py, pz, vx, vy, vz, wx, wy, wz
    for _ in range(200):
        ev = ball_substep(b, PROBE_DT / 2.0)
        if ev == 1:
            return -b[0] * s if b[0] * s < 0.0 else -1.0
        if ev == 2 or b[0] * s > 0.0:
            return -1.0
        if ev != 0 or b[2] < TABLE_H - 0.3:
            return 99.0
    return -1.0


@njit(cache=True)
def plan_serve_shot(px, py, pz, tx, ty, d1, top, side, margin):
    """发球的出球规划：先压到本方台面离网 d1 处弹起，再过网落到目标。
    给定出球水平速度，二分出向下的角度使第一跳落在 d1；再二分水平速度使第二跳落在目标深度。"""
    s = 1.0 if tx > 0.0 else -1.0
    want = tx * s
    ay = ty
    vx, vy, vz = 0.0, 0.0, 0.0
    for _ in range(4):
        for lat in range(2):
            dx = tx - px
            dy = ay - py
            d = math.sqrt(dx * dx + dy * dy) + 1e-9
            ux, uy = dx / d, dy / d
            lo_v, hi_v = 1.0, 16.0
            lz, yl, clr = -1.0, 0.0, -1.0
            for _a in range(9):
                vh = 0.5 * (lo_v + hi_v)
                lo_z, hi_z = -9.0, 2.0
                for _b in range(9):
                    mz = 0.5 * (lo_z + hi_z)
                    u = first_bounce(px, py, pz, ux * vh, uy * vh, mz, top, side, s)
                    if u > d1:
                        lo_z = mz
                    else:
                        hi_z = mz
                vx, vy, vz = ux * vh, uy * vh, 0.5 * (lo_z + hi_z)
                lz, yl, clr = fly_probe(px, py, pz, vx, vy, vz, top, side, s, 1)
                if lz < want:
                    lo_v = vh
                else:
                    hi_v = vh
            if lz > 0.0 and lz < 50.0:
                ay = min(max(ay + ty - yl, -HALF_W - 0.4), HALF_W + 0.4)
        if clr >= margin:
            break
        d1 = min(d1 + 0.15, HALF_L - 0.05)
    return vx, vy, vz


@njit(cache=True)
def fly_probe(px, py, pz, vx, vy, vz, top, side, s, own=0):
    """脑中弹道：返回 (过网后沿出球方向的落点 x·s, 落点 y, 过网高度余量)。
    碰网或落在自己一侧时落点记为 -1（太短），一直飞不下来记为 99。own=1 时允许先在本方台面弹一次（发球）。"""
    b = np.empty(9)
    wx, wy, wz = spin_vector(vx, vy, top, side)
    b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8] = px, py, pz, vx, vy, vz, wx, wy, wz
    clr = -1.0
    for _ in range(300):
        x0 = b[0]
        z0 = b[2]
        ev = ball_substep(b, PROBE_DT)
        if ev == 2:
            return -1.0, 0.0, -1.0
        if (x0 < 0.0) != (b[0] < 0.0):
            clr = min(z0, b[2]) - (TABLE_H + NET_H + BALL_R)
        if ev == 1:
            if b[0] * s > 0.0:
                return b[0] * s, b[1], clr
            if own > 0:
                own -= 1
                continue
            return -1.0, 0.0, clr
        if b[2] < TABLE_H + BALL_R and b[5] < 0.0:
            if b[0] * s > 0.0:
                return b[0] * s, b[1], clr
            return -1.0, 0.0, clr
    return 99.0, 0.0, clr


@njit(cache=True)
def spin_vector(vx, vy, top, side):
    n = math.sqrt(vx * vx + vy * vy) + 1e-9
    return top * (-vy / n), top * (vx / n), side


@njit(cache=True)
def dot3(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@njit(cache=True)
def cross3(a, b):
    r = np.empty(3)
    r[0] = a[1] * b[2] - a[2] * b[1]
    r[1] = a[2] * b[0] - a[0] * b[2]
    r[2] = a[0] * b[1] - a[1] * b[0]
    return r


@njit(cache=True)
def norm3(a):
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


@njit(cache=True)
def hand_off(blend):
    """持拍手相对身体的放松位置（球员自己视角，正手 blend=+1 在右侧，反手 blend=-1 在身前偏左）。"""
    return HAND_MID + HAND_HALF * blend


@njit(cache=True)
def hand_range(blend):
    """当前持拍状态下手能伸到的横向范围（自己视角，右为负）：正手只够身体右侧，反手只够左侧到身前。"""
    t = (blend + 1.0) * 0.5
    return BH_LO + (FH_LO - BH_LO) * t, BH_HI + (FH_HI - BH_HI) * t


@njit(cache=True)
def hs_of(S, i, j):
    """持拍手方向：右手 +1，左手 -1。球员自己视角里的“左右”都要乘它，左手球员就是右手的镜像。"""
    return -1.0 if S[i, j, HAND] < 0.0 else 1.0


@njit(cache=True)
def lat_m(S, i, j):
    """世界 y 与球员 j 自己视角横向坐标之间的换算系数（含左右手镜像）。"""
    return (1.0 if j == 0 else -1.0) * hs_of(S, i, j)


@njit(cache=True)
def face_rubber(S, i, j, bh, k):
    """这一板用哪面胶皮：正手用正面；横板反手用反面；直板的反手挡、搓（推挡）也用正面，
    只有直拍横打（反手拉、拨、拧等）用反面。k 为技术大类，发球传 -1。"""
    if not bh:
        return int(S[i, j, RFH])
    if S[i, j, GRIP] > 0.5 and (k == ST_BLOCK or k == ST_PUSH or k == -1):
        return int(S[i, j, RFH])
    return int(S[i, j, RBH])


@njit(cache=True)
def pick_index(p):
    r = np.random.random()
    for k in range(p.shape[0]):
        r -= p[k]
        if r <= 0.0:
            return k
    return p.shape[0] - 1


@njit(cache=True)
def random_equipment(S, i, j, balanced=False):
    """balanced：训练用，左右手、横直板、正反手各种胶皮完全均匀地抽，冷门组合（两面长胶等）也练得到；
    否则按真实比赛里的比例。"""
    if balanced:
        S[i, j, HAND] = -1.0 if np.random.random() < 0.5 else 1.0
        S[i, j, GRIP] = 1.0 if np.random.random() < 0.5 else 0.0
        S[i, j, RFH] = float(np.random.randint(0, N_RUB))
        S[i, j, RBH] = float(np.random.randint(0, N_RUB))
        return
    S[i, j, HAND] = -1.0 if np.random.random() < LEFTY_P else 1.0
    S[i, j, GRIP] = 1.0 if np.random.random() < PEN_P else 0.0
    S[i, j, RFH] = float(pick_index(RUB_FH_P))
    S[i, j, RBH] = float(pick_index(RUB_BH_P))


@njit(cache=True)
def racket_impact(v, w, n, vp, e=E_PAD, mu=MU_PAD, ks=1.0):
    """球与拍面碰撞。n 为拍面法向（指向出球一侧），vp 为挥拍速度。
    法向：恢复系数 e；切向：接触点滑动速度产生胶皮摩擦冲量，滑动足够小时“咬住”球（滚动），
    否则按库仑摩擦 mu 打滑。摩擦冲量同时改变球速与旋转。e、mu、ks（咬球系数倍数）取决于胶皮。"""
    u = v - vp
    un = min(dot3(u, n), -0.05)
    jn = -(1.0 + e) * un
    c = u - BALL_R * cross3(w, n)
    s = c - dot3(c, n) * n
    sn = norm3(s)
    jt = np.zeros(3)
    if sn > 1e-9:
        jt = -min(ks * K_STICK * sn, mu * jn) / sn * s
    return v + jn * n + jt, w - cross3(n, jt) / (K_INERTIA * BALL_R)


@njit(cache=True)
def plan_stroke(pos, v_in, w_est, tx, ty, vh, top, side, margin, smax, swing_max, dv=0.95, ds=0.96, d1=0.0,
                e=E_PAD, ks=1.0):
    """球员“脑中”的挥拍反解：按感知到的来球（速度 v_in、旋转 w_est）和想要的落点/旋转，
    求拍面法向 n 与挥拍速度 vp，使 racket_impact 恰好给出目标出球。
    旋转只能靠拍面相对冲量方向的倾角产生：倾角受胶皮摩擦和技术（smax）限制，挥拍速度受力量限制，
    做不到时自动降低力量与旋转。返回 n, vp。"""
    n = np.zeros(3)
    vp = np.zeros(3)
    vout = np.empty(3)
    for attempt in range(20):
        top_eff = top
        side_eff = side
        for _ in range(3):
            if d1 > 0.0:
                vx, vy, vz = plan_serve_shot(pos[0], pos[1], pos[2], tx, ty, d1, top_eff, side_eff, margin)
            else:
                vx, vy, vz = plan_shot(pos[0], pos[1], pos[2], tx, ty, vh, top_eff, margin, side_eff)
            vout[0], vout[1], vout[2] = vx, vy, vz
            wdx, wdy, wdz = spin_vector(vx, vy, top, side)
            J = vout - v_in
            jm = norm3(J) + 1e-9
            jh = J / jm
            a = np.empty(3)
            a[0] = -K_INERTIA * BALL_R * (wdx - w_est[0])
            a[1] = -K_INERTIA * BALL_R * (wdy - w_est[1])
            a[2] = -K_INERTIA * BALL_R * (wdz - w_est[2])
            a = a - dot3(a, jh) * jh
            am = norm3(a)
            sa = min(am / jm, smax)
            if am > 1e-9:
                m = cross3(jh, a / am)
                n = math.sqrt(1.0 - sa * sa) * jh + sa * m
            else:
                n = jh.copy()
            w_ach = w_est - cross3(n, J) / (K_INERTIA * BALL_R)
            hn = math.sqrt(vx * vx + vy * vy) + 1e-9
            top_eff = w_ach[0] * (-vy / hn) + w_ach[1] * (vx / hn)
            side_eff = w_ach[2]
        jn = dot3(J, n)
        jt = J - jn * n
        c = v_in - BALL_R * cross3(w_est, n)
        ct = c - dot3(c, n) * n
        vp = (dot3(v_in, n) + jn / (1.0 + e)) * n + ct + jt / (ks * K_STICK)
        if norm3(vp) <= swing_max:
            break
        vh *= dv
        top *= ds
        side *= ds
    return n, vp


@njit(cache=True)
def rub_spin_cap(r):
    """胶皮允许的最大拍面倾角（sin）：受摩擦系数（切向冲量 ≤ μ·法向冲量）和咬球能力限制。"""
    mu = RUB_P[r, 1]
    return min(mu / math.sqrt(1.0 + mu * mu), 0.6 * RUB_P[r, 2])


@njit(cache=True)
def execute_stroke(i, h, n, vp, sig_n, B, I, T, bh, r=0):
    """按执行误差扰动拍面角度与挥拍速度，再用这面胶皮做真实的物理碰撞。"""
    n = n + np.random.standard_normal(3) * sig_n
    n = n / norm3(n)
    for k in range(3):
        T[i, PN0 + 3 * h + k] = n[k]
    vp = vp * (1.0 + np.random.standard_normal() * sig_n * 0.5)
    v_in = B[i, 3:6].copy()
    w_in = B[i, 6:9].copy()
    v_out, w_out = racket_impact(v_in, w_in, n, vp, RUB_P[r, 0], RUB_P[r, 1], RUB_P[r, 2])
    B[i, 3:6] = v_out
    B[i, 6:9] = w_out
    I[i, LHIT] = h
    I[i, LSIDE] = 1 if bh else 0
    T[i, LSW] = norm3(vp)
    T[i, LRUB] = r


@njit(cache=True)
def swing_limit(st, fat, bh):
    return (12.0 + 16.0 * st) * (1.0 - 0.25 * fat) * (0.8 if bh else 1.0)


@njit(cache=True)
def react_delay(rea):
    return 0.10 + 0.25 * (1.0 - rea)


@njit(cache=True)
def vmax_of(agi, fat):
    return (1.2 + 1.2 * agi) * (1.0 - 0.35 * fat)


@njit(cache=True)
def predict(i, B, I, T, S):
    """为接球方预测两个候选击球点：早点（落台后的最高点；最高点在台面上方够不着时取球出台沿的位置）
    和晚点（下降期，球落到台面高度以下或飞出端线较远；短球取二跳前），并按反应属性加入感知误差。
    实际击球点由 update_contact 按站位在两者间选。"""
    rec = I[i, TURN]
    b = B[i].copy()
    dt = 1.0 / 240.0
    found = False
    valid = False
    apex = False
    twice = False
    lx, ly, lz, lth = 0.0, 0.0, 0.0, 0.0
    rs = -1.0 if rec == 0 else 1.0
    tx, ty, tz, th = rs * (HALF_L + 0.6), 0.0, 1.0, 0.0
    own_ok = I[i, HITS] == 0 and I[i, SVB] == 0
    for k in range(480):
        ev = ball_substep(b, dt)
        t = (k + 1) * dt
        if ev == 1:
            on_rec = (b[0] < 0.0) if rec == 0 else (b[0] > 0.0)
            if own_ok and not on_rec and not found:
                own_ok = False
                continue
            if not found:
                if on_rec:
                    found = True
                    continue
                break
            if not apex:
                tx, ty, tz, th = b[0], b[1], b[2], t
            valid = True
            twice = True
            if lth <= 0.0:
                lx, ly, lz, lth = tx, ty, tz, th
            break
        if apex and b[2] >= TABLE_H + LATE_SHORT_Z:
            lx, ly, lz, lth = b[0], b[1], b[2], t
        if found and not apex and b[5] <= 0.0:
            tx, ty, tz, th = b[0], b[1], b[2], t
            apex = True
            valid = True
            if abs(b[0]) >= HALF_L - 0.25:
                break
        if apex and abs(b[0]) >= HALF_L - 0.25:
            tx, ty, tz, th = b[0], b[1], b[2], t
            break
        if b[2] < TABLE_H - 0.3 or abs(b[0]) > 4.5:
            break
    if not twice:
        lx, ly, lz, lth = tx, ty, tz, th
    if valid:
        t = th
        for k in range(480 if not twice else 0):
            ev = ball_substep(b, dt)
            t += dt
            if ev != 0 or b[2] < LATE_ZMIN or abs(b[0]) > LATE_XMAX:
                break
            lx, ly, lz, lth = b[0], b[1], b[2], t
        sig = (0.02 + 0.12 * (1.0 - S[i, rec, REA])) * (1.0 + norm3(B[i, 3:6].copy()) / 15.0)
        ex, ey, ez = (np.random.standard_normal() * sig, np.random.standard_normal() * sig,
                      np.random.standard_normal() * sig * 0.5)
        tx += ex
        ty += ey
        tz += ez
        lx += ex
        ly += ey
        lz += ez
    T[i, EX], T[i, EY], T[i, EZ], T[i, ETH] = tx, ty, tz, T[i, RT] + th
    T[i, LX], T[i, LY], T[i, LZ], T[i, LTH] = lx, ly, lz, T[i, RT] + lth
    T[i, PX] = tx
    T[i, PY] = ty
    T[i, PZ] = tz
    T[i, PRT] = T[i, RT] + th
    T[i, EVT] = T[i, RT]
    T[i, SH0 + rec] = -1.0
    I[i, PVALID] = 1 if valid else 0


@njit(cache=True)
def toss_time(toss):
    """抛球从离手到落回出手高度的时间。"""
    return 2.0 * math.sqrt(2.0 * (TOSS_H0 + TOSS_H1 * toss) / G)


@njit(cache=True)
def serve_is_bh(k):
    return k == SV_BH or k == SV_BHFAST


@njit(cache=True)
def serve_spot(ms, a1, k):
    """发球时球的横向位置：正手发球可以站满半张台（多在反手角侧身发），反手发球站在中间偏左，
    砍式发球站在中间、可以站到正手位。"""
    if serve_is_bh(k):
        return ms * (0.12 + 0.2 * a1)
    if k == SV_CHOP:
        return ms * (-0.3 + 0.3 * a1)
    return ms * a1 * 0.65


@njit(cache=True)
def place_server(i, srv, B, P, S, bh):
    """正手发球球在持拍手一侧；反手发球球在身前略偏非持拍手一侧。"""
    xs = -1.0 if srv == 0 else 1.0
    P[i, srv, 0] = B[i, 0] + xs * 0.1
    P[i, srv, 1] = B[i, 1] - lat_m(S, i, srv) * hand_off(-1.0 if bh else 1.0)
    P[i, srv, 2] = B[i, 2] - 0.03
    for k in range(3, 6):
        P[i, srv, k] = 0.0


@njit(cache=True)
def plan_serve(i, srv, T):
    """发球战术脑：按本场（近期加权）各手法的得分率与先验估计每种发球的价值，扣掉对手对它的熟悉度，
    再按 softmax 抽签——好用的多用，被适应了就换，偶尔试试别的。"""
    rec = 1 - srv
    sc = np.empty(N_SERVE)
    for k in range(N_SERVE):
        n = T[i, SN0 + srv * N_SERVE + k]
        w = T[i, SW0 + srv * N_SERVE + k]
        p = (w + SERVE_PRIOR[k] * PRIOR_N) / (n + PRIOR_N)
        sc[k] = (p - TAC_FAM * T[i, FAM0 + rec * N_SERVE + k]) / TAC_TEMP
    mx = sc.max()
    tot = 0.0
    for k in range(N_SERVE):
        sc[k] = math.exp(sc[k] - mx)
        tot += sc[k]
    r = np.random.random() * tot
    for k in range(N_SERVE):
        r -= sc[k]
        if r <= 0.0:
            return k
    return N_SERVE - 1


@njit(cache=True)
def plan_variation(i, srv, T):
    """旋转变化的选择：本场各变化的得分率，减去“用得太多、容易被预判”的扣分，softmax 抽签。"""
    sc = np.empty(N_VAR)
    tot_n = 1e-6
    for v in range(N_VAR):
        tot_n += T[i, VN0 + srv * N_VAR + v]
    for v in range(N_VAR):
        n = T[i, VN0 + srv * N_VAR + v]
        w = T[i, VW0 + srv * N_VAR + v]
        p = (w + VAR_PRIOR[v] * PRIOR_N) / (n + PRIOR_N)
        sc[v] = (p - TAC_VREP * n / tot_n) / TAC_TEMP
    mx = sc.max()
    tot = 0.0
    for v in range(N_VAR):
        sc[v] = math.exp(sc[v] - mx)
        tot += sc[v]
    r = np.random.random() * tot
    for v in range(N_VAR):
        r -= sc[v]
        if r <= 0.0:
            return v
    return N_VAR - 1


@njit(cache=True)
def record_serve(i, w, I, T):
    k = I[i, LSRV]
    if k < 0 or w < 0:
        return
    srv = I[i, SERVER]
    for kk in range(N_SERVE):
        T[i, SN0 + srv * N_SERVE + kk] *= TAC_DECAY
        T[i, SW0 + srv * N_SERVE + kk] *= TAC_DECAY
    T[i, SN0 + srv * N_SERVE + k] += 1.0
    if w == srv:
        T[i, SW0 + srv * N_SERVE + k] += 1.0
    v = int(T[i, VPLAN0 + srv])
    for vv in range(N_VAR):
        T[i, VN0 + srv * N_VAR + vv] *= TAC_DECAY
        T[i, VW0 + srv * N_VAR + vv] *= TAC_DECAY
    T[i, VN0 + srv * N_VAR + v] += 1.0
    if w == srv:
        T[i, VW0 + srv * N_VAR + v] += 1.0


@njit(cache=True)
def reset_serve_stats(i, T):
    for k in range(2 * N_SERVE):
        T[i, SW0 + k] = 0.0
        T[i, SN0 + k] = 0.0
    for k in range(2 * N_VAR):
        T[i, VW0 + k] = 0.0
        T[i, VN0 + k] = 0.0


@njit(cache=True)
def serve_type(a):
    """发球手法的选择：持拍面（a8）决定正手/反手，a10 在该面的几种手法里选。
    选择带执行噪声（约一档宽），偶尔发出相邻手法：期望收益随动作平滑变化，策略才学得会换发球。"""
    return serve_type_of(a[8] + np.random.standard_normal() * SERVE_PICK_SIG,
                         a[10] + np.random.standard_normal() * SERVE_PICK_SIG)


@njit(cache=True)
def serve_type_of(a8, a10):
    """不带噪声的手法映射（画面上发球前的预备姿势用）。"""
    if a8 < -0.15:
        return SV_BH if a10 < 0.0 else SV_BHFAST
    return (SV_PEND, SV_HOOK, SV_SHOVEL, SV_SQUAT, SV_CHOP, SV_FAST)[min(max(int((a10 + 1.0) * 3.0), 0), 5)]


@njit(cache=True)
def serve_toss(k, a9):
    toss = (a9 + 1.0) * 0.5
    if k == SV_FAST or k == SV_BHFAST:
        toss = min(toss, FAST_TOSS_MAX)
    return toss


@njit(cache=True)
def serve_hit(i, a, B, I, T, S, F):
    """发球：抛起的球几乎静止，靠拍面摩擦“搓/拉”出旋转。手法决定侧旋方向、旋转上限、可发的上下旋范围、
    控制难度、迷惑性和发完后的还原；侧上/侧下/转与不转在同一个手法里变化。抛球越高（a9）球下落越快，
    旋转和速度更强但更难控制；回抛（a11）让动作更隐蔽，但有被判发球违例的风险。返回是否违例。"""
    srv = I[i, SERVER]
    xs = -1.0 if srv == 0 else 1.0
    st = S[i, srv, STR]
    srv_tec = S[i, srv, SRV]
    # 接发球方读旋转：反应越快，越早看清旋转
    recv_rea = S[i, 1 - srv, REA]
    k = int(T[i, SPLAN0 + srv]) if USE_TACTIC else serve_type(a)
    sp = SERVE_P[k]
    bh = k == SV_BH or k == SV_BHFAST
    toss = T[i, TOSS] if T[i, TOSS] >= 0.0 else serve_toss(k, a[9])
    hidden = a[11] > 0.5
    T[i, TOSS] = toss
    I[i, LHID] = 1 if hidden else 0
    I[i, LSRV] = k
    if hidden and np.random.random() < FAULT_P:
        return True
    B[i, 3], B[i, 4], B[i, 5] = 0.0, 0.0, -0.5 - 2.0 * toss
    B[i, 6], B[i, 7], B[i, 8] = 0.0, 0.0, 0.0
    depth = 0.3 + (a[3] + 1.0) * 0.45
    if k == SV_FAST or k == SV_BHFAST:
        depth = max(depth, FAST_DEPTH)
    hs = hs_of(S, i, srv)
    pen = S[i, srv, GRIP] > 0.5
    rub = face_rubber(S, i, srv, bh, -1)
    tx = -xs * depth
    ty = -xs * hs * a[4] * 0.6
    pw = (a[5] + 1.0) * 0.5
    vh = (3.0 + pw * (1.5 + 2.0 * st)) * sp[6] * (1.0 + 0.1 * toss)
    ms = (SRV_SPIN0 + SRV_SPIN1 * srv_tec) * sp[0] * (1.0 + 0.3 * toss)
    a6 = VAR_TOP[int(T[i, VPLAN0 + srv])] if USE_TACTIC else a[6]
    top = min(max(a6, sp[7]), sp[8]) * ms
    sgn = sp[1] if sp[1] != 0.0 else (1.0 if a[7] >= 0.0 else -1.0)
    # 侧旋量主要由手法决定（勾手、逆旋转、下蹲本来就以侧旋为主），侧旋意图只微调；发“不转”时侧旋也收住
    # 左手发球是右手的镜像：同一手法侧旋方向相反
    side = hs * sgn * (0.75 + 0.25 * abs(a[7])) * sp[2] * ms * (0.35 + 0.65 * min(abs(a6), 1.0))
    # 第一跳离网距离：短球压在本方台面靠网处，长球、急球压在本方端线附近；发力越大第一跳越靠后
    # 实测：第一跳压在本方离网 0.9-1.1 m（靠近端线）时弧线最低，短球、长球都能精确控制；
    # 第一跳太靠网，弧线必须抬高才能过网，短球就发不出来
    d1 = 0.9 + 0.1 * min(max((depth - 0.3) / 0.9, 0.0), 1.0) + 0.15 * pw
    if k == SV_FAST or k == SV_BHFAST:
        d1 = max(d1, FAST_D1)
    d1 = min(d1, HALF_L - 0.1)
    n, vp = plan_stroke(B[i, 0:3].copy(), B[i, 3:6].copy(), B[i, 6:9].copy(), tx, ty, vh, top, side,
                        0.05, min((0.4 + 0.4 * srv_tec) * min(sp[0], 1.0), rub_spin_cap(rub)),
                        swing_limit(st, F[i, srv], bh), 0.95, 0.96, d1, RUB_P[rub, 0], RUB_P[rub, 2])
    execute_stroke(i, srv, n, vp, (0.003 + 0.01 * (1.0 - srv_tec)) * sp[3] * (1.0 + 0.5 * toss)
                   * (PEN_SERVE if pen else 1.0), B, I, T, bh, rub)
    rec = 1 - srv
    fam = T[i, FAM0 + rec * N_SERVE + k]
    T[i, SFAM] = fam
    # 迷惑性 = 手法本身的隐蔽程度 × 发球方的手法水平 × 对手看清旋转的快慢（反应）
    T[i, DEC] = (sp[4] * (0.7 + 0.6 * srv_tec) * (1.0 + 0.1 * toss) * (1.15 if hidden else 1.0)
                 * (PEN_DEC if pen else 1.0)
                 * (FAM_DEC_NEW + (FAM_DEC_OLD - FAM_DEC_NEW) * fam) * (1.35 - 0.7 * recv_rea))
    T[i, ORB0 + rec] = rub
    for kk in range(N_SERVE):
        T[i, FAM0 + rec * N_SERVE + kk] *= 1.0 - FAM_RATE
    T[i, FAM0 + rec * N_SERVE + k] += FAM_RATE
    T[i, REC0 + srv] = sp[5]
    T[i, LK0 + srv] = ST_DRIVE
    bl = -1.0 if bh else 1.0
    T[i, BL0 + srv] = bl
    T[i, BT0 + srv] = bl
    return False


@njit(cache=True)
def reset_mind(i, T):
    """新比赛：连丢分、烦躁、对对手胶皮的适应都清零。"""
    for j in range(2):
        T[i, STK0 + j] = 0.0
        T[i, FRU0 + j] = 0.0
        T[i, ADP0 + j] = 0.0
        T[i, ORB0 + j] = 0.0


@njit(cache=True)
def pressure(i, h, I, T):
    """击球方此刻的心理压力：局点/赛点 + 连丢分 + 被怪胶打烦。"""
    s0, s1 = I[i, SC0], I[i, SC1]
    p = 0.0
    if max(s0, s1) >= GAME_PTS - 1:
        p = CLUTCH_GP
        for j in range(2):
            if I[i, SC0 + j] >= GAME_PTS - 1 and I[i, SC0 + j] > I[i, SC0 + 1 - j] and I[i, G0 + j] >= MATCH_WIN - 1:
                p = CLUTCH_MP
    return p + min(TILT_PT * T[i, STK0 + h], TILT_MAX) + T[i, FRU0 + h]


def pressure_py(I, T, h):
    """pressure 的纯 Python 版（界面显示用；numba 版要传二维数组）。"""
    s0, s1 = int(I[SC0]), int(I[SC1])
    p = 0.0
    if max(s0, s1) >= GAME_PTS - 1:
        p = CLUTCH_GP
        for j in range(2):
            if int(I[SC0 + j]) >= GAME_PTS - 1 and int(I[SC0 + j]) > int(I[SC0 + 1 - j]) \
                    and int(I[G0 + j]) >= MATCH_WIN - 1:
                p = CLUTCH_MP
    return p + min(TILT_PT * float(T[STK0 + h]), TILT_MAX) + float(T[FRU0 + h])


@njit(cache=True)
def start_point(i, train, B, P, I, T, S, F):
    if train:
        srv = np.random.randint(0, 2)
        if np.random.random() < 0.15:
            for k in range(2 * N_SERVE):
                T[i, FAM0 + k] = 0.0
            reset_serve_stats(i, T)
            for j in range(2):
                for k in range(N_STATS):
                    S[i, j, k] = np.random.random()
                S[i, j, SRV] = np.random.random()
                S[i, j, MEN] = np.random.random()
                S[i, j, TCH] = np.random.random()
                plain = np.random.random() < 0.25
                for k in range(N_STATS, N_STATS + N_STYLE):
                    S[i, j, k] = 0.0 if plain else np.random.random() * 2.0 - 1.0
                random_equipment(S, i, j, np.random.random() < BAL_FRAC)
                F[i, j] = np.random.random() * 0.9
            reset_mind(i, T)
    else:
        tot = I[i, SC0] + I[i, SC1]
        fs = I[i, FIRST]
        if tot < 2 * (GAME_PTS - 1):
            srv = fs ^ ((tot // 2) & 1)
        else:
            srv = fs ^ (tot & 1)
    for j in range(2):
        F[i, j] = max(0.0, F[i, j] - FAT_REC_PT)
        T[i, LM0 + j] = lat_m(S, i, j)

    xs = -1.0 if srv == 0 else 1.0
    B[i, 0] = xs * (HALF_L + 0.15)
    B[i, 1] = (np.random.random() - 0.5) * 0.9
    B[i, 2] = TABLE_H + 0.22
    for k in range(3, 9):
        B[i, k] = 0.0
    for j in range(2):
        sj = -1.0 if j == 0 else 1.0
        if train:
            P[i, j, 0] = sj * (HALF_L + 0.35 + np.random.random() * 0.7)
            P[i, j, 1] = (np.random.random() - 0.5) * 0.8
        else:
            P[i, j, 0] = sj * (HALF_L + 0.55)
            P[i, j, 1] = 0.0
        P[i, j, 2] = 1.0
        for k in range(3, 6):
            P[i, j, k] = 0.0
    for j in range(2):
        bl = 1.0
        if train and j != srv and np.random.random() < 0.5:
            bl = -1.0
        T[i, BL0 + j] = bl
        T[i, BT0 + j] = bl
        T[i, LN0 + j] = 0.0
        T[i, CX0 + j] = 0.0
        T[i, CD0 + j] = 0.0
        T[i, CU0 + j] = 0.0
        T[i, SET0 + j] = SETTLE_T
        T[i, SPL0 + j] = 0.0
        T[i, REC0 + j] = 0.0
        T[i, DV0 + j] = 0.0
        T[i, LK0 + j] = 0.0
    T[i, STB] = 0.0
    T[i, DEC] = 1.0
    I[i, LFAKE] = 0
    I[i, LSTK] = -1
    I[i, LSRV] = -1
    I[i, LTECH] = -1
    T[i, TOSS] = -1.0
    T[i, SPLAN0 + srv] = plan_serve(i, srv, T)
    T[i, VPLAN0 + srv] = plan_variation(i, srv, T)
    sk = int(T[i, SPLAN0 + srv])
    bh = serve_is_bh(sk)
    B[i, 2] = TABLE_H + SERVE_Z[sk]
    if bh or sk == SV_CHOP:
        B[i, 1] = serve_spot(lat_m(S, i, srv), (np.random.random() - 0.5) * 1.4, sk)
    if bh:
        T[i, BL0 + srv] = -1.0
        T[i, BT0 + srv] = -1.0
    place_server(i, srv, B, P, S, bh)

    I[i, TURN] = 1 - srv
    I[i, BOUNCED] = 0
    I[i, NETT] = 0
    I[i, HITS] = 0
    T[i, PCHAIN] = 0.0
    I[i, SVB] = 0
    I[i, SERVER] = srv
    I[i, PHASE] = PH_SERVE
    I[i, PVALID] = 0
    T[i, RT] = 0.0
    T[i, TIMER] = (0.15 + np.random.random() * 1.65) if train else SERVE_WAIT
    T[i, EVT] = 0.0
    T[i, SH0] = -1.0
    T[i, SH1] = -1.0


@njit(cache=True)
def mark_fake(i, h, lat, cue, P, I, T, S):
    """出手时：引拍明显指向一侧，对手的重心正倒向引拍那一侧，球却去了另一侧——假动作骗到了重心。
    只看“重心压在反方向”不够：对手自己随挥后的重心偏移几乎每拍都满足。"""
    q = 1 - h
    mq, mh = lat_m(S, i, q), lat_m(S, i, h)
    need = mq * (mh * lat - P[i, q, 1])
    lure = mq * (mh * cue - P[i, q, 1])
    cn = T[i, LN0 + q] / LEAN_MAX
    I[i, LFAKE] = 0
    if (abs(lat - cue) > FAKE_DEV and cn * lure > 0.0 and cn * need < 0.0 and abs(cn) > FAKE_LEAN
            and abs(need) > FAKE_NEED and abs(lure) > 0.1):
        I[i, LFAKE] = h + 1
        T[i, LFT] = T[i, RT]


@njit(cache=True)
def classify_stroke(b, pw, top):
    """按击球点（离台远近、高度）和击球意图（力度、上/下旋）判定技术。"""
    dxt = abs(b[0]) - HALF_L
    hz = b[2] - TABLE_H
    if hz > SMASH_Z and pw > 0.55 and top > -0.3:
        return ST_SMASH
    if top < -0.25:
        return ST_CHOP if dxt > 0.45 else ST_PUSH
    if dxt > 1.1 and pw < 0.45:
        return ST_LOB
    if dxt < -0.25:
        return ST_FLICK
    if pw < 0.35 and dxt < 0.5:
        return ST_BLOCK
    if top < 0.35:
        return ST_DRIVE
    return ST_LOOP


@njit(cache=True)
def tech_of(k, bh, pw, top, side, depth, top_in):
    """技术细分。depth 为目标落点离网距离（米），top_in 为来球上旋角速度（rad/s，负为下旋）。
    台内：强下旋短球摆短 / 劈长，不转或弱下旋拧拉、挑打；反手大力拧强下旋是霸王拧；
    反手位轻摩擦带侧旋、突然变短的是抹。相持：拉下旋叫提拉，大力拉上旋是前冲，反手近台快拉上旋是撕；
    反手侧旋挡（拍面半侧立向侧下切）是侧切。"""
    back = top_in < -60.0
    if k == ST_PUSH:
        if depth < 0.6:
            return TQ_SHORT
        if depth > 0.95 and pw > 0.35:
            return TQ_LONG
        return TQ_PUSH
    if k == ST_FLICK:
        if pw < 0.4 and abs(side) > 0.4:
            return TQ_WIPE
        if bh:
            return TQ_BANANA if (pw > 0.7 and top_in < -120.0) else TQ_TWIST
        return TQ_FLICK
    if k == ST_LOOP:
        if back:
            return TQ_LIFT
        if bh and pw > 0.6:
            return TQ_RIP
        if pw > 0.7:
            return TQ_DRIVELOOP
        return TQ_LOOP
    if k == ST_BLOCK:
        if bh and abs(side) > 0.45 and top < 0.1:
            return TQ_SIDECUT
        return TQ_BLOCK
    if k == ST_DRIVE:
        return TQ_DRIVE
    if k == ST_CHOP:
        return TQ_CHOP
    if k == ST_SMASH:
        return TQ_SMASH
    return TQ_LOB


@njit(cache=True)
def incoming_top(v, w):
    hn = math.sqrt(v[0] * v[0] + v[1] * v[1]) + 1e-9
    return w[0] * (-v[1] / hn) + w[1] * (v[0] / hn)


@njit(cache=True)
def do_hit(i, h, a, stretch, B, P, I, T, S, F):
    """球员 h 击球。动作里的 落点深度/横向/力度/上旋/侧旋 是“意图”，
    先按感知到的来球旋转反解挥拍（plan_stroke），再加执行误差做真实的拍面碰撞（execute_stroke）。
    正反手由当前持拍状态决定：反手挥拍速度和可用旋转更低；正反手切换途中击球动作变形、误差大。
    伸手够球（stretch）、跑动中击球、来球速度快都会让动作变形：误差变大、挥拍速度受限，反手接重球更吃亏。"""
    m = 1.0 if h == 0 else -1.0
    hs = hs_of(S, i, h)
    pen = S[i, h, GRIP] > 0.5
    st = S[i, h, STR]
    tec = S[i, h, TEC]
    # 读来球旋转：这一分的第一板（接发球）看反应，相持中看击球技术
    rd = S[i, h, REA] if I[i, HITS] == 0 else tec
    fat = F[i, h]
    blend = T[i, BL0 + h]
    bh = blend < 0.0
    awk = max(0.0, 0.9 - abs(blend)) / 0.9
    depth = DEPTH_MIN + (a[3] + 1.0) * 0.5 * (1.2 - DEPTH_MIN)
    short = max(0.0, 0.6 - depth) / 0.3
    depth = max(depth + np.random.standard_normal() * (0.03 + 0.08 * (1.0 - tec)) * (1.0 + 1.5 * short), 0.12)
    cue = T[i, CU0 + h]
    dev = 0.25 + 0.45 * tec
    lat = min(max(a[4] * LAT_MAX, cue - dev), cue + dev)
    mark_fake(i, h, lat, cue, P, I, T, S)
    T[i, CU0 + h] = 0.0
    pw = (a[5] + 1.0) * 0.5
    F[i, h] = min(1.0, F[i, h] + FAT_HIT * (0.4 + pw) * (FAT_BASE - S[i, h, STA]))
    k = classify_stroke(B[i], pw, a[6])
    sp = STROKE_P[k]
    rub = face_rubber(S, i, h, bh, k)
    tx = m * depth
    ty = m * hs * lat
    vh = min((3.0 + pw * (5.0 + 17.0 * st)) * (SMASH_V if k == ST_SMASH else 1.0) * (1.0 - 0.3 * fat), sp[4])
    T[i, DBG0] = vh
    ms = (STK_SPIN0 + STK_SPIN1 * tec) * (0.9 if bh else 1.0) * sp[7]
    top = a[6] * ms
    side = hs * a[7] * 0.4 * ms

    w_in = B[i, 6:9].copy()
    v_in = B[i, 3:6].copy()
    floor = BORROW[k] * math.sqrt(v_in[0] ** 2 + v_in[1] ** 2)
    absorb = 0.0
    if vh < floor:
        absorb = (floor - vh) / (floor + 1.0)
        vh = floor - (floor - vh) * (0.25 + 0.35 * tec)
    dec = T[i, DEC]
    T[i, DEC] = 1.0
    men, tch = S[i, h, MEN], S[i, h, TCH]
    odd = RUB_ODD[int(T[i, ORB0 + h])]
    adp = T[i, ADP0 + h]
    if I[i, HITS] > 0:
        # 相持中怪胶打来的球：手感好、打得多了就读得清（发球的迷惑性另算，不在这里缩放）
        dec = 1.0 + (dec - 1.0) * max(1.0 + TCH_READ * (0.5 - tch), 0.0) * (1.0 - adp)
    T[i, ADP0 + h] = min(adp + (ADAPT0 + ADAPT1 * tch) * odd * (1.0 - adp / ADAPT_MAX), ADAPT_MAX)
    sig_rel = (0.06 + 0.45 * (1.0 - rd)) * (1.0 + 0.5 * fat) * 0.6 * dec * READ_SCALE
    w_est = (w_in * (1.0 + np.random.standard_normal(3) * sig_rel)
             + np.random.standard_normal(3) * 10.0 * (1.0 - rd) * dec * READ_SCALE)
    margin = 0.02 + 0.06 * (1.0 - tec) + sp[5]
    if k == ST_BLOCK:
        margin += NOPACE_MARGIN * min(max((NOPACE_V - math.sqrt(v_in[0] ** 2 + v_in[1] ** 2)) / 5.0, 0.0), 1.0)
    smax = min((0.3 + 0.3 * tec) * (0.85 if bh else 1.0) * sp[1] * (PEN_BH_SPIN if pen and bh else 1.0),
               rub_spin_cap(rub))
    # 直板：正手台内小球、挡球更细腻；反手推挡稳，直拍横打（拉、拨、拧）挥拍受限、误差大
    small = k == ST_PUSH or k == ST_FLICK or k == ST_BLOCK
    if not pen:
        grip_sig = 1.0
    elif not bh:
        grip_sig = PEN_FH_SHORT if small else 1.0
    else:
        grip_sig = PEN_PUSHBLOCK_SIG if (k == ST_BLOCK or k == ST_PUSH) else PEN_BH_SIG
    sk = min(stretch / 0.2, 2.0)
    dive = not bh and pw > DIVE_PW and T[i, LM0 + h] * (B[i, 1] - P[i, h, 1]) < FH_CLO - DIVE_ST
    sk_sw = 0.0 if dive else min(sk, 1.0)
    sk_sig = 0.4 * sk if dive else sk
    run = math.sqrt(min(m * P[i, h, 3], 0.0) ** 2 + P[i, h, 4] ** 2) / vmax_of(S[i, h, AGI], 0.0)
    settle = min(T[i, SET0 + h] / SETTLE_T, 1.0)
    unset = (1.0 - settle) * sp[6]
    cn = T[i, LN0 + h] / LEAN_MAX
    load = max(0.0, cn) if bh else max(0.0, -cn)
    vf = min(max(m * P[i, h, 3] / 0.8, -1.0), 1.0)
    body = (0.5 + 0.5 * settle) * (0.6 + 0.4 * load) * (1.0 + (0.3 if vf > 0.0 else 0.2) * vf)
    share = sp[10] * (BH_BODY if bh else 1.0)
    T[i, LBODY] = share * body
    avail = T[i, RT] - T[i, EVT] - react_delay(S[i, h, REA])
    lack = min(max((PREP_T[k] - avail) / PREP_T[k], 0.0), 1.0)
    T[i, LPREP] = lack
    margin += LACK_MARGIN * lack
    top_in = incoming_top(v_in, w_in)
    back_in = max(0.0, -top_in) / SPIN_REF
    I[i, LTECH] = tech_of(k, bh, pw, a[6], a[7], depth, top_in)
    if k == ST_PUSH:
        r = np.random.random()
        margin += min(POP_MARGIN * r * r * (0.3 + min(back_in, 1.5)) * (1.3 - tec)
                      * (1.0 + POP_CHAIN * min(T[i, PCHAIN], 6.0)), POP_CAP)
        T[i, PCHAIN] += 1.0
    else:
        T[i, PCHAIN] = 0.0
    hi = min(max((B[i, 2] - TABLE_H - HI_Z0) / (HI_Z1 - HI_Z0), 0.0), 1.0)
    atk = k == ST_LOOP or k == ST_DRIVE or k == ST_FLICK or k == ST_SMASH
    vh = min(vh * (1.0 - share + share * body) * (1.0 - 0.5 * lack), sp[4])
    swing = (swing_limit(st, fat, bh) * sp[0] * (1.0 - share + share * body)
             * (1.0 - 0.5 * awk) * (1.0 - 0.4 * sk_sw) * (1.0 - 0.15 * min(run, 1.0))
             * (1.0 - 0.5 * lack) * (PEN_BH_SWING if pen and bh and not (k == ST_BLOCK or k == ST_PUSH) else 1.0))
    T[i, DBG0 + 1] = vh
    T[i, DBG0 + 2] = swing
    T[i, DBG0 + 3] = B[i, 0]
    T[i, DBG0 + 4] = 1.0 - share + share * body
    T[i, DBG0 + 5] = 1.0 - 0.5 * awk
    T[i, DBG0 + 6] = 1.0 - 0.4 * min(sk, 1.0)
    T[i, DBG0 + 7] = 1.0 - 0.15 * min(run, 1.0)
    T[i, DBG0 + 8] = 1.0 - 0.5 * lack
    n, vp = plan_stroke(B[i, 0:3].copy(), v_in, w_est, tx, ty, vh, top, side,
                        margin, smax, swing,
                        0.97 if (k == ST_PUSH or k == ST_CHOP) else 0.95,
                        0.88 if (k == ST_PUSH or k == ST_CHOP) else 0.96, 0.0, RUB_P[rub, 0], RUB_P[rub, 2])
    sens = RUB_SENS[rub] * (1.0 + TCH_SENS * (0.5 - tch))
    heavy = max(0.0, norm3(v_in) - 6.0) / 8.0 * sens
    push_vs_top = VS_TOP[k] * max(0.0, incoming_top(v_in, w_in)) / SPIN_REF * (1.3 - tec) * sens
    vs_back = VS_BACK[k] * back_in * (1.3 - tec) * ((1.0 - HI_BACK * hi) if atk else 1.0) * sens
    sig_n = (ERR_SCALE * (0.004 + 0.018 * (1.0 - tec)) * (0.6 + 0.7 * pw) * (1.0 + 0.5 * fat) * sp[2]
             * (1.1 if bh else 1.0) * (1.0 + 3.0 * awk)
             * (1.0 + 1.5 * sk_sig) * (1.0 + RUN_SIG * min(run, 1.5)) * (1.0 + 0.8 * unset) * (1.0 + LACK_SIG * lack) * (1.0 + ABSORB_SIG * absorb)
             * (1.0 + 2.5 * heavy * sp[3] * (1.3 - tec) * (1.1 if bh else 1.0))
             * (1.0 + push_vs_top) * (1.0 + vs_back) * ((1.0 - HI_SIG * hi) if atk else 1.0) * grip_sig
             * ((1.0 + TCH_SMALL * (0.5 - tch)) if (k == ST_PUSH or k == ST_FLICK) else 1.0)
             * (1.0 + MEN_SIG * pressure(i, h, I, T) * (1.0 - men)))
    execute_stroke(i, h, n, vp, sig_n, B, I, T, bh, rub)
    T[i, ORB0 + 1 - h] = rub
    # 怪胶回出的球旋转难判断：对手下一板读旋转的误差按胶皮放大（同发球的迷惑性）
    T[i, DEC] = RUB_DEC[rub]
    I[i, LSTK] = k
    T[i, LK0 + h] = k
    T[i, REC0 + h] = sp[8] + 0.15 * share * load
    if dive:
        T[i, DV0 + h] = DIVE_T
        T[i, REC0 + h] = DIVE_T + DIVE_REC
        P[i, h, 3] *= 0.2
        P[i, h, 4] *= 0.2
    T[i, LN0 + h] = -T[i, LN0 + h] * min(1.5 * share, 1.0) if load > 0.0 else T[i, LN0 + h]
    feat = ((1.0 if bh else -1.0) * S[i, h, STY_BH] + a[6] * S[i, h, STY_SPIN]
            + a[5] * S[i, h, STY_POW] + (2.0 * abs(lat) / LAT_MAX - 1.0) * S[i, h, STY_WIDE])
    T[i, STB] = STYLE_W * feat / N_STYLE
    I[i, TURN] = 1 - h
    I[i, BOUNCED] = 0
    I[i, NETT] = 0
    I[i, HITS] += 1
    T[i, SH0] = -1.0
    T[i, SH1] = -1.0
    predict(i, B, I, T, S)


@njit(cache=True)
def end_point(i, w, reason, train, I, T, F, rew, done, winner, reason_out, hits_out):
    done[i] = True
    winner[i] = w
    reason_out[i] = reason
    hits_out[i] = I[i, HITS]
    record_serve(i, w, I, T)
    I[i, LWIN] = w
    I[i, LREASON] = reason
    if w < 0:
        if reason == R_TIMEOUT:
            # 谁也不上手、把球拖到超时：和输掉这一分同样疼，逼双方主动进攻
            rew[i, 0] -= 1.0
            rew[i, 1] -= 1.0
        elif reason != R_LET:
            rew[i, 0] -= 0.5
            rew[i, 1] -= 0.5
    else:
        pts = WINNER_BONUS if reason == R_MISS else 1.0
        rew[i, w] += pts
        rew[i, 1 - w] -= pts
        # 心态：输分的连丢分 +1、赢分的清零；烦躁逐分消退，丢在怪胶打来的球上会更烦（心态的抵消在用的时候算）
        lo = 1 - w
        T[i, STK0 + lo] += 1.0
        T[i, STK0 + w] = 0.0
        for j in range(2):
            T[i, FRU0 + j] *= FRUS_DECAY
        T[i, FRU0 + lo] = min(T[i, FRU0 + lo] + FRUS_ADD * RUB_ODD[int(T[i, ORB0 + lo])], 1.0)
        I[i, SC0 + w] += 1
        s0, s1 = I[i, SC0], I[i, SC1]
        if max(s0, s1) >= GAME_PTS and abs(s0 - s1) >= 2:
            I[i, G0 + w] += 1
            I[i, SC0] = 0
            I[i, SC1] = 0
            I[i, FIRST] ^= 1
            if I[i, G0 + w] >= MATCH_WIN:
                I[i, M0 + w] += 1
                I[i, G0] = 0
                I[i, G1] = 0
                for j in range(2):
                    F[i, j] = 0.0
                reset_mind(i, T)
            else:
                for j in range(2):
                    F[i, j] = max(0.0, F[i, j] - FAT_REC_GAME)
    if not train:
        I[i, PHASE] = PH_DEAD
        T[i, TIMER] = 1.2


@njit(cache=True)
def hand_y(P, T, i, j):
    return P[i, j, 1] + T[i, LM0 + j] * hand_off(T[i, BL0 + j])


@njit(cache=True)
def comfort_range(blend):
    t = (blend + 1.0) * 0.5
    return BH_CLO + (FH_CLO - BH_CLO) * t, BH_CHI + (FH_CHI - BH_CHI) * t


@njit(cache=True)
def reach_hand(P, T, i, j, tx, ty, ext=0.0):
    """手朝 (tx, ty) 伸出：横向受当前正/反手可达范围限制，前后可向台内探 FWD_HI、向后收 FWD_LO。
    ext：正手侧额外能扑出去的距离（倒地爆冲）。
    返回手的世界坐标 x, y，以及超出舒适击球区的“够球”程度（米）。"""
    m = 1.0 if j == 0 else -1.0
    ml = T[i, LM0 + j]
    lat = ml * (ty - P[i, j, 1])
    fwd = m * (tx - P[i, j, 0])
    lo, hi = hand_range(T[i, BL0 + j])
    lo -= ext
    lat = min(max(lat, lo), hi)
    fhi = FWD_HI + HIGH_REACH * min(max((T[i, PZ] - TABLE_H - 0.25) / 0.35, 0.0), 1.0)
    fwd = min(max(fwd, FWD_LO), fhi)
    clo, chi = comfort_range(T[i, BL0 + j])
    stretch = max(clo - lat, 0.0) + max(lat - chi, 0.0) + max(FWD_CLO - fwd, 0.0) + FWD_STRETCH_W * max(fwd - FWD_CHI, 0.0)
    return P[i, j, 0] + m * fwd, P[i, j, 1] + ml * lat, stretch


@njit(cache=True)
def update_contact(i, P, I, T):
    """按接球方的站位在“早点”（上升期/最高点）和“晚点”（下降期）之间选实际击球点：
    贴台就打早点，退台则等球落到身前舒适位置再打——用距离换准备时间。"""
    if I[i, PVALID] == 0:
        return
    j = I[i, TURN]
    want = abs(P[i, j, 0]) - CONTACT_FWD
    ex, lx = abs(T[i, EX]), abs(T[i, LX])
    u = 0.0
    if lx > ex + 1e-6:
        u = min(max((want - ex) / (lx - ex), 0.0), 1.0)
    T[i, PX] = T[i, EX] + (T[i, LX] - T[i, EX]) * u
    T[i, PY] = T[i, EY] + (T[i, LY] - T[i, EY]) * u
    T[i, PZ] = T[i, EZ] + (T[i, LZ] - T[i, EZ]) * u * u
    T[i, PRT] = T[i, ETH] + (T[i, LTH] - T[i, ETH]) * u


@njit(cache=True)
def contact_x(B, I, T, S, i, j):
    """手伸向的击球点前后位置：判断出来球后对准预测的击球点（最高点或出台点），否则跟着球。"""
    if I[i, PVALID] == 1 and T[i, RT] - T[i, EVT] >= react_delay(S[i, j, REA]):
        return T[i, PX]
    return B[i, 0]


@njit(cache=True)
def shown_hand(B, P, I, T, S, i, j):
    if I[i, PHASE] == PH_FLY and I[i, TURN] == j and abs(B[i, 0] - P[i, j, 0]) < 1.2:
        hx, hy, _ = reach_hand(P, T, i, j, contact_x(B, I, T, S, i, j), B[i, 1])
        return hx, hy
    return P[i, j, 0], hand_y(P, T, i, j)


@njit(cache=True)
def clamp_paddle(P, T, i, j):
    sj = -1.0 if j == 0 else 1.0
    lx = sj * P[i, j, 0]
    xmin = BODY_MIN if abs(P[i, j, 1]) < HALF_W + 0.25 else 0.3
    if lx < xmin:
        P[i, j, 0] = sj * xmin
        if P[i, j, 3] * sj < 0.0:
            P[i, j, 3] = 0.0
    elif lx > 3.6:
        P[i, j, 0] = sj * 3.6
        if P[i, j, 3] * sj > 0.0:
            P[i, j, 3] = 0.0
    if P[i, j, 1] < -2.2:
        P[i, j, 1] = -2.2
        P[i, j, 4] = max(P[i, j, 4], 0.0)
    elif P[i, j, 1] > 2.2:
        P[i, j, 1] = 2.2
        P[i, j, 4] = min(P[i, j, 4], 0.0)
    zmin = 0.25
    hy = hand_y(P, T, i, j)
    if sj * P[i, j, 0] < HALF_L + 0.55 and (abs(P[i, j, 1]) < HALF_W + 0.05 or abs(hy) < HALF_W + 0.05):
        zmin = TABLE_H + 0.04
    if P[i, j, 2] < zmin:
        P[i, j, 2] = zmin
        P[i, j, 5] = max(P[i, j, 5], 0.0)
    elif P[i, j, 2] > 2.0:
        P[i, j, 2] = 2.0
        P[i, j, 5] = min(P[i, j, 5], 0.0)


@njit(cache=True)
def clamp_receiver(P, T, i, j):
    """发球出手前，接发球方只能在等候区里调整站位。"""
    sj = -1.0 if j == 0 else 1.0
    lx = sj * P[i, j, 0] - HALF_L
    if lx < RECV_X0 or lx > RECV_X1:
        P[i, j, 0] = sj * (HALF_L + min(max(lx, RECV_X0), RECV_X1))
        P[i, j, 3] = 0.0
    ml = T[i, LM0 + j]
    ly = ml * P[i, j, 1]
    if abs(ly - RECV_Y) > RECV_DY:
        P[i, j, 1] = ml * min(max(ly, RECV_Y - RECV_DY), RECV_Y + RECV_DY)
        P[i, j, 4] = 0.0


@njit(cache=True)
def seg_dist(ax, ay, az, bx, by, bz):
    dx, dy, dz = bx - ax, by - ay, bz - az
    dd = dx * dx + dy * dy + dz * dz
    t = 0.0
    if dd > 1e-12:
        t = min(max(-(ax * dx + ay * dy + az * dz) / dd, 0.0), 1.0)
    cx, cy, cz = ax + t * dx, ay + t * dy, az + t * dz
    return math.sqrt(cx * cx + cy * cy + cz * cz)


@njit(cache=True)
def move_player(i, j, a, des, acc, vmax, lrate, agi, P, T, pdt):
    """步法与重心。前后、手的高度：普通加速度控制。横向（自己视角）：
    垫步——随时可变向，起动快慢取决于重心；交叉步——更快，但迈出后要走完这一步（约 0.25 秒），
    期间收不住、重心被带向移动方向。重心 LN 可以主动提前压向一侧（预判）：往重心一侧起动更快，
    往反方向起动要先把重心移回来，压得越狠被骗时越难动。"""
    m = 1.0 if j == 0 else -1.0
    step_in = m * des[0] > 0.0 and -m * P[i, j, 0] < BODY_MIN + STEPIN_D
    for k in (0, 2):
        dv = des[k] * (STEPIN_V if k == 0 and step_in else 1.0) - P[i, j, 3 + k]
        mx = acc * (STEPIN_ACC if k == 0 and step_in else 1.0) * pdt
        P[i, j, 3 + k] += min(max(dv, -mx), mx)
    ml = T[i, LM0 + j]
    vy = ml * P[i, j, 4]
    cn = T[i, LN0 + j] / LEAN_MAX
    if T[i, CX0 + j] <= 0.0 and a[11] > 0.3 and abs(a[1]) > 0.5:
        T[i, CX0 + j] = 0.3 - 0.08 * agi
        T[i, CD0 + j] = 1.0 if a[1] > 0.0 else -1.0
    if T[i, CX0 + j] > 0.0:
        cd = T[i, CD0 + j]
        vt = cd * vmax * CROSS_V
        g = CROSS_ACC if (vt - vy) * cd >= 0.0 else 0.25
        lt = cd * LEAN_MAX
        T[i, CX0 + j] -= pdt
    else:
        vt = des[1]
        s = vt - vy
        if cn * s >= 0.0:
            g = 1.0 + 0.5 * abs(cn)
        else:
            g = max(0.05, 1.0 - 1.3 * abs(cn))
        lt = min(max(a[9] * LEAN_MAX + 0.25 * LEAN_MAX * vy / vmax, -LEAN_MAX), LEAN_MAX)
    mx = acc * g * pdt
    vy += min(max(vt - vy, -mx), mx)
    P[i, j, 4] = ml * vy
    dl = lt - T[i, LN0 + j]
    T[i, LN0 + j] += min(max(dl, -lrate * pdt), lrate * pdt)
    for k in range(3):
        P[i, j, k] += P[i, j, 3 + k] * pdt


@njit(cache=True)
def step_env(i, B, P, I, T, S, F, A, rew, done, winner, reason_out, hits_out, hist, train, nsub, dt):
    rew[i, 0] = 0.0
    rew[i, 1] = 0.0
    done[i] = False
    winner[i] = -1
    reason_out[i] = R_NONE
    pdt = dt / nsub

    a = np.empty((2, ACT_DIM))
    des = np.empty((2, 3))
    acc = np.empty(2)
    srate = np.empty(2)
    vm = np.empty(2)
    lrate = np.empty(2)
    for j in range(2):
        m = 1.0 if j == 0 else -1.0
        for k in range(ACT_DIM):
            a[j, k] = min(max(A[i, j, k], -1.0), 1.0)
        agi = S[i, j, AGI]
        fat = F[i, j]
        vmax = vmax_of(agi, fat)
        vm[j] = vmax
        acc[j] = (4.0 + 8.0 * agi) * (1.0 - 0.3 * fat)
        lrate[j] = (1.3 + 1.8 * agi) * LEAN_MAX
        des[j, 0] = m * a[j, 0] * vmax
        des[j, 1] = a[j, 1] * vmax
        des[j, 2] = a[j, 2] * vmax * 0.7
        rew[i, j] -= 0.0005 * (a[j, 0] ** 2 + a[j, 1] ** 2 + a[j, 2] ** 2)
        if a[j, 8] > HAND_FORCE:
            T[i, BT0 + j] = 1.0
        elif a[j, 8] < -HAND_FORCE:
            T[i, BT0 + j] = -1.0
        elif (I[i, PHASE] == PH_FLY and I[i, TURN] == j and I[i, PVALID] == 1
              and T[i, RT] - T[i, EVT] >= react_delay(S[i, j, REA])):
            T[i, BT0 + j] = 1.0 if T[i, LM0 + j] * (T[i, PY] - P[i, j, 1]) < -0.07 else -1.0
        srate[j] = 2.0 / ((0.2 + 0.25 * (1.0 - agi)) * (1.0 + 0.4 * fat))

    for s in range(nsub):
        if I[i, PHASE] == PH_FLY:
            update_contact(i, P, I, T)
        for j in range(2):
            recovering = T[i, REC0 + j] > 0.0
            acc_j = acc[j] * REC_ACC if recovering else acc[j]
            if T[i, DV0 + j] > 0.0:
                acc_j = acc[j] * DIVE_ACC
                T[i, DV0 + j] -= pdt
            srate_j = srate[j] / STROKE_P[int(T[i, LK0 + j]), 9] if recovering else srate[j]
            if recovering:
                T[i, REC0 + j] -= pdt
            was_set = T[i, SET0 + j] >= SETTLE_T
            if not recovering and (was_set or T[i, SPL0 + j] > 0.0):
                acc_j *= SPLIT_ACC
            T[i, SPL0 + j] = max(T[i, SPL0 + j] - pdt, 0.0)
            move_player(i, j, a[j], des[j], acc_j, vm[j], lrate[j], S[i, j, AGI], P, T, pdt)
            spd = math.sqrt(P[i, j, 3] ** 2 + P[i, j, 4] ** 2)
            if spd < 0.35 * vm[j] and T[i, CX0 + j] <= 0.0:
                T[i, SET0 + j] = min(T[i, SET0 + j] + pdt, 1.0)
            else:
                if was_set:
                    T[i, SPL0 + j] = SPLIT_T
                T[i, SET0 + j] = 0.0
            if I[i, PHASE] == PH_FLY and I[i, TURN] == j:
                dc = a[j, 10] * LAT_MAX - T[i, CU0 + j]
                T[i, CU0 + j] += min(max(dc, -CUE_RATE * pdt), CUE_RATE * pdt)
            d = T[i, BT0 + j] - T[i, BL0 + j]
            ds = srate_j * pdt
            T[i, BL0 + j] += min(max(d, -ds), ds)
            clamp_paddle(P, T, i, j)
            if I[i, PHASE] == PH_SERVE and j != I[i, SERVER]:
                clamp_receiver(P, T, i, j)

        ended = False
        ph = I[i, PHASE]
        if ph == PH_SERVE:
            srv = I[i, SERVER]
            sk = int(T[i, SPLAN0 + srv]) if USE_TACTIC else serve_type_of(a[srv, 8], a[srv, 10])
            sbh = serve_is_bh(sk)
            T[i, BL0 + srv] = -1.0 if sbh else 1.0
            T[i, BT0 + srv] = T[i, BL0 + srv]
            if T[i, TOSS] < 0.0:
                tz = serve_toss(int(T[i, SPLAN0 + srv]) if USE_TACTIC else serve_type(a[srv]), a[srv, 9])
                if T[i, TIMER] <= toss_time(tz):
                    # 抛球出手：抛球高度锁定，此后站位不再移动
                    T[i, TOSS] = tz
                else:
                    dy = serve_spot(lat_m(S, i, srv), a[srv, 1], sk) - B[i, 1]
                    B[i, 1] += min(max(dy, -1.2 * pdt), 1.2 * pdt)
            place_server(i, srv, B, P, S, sbh)
            T[i, TIMER] -= pdt
            if T[i, TIMER] <= 0.0:
                if serve_hit(i, a[srv], B, I, T, S, F):
                    end_point(i, 1 - srv, R_FAULT, train, I, T, F, rew, done, winner, reason_out, hits_out)
                    ended = True
                else:
                    I[i, PHASE] = PH_FLY
                    predict(i, B, I, T, S)
                    # 接发球方认出旋转的时间：熟悉度越高越快，反应越快也越快
                    T[i, EVT] += ((FAM_SURPRISE - (FAM_SURPRISE + FAM_READ) * T[i, SFAM])
                                  * (1.4 - 0.8 * S[i, 1 - srv, REA]))
        elif ph == PH_FLY:
            x0, y0, z0 = B[i, 0], B[i, 1], B[i, 2]
            ev = ball_substep(B[i], pdt)
            rec = I[i, TURN]
            hitter = 1 - rec
            if ev == 2:
                I[i, NETT] = 1
                predict(i, B, I, T, S)
            elif ev == 1:
                on_rec = (B[i, 0] < 0.0) if rec == 0 else (B[i, 0] > 0.0)
                if I[i, HITS] == 0 and I[i, SVB] == 0:
                    if on_rec:
                        end_point(i, rec, R_SFAULT, train, I, T, F, rew, done, winner, reason_out, hits_out)
                        ended = True
                    else:
                        I[i, SVB] = 1
                elif not on_rec:
                    end_point(i, rec, R_NET if I[i, NETT] else R_OWN, train, I, T, F,
                              rew, done, winner, reason_out, hits_out)
                    ended = True
                elif I[i, HITS] == 0 and I[i, NETT] and I[i, BOUNCED] == 0:
                    end_point(i, -1, R_LET, train, I, T, F, rew, done, winner, reason_out, hits_out)
                    ended = True
                else:
                    I[i, BOUNCED] += 1
                    if I[i, BOUNCED] == 1:
                        rew[i, hitter] += RALLY_W + T[i, STB]
                        T[i, STB] = 0.0
                    else:
                        end_point(i, hitter, R_MISS, train, I, T, F, rew, done, winner, reason_out, hits_out)
                        ended = True
            if not ended and I[i, BOUNCED] == 1 and T[i, DV0 + rec] <= 0.0:
                ext = DIVE_EXT if ((a[rec, 5] + 1.0) * 0.5 > DIVE_PW and T[i, BL0 + rec] > 0.5) else 0.0
                px, py, stretch = reach_hand(P, T, i, rec, contact_x(B, I, T, S, i, rec), B[i, 1], ext)
                pz = P[i, rec, 2]
                d = seg_dist(x0 - px, y0 - py, z0 - pz, B[i, 0] - px, B[i, 1] - py, B[i, 2] - pz)
                if d < REACH:
                    do_hit(i, rec, a[rec], stretch, B, P, I, T, S, F)
            if not ended:
                bx, by, bz = B[i, 0], B[i, 1], B[i, 2]
                if bz < TABLE_H - 0.3 or abs(bx) > 4.5 or abs(by) > 3.0 or ev == 3:
                    rec = I[i, TURN]
                    if I[i, BOUNCED] == 0:
                        end_point(i, rec, R_NET if I[i, NETT] else R_OUT, train, I, T, F,
                                  rew, done, winner, reason_out, hits_out)
                    else:
                        end_point(i, 1 - rec, R_MISS, train, I, T, F, rew, done, winner, reason_out, hits_out)
                    ended = True
            T[i, RT] += pdt
            if not ended and T[i, RT] > (RALLY_TIMEOUT if train else RALLY_TIMEOUT_PLAY):
                end_point(i, -1, R_TIMEOUT, train, I, T, F, rew, done, winner, reason_out, hits_out)
                ended = True
        else:
            ball_substep(B[i], pdt)
            T[i, TIMER] -= pdt
            if T[i, TIMER] <= 0.0:
                start_point(i, False, B, P, I, T, S, F)

        for k in range(3):
            hist[i, s, k] = B[i, k]
            hist[i, s, 3 + k] = P[i, 0, k]
            hist[i, s, 6 + k] = P[i, 1, k]
            hist[i, s, 9 + k] = P[i, 0, k]
            hist[i, s, 12 + k] = P[i, 1, k]
        for j in range(2):
            hx, hy = shown_hand(B, P, I, T, S, i, j)
            hist[i, s, 9 + 3 * j] = hx
            hist[i, s, 10 + 3 * j] = hy
        hist[i, s, 15] = T[i, BL0]
        hist[i, s, 16] = T[i, BL1]
        for j in range(2):
            hist[i, s, 17 + j] = T[i, LN0 + j] / LEAN_MAX
            hist[i, s, 19 + j] = T[i, CD0 + j] if T[i, CX0 + j] > 0.0 else 0.0
            hist[i, s, 21 + j] = T[i, CU0 + j]
        cueing = I[i, PHASE] == PH_FLY
        hist[i, s, 23] = float(I[i, TURN]) if cueing else -1.0

        if ended and train:
            start_point(i, True, B, P, I, T, S, F)
            for s2 in range(s + 1, nsub):
                for k in range(HIST_DIM):
                    hist[i, s2, k] = hist[i, s, k]
            return

    if I[i, PHASE] == PH_FLY:
        rec = I[i, TURN]
        for j in range(2):
            sj = -1.0 if j == 0 else 1.0
            if j == rec:
                if I[i, PVALID] == 1 and T[i, RT] - T[i, EVT] >= react_delay(S[i, j, REA]):
                    hx, hy, _ = reach_hand(P, T, i, j, T[i, PX], T[i, PY])
                    d = math.sqrt((T[i, PX] - hx) ** 2 + (T[i, PY] - hy) ** 2 + (T[i, PZ] - P[i, j, 2]) ** 2)
                    if T[i, SH0 + j] >= 0.0:
                        rew[i, j] += 0.3 * (T[i, SH0 + j] - d)
                    T[i, SH0 + j] = d
            else:
                d = math.sqrt((sj * (HALF_L + 0.6) - P[i, j, 0]) ** 2 + P[i, j, 1] ** 2
                              + (1.0 - P[i, j, 2]) ** 2)
                rew[i, j] -= READY_W * d

    for j in range(2):
        vb = vmax_of(S[i, j, AGI], 0.0)
        sp2 = (P[i, j, 3] ** 2 + P[i, j, 4] ** 2 + P[i, j, 5] ** 2) / (vb * vb)
        F[i, j] = min(1.0, F[i, j] + dt * FAT_RUN * (FAT_BASE - S[i, j, STA]) * sp2)


@njit(parallel=True, cache=True)
def step_kernel(B, P, I, T, S, F, A, rew, done, winner, reason_out, hits_out, hist, train, nsub, dt):
    for i in prange(B.shape[0]):
        step_env(i, B, P, I, T, S, F, A, rew, done, winner, reason_out, hits_out, hist, train, nsub, dt)


@njit(cache=True)
def reset_kernel(B, P, I, T, S, F, train):
    for i in range(B.shape[0]):
        for k in range(NI):
            I[i, k] = 0
        I[i, FIRST] = np.random.randint(0, 2)
        I[i, LWIN] = -1
        F[i, 0] = 0.0
        F[i, 1] = 0.0
        for k in range(2 * N_SERVE):
            T[i, FAM0 + k] = 0.0
        reset_serve_stats(i, T)
        reset_mind(i, T)
        start_point(i, train, B, P, I, T, S, F)


@njit(parallel=True, cache=True)
def obs_kernel(B, P, I, T, S, F, O):
    for i in prange(B.shape[0]):
        ph = I[i, PHASE]
        rec = I[i, TURN]
        for j in range(2):
            m = 1.0 if j == 0 else -1.0
            o = O[i, j]
            q = 1 - j
            ml = T[i, LM0 + j]
            hs = ml * m
            # 对手视角横向量换到我的视角：两个换算系数相乘（都右手时为 -1）
            rel = ml * T[i, LM0 + q]
            o[0] = m * P[i, j, 0]
            o[1] = ml * P[i, j, 1]
            o[2] = P[i, j, 2]
            o[3] = m * P[i, j, 3] * 0.25
            o[4] = ml * P[i, j, 4] * 0.25
            o[5] = P[i, j, 5] * 0.25
            o[6] = m * B[i, 0]
            o[7] = ml * B[i, 1]
            o[8] = B[i, 2]
            o[9] = m * B[i, 3] * 0.1
            o[10] = ml * B[i, 4] * 0.1
            o[11] = B[i, 5] * 0.1
            o[12] = hs * m * B[i, 6] * 0.01
            o[13] = m * B[i, 7] * 0.01
            o[14] = hs * B[i, 8] * 0.01
            o[15] = m * P[i, q, 0]
            o[16] = ml * P[i, q, 1]
            o[17] = P[i, q, 2]
            mine = ph == PH_FLY and rec == j
            o[18] = 1.0 if mine else 0.0
            o[19] = float(I[i, BOUNCED]) if mine else 0.0
            o[20] = 1.0 if ph == PH_SERVE else 0.0
            vis = mine and I[i, PVALID] == 1 and T[i, RT] - T[i, EVT] >= react_delay(S[i, j, REA])
            if vis:
                o[21] = 1.0
                hx, hy, _ = reach_hand(P, T, i, j, T[i, PX], T[i, PY])
                o[22] = m * (T[i, PX] - hx)
                o[23] = ml * (T[i, PY] - hy)
                o[24] = T[i, PZ] - P[i, j, 2]
                o[25] = min(max(T[i, PRT] - T[i, RT], 0.0), 1.5)
            else:
                for k in range(21, 26):
                    o[k] = 0.0
            for k in range(N_STATS):
                o[26 + k] = S[i, j, k]
                o[31 + k] = S[i, q, k]
            o[36] = F[i, j]
            o[37] = F[i, q]
            o[38] = 1.0 if I[i, SERVER] == j else 0.0
            o[39] = T[i, BL0 + j]
            o[40] = T[i, BL0 + q]
            o[41] = ml * (T[i, PY] - P[i, j, 1]) if vis else 0.0
            o[42] = m * (T[i, PX] - P[i, j, 0]) if vis else 0.0
            o[43] = T[i, LN0 + j] / LEAN_MAX
            o[44] = T[i, CD0 + j] * T[i, CX0 + j] / 0.3
            o[45] = rel * T[i, LN0 + q] / LEAN_MAX
            o[46] = rel * T[i, CD0 + q] * T[i, CX0 + q] / 0.3
            opp_cue = ph == PH_FLY and rec == q
            o[47] = 1.0 if opp_cue else 0.0
            o[48] = rel * T[i, CU0 + q] if opp_cue else 0.0
            o[49] = T[i, CU0 + j]
            for k in range(N_STYLE):
                o[50 + k] = S[i, j, N_STATS + k]
            o[54] = min(T[i, SET0 + j] / SETTLE_T, 1.0)
            o[55] = max(T[i, REC0 + j], 0.0) / 0.35
            o[56] = max(T[i, REC0 + q], 0.0) / 0.35
            is_serve = ph == PH_FLY and I[i, HITS] == 0 and I[i, LSRV] >= 0
            for k in range(N_SERVE):
                o[57 + k] = 1.0 if is_serve and I[i, LSRV] == k else 0.0
            o[65] = T[i, TOSS] if is_serve else 0.0
            o[66] = float(I[i, LHID]) if is_serve else 0.0
            o[67] = 1.0 if is_serve else 0.0
            o[68] = m * P[i, q, 3] * 0.25
            o[69] = ml * P[i, q, 4] * 0.25
            o[70] = min(T[i, SET0 + q] / SETTLE_T, 1.0)
            for k in range(N_SERVE):
                o[71 + k] = T[i, FAM0 + q * N_SERVE + k]
                o[79 + k] = T[i, FAM0 + j * N_SERVE + k]
                if USE_TACTIC and ph == PH_SERVE and I[i, SERVER] == j and int(T[i, SPLAN0 + j]) == k:
                    o[57 + k] = 1.0
            # 追加在末尾：双方的发球技术。保持既有 87 维不变，热启动可以直接沿用
            o[87] = S[i, j, SRV]
            o[88] = S[i, q, SRV]
            # 器材：对手是否与我同手（-1 为一左一右）、双方握法、四面胶皮的弹性/摩擦/咬球
            o[89] = hs * hs_of(S, i, q)
            o[90] = S[i, j, GRIP]
            o[91] = S[i, q, GRIP]
            for f in range(4):
                r = int(S[i, j if f < 2 else q, RFH if f % 2 == 0 else RBH])
                o[92 + 3 * f] = (RUB_P[r, 0] - 0.8) * 5.0
                o[93 + 3 * f] = RUB_P[r, 1]
                o[94 + 3 * f] = RUB_P[r, 2]
            # 追加：双方的心态、手感（新增在末尾，热启动可直接沿用旧维度）
            o[104] = S[i, j, MEN]
            o[105] = S[i, j, TCH]
            o[106] = S[i, q, MEN]
            o[107] = S[i, q, TCH]
