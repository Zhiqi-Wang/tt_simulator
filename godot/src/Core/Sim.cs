namespace PingPong;

/// <summary>模拟的常量与状态数组下标（与 pingpong/sim.py、core/ppcore.cpp 一致；启动时用 pp_dims 校验）。</summary>
public static class Sim
{
    public const double G = 9.81;
    public const double BALL_R = 0.02;
    public const double HALF_L = 1.37, HALF_W = 0.7625, TABLE_H = 0.76;
    public const double NET_H = 0.1525, NET_HALF_W = 0.9144;

    public const double FH_CLO = -0.48, FH_CHI = -0.18, BH_CLO = -0.05, BH_CHI = 0.28;
    public const double FWD_CLO = -0.15, FWD_CHI = 0.3;
    public const double DEPTH_MIN = 0.3;
    public const double DIVE_PW = 0.6, DIVE_ST = 0.12, DIVE_T = 0.9;
    public const double SERVE_WAIT = 1.8;
    public const double ADAPT_MAX = 0.7;
    public const double MEN_SIG = 1.2;
    public const bool USE_TACTIC = true;
    public const int GAME_PTS = 5, MATCH_WIN = 2;

    // 球员属性 S[j, *]
    public const int STR = 0, AGI = 1, TEC = 2, REA = 3, STA = 4, N_STATS = 5;
    public const int N_STYLE = 4, SRV = 9;
    public const int HAND = 10, GRIP = 11, RFH = 12, RBH = 13, MEN = 14, TCH = 15, N_S = 16;
    public static readonly int[] STAT_ORDER = { STR, AGI, TEC, SRV, REA, STA, MEN, TCH };
    public static readonly string[] STAT_ORDER_NAMES = { "力量", "敏捷", "技术", "发球", "反应", "体力", "心态", "适应" };
    public static readonly string[] STYLE_NAMES = { "反手偏好", "旋转偏好", "力量偏好", "大角度偏好" };
    public static readonly string[] GRIP_NAMES = { "横板", "直板" };
    public const int RUB_INV = 0, RUB_PIPS = 1, RUB_RAW = 2, RUB_LONG = 3, RUB_ANTI = 4, N_RUB = 5;
    public static readonly string[] RUB_NAMES = { "反胶", "正胶", "生胶", "长胶", "防弧" };

    // 击球大类与技术细分
    public const int ST_LOOP = 0, ST_DRIVE = 1, ST_BLOCK = 2, ST_PUSH = 3, ST_FLICK = 4, ST_CHOP = 5, ST_SMASH = 6,
        ST_LOB = 7;
    public static readonly string[] STROKE_NAMES = { "拉球", "快攻", "挡球", "搓球", "挑打", "削球", "扣杀", "放高球" };
    public const int TQ_LOOP = 0, TQ_LIFT = 1, TQ_DRIVELOOP = 2, TQ_RIP = 3, TQ_DRIVE = 4, TQ_BLOCK = 5,
        TQ_SIDECUT = 6, TQ_SHORT = 7, TQ_LONG = 8, TQ_PUSH = 9, TQ_BANANA = 10, TQ_TWIST = 11, TQ_WIPE = 12,
        TQ_FLICK = 13, TQ_CHOP = 14, TQ_SMASH = 15, TQ_LOB = 16;
    public static readonly string[] TECH_NAMES =
    {
        "拉球", "提拉", "前冲弧圈", "撕", "快攻", "挡球", "侧切", "摆短", "劈长", "搓球",
        "霸王拧", "拧拉", "抹", "挑打", "削球", "扣杀", "放高球",
    };
    public static bool TechNoSide(int t) => t == TQ_RIP || t == TQ_SIDECUT || t == TQ_BANANA || t == TQ_TWIST;

    // 发球
    public const int SV_PEND = 0, SV_HOOK = 1, SV_SHOVEL = 2, SV_SQUAT = 3, SV_CHOP = 4, SV_FAST = 5, SV_BH = 6,
        SV_BHFAST = 7, N_SERVE = 8, N_VAR = 3;
    public static readonly string[] SERVE_NAMES =
    {
        "正手侧旋发球", "勾手发球", "逆旋转发球", "下蹲砍式发球", "砍式发球", "正手奔球", "反手侧旋发球", "反手急球",
    };
    public static readonly double[] VAR_TOP = { -1.0, 0.0, 1.0 };

    // 整数状态 I
    public const int TURN = 0, BOUNCED = 1, NETT = 2, HITS = 3, SERVER = 4, PHASE = 5, SC0 = 6, SC1 = 7, G0 = 8,
        G1 = 9, FIRST = 10, LWIN = 11, LREASON = 12, PVALID = 13, LHIT = 14, LSIDE = 15, LFAKE = 16, LSTK = 17,
        LSRV = 18, LHID = 19, SVB = 20, M0 = 21, M1 = 22, LTECH = 23, NI = 24;

    // 浮点状态 T
    public const int RT = 0, TIMER = 1, EVT = 2, PRT = 3, PX = 4, PY = 5, PZ = 6, SH0 = 7, SH1 = 8, BL0 = 9, BL1 = 10,
        BT0 = 11, BT1 = 12, LSW = 13, LN0 = 14, LN1 = 15, CX0 = 16, CX1 = 17, CD0 = 18, CD1 = 19, CU0 = 20, CU1 = 21,
        LFT = 22, SET0 = 23, SET1 = 24, STB = 25, REC0 = 26, REC1 = 27, LK0 = 28, LK1 = 29, DEC = 30, TOSS = 31,
        LBODY = 32, SPL0 = 33, SPL1 = 34, LPREP = 35, EX = 36, EY = 37, EZ = 38, ETH = 39, LX = 40, LY = 41, LZ = 42,
        LTH = 43, SFAM = 44, FAM0 = 45;
    public const int DBG0 = FAM0 + 2 * N_SERVE;
    public const int SW0 = DBG0 + 9;
    public const int SN0 = SW0 + 2 * N_SERVE;
    public const int SPLAN0 = SN0 + 2 * N_SERVE;
    public const int VW0 = SPLAN0 + 2;
    public const int VN0 = VW0 + 2 * N_VAR;
    public const int VPLAN0 = VN0 + 2 * N_VAR;
    public const int PCHAIN = VPLAN0 + 2;
    public const int PN0 = PCHAIN + 1;
    public const int LRUB = PN0 + 6;
    public const int LM0 = PN0 + 7;
    public const int DV0 = PN0 + 9;
    public const int STK0 = PN0 + 11;
    public const int FRU0 = PN0 + 13;
    public const int ADP0 = PN0 + 15;
    public const int ORB0 = PN0 + 17;
    public const int NT = PN0 + 19;

    public const int PH_SERVE = 0, PH_FLY = 1, PH_DEAD = 2;
    public const int R_NONE = 0, R_OUT = 1, R_NET = 2, R_OWN = 3, R_MISS = 4, R_TIMEOUT = 5, R_FAULT = 6, R_LET = 7,
        R_SFAULT = 8;
    public static readonly string[] REASON_TEXT =
    {
        "", "出界", "下网", "落在己方台面", "没接到", "超时", "发球违例（回抛）", "擦网重发", "发球违例（没先落本方台面）",
    };

    public const int OBS_DIM = 108, ACT_DIM = 12, HIST_DIM = 24;
}
