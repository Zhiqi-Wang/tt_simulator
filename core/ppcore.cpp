// 乒乓球物理引擎 + 比赛规则（pingpong/sim.py 的 C++ 版，逐行对应）。
// 训练（Python ctypes）和游戏（Godot C# P/Invoke）共用这一份。
// 数组布局与 numpy 版完全一致（行优先）：B[n,9] P[n,2,6] I[n,NI](int64) T[n,NT] S[n,2,N_S] F[n,2]
// A[n,2,ACT_DIM] rew[n,2] done[n](uint8) winner/reason/hits[n](int64) hist[n,nsub,HIST_DIM] O[n,2,OBS_DIM]
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifdef _WIN32
#define PP_API extern "C" __declspec(dllexport)
#else
#define PP_API extern "C" __attribute__((visibility("default")))
#endif

namespace {

// ---------------------------------------------------------------- 常数
constexpr double PI = 3.14159265358979323846;
constexpr double G = 9.81;
constexpr double BALL_R = 0.02;
constexpr double BALL_M = 0.0027;
constexpr double HALF_L = 1.37;
constexpr double HALF_W = 0.7625;
constexpr double TABLE_H = 0.76;
constexpr double NET_H = 0.1525;
constexpr double NET_HALF_W = 0.9144;
constexpr double AIR_RHO = 1.2;
const double KD = 0.5 * AIR_RHO * 0.5 * PI * BALL_R * BALL_R / BALL_M;
const double KM = 0.5 * AIR_RHO * 0.6 * PI * BALL_R * BALL_R * BALL_R / BALL_M;
constexpr double S_SAT = 0.5;
constexpr double E_TABLE = 0.89;
constexpr double MU_TABLE = 0.25;
constexpr double K_INERTIA = 2.0 / 3.0;
constexpr double SPIN_DECAY = 0.05;
constexpr double E_FLOOR = 0.6;
constexpr double E_PAD = 0.86;
constexpr double MU_PAD = 0.9;
constexpr double K_STICK = K_INERTIA / (1.0 + K_INERTIA);
constexpr double REACH = 0.14;
constexpr double RALLY_TIMEOUT = 20.0;
constexpr double RALLY_TIMEOUT_PLAY = 90.0;
constexpr double HAND_MID = -0.13;
constexpr double HAND_HALF = -0.19;
constexpr double FH_LO = -0.68, FH_HI = -0.08;
constexpr double BH_LO = -0.12, BH_HI = 0.45;
constexpr double FH_CLO = -0.48, FH_CHI = -0.18;
constexpr double BH_CLO = -0.05, BH_CHI = 0.28;
constexpr double FWD_LO = -0.2, FWD_HI = 1.1;
constexpr double DEPTH_MIN = 0.3;
constexpr double HIGH_REACH = 0.35;
constexpr double FWD_STRETCH_W = 0.5;
constexpr double FWD_CLO = -0.15, FWD_CHI = 0.3;
constexpr double BODY_MIN = HALF_L + 0.1;
constexpr double READY_W = 0.003;
constexpr double RECV_X0 = 0.3, RECV_X1 = 0.7, RECV_Y = 0.1, RECV_DY = 0.25;
constexpr double WINNER_BONUS = 1.0;
constexpr double RALLY_W = 0.08;
constexpr double LEAN_MAX = 0.12;
constexpr double FAKE_DEV = 0.3, FAKE_LEAN = 0.7, FAKE_NEED = 0.3;
constexpr double CROSS_V = 1.5;
constexpr double CROSS_ACC = 1.15;
constexpr double CUE_RATE = 2.0;
constexpr double LAT_MAX = 0.7;
constexpr double SETTLE_T = 0.25;
constexpr double SPLIT_ACC = 1.8;
constexpr double SPLIT_T = 0.3;
constexpr double HAND_FORCE = 0.5;
constexpr double STEPIN_D = 0.8, STEPIN_V = 1.4, STEPIN_ACC = 1.6;
constexpr double STYLE_W = 0.04;

constexpr int STR = 0, AGI = 1, TEC = 2, REA = 3, STA = 4;
constexpr int N_STATS = 5;
constexpr int STY_BH = 5, STY_SPIN = 6, STY_POW = 7, STY_WIDE = 8;
constexpr int N_STYLE = 4;
constexpr int SRV = 9;
constexpr int HAND = 10, GRIP = 11, RFH = 12, RBH = 13;
constexpr int MEN = 14, TCH = 15;
constexpr int N_S = N_STATS + N_STYLE + 1 + 4 + 2;

constexpr int RUB_INV = 0, RUB_PIPS = 1, RUB_RAW = 2, RUB_LONG = 3, RUB_ANTI = 4;
constexpr int N_RUB = 5;
// 以下几项在 numba 版里可用环境变量调，这里用 pp_set_param 调
double RUB_DEC[N_RUB] = {1.0, 1.3, 1.3, 3.5, 2.8};
double RUB_SENS[N_RUB] = {1.0, 0.7, 0.55, 0.35, 0.3};
const double RUB_ODD[N_RUB] = {0.0, 0.3, 0.3, 1.0, 0.8};
double MEN_SIG = 1.2;
constexpr double CLUTCH_GP = 0.5, CLUTCH_MP = 1.0, TILT_PT = 0.2, TILT_MAX = 0.8;
constexpr double FRUS_ADD = 0.35, FRUS_DECAY = 0.8;
double TCH_READ = 1.6;
constexpr double TCH_SENS = 0.6, TCH_SMALL = 0.4;
constexpr double ADAPT0 = 0.02, ADAPT1 = 0.06, ADAPT_MAX = 0.7;
const double RUB_P[N_RUB][3] = {
    {0.86, 1.20, 1.45},
    {0.88, 0.55, 0.80},
    {0.90, 0.40, 0.55},
    {0.72, 0.14, 0.15},
    {0.70, 0.20, 0.25},
};
constexpr double BAL_FRAC = 0.5;
const double RUB_FH_P[N_RUB] = {0.82, 0.10, 0.06, 0.01, 0.01};
const double RUB_BH_P[N_RUB] = {0.60, 0.10, 0.10, 0.13, 0.07};
constexpr double LEFTY_P = 0.25, PEN_P = 0.25;
constexpr double PEN_FH_SHORT = 0.85, PEN_SERVE = 0.88, PEN_DEC = 1.1;
constexpr double PEN_BH_SWING = 0.88, PEN_BH_SIG = 1.12, PEN_BH_SPIN = 0.9, PEN_PUSHBLOCK_SIG = 0.92;

constexpr int GAME_PTS = 5;
constexpr int MATCH_WIN = 2;
constexpr double FAT_BASE = 1.25;
constexpr double FAT_RUN = 0.012;
constexpr double FAT_HIT = 0.0032;
constexpr double FAT_REC_PT = 0.005;
constexpr double FAT_REC_GAME = 0.08;

constexpr int ST_LOOP = 0, ST_DRIVE = 1, ST_BLOCK = 2, ST_PUSH = 3, ST_FLICK = 4, ST_CHOP = 5, ST_SMASH = 6,
              ST_LOB = 7;
constexpr int TQ_LOOP = 0, TQ_LIFT = 1, TQ_DRIVELOOP = 2, TQ_RIP = 3, TQ_DRIVE = 4, TQ_BLOCK = 5, TQ_SIDECUT = 6,
              TQ_SHORT = 7, TQ_LONG = 8, TQ_PUSH = 9, TQ_BANANA = 10, TQ_TWIST = 11, TQ_WIPE = 12, TQ_FLICK = 13,
              TQ_CHOP = 14, TQ_SMASH = 15, TQ_LOB = 16;
const double STROKE_P[8][11] = {
    {1.00, 1.00, 1.00, 1.00, 99.0, 0.00, 1.0, 1.0, 0.30, 1.3, 0.50},
    {0.85, 0.60, 0.90, 0.70, 99.0, 0.00, 0.6, 0.6, 0.18, 1.0, 0.30},
    {0.45, 0.40, 0.70, 0.35, 99.0, 0.00, 0.2, 0.4, 0.08, 0.7, 0.10},
    {0.40, 1.00, 0.45, 0.80, 6.0, 0.00, 0.3, 0.8, 0.12, 0.8, 0.10},
    {0.60, 0.80, 1.20, 1.00, 99.0, 0.00, 0.4, 0.8, 0.15, 0.9, 0.20},
    {0.70, 1.00, 0.85, 0.50, 9.0, 0.03, 0.8, 1.0, 0.30, 1.2, 0.35},
    {1.25, 0.40, 1.10, 1.00, 99.0, 0.00, 1.0, 0.3, 0.35, 1.4, 0.50},
    {0.80, 0.80, 0.80, 0.60, 7.0, 0.60, 0.6, 0.8, 0.30, 1.2, 0.40},
};
constexpr double STK_SPIN0 = 150.0, STK_SPIN1 = 750.0;
constexpr double SRV_SPIN0 = 150.0, SRV_SPIN1 = 550.0;
constexpr double SPIN_REF = 150.0;
constexpr double REC_ACC = 0.3;
double DIVE_PW = 0.6;
constexpr double DIVE_ST = 0.12, DIVE_EXT = 0.3;
constexpr double DIVE_T = 0.9, DIVE_REC = 0.4, DIVE_ACC = 0.05;
constexpr double BH_BODY = 0.6;
const double PREP_T[8] = {0.38, 0.22, 0.18, 0.20, 0.25, 0.35, 0.40, 0.30};
constexpr double RUN_SIG = 0.8;
constexpr double LACK_MARGIN = 0.12;
constexpr double LACK_SIG = 0.7;
constexpr double SMASH_V = 1.25;
double ERR_SCALE = 0.6;
double READ_SCALE = 0.6;
constexpr int PLAN_BISECT = 8;
constexpr double EDGE_W = 0.6 * BALL_R;
constexpr double SERVE_WAIT = 1.8;
constexpr double TOSS_H0 = 0.3, TOSS_H1 = 1.5;
constexpr double FAST_TOSS_MAX = 0.25;
constexpr double PROBE_DT = 1.0 / 120.0;
const double BORROW[8] = {0.25, 0.45, 0.7, 0.2, 0.3, 0.15, 0.0, 0.2};
constexpr double ABSORB_SIG = 1.0;
const double VS_BACK[8] = {0.7, 1.2, 6.0, 0.0, 1.2, 0.0, 0.8, 0.3};
const double VS_TOP[8] = {0.2, 0.4, 1.0, 1.0, 0.6, 0.3, 0.3, 0.2};
constexpr double NOPACE_V = 9.0;
constexpr double NOPACE_MARGIN = 0.12;
constexpr double POP_MARGIN = 0.6;
constexpr double POP_CHAIN = 0.5;
constexpr double POP_CAP = 0.45;
constexpr double HI_Z0 = 0.12, HI_Z1 = 0.35;
constexpr double HI_BACK = 0.7;
constexpr double HI_SIG = 0.35;
constexpr double SMASH_Z = 0.30;

constexpr int SV_PEND = 0, SV_HOOK = 1, SV_SHOVEL = 2, SV_SQUAT = 3, SV_CHOP = 4, SV_FAST = 5, SV_BH = 6,
              SV_BHFAST = 7;
constexpr int N_SERVE = 8;
const double SERVE_P[N_SERVE][9] = {
    {1.00, -1.0, 0.80, 1.00, 1.00, 0.25, 1.0, -1.0, 1.0},
    {0.90, 1.0, 1.00, 0.95, 1.35, 0.30, 1.0, -1.0, 1.0},
    {1.05, 1.0, 1.00, 1.30, 1.15, 0.30, 1.0, -1.0, 1.0},
    {1.25, 0.0, 1.20, 1.30, 1.25, 0.50, 1.0, -1.0, 0.8},
    {1.00, 0.0, 0.25, 1.00, 1.10, 0.22, 1.15, -1.0, 0.25},
    {0.70, 1.0, 0.50, 1.00, 0.80, 0.20, 1.5, 0.3, 1.0},
    {0.80, 1.0, 0.90, 0.90, 0.90, 0.10, 0.9, -1.0, 1.0},
    {0.60, 0.0, 0.20, 1.00, 0.85, 0.10, 1.35, -1.0, 1.0},
};
const double SERVE_Z[N_SERVE] = {0.18, 0.18, 0.18, 0.20, 0.18, 0.12, 0.17, 0.12};
constexpr double FAST_DEPTH = 1.1, FAST_D1 = 1.2;
constexpr double FAULT_P = 0.05;
const double SERVE_PRIOR[N_SERVE] = {0.75, 0.75, 0.75, 0.74, 0.62, 0.60, 0.55, 0.55};
constexpr double PRIOR_N = 4.0;
constexpr double TAC_DECAY = 0.92;
constexpr double TAC_FAM = 0.25;
constexpr double TAC_TEMP = 0.06;
const double VAR_TOP[3] = {-1.0, 0.0, 1.0};
const double VAR_PRIOR[3] = {0.6, 0.6, 0.6};
constexpr double TAC_VREP = 0.2;

// I 的下标
enum {
    TURN, BOUNCED, NETT, HITS, SERVER, PHASE, SC0, SC1, G0, G1, FIRST, LWIN, LREASON, PVALID,
    LHIT, LSIDE, LFAKE, LSTK, LSRV, LHID, SVB, M0, M1, LTECH, NI
};
// T 的下标
enum {
    RT, TIMER, EVT, PRT, PX, PY, PZ, SH0, SH1, BL0, BL1, BT0, BT1, LSW,
    LN0, LN1, CX0, CX1, CD0, CD1, CU0, CU1, LFT, SET0, SET1, STB, REC0, REC1, LK0, LK1, DEC, TOSS, LBODY,
    SPL0, SPL1, LPREP, EX, EY, EZ, ETH, LX, LY, LZ, LTH, SFAM, FAM0
};
constexpr int DBG0 = FAM0 + 2 * N_SERVE;
constexpr int SW0 = DBG0 + 9;
constexpr int SN0 = SW0 + 2 * N_SERVE;
constexpr int SPLAN0 = SN0 + 2 * N_SERVE;
constexpr int N_VAR = 3;
constexpr int VW0 = SPLAN0 + 2;
constexpr int VN0 = VW0 + 2 * N_VAR;
constexpr int VPLAN0 = VN0 + 2 * N_VAR;
constexpr int PCHAIN = VPLAN0 + 2;
constexpr int PN0 = PCHAIN + 1;
constexpr int LRUB = PN0 + 6;
constexpr int LM0 = PN0 + 7;
constexpr int DV0 = PN0 + 9;
constexpr int STK0 = PN0 + 11;
constexpr int FRU0 = PN0 + 13;
constexpr int ADP0 = PN0 + 15;
constexpr int ORB0 = PN0 + 17;
constexpr int NT = PN0 + 19;
constexpr double FAM_RATE = 0.25;
constexpr double FAM_DEC_NEW = 1.6, FAM_DEC_OLD = 0.3;
constexpr double FAM_READ = 0.3, FAM_SURPRISE = 0.05;
constexpr double LATE_ZMIN = TABLE_H - 0.1;
constexpr double LATE_XMAX = HALF_L + 1.8;
constexpr double CONTACT_FWD = 0.1;
constexpr double LATE_SHORT_Z = 0.08;

constexpr int PH_SERVE = 0, PH_FLY = 1, PH_DEAD = 2;
constexpr int R_NONE = 0, R_OUT = 1, R_NET = 2, R_OWN = 3, R_MISS = 4, R_TIMEOUT = 5, R_FAULT = 6, R_LET = 7,
              R_SFAULT = 8;
constexpr int OBS_DIM = 108;
constexpr int ACT_DIM = 12;
constexpr int HIST_DIM = 24;

// ---------------------------------------------------------------- 随机数
// xoshiro256** + Box-Muller。tape 模式：均匀数、正态数各读一条外部给定的序列（和 numba 版逐步对照用）
struct Tape {
    const double* u = nullptr;
    int64_t nu = 0;
    const double* g = nullptr;
    int64_t ng = 0;
    int64_t* idx = nullptr;  // idx[0] 均匀数读到哪，idx[1] 正态数读到哪
};
Tape g_tape;

struct Rng {
    uint64_t s[4] = {1, 2, 3, 4};
    bool has_spare = false;
    double spare = 0.0;

    static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    void seed(uint64_t v) {
        for (int k = 0; k < 4; ++k) {  // splitmix64
            v += 0x9E3779B97F4A7C15ull;
            uint64_t z = v;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
            s[k] = z ^ (z >> 31);
        }
        has_spare = false;
    }
    uint64_t next() {
        uint64_t r = rotl(s[1] * 5, 7) * 9;
        uint64_t t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return r;
    }
    double random() {
        if (g_tape.u) {
            double v = g_tape.u[g_tape.idx[0] % g_tape.nu];
            g_tape.idx[0]++;
            return v;
        }
        return (next() >> 11) * (1.0 / 9007199254740992.0);
    }
    double normal() {
        if (g_tape.g) {
            double v = g_tape.g[g_tape.idx[1] % g_tape.ng];
            g_tape.idx[1]++;
            return v;
        }
        if (has_spare) {
            has_spare = false;
            return spare;
        }
        double u1, u2;
        do {
            u1 = random();
        } while (u1 <= 1e-300);
        u2 = random();
        double r = std::sqrt(-2.0 * std::log(u1));
        spare = r * std::sin(2.0 * PI * u2);
        has_spare = true;
        return r * std::cos(2.0 * PI * u2);
    }
    // [lo, hi)
    int64_t randint(int64_t lo, int64_t hi) {
        int64_t v = lo + (int64_t)(random() * (double)(hi - lo));
        return std::min(v, hi - 1);
    }
};

// ---------------------------------------------------------------- 小向量
struct V3 {
    double x = 0, y = 0, z = 0;
    V3() = default;
    V3(double a, double b, double c) : x(a), y(b), z(c) {}
    double operator[](int k) const { return k == 0 ? x : (k == 1 ? y : z); }
};
inline V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline V3 operator*(double s, V3 a) { return {s * a.x, s * a.y, s * a.z}; }
inline V3 operator*(V3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
inline V3 operator/(V3 a, double s) { return {a.x / s, a.y / s, a.z / s}; }
inline double dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline V3 cross3(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline double norm3(V3 a) { return std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z); }

inline double clampd(double v, double lo, double hi) { return std::min(std::max(v, lo), hi); }

// ---------------------------------------------------------------- 世界（一批并行的球台）
struct World {
    int n = 0;
    double* B = nullptr;
    double* P = nullptr;
    int64_t* I = nullptr;
    double* T = nullptr;
    double* S = nullptr;
    double* F = nullptr;
    std::vector<Rng>* rng = nullptr;

    double& b(int i, int k) const { return B[(int64_t)i * 9 + k]; }
    double* bp(int i) const { return B + (int64_t)i * 9; }
    double& p(int i, int j, int k) const { return P[((int64_t)i * 2 + j) * 6 + k]; }
    int64_t& in(int i, int k) const { return I[(int64_t)i * NI + k]; }
    double& t(int i, int k) const { return T[(int64_t)i * NT + k]; }
    double& s(int i, int j, int k) const { return S[((int64_t)i * 2 + j) * N_S + k]; }
    double& f(int i, int j) const { return F[(int64_t)i * 2 + j]; }
    Rng& r(int i) const { return (*rng)[i]; }
};

// ---------------------------------------------------------------- 球的物理
int ball_substep(double* b, double dt, Rng& rng) {
    double x = b[0], y = b[1], z = b[2];
    double vx = b[3], vy = b[4], vz = b[5];
    double wx = b[6], wy = b[7], wz = b[8];
    double sp = std::sqrt(vx * vx + vy * vy + vz * vz);
    double km = KM / (1.0 + BALL_R * std::sqrt(wx * wx + wy * wy + wz * wz) / (std::max(sp, 0.5) * S_SAT));
    double ax = -KD * sp * vx + km * (wy * vz - wz * vy);
    double ay = -KD * sp * vy + km * (wz * vx - wx * vz);
    double az = -KD * sp * vz + km * (wx * vy - wy * vx) - G;
    vx += ax * dt;
    vy += ay * dt;
    vz += az * dt;
    double nx = x + vx * dt;
    double ny = y + vy * dt;
    double nz = z + vz * dt;
    double decay = 1.0 - SPIN_DECAY * dt;
    wx *= decay;
    wy *= decay;
    wz *= decay;
    int ev = 0;

    if ((x < 0.0) != (nx < 0.0)) {
        double f = x / (x - nx);
        double zc = z + f * (nz - z);
        double yc = y + f * (ny - y);
        if (zc < TABLE_H + NET_H + BALL_R && zc > TABLE_H - BALL_R && std::fabs(yc) < NET_HALF_W + BALL_R) {
            double cord = (zc - (TABLE_H + NET_H - BALL_R)) / (2.0 * BALL_R);
            if (cord > 0.0 && rng.random() < std::pow(cord, 0.7)) {
                vx *= 0.25 + 0.35 * rng.random();
                vy = vy * 0.6 + rng.normal() * 0.25;
                vz = std::fabs(vz) * 0.3 + 0.3 + 1.2 * rng.random() * cord;
                nz = std::max(nz, TABLE_H + NET_H + BALL_R);
                wx *= 0.5;
                wy *= 0.5;
                wz *= 0.5;
            } else {
                double s = x < 0.0 ? -1.0 : 1.0;
                nx = s * BALL_R * 1.05;
                vx = -0.12 * vx;
                vy *= 0.5;
                vz *= 0.5;
                wx *= 0.3;
                wy *= 0.3;
                wz *= 0.3;
            }
            ev = 2;
        }
    }

    if (std::fabs(nx) <= HALF_L && std::fabs(ny) <= HALF_W && nz - BALL_R < TABLE_H && vz < 0.0 &&
        z - BALL_R >= TABLE_H - 0.01) {
        double vin = -vz;
        vz = E_TABLE * vin;
        double ux = vx - BALL_R * wy;
        double uy = vy + BALL_R * wx;
        double un = std::sqrt(ux * ux + uy * uy);
        if (un > 1e-9) {
            double jt = std::min(0.4 * un, MU_TABLE * (1.0 + E_TABLE) * vin);
            double hx = ux / un;
            double hy = uy / un;
            vx -= jt * hx;
            vy -= jt * hy;
            double c = jt / (K_INERTIA * BALL_R);
            wx -= c * hy;
            wy += c * hx;
        }
        nz = TABLE_H + BALL_R;
        ev = 1;
    } else if (nz - BALL_R < TABLE_H && vz < 0.0 && z - BALL_R >= TABLE_H - 0.01 &&
               std::fabs(nx) <= HALF_L + EDGE_W && std::fabs(ny) <= HALF_W + EDGE_W) {
        vz = E_TABLE * (-vz) * 0.35 * rng.random();
        vx *= 0.95;
        vy *= 0.95;
        nz = TABLE_H + BALL_R;
        ev = 1;
    } else if (nz < BALL_R && vz < 0.0) {
        vz = -E_FLOOR * vz;
        nz = BALL_R;
        vx *= 0.85;
        vy *= 0.85;
        ev = 3;
    }

    b[0] = nx; b[1] = ny; b[2] = nz;
    b[3] = vx; b[4] = vy; b[5] = vz;
    b[6] = wx; b[7] = wy; b[8] = wz;
    return ev;
}

V3 aim_velocity(double px, double py, double pz, double tx, double ty, double tz, double vh, double top,
                double side) {
    double dx = tx - px;
    double dy = ty - py;
    double d = std::sqrt(dx * dx + dy * dy) + 1e-6;
    double t = std::expm1(KD * d) / (KD * vh);
    double km = KM / (1.0 + BALL_R * std::sqrt(top * top + side * side) / (std::max(vh, 0.5) * S_SAT));
    double ge = clampd(G + km * top * vh, 2.0, 60.0);
    double vz = (tz - pz + 0.5 * ge * t * t) / t;
    double shift = 0.35 * km * side * vh * t * t;
    double ax = tx + shift * dy / d;
    double ay = ty - shift * dx / d;
    double ex = ax - px;
    double ey = ay - py;
    double e = std::sqrt(ex * ex + ey * ey) + 1e-6;
    return {ex / e * vh, ey / e * vh, vz};
}

V3 spin_vector(double vx, double vy, double top, double side) {
    double n = std::sqrt(vx * vx + vy * vy) + 1e-9;
    return {top * (-vy / n), top * (vx / n), side};
}

struct Probe {
    double l, y, clr;
};

// 脑中弹道用的球不会擦网（擦网分支要随机数）：numba 版这里也会抽随机数，所以必须用同一个发生器
Probe fly_probe(double px, double py, double pz, double vx, double vy, double vz, double top, double side,
                double s, int own, Rng& rng) {
    double b[9];
    V3 w = spin_vector(vx, vy, top, side);
    b[0] = px; b[1] = py; b[2] = pz; b[3] = vx; b[4] = vy; b[5] = vz; b[6] = w.x; b[7] = w.y; b[8] = w.z;
    double clr = -1.0;
    for (int k = 0; k < 300; ++k) {
        double x0 = b[0];
        double z0 = b[2];
        int ev = ball_substep(b, PROBE_DT, rng);
        if (ev == 2) return {-1.0, 0.0, -1.0};
        if ((x0 < 0.0) != (b[0] < 0.0)) clr = std::min(z0, b[2]) - (TABLE_H + NET_H + BALL_R);
        if (ev == 1) {
            if (b[0] * s > 0.0) return {b[0] * s, b[1], clr};
            if (own > 0) {
                own -= 1;
                continue;
            }
            return {-1.0, 0.0, clr};
        }
        if (b[2] < TABLE_H + BALL_R && b[5] < 0.0) {
            if (b[0] * s > 0.0) return {b[0] * s, b[1], clr};
            return {-1.0, 0.0, clr};
        }
    }
    return {99.0, 0.0, clr};
}

V3 plan_shot(double px, double py, double pz, double tx, double ty, double vh, double top, double margin,
             double side, Rng& rng) {
    V3 v = aim_velocity(px, py, pz, tx, ty, TABLE_H + BALL_R, vh, top, side);
    double vx = v.x, vy = v.y, vz = v.z;
    if ((px < 0.0) == (tx < 0.0)) return v;
    double s = tx > 0.0 ? 1.0 : -1.0;
    double want = tx * s;
    double ay = ty;
    for (int it = 0; it < 10; ++it) {
        double vz0 = aim_velocity(px, py, pz, tx, ay, TABLE_H + BALL_R, vh, top, side).z;
        double lz = 0.0;
        double clr = -1.0;
        for (int lat = 0; lat < 2; ++lat) {
            double dx = tx - px;
            double dy = ay - py;
            double d = std::sqrt(dx * dx + dy * dy) + 1e-9;
            vx = dx / d * vh;
            vy = dy / d * vh;
            double lo = vz0 - 2.0;
            double hi = vz0 + 3.0;
            for (int bb = 0; bb < PLAN_BISECT; ++bb) {
                double mid = 0.5 * (lo + hi);
                Probe pr = fly_probe(px, py, pz, vx, vy, mid, top, side, s, 0, rng);
                if (pr.l < want) lo = mid; else hi = mid;
            }
            vz = hi;
            Probe pr = fly_probe(px, py, pz, vx, vy, vz, top, side, s, 0, rng);
            lz = pr.l;
            clr = pr.clr;
            if (lz > 0.0 && lz < 50.0) ay = clampd(ay + ty - pr.y, -HALF_W - 0.6, HALF_W + 0.6);
        }
        if (clr >= margin && std::fabs(lz - want) < 0.15) return {vx, vy, vz};
        if (vh < 2.2) return {vx, vy, vz};
        vh *= 0.88;
    }
    return {vx, vy, vz};
}

double first_bounce(double px, double py, double pz, double vx, double vy, double vz, double top, double side,
                    double s, Rng& rng) {
    double b[9];
    V3 w = spin_vector(vx, vy, top, side);
    b[0] = px; b[1] = py; b[2] = pz; b[3] = vx; b[4] = vy; b[5] = vz; b[6] = w.x; b[7] = w.y; b[8] = w.z;
    for (int k = 0; k < 200; ++k) {
        int ev = ball_substep(b, PROBE_DT / 2.0, rng);
        if (ev == 1) return b[0] * s < 0.0 ? -b[0] * s : -1.0;
        if (ev == 2 || b[0] * s > 0.0) return -1.0;
        if (ev != 0 || b[2] < TABLE_H - 0.3) return 99.0;
    }
    return -1.0;
}

V3 plan_serve_shot(double px, double py, double pz, double tx, double ty, double d1, double top, double side,
                   double margin, Rng& rng) {
    double s = tx > 0.0 ? 1.0 : -1.0;
    double want = tx * s;
    double ay = ty;
    double vx = 0.0, vy = 0.0, vz = 0.0;
    for (int it = 0; it < 4; ++it) {
        double clr = -1.0;
        for (int lat = 0; lat < 2; ++lat) {
            double dx = tx - px;
            double dy = ay - py;
            double d = std::sqrt(dx * dx + dy * dy) + 1e-9;
            double ux = dx / d, uy = dy / d;
            double lo_v = 1.0, hi_v = 16.0;
            double lz = -1.0, yl = 0.0;
            clr = -1.0;
            for (int a = 0; a < 9; ++a) {
                double vh = 0.5 * (lo_v + hi_v);
                double lo_z = -9.0, hi_z = 2.0;
                for (int bb = 0; bb < 9; ++bb) {
                    double mz = 0.5 * (lo_z + hi_z);
                    double u = first_bounce(px, py, pz, ux * vh, uy * vh, mz, top, side, s, rng);
                    if (u > d1) lo_z = mz; else hi_z = mz;
                }
                vx = ux * vh;
                vy = uy * vh;
                vz = 0.5 * (lo_z + hi_z);
                Probe pr = fly_probe(px, py, pz, vx, vy, vz, top, side, s, 1, rng);
                lz = pr.l;
                yl = pr.y;
                clr = pr.clr;
                if (lz < want) lo_v = vh; else hi_v = vh;
            }
            if (lz > 0.0 && lz < 50.0) ay = clampd(ay + ty - yl, -HALF_W - 0.4, HALF_W + 0.4);
        }
        if (clr >= margin) break;
        d1 = std::min(d1 + 0.15, HALF_L - 0.05);
    }
    return {vx, vy, vz};
}

double hand_off(double blend) { return HAND_MID + HAND_HALF * blend; }

void hand_range(double blend, double& lo, double& hi) {
    double t = (blend + 1.0) * 0.5;
    lo = BH_LO + (FH_LO - BH_LO) * t;
    hi = BH_HI + (FH_HI - BH_HI) * t;
}

double hs_of(const World& w, int i, int j) { return w.s(i, j, HAND) < 0.0 ? -1.0 : 1.0; }

double lat_m(const World& w, int i, int j) { return (j == 0 ? 1.0 : -1.0) * hs_of(w, i, j); }

int face_rubber(const World& w, int i, int j, bool bh, int k) {
    if (!bh) return (int)w.s(i, j, RFH);
    if (w.s(i, j, GRIP) > 0.5 && (k == ST_BLOCK || k == ST_PUSH || k == -1)) return (int)w.s(i, j, RFH);
    return (int)w.s(i, j, RBH);
}

int pick_index(const double* p, int n, Rng& rng) {
    double r = rng.random();
    for (int k = 0; k < n; ++k) {
        r -= p[k];
        if (r <= 0.0) return k;
    }
    return n - 1;
}

void random_equipment(World& w, int i, int j, bool balanced) {
    Rng& rng = w.r(i);
    if (balanced) {
        w.s(i, j, HAND) = rng.random() < 0.5 ? -1.0 : 1.0;
        w.s(i, j, GRIP) = rng.random() < 0.5 ? 1.0 : 0.0;
        w.s(i, j, RFH) = (double)rng.randint(0, N_RUB);
        w.s(i, j, RBH) = (double)rng.randint(0, N_RUB);
        return;
    }
    w.s(i, j, HAND) = rng.random() < LEFTY_P ? -1.0 : 1.0;
    w.s(i, j, GRIP) = rng.random() < PEN_P ? 1.0 : 0.0;
    w.s(i, j, RFH) = (double)pick_index(RUB_FH_P, N_RUB, rng);
    w.s(i, j, RBH) = (double)pick_index(RUB_BH_P, N_RUB, rng);
}

void racket_impact(V3 v, V3 wv, V3 n, V3 vp, double e, double mu, double ks, V3& v_out, V3& w_out) {
    V3 u = v - vp;
    double un = std::min(dot3(u, n), -0.05);
    double jn = -(1.0 + e) * un;
    V3 c = u - BALL_R * cross3(wv, n);
    V3 s = c - dot3(c, n) * n;
    double sn = norm3(s);
    V3 jt(0, 0, 0);
    if (sn > 1e-9) jt = (-std::min(ks * K_STICK * sn, mu * jn) / sn) * s;
    v_out = v + jn * n + jt;
    w_out = wv - cross3(n, jt) / (K_INERTIA * BALL_R);
}

void plan_stroke(V3 pos, V3 v_in, V3 w_est, double tx, double ty, double vh, double top, double side,
                 double margin, double smax, double swing_max, double dv, double ds, double d1, double e, double ks,
                 Rng& rng, V3& n_out, V3& vp_out) {
    V3 n(0, 0, 0), vp(0, 0, 0), vout;
    for (int attempt = 0; attempt < 20; ++attempt) {
        double top_eff = top;
        double side_eff = side;
        V3 J;
        for (int it = 0; it < 3; ++it) {
            V3 v;
            if (d1 > 0.0)
                v = plan_serve_shot(pos.x, pos.y, pos.z, tx, ty, d1, top_eff, side_eff, margin, rng);
            else
                v = plan_shot(pos.x, pos.y, pos.z, tx, ty, vh, top_eff, margin, side_eff, rng);
            vout = v;
            V3 wd = spin_vector(v.x, v.y, top, side);
            J = vout - v_in;
            double jm = norm3(J) + 1e-9;
            V3 jh = J / jm;
            V3 a(-K_INERTIA * BALL_R * (wd.x - w_est.x), -K_INERTIA * BALL_R * (wd.y - w_est.y),
                 -K_INERTIA * BALL_R * (wd.z - w_est.z));
            a = a - dot3(a, jh) * jh;
            double am = norm3(a);
            double sa = std::min(am / jm, smax);
            if (am > 1e-9) {
                V3 m = cross3(jh, a / am);
                n = std::sqrt(1.0 - sa * sa) * jh + sa * m;
            } else {
                n = jh;
            }
            V3 w_ach = w_est - cross3(n, J) / (K_INERTIA * BALL_R);
            double hn = std::sqrt(v.x * v.x + v.y * v.y) + 1e-9;
            top_eff = w_ach.x * (-v.y / hn) + w_ach.y * (v.x / hn);
            side_eff = w_ach.z;
        }
        double jn = dot3(J, n);
        V3 jt = J - jn * n;
        V3 c = v_in - BALL_R * cross3(w_est, n);
        V3 ct = c - dot3(c, n) * n;
        vp = (dot3(v_in, n) + jn / (1.0 + e)) * n + ct + jt / (ks * K_STICK);
        if (norm3(vp) <= swing_max) break;
        vh *= dv;
        top *= ds;
        side *= ds;
    }
    n_out = n;
    vp_out = vp;
}

double rub_spin_cap(int r) {
    double mu = RUB_P[r][1];
    return std::min(mu / std::sqrt(1.0 + mu * mu), 0.6 * RUB_P[r][2]);
}

V3 bvec(const World& w, int i, int k0) { return {w.b(i, k0), w.b(i, k0 + 1), w.b(i, k0 + 2)}; }

void execute_stroke(World& w, int i, int h, V3 n, V3 vp, double sig_n, bool bh, int r) {
    Rng& rng = w.r(i);
    double g0 = rng.normal(), g1 = rng.normal(), g2 = rng.normal();
    n = n + V3(g0, g1, g2) * sig_n;
    n = n / norm3(n);
    w.t(i, PN0 + 3 * h + 0) = n.x;
    w.t(i, PN0 + 3 * h + 1) = n.y;
    w.t(i, PN0 + 3 * h + 2) = n.z;
    vp = vp * (1.0 + rng.normal() * sig_n * 0.5);
    V3 v_in = bvec(w, i, 3);
    V3 w_in = bvec(w, i, 6);
    V3 v_out, w_out;
    racket_impact(v_in, w_in, n, vp, RUB_P[r][0], RUB_P[r][1], RUB_P[r][2], v_out, w_out);
    w.b(i, 3) = v_out.x; w.b(i, 4) = v_out.y; w.b(i, 5) = v_out.z;
    w.b(i, 6) = w_out.x; w.b(i, 7) = w_out.y; w.b(i, 8) = w_out.z;
    w.in(i, LHIT) = h;
    w.in(i, LSIDE) = bh ? 1 : 0;
    w.t(i, LSW) = norm3(vp);
    w.t(i, LRUB) = r;
}

double swing_limit(double st, double fat, bool bh) {
    return (12.0 + 16.0 * st) * (1.0 - 0.25 * fat) * (bh ? 0.8 : 1.0);
}

double react_delay(double rea) { return 0.10 + 0.25 * (1.0 - rea); }

double vmax_of(double agi, double fat) { return (1.2 + 1.2 * agi) * (1.0 - 0.35 * fat); }

void predict(World& w, int i) {
    Rng& rng = w.r(i);
    int rec = (int)w.in(i, TURN);
    double b[9];
    std::memcpy(b, w.bp(i), sizeof(b));
    const double dt = 1.0 / 240.0;
    bool found = false, valid = false, apex = false, twice = false;
    double lx = 0.0, ly = 0.0, lz = 0.0, lth = 0.0;
    double rs = rec == 0 ? -1.0 : 1.0;
    double tx = rs * (HALF_L + 0.6), ty = 0.0, tz = 1.0, th = 0.0;
    bool own_ok = w.in(i, HITS) == 0 && w.in(i, SVB) == 0;
    for (int k = 0; k < 480; ++k) {
        int ev = ball_substep(b, dt, rng);
        double t = (k + 1) * dt;
        if (ev == 1) {
            bool on_rec = rec == 0 ? (b[0] < 0.0) : (b[0] > 0.0);
            if (own_ok && !on_rec && !found) {
                own_ok = false;
                continue;
            }
            if (!found) {
                if (on_rec) {
                    found = true;
                    continue;
                }
                break;
            }
            if (!apex) {
                tx = b[0]; ty = b[1]; tz = b[2]; th = t;
            }
            valid = true;
            twice = true;
            if (lth <= 0.0) {
                lx = tx; ly = ty; lz = tz; lth = th;
            }
            break;
        }
        if (apex && b[2] >= TABLE_H + LATE_SHORT_Z) {
            lx = b[0]; ly = b[1]; lz = b[2]; lth = t;
        }
        if (found && !apex && b[5] <= 0.0) {
            tx = b[0]; ty = b[1]; tz = b[2]; th = t;
            apex = true;
            valid = true;
            if (std::fabs(b[0]) >= HALF_L - 0.25) break;
        }
        if (apex && std::fabs(b[0]) >= HALF_L - 0.25) {
            tx = b[0]; ty = b[1]; tz = b[2]; th = t;
            break;
        }
        if (b[2] < TABLE_H - 0.3 || std::fabs(b[0]) > 4.5) break;
    }
    if (!twice) {
        lx = tx; ly = ty; lz = tz; lth = th;
    }
    if (valid) {
        double t = th;
        int lim = twice ? 0 : 480;
        for (int k = 0; k < lim; ++k) {
            int ev = ball_substep(b, dt, rng);
            t += dt;
            if (ev != 0 || b[2] < LATE_ZMIN || std::fabs(b[0]) > LATE_XMAX) break;
            lx = b[0]; ly = b[1]; lz = b[2]; lth = t;
        }
        double sig = (0.02 + 0.12 * (1.0 - w.s(i, rec, REA))) * (1.0 + norm3(bvec(w, i, 3)) / 15.0);
        double ex = rng.normal() * sig;
        double ey = rng.normal() * sig;
        double ez = rng.normal() * sig * 0.5;
        tx += ex; ty += ey; tz += ez;
        lx += ex; ly += ey; lz += ez;
    }
    w.t(i, EX) = tx; w.t(i, EY) = ty; w.t(i, EZ) = tz; w.t(i, ETH) = w.t(i, RT) + th;
    w.t(i, LX) = lx; w.t(i, LY) = ly; w.t(i, LZ) = lz; w.t(i, LTH) = w.t(i, RT) + lth;
    w.t(i, PX) = tx;
    w.t(i, PY) = ty;
    w.t(i, PZ) = tz;
    w.t(i, PRT) = w.t(i, RT) + th;
    w.t(i, EVT) = w.t(i, RT);
    w.t(i, SH0 + rec) = -1.0;
    w.in(i, PVALID) = valid ? 1 : 0;
}

double toss_time(double toss) { return 2.0 * std::sqrt(2.0 * (TOSS_H0 + TOSS_H1 * toss) / G); }

bool serve_is_bh(int k) { return k == SV_BH || k == SV_BHFAST; }

double serve_spot(double ms, double a1, int k) {
    if (serve_is_bh(k)) return ms * (0.12 + 0.2 * a1);
    if (k == SV_CHOP) return ms * (-0.3 + 0.3 * a1);
    return ms * a1 * 0.65;
}

void place_server(World& w, int i, int srv, bool bh) {
    double xs = srv == 0 ? -1.0 : 1.0;
    w.p(i, srv, 0) = w.b(i, 0) + xs * 0.1;
    w.p(i, srv, 1) = w.b(i, 1) - lat_m(w, i, srv) * hand_off(bh ? -1.0 : 1.0);
    w.p(i, srv, 2) = w.b(i, 2) - 0.03;
    for (int k = 3; k < 6; ++k) w.p(i, srv, k) = 0.0;
}

int plan_serve(World& w, int i, int srv) {
    int rec = 1 - srv;
    double sc[N_SERVE];
    for (int k = 0; k < N_SERVE; ++k) {
        double n = w.t(i, SN0 + srv * N_SERVE + k);
        double ww = w.t(i, SW0 + srv * N_SERVE + k);
        double p = (ww + SERVE_PRIOR[k] * PRIOR_N) / (n + PRIOR_N);
        sc[k] = (p - TAC_FAM * w.t(i, FAM0 + rec * N_SERVE + k)) / TAC_TEMP;
    }
    double mx = *std::max_element(sc, sc + N_SERVE);
    double tot = 0.0;
    for (int k = 0; k < N_SERVE; ++k) {
        sc[k] = std::exp(sc[k] - mx);
        tot += sc[k];
    }
    double r = w.r(i).random() * tot;
    for (int k = 0; k < N_SERVE; ++k) {
        r -= sc[k];
        if (r <= 0.0) return k;
    }
    return N_SERVE - 1;
}

int plan_variation(World& w, int i, int srv) {
    double sc[N_VAR];
    double tot_n = 1e-6;
    for (int v = 0; v < N_VAR; ++v) tot_n += w.t(i, VN0 + srv * N_VAR + v);
    for (int v = 0; v < N_VAR; ++v) {
        double n = w.t(i, VN0 + srv * N_VAR + v);
        double ww = w.t(i, VW0 + srv * N_VAR + v);
        double p = (ww + VAR_PRIOR[v] * PRIOR_N) / (n + PRIOR_N);
        sc[v] = (p - TAC_VREP * n / tot_n) / TAC_TEMP;
    }
    double mx = *std::max_element(sc, sc + N_VAR);
    double tot = 0.0;
    for (int v = 0; v < N_VAR; ++v) {
        sc[v] = std::exp(sc[v] - mx);
        tot += sc[v];
    }
    double r = w.r(i).random() * tot;
    for (int v = 0; v < N_VAR; ++v) {
        r -= sc[v];
        if (r <= 0.0) return v;
    }
    return N_VAR - 1;
}

void record_serve(World& w, int i, int win) {
    int64_t k = w.in(i, LSRV);
    if (k < 0 || win < 0) return;
    int srv = (int)w.in(i, SERVER);
    for (int kk = 0; kk < N_SERVE; ++kk) {
        w.t(i, SN0 + srv * N_SERVE + kk) *= TAC_DECAY;
        w.t(i, SW0 + srv * N_SERVE + kk) *= TAC_DECAY;
    }
    w.t(i, SN0 + srv * N_SERVE + (int)k) += 1.0;
    if (win == srv) w.t(i, SW0 + srv * N_SERVE + (int)k) += 1.0;
    int v = (int)w.t(i, VPLAN0 + srv);
    for (int vv = 0; vv < N_VAR; ++vv) {
        w.t(i, VN0 + srv * N_VAR + vv) *= TAC_DECAY;
        w.t(i, VW0 + srv * N_VAR + vv) *= TAC_DECAY;
    }
    w.t(i, VN0 + srv * N_VAR + v) += 1.0;
    if (win == srv) w.t(i, VW0 + srv * N_VAR + v) += 1.0;
}

void reset_serve_stats(World& w, int i) {
    for (int k = 0; k < 2 * N_SERVE; ++k) {
        w.t(i, SW0 + k) = 0.0;
        w.t(i, SN0 + k) = 0.0;
    }
    for (int k = 0; k < 2 * N_VAR; ++k) {
        w.t(i, VW0 + k) = 0.0;
        w.t(i, VN0 + k) = 0.0;
    }
}

int serve_type_of(double a8, double a10) {
    if (a8 < -0.15) return a10 < 0.0 ? SV_BH : SV_BHFAST;
    static const int tbl[6] = {SV_PEND, SV_HOOK, SV_SHOVEL, SV_SQUAT, SV_CHOP, SV_FAST};
    int idx = (int)((a10 + 1.0) * 3.0);
    return tbl[std::min(std::max(idx, 0), 5)];
}

double serve_toss(int k, double a9) {
    double toss = (a9 + 1.0) * 0.5;
    if (k == SV_FAST || k == SV_BHFAST) toss = std::min(toss, FAST_TOSS_MAX);
    return toss;
}

bool serve_hit(World& w, int i, const double* a) {
    Rng& rng = w.r(i);
    int srv = (int)w.in(i, SERVER);
    double xs = srv == 0 ? -1.0 : 1.0;
    double st = w.s(i, srv, STR);
    double srv_tec = w.s(i, srv, SRV);
    double recv_rea = w.s(i, 1 - srv, REA);
    int k = (int)w.t(i, SPLAN0 + srv);
    const double* sp = SERVE_P[k];
    bool bh = k == SV_BH || k == SV_BHFAST;
    double toss = w.t(i, TOSS) >= 0.0 ? w.t(i, TOSS) : serve_toss(k, a[9]);
    bool hidden = a[11] > 0.5;
    w.t(i, TOSS) = toss;
    w.in(i, LHID) = hidden ? 1 : 0;
    w.in(i, LSRV) = k;
    if (hidden && rng.random() < FAULT_P) return true;
    w.b(i, 3) = 0.0; w.b(i, 4) = 0.0; w.b(i, 5) = -0.5 - 2.0 * toss;
    w.b(i, 6) = 0.0; w.b(i, 7) = 0.0; w.b(i, 8) = 0.0;
    double depth = 0.3 + (a[3] + 1.0) * 0.45;
    if (k == SV_FAST || k == SV_BHFAST) depth = std::max(depth, FAST_DEPTH);
    double hs = hs_of(w, i, srv);
    bool pen = w.s(i, srv, GRIP) > 0.5;
    int rub = face_rubber(w, i, srv, bh, -1);
    double tx = -xs * depth;
    double ty = -xs * hs * a[4] * 0.6;
    double pw = (a[5] + 1.0) * 0.5;
    double vh = (3.0 + pw * (1.5 + 2.0 * st)) * sp[6] * (1.0 + 0.1 * toss);
    double ms = (SRV_SPIN0 + SRV_SPIN1 * srv_tec) * sp[0] * (1.0 + 0.3 * toss);
    double a6 = VAR_TOP[(int)w.t(i, VPLAN0 + srv)];
    double top = clampd(a6, sp[7], sp[8]) * ms;
    double sgn = sp[1] != 0.0 ? sp[1] : (a[7] >= 0.0 ? 1.0 : -1.0);
    double side = hs * sgn * (0.75 + 0.25 * std::fabs(a[7])) * sp[2] * ms * (0.35 + 0.65 * std::min(std::fabs(a6), 1.0));
    double d1 = 0.9 + 0.1 * clampd((depth - 0.3) / 0.9, 0.0, 1.0) + 0.15 * pw;
    if (k == SV_FAST || k == SV_BHFAST) d1 = std::max(d1, FAST_D1);
    d1 = std::min(d1, HALF_L - 0.1);
    V3 n, vp;
    plan_stroke(bvec(w, i, 0), bvec(w, i, 3), bvec(w, i, 6), tx, ty, vh, top, side, 0.05,
                std::min((0.4 + 0.4 * srv_tec) * std::min(sp[0], 1.0), rub_spin_cap(rub)),
                swing_limit(st, w.f(i, srv), bh), 0.95, 0.96, d1, RUB_P[rub][0], RUB_P[rub][2], rng, n, vp);
    execute_stroke(w, i, srv, n, vp,
                   (0.003 + 0.01 * (1.0 - srv_tec)) * sp[3] * (1.0 + 0.5 * toss) * (pen ? PEN_SERVE : 1.0), bh, rub);
    int rec = 1 - srv;
    double fam = w.t(i, FAM0 + rec * N_SERVE + k);
    w.t(i, SFAM) = fam;
    w.t(i, DEC) = (sp[4] * (0.7 + 0.6 * srv_tec) * (1.0 + 0.1 * toss) * (hidden ? 1.15 : 1.0) * (pen ? PEN_DEC : 1.0) *
                   (FAM_DEC_NEW + (FAM_DEC_OLD - FAM_DEC_NEW) * fam) * (1.35 - 0.7 * recv_rea));
    w.t(i, ORB0 + rec) = rub;
    for (int kk = 0; kk < N_SERVE; ++kk) w.t(i, FAM0 + rec * N_SERVE + kk) *= 1.0 - FAM_RATE;
    w.t(i, FAM0 + rec * N_SERVE + k) += FAM_RATE;
    w.t(i, REC0 + srv) = sp[5];
    w.t(i, LK0 + srv) = ST_DRIVE;
    double bl = bh ? -1.0 : 1.0;
    w.t(i, BL0 + srv) = bl;
    w.t(i, BT0 + srv) = bl;
    return false;
}

void reset_mind(World& w, int i) {
    for (int j = 0; j < 2; ++j) {
        w.t(i, STK0 + j) = 0.0;
        w.t(i, FRU0 + j) = 0.0;
        w.t(i, ADP0 + j) = 0.0;
        w.t(i, ORB0 + j) = 0.0;
    }
}

double pressure(const World& w, int i, int h) {
    int64_t s0 = w.in(i, SC0), s1 = w.in(i, SC1);
    double p = 0.0;
    if (std::max(s0, s1) >= GAME_PTS - 1) {
        p = CLUTCH_GP;
        for (int j = 0; j < 2; ++j) {
            if (w.in(i, SC0 + j) >= GAME_PTS - 1 && w.in(i, SC0 + j) > w.in(i, SC0 + 1 - j) &&
                w.in(i, G0 + j) >= MATCH_WIN - 1)
                p = CLUTCH_MP;
        }
    }
    return p + std::min(TILT_PT * w.t(i, STK0 + h), TILT_MAX) + w.t(i, FRU0 + h);
}

void start_point(World& w, int i, bool train) {
    Rng& rng = w.r(i);
    int srv;
    if (train) {
        srv = (int)rng.randint(0, 2);
        if (rng.random() < 0.15) {
            for (int k = 0; k < 2 * N_SERVE; ++k) w.t(i, FAM0 + k) = 0.0;
            reset_serve_stats(w, i);
            for (int j = 0; j < 2; ++j) {
                for (int k = 0; k < N_STATS; ++k) w.s(i, j, k) = rng.random();
                w.s(i, j, SRV) = rng.random();
                w.s(i, j, MEN) = rng.random();
                w.s(i, j, TCH) = rng.random();
                bool plain = rng.random() < 0.25;
                for (int k = N_STATS; k < N_STATS + N_STYLE; ++k) w.s(i, j, k) = plain ? 0.0 : rng.random() * 2.0 - 1.0;
                bool bal = rng.random() < BAL_FRAC;
                random_equipment(w, i, j, bal);
                w.f(i, j) = rng.random() * 0.9;
            }
            reset_mind(w, i);
        }
    } else {
        int64_t tot = w.in(i, SC0) + w.in(i, SC1);
        int64_t fs = w.in(i, FIRST);
        if (tot < 2 * (GAME_PTS - 1))
            srv = (int)(fs ^ ((tot / 2) & 1));
        else
            srv = (int)(fs ^ (tot & 1));
    }
    for (int j = 0; j < 2; ++j) {
        w.f(i, j) = std::max(0.0, w.f(i, j) - FAT_REC_PT);
        w.t(i, LM0 + j) = lat_m(w, i, j);
    }

    double xs = srv == 0 ? -1.0 : 1.0;
    w.b(i, 0) = xs * (HALF_L + 0.15);
    w.b(i, 1) = (rng.random() - 0.5) * 0.9;
    w.b(i, 2) = TABLE_H + 0.22;
    for (int k = 3; k < 9; ++k) w.b(i, k) = 0.0;
    for (int j = 0; j < 2; ++j) {
        double sj = j == 0 ? -1.0 : 1.0;
        if (train) {
            w.p(i, j, 0) = sj * (HALF_L + 0.35 + rng.random() * 0.7);
            w.p(i, j, 1) = (rng.random() - 0.5) * 0.8;
        } else {
            w.p(i, j, 0) = sj * (HALF_L + 0.55);
            w.p(i, j, 1) = 0.0;
        }
        w.p(i, j, 2) = 1.0;
        for (int k = 3; k < 6; ++k) w.p(i, j, k) = 0.0;
    }
    for (int j = 0; j < 2; ++j) {
        double bl = 1.0;
        if (train && j != srv && rng.random() < 0.5) bl = -1.0;
        w.t(i, BL0 + j) = bl;
        w.t(i, BT0 + j) = bl;
        w.t(i, LN0 + j) = 0.0;
        w.t(i, CX0 + j) = 0.0;
        w.t(i, CD0 + j) = 0.0;
        w.t(i, CU0 + j) = 0.0;
        w.t(i, SET0 + j) = SETTLE_T;
        w.t(i, SPL0 + j) = 0.0;
        w.t(i, REC0 + j) = 0.0;
        w.t(i, DV0 + j) = 0.0;
        w.t(i, LK0 + j) = 0.0;
    }
    w.t(i, STB) = 0.0;
    w.t(i, DEC) = 1.0;
    w.in(i, LFAKE) = 0;
    w.in(i, LSTK) = -1;
    w.in(i, LSRV) = -1;
    w.in(i, LTECH) = -1;
    w.t(i, TOSS) = -1.0;
    w.t(i, SPLAN0 + srv) = plan_serve(w, i, srv);
    w.t(i, VPLAN0 + srv) = plan_variation(w, i, srv);
    int sk = (int)w.t(i, SPLAN0 + srv);
    bool bh = serve_is_bh(sk);
    w.b(i, 2) = TABLE_H + SERVE_Z[sk];
    if (bh || sk == SV_CHOP) w.b(i, 1) = serve_spot(lat_m(w, i, srv), (rng.random() - 0.5) * 1.4, sk);
    if (bh) {
        w.t(i, BL0 + srv) = -1.0;
        w.t(i, BT0 + srv) = -1.0;
    }
    place_server(w, i, srv, bh);

    w.in(i, TURN) = 1 - srv;
    w.in(i, BOUNCED) = 0;
    w.in(i, NETT) = 0;
    w.in(i, HITS) = 0;
    w.t(i, PCHAIN) = 0.0;
    w.in(i, SVB) = 0;
    w.in(i, SERVER) = srv;
    w.in(i, PHASE) = PH_SERVE;
    w.in(i, PVALID) = 0;
    w.t(i, RT) = 0.0;
    w.t(i, TIMER) = train ? (0.15 + rng.random() * 1.65) : SERVE_WAIT;
    w.t(i, EVT) = 0.0;
    w.t(i, SH0) = -1.0;
    w.t(i, SH1) = -1.0;
}

void mark_fake(World& w, int i, int h, double lat, double cue) {
    int q = 1 - h;
    double mq = lat_m(w, i, q), mh = lat_m(w, i, h);
    double need = mq * (mh * lat - w.p(i, q, 1));
    double lure = mq * (mh * cue - w.p(i, q, 1));
    double cn = w.t(i, LN0 + q) / LEAN_MAX;
    w.in(i, LFAKE) = 0;
    if (std::fabs(lat - cue) > FAKE_DEV && cn * lure > 0.0 && cn * need < 0.0 && std::fabs(cn) > FAKE_LEAN &&
        std::fabs(need) > FAKE_NEED && std::fabs(lure) > 0.1) {
        w.in(i, LFAKE) = h + 1;
        w.t(i, LFT) = w.t(i, RT);
    }
}

int classify_stroke(const double* b, double pw, double top) {
    double dxt = std::fabs(b[0]) - HALF_L;
    double hz = b[2] - TABLE_H;
    if (hz > SMASH_Z && pw > 0.55 && top > -0.3) return ST_SMASH;
    if (top < -0.25) return dxt > 0.45 ? ST_CHOP : ST_PUSH;
    if (dxt > 1.1 && pw < 0.45) return ST_LOB;
    if (dxt < -0.25) return ST_FLICK;
    if (pw < 0.35 && dxt < 0.5) return ST_BLOCK;
    if (top < 0.35) return ST_DRIVE;
    return ST_LOOP;
}

int tech_of(int k, bool bh, double pw, double top, double side, double depth, double top_in) {
    bool back = top_in < -60.0;
    if (k == ST_PUSH) {
        if (depth < 0.6) return TQ_SHORT;
        if (depth > 0.95 && pw > 0.35) return TQ_LONG;
        return TQ_PUSH;
    }
    if (k == ST_FLICK) {
        if (pw < 0.4 && std::fabs(side) > 0.4) return TQ_WIPE;
        if (bh) return (pw > 0.7 && top_in < -120.0) ? TQ_BANANA : TQ_TWIST;
        return TQ_FLICK;
    }
    if (k == ST_LOOP) {
        if (back) return TQ_LIFT;
        if (bh && pw > 0.6) return TQ_RIP;
        if (pw > 0.7) return TQ_DRIVELOOP;
        return TQ_LOOP;
    }
    if (k == ST_BLOCK) {
        if (bh && std::fabs(side) > 0.45 && top < 0.1) return TQ_SIDECUT;
        return TQ_BLOCK;
    }
    if (k == ST_DRIVE) return TQ_DRIVE;
    if (k == ST_CHOP) return TQ_CHOP;
    if (k == ST_SMASH) return TQ_SMASH;
    return TQ_LOB;
}

double incoming_top(V3 v, V3 wv) {
    double hn = std::sqrt(v.x * v.x + v.y * v.y) + 1e-9;
    return wv.x * (-v.y / hn) + wv.y * (v.x / hn);
}

void do_hit(World& w, int i, int h, const double* a, double stretch) {
    Rng& rng = w.r(i);
    double m = h == 0 ? 1.0 : -1.0;
    double hs = hs_of(w, i, h);
    bool pen = w.s(i, h, GRIP) > 0.5;
    double st = w.s(i, h, STR);
    double tec = w.s(i, h, TEC);
    double rd = w.in(i, HITS) == 0 ? w.s(i, h, REA) : tec;
    double fat = w.f(i, h);
    double blend = w.t(i, BL0 + h);
    bool bh = blend < 0.0;
    double awk = std::max(0.0, 0.9 - std::fabs(blend)) / 0.9;
    double depth = DEPTH_MIN + (a[3] + 1.0) * 0.5 * (1.2 - DEPTH_MIN);
    double shrt = std::max(0.0, 0.6 - depth) / 0.3;
    depth = std::max(depth + rng.normal() * (0.03 + 0.08 * (1.0 - tec)) * (1.0 + 1.5 * shrt), 0.12);
    double cue = w.t(i, CU0 + h);
    double dev = 0.25 + 0.45 * tec;
    double lat = clampd(a[4] * LAT_MAX, cue - dev, cue + dev);
    mark_fake(w, i, h, lat, cue);
    w.t(i, CU0 + h) = 0.0;
    double pw = (a[5] + 1.0) * 0.5;
    w.f(i, h) = std::min(1.0, w.f(i, h) + FAT_HIT * (0.4 + pw) * (FAT_BASE - w.s(i, h, STA)));
    int k = classify_stroke(w.bp(i), pw, a[6]);
    const double* sp = STROKE_P[k];
    int rub = face_rubber(w, i, h, bh, k);
    double tx = m * depth;
    double ty = m * hs * lat;
    double vh = std::min((3.0 + pw * (5.0 + 17.0 * st)) * (k == ST_SMASH ? SMASH_V : 1.0) * (1.0 - 0.3 * fat), sp[4]);
    w.t(i, DBG0) = vh;
    double ms = (STK_SPIN0 + STK_SPIN1 * tec) * (bh ? 0.9 : 1.0) * sp[7];
    double top = a[6] * ms;
    double side = hs * a[7] * 0.4 * ms;

    V3 w_in = bvec(w, i, 6);
    V3 v_in = bvec(w, i, 3);
    double floor_ = BORROW[k] * std::sqrt(v_in.x * v_in.x + v_in.y * v_in.y);
    double absorb = 0.0;
    if (vh < floor_) {
        absorb = (floor_ - vh) / (floor_ + 1.0);
        vh = floor_ - (floor_ - vh) * (0.25 + 0.35 * tec);
    }
    double dec = w.t(i, DEC);
    w.t(i, DEC) = 1.0;
    double men = w.s(i, h, MEN), tch = w.s(i, h, TCH);
    double odd = RUB_ODD[(int)w.t(i, ORB0 + h)];
    double adp = w.t(i, ADP0 + h);
    if (w.in(i, HITS) > 0) dec = 1.0 + (dec - 1.0) * std::max(1.0 + TCH_READ * (0.5 - tch), 0.0) * (1.0 - adp);
    w.t(i, ADP0 + h) = std::min(adp + (ADAPT0 + ADAPT1 * tch) * odd * (1.0 - adp / ADAPT_MAX), ADAPT_MAX);
    double sig_rel = (0.06 + 0.45 * (1.0 - rd)) * (1.0 + 0.5 * fat) * 0.6 * dec * READ_SCALE;
    double n0 = rng.normal(), n1 = rng.normal(), n2 = rng.normal();
    double m0 = rng.normal(), m1 = rng.normal(), m2 = rng.normal();
    double amp = 10.0 * (1.0 - rd) * dec * READ_SCALE;
    V3 w_est(w_in.x * (1.0 + n0 * sig_rel) + m0 * amp, w_in.y * (1.0 + n1 * sig_rel) + m1 * amp,
             w_in.z * (1.0 + n2 * sig_rel) + m2 * amp);
    double margin = 0.02 + 0.06 * (1.0 - tec) + sp[5];
    if (k == ST_BLOCK)
        margin += NOPACE_MARGIN * clampd((NOPACE_V - std::sqrt(v_in.x * v_in.x + v_in.y * v_in.y)) / 5.0, 0.0, 1.0);
    double smax = std::min((0.3 + 0.3 * tec) * (bh ? 0.85 : 1.0) * sp[1] * (pen && bh ? PEN_BH_SPIN : 1.0),
                           rub_spin_cap(rub));
    bool small = k == ST_PUSH || k == ST_FLICK || k == ST_BLOCK;
    double grip_sig;
    if (!pen)
        grip_sig = 1.0;
    else if (!bh)
        grip_sig = small ? PEN_FH_SHORT : 1.0;
    else
        grip_sig = (k == ST_BLOCK || k == ST_PUSH) ? PEN_PUSHBLOCK_SIG : PEN_BH_SIG;
    double sk = std::min(stretch / 0.2, 2.0);
    bool dive = !bh && pw > DIVE_PW && w.t(i, LM0 + h) * (w.b(i, 1) - w.p(i, h, 1)) < FH_CLO - DIVE_ST;
    double sk_sw = dive ? 0.0 : std::min(sk, 1.0);
    double sk_sig = dive ? 0.4 * sk : sk;
    double rx = std::min(m * w.p(i, h, 3), 0.0);
    double run = std::sqrt(rx * rx + w.p(i, h, 4) * w.p(i, h, 4)) / vmax_of(w.s(i, h, AGI), 0.0);
    double settle = std::min(w.t(i, SET0 + h) / SETTLE_T, 1.0);
    double unset = (1.0 - settle) * sp[6];
    double cn = w.t(i, LN0 + h) / LEAN_MAX;
    double load = bh ? std::max(0.0, cn) : std::max(0.0, -cn);
    double vf = clampd(m * w.p(i, h, 3) / 0.8, -1.0, 1.0);
    double body = (0.5 + 0.5 * settle) * (0.6 + 0.4 * load) * (1.0 + (vf > 0.0 ? 0.3 : 0.2) * vf);
    double share = sp[10] * (bh ? BH_BODY : 1.0);
    w.t(i, LBODY) = share * body;
    double avail = w.t(i, RT) - w.t(i, EVT) - react_delay(w.s(i, h, REA));
    double lack = clampd((PREP_T[k] - avail) / PREP_T[k], 0.0, 1.0);
    w.t(i, LPREP) = lack;
    margin += LACK_MARGIN * lack;
    double top_in = incoming_top(v_in, w_in);
    double back_in = std::max(0.0, -top_in) / SPIN_REF;
    w.in(i, LTECH) = tech_of(k, bh, pw, a[6], a[7], depth, top_in);
    if (k == ST_PUSH) {
        double r = rng.random();
        margin += std::min(POP_MARGIN * r * r * (0.3 + std::min(back_in, 1.5)) * (1.3 - tec) *
                               (1.0 + POP_CHAIN * std::min(w.t(i, PCHAIN), 6.0)),
                           POP_CAP);
        w.t(i, PCHAIN) += 1.0;
    } else {
        w.t(i, PCHAIN) = 0.0;
    }
    double hi = clampd((w.b(i, 2) - TABLE_H - HI_Z0) / (HI_Z1 - HI_Z0), 0.0, 1.0);
    bool atk = k == ST_LOOP || k == ST_DRIVE || k == ST_FLICK || k == ST_SMASH;
    vh = std::min(vh * (1.0 - share + share * body) * (1.0 - 0.5 * lack), sp[4]);
    double swing = (swing_limit(st, fat, bh) * sp[0] * (1.0 - share + share * body) * (1.0 - 0.5 * awk) *
                    (1.0 - 0.4 * sk_sw) * (1.0 - 0.15 * std::min(run, 1.0)) * (1.0 - 0.5 * lack) *
                    ((pen && bh && !(k == ST_BLOCK || k == ST_PUSH)) ? PEN_BH_SWING : 1.0));
    w.t(i, DBG0 + 1) = vh;
    w.t(i, DBG0 + 2) = swing;
    w.t(i, DBG0 + 3) = w.b(i, 0);
    w.t(i, DBG0 + 4) = 1.0 - share + share * body;
    w.t(i, DBG0 + 5) = 1.0 - 0.5 * awk;
    w.t(i, DBG0 + 6) = 1.0 - 0.4 * std::min(sk, 1.0);
    w.t(i, DBG0 + 7) = 1.0 - 0.15 * std::min(run, 1.0);
    w.t(i, DBG0 + 8) = 1.0 - 0.5 * lack;
    bool soft = k == ST_PUSH || k == ST_CHOP;
    V3 n, vp;
    plan_stroke(bvec(w, i, 0), v_in, w_est, tx, ty, vh, top, side, margin, smax, swing, soft ? 0.97 : 0.95,
                soft ? 0.88 : 0.96, 0.0, RUB_P[rub][0], RUB_P[rub][2], rng, n, vp);
    double sens = RUB_SENS[rub] * (1.0 + TCH_SENS * (0.5 - tch));
    double heavy = std::max(0.0, norm3(v_in) - 6.0) / 8.0 * sens;
    double push_vs_top = VS_TOP[k] * std::max(0.0, incoming_top(v_in, w_in)) / SPIN_REF * (1.3 - tec) * sens;
    double vs_back = VS_BACK[k] * back_in * (1.3 - tec) * (atk ? (1.0 - HI_BACK * hi) : 1.0) * sens;
    double sig_n = (ERR_SCALE * (0.004 + 0.018 * (1.0 - tec)) * (0.6 + 0.7 * pw) * (1.0 + 0.5 * fat) * sp[2] *
                    (bh ? 1.1 : 1.0) * (1.0 + 3.0 * awk) * (1.0 + 1.5 * sk_sig) *
                    (1.0 + RUN_SIG * std::min(run, 1.5)) * (1.0 + 0.8 * unset) * (1.0 + LACK_SIG * lack) *
                    (1.0 + ABSORB_SIG * absorb) * (1.0 + 2.5 * heavy * sp[3] * (1.3 - tec) * (bh ? 1.1 : 1.0)) *
                    (1.0 + push_vs_top) * (1.0 + vs_back) * (atk ? (1.0 - HI_SIG * hi) : 1.0) * grip_sig *
                    ((k == ST_PUSH || k == ST_FLICK) ? (1.0 + TCH_SMALL * (0.5 - tch)) : 1.0) *
                    (1.0 + MEN_SIG * pressure(w, i, h) * (1.0 - men)));
    execute_stroke(w, i, h, n, vp, sig_n, bh, rub);
    w.t(i, ORB0 + 1 - h) = rub;
    w.t(i, DEC) = RUB_DEC[rub];
    w.in(i, LSTK) = k;
    w.t(i, LK0 + h) = k;
    w.t(i, REC0 + h) = sp[8] + 0.15 * share * load;
    if (dive) {
        w.t(i, DV0 + h) = DIVE_T;
        w.t(i, REC0 + h) = DIVE_T + DIVE_REC;
        w.p(i, h, 3) *= 0.2;
        w.p(i, h, 4) *= 0.2;
    }
    w.t(i, LN0 + h) = load > 0.0 ? -w.t(i, LN0 + h) * std::min(1.5 * share, 1.0) : w.t(i, LN0 + h);
    double feat = ((bh ? 1.0 : -1.0) * w.s(i, h, STY_BH) + a[6] * w.s(i, h, STY_SPIN) + a[5] * w.s(i, h, STY_POW) +
                   (2.0 * std::fabs(lat) / LAT_MAX - 1.0) * w.s(i, h, STY_WIDE));
    w.t(i, STB) = STYLE_W * feat / N_STYLE;
    w.in(i, TURN) = 1 - h;
    w.in(i, BOUNCED) = 0;
    w.in(i, NETT) = 0;
    w.in(i, HITS) += 1;
    w.t(i, SH0) = -1.0;
    w.t(i, SH1) = -1.0;
    predict(w, i);
}

struct Out {
    double* rew;
    uint8_t* done;
    int64_t* winner;
    int64_t* reason;
    int64_t* hits;
};

void end_point(World& w, int i, int win, int reason, bool train, Out& o) {
    o.done[i] = 1;
    o.winner[i] = win;
    o.reason[i] = reason;
    o.hits[i] = w.in(i, HITS);
    record_serve(w, i, win);
    w.in(i, LWIN) = win;
    w.in(i, LREASON) = reason;
    double* rew = o.rew + (int64_t)i * 2;
    if (win < 0) {
        if (reason == R_TIMEOUT) {
            rew[0] -= 1.0;
            rew[1] -= 1.0;
        } else if (reason != R_LET) {
            rew[0] -= 0.5;
            rew[1] -= 0.5;
        }
    } else {
        double pts = reason == R_MISS ? WINNER_BONUS : 1.0;
        rew[win] += pts;
        rew[1 - win] -= pts;
        int lo = 1 - win;
        w.t(i, STK0 + lo) += 1.0;
        w.t(i, STK0 + win) = 0.0;
        for (int j = 0; j < 2; ++j) w.t(i, FRU0 + j) *= FRUS_DECAY;
        w.t(i, FRU0 + lo) = std::min(w.t(i, FRU0 + lo) + FRUS_ADD * RUB_ODD[(int)w.t(i, ORB0 + lo)], 1.0);
        w.in(i, SC0 + win) += 1;
        int64_t s0 = w.in(i, SC0), s1 = w.in(i, SC1);
        if (std::max(s0, s1) >= GAME_PTS && std::llabs(s0 - s1) >= 2) {
            w.in(i, G0 + win) += 1;
            w.in(i, SC0) = 0;
            w.in(i, SC1) = 0;
            w.in(i, FIRST) ^= 1;
            if (w.in(i, G0 + win) >= MATCH_WIN) {
                w.in(i, M0 + win) += 1;
                w.in(i, G0) = 0;
                w.in(i, G1) = 0;
                for (int j = 0; j < 2; ++j) w.f(i, j) = 0.0;
                reset_mind(w, i);
            } else {
                for (int j = 0; j < 2; ++j) w.f(i, j) = std::max(0.0, w.f(i, j) - FAT_REC_GAME);
            }
        }
    }
    if (!train) {
        w.in(i, PHASE) = PH_DEAD;
        w.t(i, TIMER) = 1.2;
    }
}

double hand_y(const World& w, int i, int j) { return w.p(i, j, 1) + w.t(i, LM0 + j) * hand_off(w.t(i, BL0 + j)); }

void comfort_range(double blend, double& lo, double& hi) {
    double t = (blend + 1.0) * 0.5;
    lo = BH_CLO + (FH_CLO - BH_CLO) * t;
    hi = BH_CHI + (FH_CHI - BH_CHI) * t;
}

void reach_hand(const World& w, int i, int j, double tx, double ty, double ext, double& hx, double& hy,
                double& stretch) {
    double m = j == 0 ? 1.0 : -1.0;
    double ml = w.t(i, LM0 + j);
    double lat = ml * (ty - w.p(i, j, 1));
    double fwd = m * (tx - w.p(i, j, 0));
    double lo, hi;
    hand_range(w.t(i, BL0 + j), lo, hi);
    lo -= ext;
    lat = clampd(lat, lo, hi);
    double fhi = FWD_HI + HIGH_REACH * clampd((w.t(i, PZ) - TABLE_H - 0.25) / 0.35, 0.0, 1.0);
    fwd = clampd(fwd, FWD_LO, fhi);
    double clo, chi;
    comfort_range(w.t(i, BL0 + j), clo, chi);
    stretch = std::max(clo - lat, 0.0) + std::max(lat - chi, 0.0) + std::max(FWD_CLO - fwd, 0.0) +
              FWD_STRETCH_W * std::max(fwd - FWD_CHI, 0.0);
    hx = w.p(i, j, 0) + m * fwd;
    hy = w.p(i, j, 1) + ml * lat;
}

void update_contact(World& w, int i) {
    if (w.in(i, PVALID) == 0) return;
    int j = (int)w.in(i, TURN);
    double want = std::fabs(w.p(i, j, 0)) - CONTACT_FWD;
    double ex = std::fabs(w.t(i, EX)), lx = std::fabs(w.t(i, LX));
    double u = 0.0;
    if (lx > ex + 1e-6) u = clampd((want - ex) / (lx - ex), 0.0, 1.0);
    w.t(i, PX) = w.t(i, EX) + (w.t(i, LX) - w.t(i, EX)) * u;
    w.t(i, PY) = w.t(i, EY) + (w.t(i, LY) - w.t(i, EY)) * u;
    w.t(i, PZ) = w.t(i, EZ) + (w.t(i, LZ) - w.t(i, EZ)) * u * u;
    w.t(i, PRT) = w.t(i, ETH) + (w.t(i, LTH) - w.t(i, ETH)) * u;
}

double contact_x(const World& w, int i, int j) {
    if (w.in(i, PVALID) == 1 && w.t(i, RT) - w.t(i, EVT) >= react_delay(w.s(i, j, REA))) return w.t(i, PX);
    return w.b(i, 0);
}

void shown_hand(const World& w, int i, int j, double& hx, double& hy) {
    if (w.in(i, PHASE) == PH_FLY && w.in(i, TURN) == j && std::fabs(w.b(i, 0) - w.p(i, j, 0)) < 1.2) {
        double st;
        reach_hand(w, i, j, contact_x(w, i, j), w.b(i, 1), 0.0, hx, hy, st);
        return;
    }
    hx = w.p(i, j, 0);
    hy = hand_y(w, i, j);
}

void clamp_paddle(World& w, int i, int j) {
    double sj = j == 0 ? -1.0 : 1.0;
    double lx = sj * w.p(i, j, 0);
    double xmin = std::fabs(w.p(i, j, 1)) < HALF_W + 0.25 ? BODY_MIN : 0.3;
    if (lx < xmin) {
        w.p(i, j, 0) = sj * xmin;
        if (w.p(i, j, 3) * sj < 0.0) w.p(i, j, 3) = 0.0;
    } else if (lx > 3.6) {
        w.p(i, j, 0) = sj * 3.6;
        if (w.p(i, j, 3) * sj > 0.0) w.p(i, j, 3) = 0.0;
    }
    if (w.p(i, j, 1) < -2.2) {
        w.p(i, j, 1) = -2.2;
        w.p(i, j, 4) = std::max(w.p(i, j, 4), 0.0);
    } else if (w.p(i, j, 1) > 2.2) {
        w.p(i, j, 1) = 2.2;
        w.p(i, j, 4) = std::min(w.p(i, j, 4), 0.0);
    }
    double zmin = 0.25;
    double hy = hand_y(w, i, j);
    if (sj * w.p(i, j, 0) < HALF_L + 0.55 && (std::fabs(w.p(i, j, 1)) < HALF_W + 0.05 || std::fabs(hy) < HALF_W + 0.05))
        zmin = TABLE_H + 0.04;
    if (w.p(i, j, 2) < zmin) {
        w.p(i, j, 2) = zmin;
        w.p(i, j, 5) = std::max(w.p(i, j, 5), 0.0);
    } else if (w.p(i, j, 2) > 2.0) {
        w.p(i, j, 2) = 2.0;
        w.p(i, j, 5) = std::min(w.p(i, j, 5), 0.0);
    }
}

void clamp_receiver(World& w, int i, int j) {
    double sj = j == 0 ? -1.0 : 1.0;
    double lx = sj * w.p(i, j, 0) - HALF_L;
    if (lx < RECV_X0 || lx > RECV_X1) {
        w.p(i, j, 0) = sj * (HALF_L + clampd(lx, RECV_X0, RECV_X1));
        w.p(i, j, 3) = 0.0;
    }
    double ml = w.t(i, LM0 + j);
    double ly = ml * w.p(i, j, 1);
    if (std::fabs(ly - RECV_Y) > RECV_DY) {
        w.p(i, j, 1) = ml * clampd(ly, RECV_Y - RECV_DY, RECV_Y + RECV_DY);
        w.p(i, j, 4) = 0.0;
    }
}

double seg_dist(double ax, double ay, double az, double bx, double by, double bz) {
    double dx = bx - ax, dy = by - ay, dz = bz - az;
    double dd = dx * dx + dy * dy + dz * dz;
    double t = 0.0;
    if (dd > 1e-12) t = clampd(-(ax * dx + ay * dy + az * dz) / dd, 0.0, 1.0);
    double cx = ax + t * dx, cy = ay + t * dy, cz = az + t * dz;
    return std::sqrt(cx * cx + cy * cy + cz * cz);
}

void move_player(World& w, int i, int j, const double* a, const double* des, double acc, double vmax, double lrate,
                 double agi, double pdt) {
    double m = j == 0 ? 1.0 : -1.0;
    bool step_in = m * des[0] > 0.0 && -m * w.p(i, j, 0) < BODY_MIN + STEPIN_D;
    for (int k : {0, 2}) {
        double dv = des[k] * ((k == 0 && step_in) ? STEPIN_V : 1.0) - w.p(i, j, 3 + k);
        double mx = acc * ((k == 0 && step_in) ? STEPIN_ACC : 1.0) * pdt;
        w.p(i, j, 3 + k) += clampd(dv, -mx, mx);
    }
    double ml = w.t(i, LM0 + j);
    double vy = ml * w.p(i, j, 4);
    double cn = w.t(i, LN0 + j) / LEAN_MAX;
    if (w.t(i, CX0 + j) <= 0.0 && a[11] > 0.3 && std::fabs(a[1]) > 0.5) {
        w.t(i, CX0 + j) = 0.3 - 0.08 * agi;
        w.t(i, CD0 + j) = a[1] > 0.0 ? 1.0 : -1.0;
    }
    double vt, g, lt;
    if (w.t(i, CX0 + j) > 0.0) {
        double cd = w.t(i, CD0 + j);
        vt = cd * vmax * CROSS_V;
        g = (vt - vy) * cd >= 0.0 ? CROSS_ACC : 0.25;
        lt = cd * LEAN_MAX;
        w.t(i, CX0 + j) -= pdt;
    } else {
        vt = des[1];
        double s = vt - vy;
        if (cn * s >= 0.0)
            g = 1.0 + 0.5 * std::fabs(cn);
        else
            g = std::max(0.05, 1.0 - 1.3 * std::fabs(cn));
        lt = clampd(a[9] * LEAN_MAX + 0.25 * LEAN_MAX * vy / vmax, -LEAN_MAX, LEAN_MAX);
    }
    double mx = acc * g * pdt;
    vy += clampd(vt - vy, -mx, mx);
    w.p(i, j, 4) = ml * vy;
    double dl = lt - w.t(i, LN0 + j);
    w.t(i, LN0 + j) += clampd(dl, -lrate * pdt, lrate * pdt);
    for (int k = 0; k < 3; ++k) w.p(i, j, k) += w.p(i, j, 3 + k) * pdt;
}

void step_env(World& w, int i, const double* A, Out& o, double* hist, bool train, int nsub, double dt) {
    double* rew = o.rew + (int64_t)i * 2;
    rew[0] = 0.0;
    rew[1] = 0.0;
    o.done[i] = 0;
    o.winner[i] = -1;
    o.reason[i] = R_NONE;
    double pdt = dt / nsub;

    double a[2][ACT_DIM], des[2][3], acc[2], srate[2], vm[2], lrate[2];
    for (int j = 0; j < 2; ++j) {
        double m = j == 0 ? 1.0 : -1.0;
        for (int k = 0; k < ACT_DIM; ++k) a[j][k] = clampd(A[((int64_t)i * 2 + j) * ACT_DIM + k], -1.0, 1.0);
        double agi = w.s(i, j, AGI);
        double fat = w.f(i, j);
        double vmax = vmax_of(agi, fat);
        vm[j] = vmax;
        acc[j] = (4.0 + 8.0 * agi) * (1.0 - 0.3 * fat);
        lrate[j] = (1.3 + 1.8 * agi) * LEAN_MAX;
        des[j][0] = m * a[j][0] * vmax;
        des[j][1] = a[j][1] * vmax;
        des[j][2] = a[j][2] * vmax * 0.7;
        rew[j] -= 0.0005 * (a[j][0] * a[j][0] + a[j][1] * a[j][1] + a[j][2] * a[j][2]);
        if (a[j][8] > HAND_FORCE)
            w.t(i, BT0 + j) = 1.0;
        else if (a[j][8] < -HAND_FORCE)
            w.t(i, BT0 + j) = -1.0;
        else if (w.in(i, PHASE) == PH_FLY && w.in(i, TURN) == j && w.in(i, PVALID) == 1 &&
                 w.t(i, RT) - w.t(i, EVT) >= react_delay(w.s(i, j, REA)))
            w.t(i, BT0 + j) = w.t(i, LM0 + j) * (w.t(i, PY) - w.p(i, j, 1)) < -0.07 ? 1.0 : -1.0;
        srate[j] = 2.0 / ((0.2 + 0.25 * (1.0 - agi)) * (1.0 + 0.4 * fat));
    }

    for (int s = 0; s < nsub; ++s) {
        if (w.in(i, PHASE) == PH_FLY) update_contact(w, i);
        for (int j = 0; j < 2; ++j) {
            bool recovering = w.t(i, REC0 + j) > 0.0;
            double acc_j = recovering ? acc[j] * REC_ACC : acc[j];
            if (w.t(i, DV0 + j) > 0.0) {
                acc_j = acc[j] * DIVE_ACC;
                w.t(i, DV0 + j) -= pdt;
            }
            double srate_j = recovering ? srate[j] / STROKE_P[(int)w.t(i, LK0 + j)][9] : srate[j];
            if (recovering) w.t(i, REC0 + j) -= pdt;
            bool was_set = w.t(i, SET0 + j) >= SETTLE_T;
            if (!recovering && (was_set || w.t(i, SPL0 + j) > 0.0)) acc_j *= SPLIT_ACC;
            w.t(i, SPL0 + j) = std::max(w.t(i, SPL0 + j) - pdt, 0.0);
            move_player(w, i, j, a[j], des[j], acc_j, vm[j], lrate[j], w.s(i, j, AGI), pdt);
            double spd = std::sqrt(w.p(i, j, 3) * w.p(i, j, 3) + w.p(i, j, 4) * w.p(i, j, 4));
            if (spd < 0.35 * vm[j] && w.t(i, CX0 + j) <= 0.0) {
                w.t(i, SET0 + j) = std::min(w.t(i, SET0 + j) + pdt, 1.0);
            } else {
                if (was_set) w.t(i, SPL0 + j) = SPLIT_T;
                w.t(i, SET0 + j) = 0.0;
            }
            if (w.in(i, PHASE) == PH_FLY && w.in(i, TURN) == j) {
                double dc = a[j][10] * LAT_MAX - w.t(i, CU0 + j);
                w.t(i, CU0 + j) += clampd(dc, -CUE_RATE * pdt, CUE_RATE * pdt);
            }
            double d = w.t(i, BT0 + j) - w.t(i, BL0 + j);
            double ds = srate_j * pdt;
            w.t(i, BL0 + j) += clampd(d, -ds, ds);
            clamp_paddle(w, i, j);
            if (w.in(i, PHASE) == PH_SERVE && j != w.in(i, SERVER)) clamp_receiver(w, i, j);
        }

        bool ended = false;
        int64_t ph = w.in(i, PHASE);
        if (ph == PH_SERVE) {
            int srv = (int)w.in(i, SERVER);
            int sk = (int)w.t(i, SPLAN0 + srv);
            bool sbh = serve_is_bh(sk);
            w.t(i, BL0 + srv) = sbh ? -1.0 : 1.0;
            w.t(i, BT0 + srv) = w.t(i, BL0 + srv);
            if (w.t(i, TOSS) < 0.0) {
                double tz = serve_toss((int)w.t(i, SPLAN0 + srv), a[srv][9]);
                if (w.t(i, TIMER) <= toss_time(tz)) {
                    w.t(i, TOSS) = tz;
                } else {
                    double dy = serve_spot(lat_m(w, i, srv), a[srv][1], sk) - w.b(i, 1);
                    w.b(i, 1) += clampd(dy, -1.2 * pdt, 1.2 * pdt);
                }
            }
            place_server(w, i, srv, sbh);
            w.t(i, TIMER) -= pdt;
            if (w.t(i, TIMER) <= 0.0) {
                if (serve_hit(w, i, a[srv])) {
                    end_point(w, i, 1 - srv, R_FAULT, train, o);
                    ended = true;
                } else {
                    w.in(i, PHASE) = PH_FLY;
                    predict(w, i);
                    w.t(i, EVT) += ((FAM_SURPRISE - (FAM_SURPRISE + FAM_READ) * w.t(i, SFAM)) *
                                    (1.4 - 0.8 * w.s(i, 1 - srv, REA)));
                }
            }
        } else if (ph == PH_FLY) {
            double x0 = w.b(i, 0), y0 = w.b(i, 1), z0 = w.b(i, 2);
            int ev = ball_substep(w.bp(i), pdt, w.r(i));
            int rec = (int)w.in(i, TURN);
            int hitter = 1 - rec;
            if (ev == 2) {
                w.in(i, NETT) = 1;
                predict(w, i);
            } else if (ev == 1) {
                bool on_rec = rec == 0 ? (w.b(i, 0) < 0.0) : (w.b(i, 0) > 0.0);
                if (w.in(i, HITS) == 0 && w.in(i, SVB) == 0) {
                    if (on_rec) {
                        end_point(w, i, rec, R_SFAULT, train, o);
                        ended = true;
                    } else {
                        w.in(i, SVB) = 1;
                    }
                } else if (!on_rec) {
                    end_point(w, i, rec, w.in(i, NETT) ? R_NET : R_OWN, train, o);
                    ended = true;
                } else if (w.in(i, HITS) == 0 && w.in(i, NETT) && w.in(i, BOUNCED) == 0) {
                    end_point(w, i, -1, R_LET, train, o);
                    ended = true;
                } else {
                    w.in(i, BOUNCED) += 1;
                    if (w.in(i, BOUNCED) == 1) {
                        rew[hitter] += RALLY_W + w.t(i, STB);
                        w.t(i, STB) = 0.0;
                    } else {
                        end_point(w, i, hitter, R_MISS, train, o);
                        ended = true;
                    }
                }
            }
            if (!ended && w.in(i, BOUNCED) == 1 && w.t(i, DV0 + rec) <= 0.0) {
                double ext = ((a[rec][5] + 1.0) * 0.5 > DIVE_PW && w.t(i, BL0 + rec) > 0.5) ? DIVE_EXT : 0.0;
                double px, py, stretch;
                reach_hand(w, i, rec, contact_x(w, i, rec), w.b(i, 1), ext, px, py, stretch);
                double pz = w.p(i, rec, 2);
                double d = seg_dist(x0 - px, y0 - py, z0 - pz, w.b(i, 0) - px, w.b(i, 1) - py, w.b(i, 2) - pz);
                if (d < REACH) do_hit(w, i, rec, a[rec], stretch);
            }
            if (!ended) {
                double bx = w.b(i, 0), by = w.b(i, 1), bz = w.b(i, 2);
                if (bz < TABLE_H - 0.3 || std::fabs(bx) > 4.5 || std::fabs(by) > 3.0 || ev == 3) {
                    int rec2 = (int)w.in(i, TURN);
                    if (w.in(i, BOUNCED) == 0)
                        end_point(w, i, rec2, w.in(i, NETT) ? R_NET : R_OUT, train, o);
                    else
                        end_point(w, i, 1 - rec2, R_MISS, train, o);
                    ended = true;
                }
            }
            w.t(i, RT) += pdt;
            if (!ended && w.t(i, RT) > (train ? RALLY_TIMEOUT : RALLY_TIMEOUT_PLAY)) {
                end_point(w, i, -1, R_TIMEOUT, train, o);
                ended = true;
            }
        } else {
            ball_substep(w.bp(i), pdt, w.r(i));
            w.t(i, TIMER) -= pdt;
            if (w.t(i, TIMER) <= 0.0) start_point(w, i, false);
        }

        double* hs = hist + ((int64_t)i * nsub + s) * HIST_DIM;
        for (int k = 0; k < 3; ++k) {
            hs[k] = w.b(i, k);
            hs[3 + k] = w.p(i, 0, k);
            hs[6 + k] = w.p(i, 1, k);
            hs[9 + k] = w.p(i, 0, k);
            hs[12 + k] = w.p(i, 1, k);
        }
        for (int j = 0; j < 2; ++j) {
            double hx, hy;
            shown_hand(w, i, j, hx, hy);
            hs[9 + 3 * j] = hx;
            hs[10 + 3 * j] = hy;
        }
        hs[15] = w.t(i, BL0);
        hs[16] = w.t(i, BL1);
        for (int j = 0; j < 2; ++j) {
            hs[17 + j] = w.t(i, LN0 + j) / LEAN_MAX;
            hs[19 + j] = w.t(i, CX0 + j) > 0.0 ? w.t(i, CD0 + j) : 0.0;
            hs[21 + j] = w.t(i, CU0 + j);
        }
        bool cueing = w.in(i, PHASE) == PH_FLY;
        hs[23] = cueing ? (double)w.in(i, TURN) : -1.0;

        if (ended && train) {
            start_point(w, i, true);
            for (int s2 = s + 1; s2 < nsub; ++s2) {
                double* h2 = hist + ((int64_t)i * nsub + s2) * HIST_DIM;
                std::memcpy(h2, hs, sizeof(double) * HIST_DIM);
            }
            return;
        }
    }

    if (w.in(i, PHASE) == PH_FLY) {
        int rec = (int)w.in(i, TURN);
        for (int j = 0; j < 2; ++j) {
            double sj = j == 0 ? -1.0 : 1.0;
            if (j == rec) {
                if (w.in(i, PVALID) == 1 && w.t(i, RT) - w.t(i, EVT) >= react_delay(w.s(i, j, REA))) {
                    double hx, hy, st;
                    reach_hand(w, i, j, w.t(i, PX), w.t(i, PY), 0.0, hx, hy, st);
                    double dx = w.t(i, PX) - hx, dy = w.t(i, PY) - hy, dz = w.t(i, PZ) - w.p(i, j, 2);
                    double d = std::sqrt(dx * dx + dy * dy + dz * dz);
                    if (w.t(i, SH0 + j) >= 0.0) rew[j] += 0.3 * (w.t(i, SH0 + j) - d);
                    w.t(i, SH0 + j) = d;
                }
            } else {
                double dx = sj * (HALF_L + 0.6) - w.p(i, j, 0), dy = w.p(i, j, 1), dz = 1.0 - w.p(i, j, 2);
                double d = std::sqrt(dx * dx + dy * dy + dz * dz);
                rew[j] -= READY_W * d;
            }
        }
    }

    for (int j = 0; j < 2; ++j) {
        double vb = vmax_of(w.s(i, j, AGI), 0.0);
        double sp2 = (w.p(i, j, 3) * w.p(i, j, 3) + w.p(i, j, 4) * w.p(i, j, 4) + w.p(i, j, 5) * w.p(i, j, 5)) / (vb * vb);
        w.f(i, j) = std::min(1.0, w.f(i, j) + dt * FAT_RUN * (FAT_BASE - w.s(i, j, STA)) * sp2);
    }
}

void observe_env(const World& w, int i, double* O) {
    int64_t ph = w.in(i, PHASE);
    int64_t rec = w.in(i, TURN);
    for (int j = 0; j < 2; ++j) {
        double m = j == 0 ? 1.0 : -1.0;
        double* o = O + ((int64_t)i * 2 + j) * OBS_DIM;
        int q = 1 - j;
        double ml = w.t(i, LM0 + j);
        double hs = ml * m;
        double rel = ml * w.t(i, LM0 + q);
        o[0] = m * w.p(i, j, 0);
        o[1] = ml * w.p(i, j, 1);
        o[2] = w.p(i, j, 2);
        o[3] = m * w.p(i, j, 3) * 0.25;
        o[4] = ml * w.p(i, j, 4) * 0.25;
        o[5] = w.p(i, j, 5) * 0.25;
        o[6] = m * w.b(i, 0);
        o[7] = ml * w.b(i, 1);
        o[8] = w.b(i, 2);
        o[9] = m * w.b(i, 3) * 0.1;
        o[10] = ml * w.b(i, 4) * 0.1;
        o[11] = w.b(i, 5) * 0.1;
        o[12] = hs * m * w.b(i, 6) * 0.01;
        o[13] = m * w.b(i, 7) * 0.01;
        o[14] = hs * w.b(i, 8) * 0.01;
        o[15] = m * w.p(i, q, 0);
        o[16] = ml * w.p(i, q, 1);
        o[17] = w.p(i, q, 2);
        bool mine = ph == PH_FLY && rec == j;
        o[18] = mine ? 1.0 : 0.0;
        o[19] = mine ? (double)w.in(i, BOUNCED) : 0.0;
        o[20] = ph == PH_SERVE ? 1.0 : 0.0;
        bool vis = mine && w.in(i, PVALID) == 1 && w.t(i, RT) - w.t(i, EVT) >= react_delay(w.s(i, j, REA));
        if (vis) {
            o[21] = 1.0;
            double hx, hy, st;
            reach_hand(w, i, j, w.t(i, PX), w.t(i, PY), 0.0, hx, hy, st);
            o[22] = m * (w.t(i, PX) - hx);
            o[23] = ml * (w.t(i, PY) - hy);
            o[24] = w.t(i, PZ) - w.p(i, j, 2);
            o[25] = clampd(w.t(i, PRT) - w.t(i, RT), 0.0, 1.5);
        } else {
            for (int k = 21; k < 26; ++k) o[k] = 0.0;
        }
        for (int k = 0; k < N_STATS; ++k) {
            o[26 + k] = w.s(i, j, k);
            o[31 + k] = w.s(i, q, k);
        }
        o[36] = w.f(i, j);
        o[37] = w.f(i, q);
        o[38] = w.in(i, SERVER) == j ? 1.0 : 0.0;
        o[39] = w.t(i, BL0 + j);
        o[40] = w.t(i, BL0 + q);
        o[41] = vis ? ml * (w.t(i, PY) - w.p(i, j, 1)) : 0.0;
        o[42] = vis ? m * (w.t(i, PX) - w.p(i, j, 0)) : 0.0;
        o[43] = w.t(i, LN0 + j) / LEAN_MAX;
        o[44] = w.t(i, CD0 + j) * w.t(i, CX0 + j) / 0.3;
        o[45] = rel * w.t(i, LN0 + q) / LEAN_MAX;
        o[46] = rel * w.t(i, CD0 + q) * w.t(i, CX0 + q) / 0.3;
        bool opp_cue = ph == PH_FLY && rec == q;
        o[47] = opp_cue ? 1.0 : 0.0;
        o[48] = opp_cue ? rel * w.t(i, CU0 + q) : 0.0;
        o[49] = w.t(i, CU0 + j);
        for (int k = 0; k < N_STYLE; ++k) o[50 + k] = w.s(i, j, N_STATS + k);
        o[54] = std::min(w.t(i, SET0 + j) / SETTLE_T, 1.0);
        o[55] = std::max(w.t(i, REC0 + j), 0.0) / 0.35;
        o[56] = std::max(w.t(i, REC0 + q), 0.0) / 0.35;
        bool is_serve = ph == PH_FLY && w.in(i, HITS) == 0 && w.in(i, LSRV) >= 0;
        for (int k = 0; k < N_SERVE; ++k) o[57 + k] = (is_serve && w.in(i, LSRV) == k) ? 1.0 : 0.0;
        o[65] = is_serve ? w.t(i, TOSS) : 0.0;
        o[66] = is_serve ? (double)w.in(i, LHID) : 0.0;
        o[67] = is_serve ? 1.0 : 0.0;
        o[68] = m * w.p(i, q, 3) * 0.25;
        o[69] = ml * w.p(i, q, 4) * 0.25;
        o[70] = std::min(w.t(i, SET0 + q) / SETTLE_T, 1.0);
        for (int k = 0; k < N_SERVE; ++k) {
            o[71 + k] = w.t(i, FAM0 + q * N_SERVE + k);
            o[79 + k] = w.t(i, FAM0 + j * N_SERVE + k);
            if (ph == PH_SERVE && w.in(i, SERVER) == j && (int)w.t(i, SPLAN0 + j) == k) o[57 + k] = 1.0;
        }
        o[87] = w.s(i, j, SRV);
        o[88] = w.s(i, q, SRV);
        o[89] = hs * hs_of(w, i, q);
        o[90] = w.s(i, j, GRIP);
        o[91] = w.s(i, q, GRIP);
        for (int f = 0; f < 4; ++f) {
            int r = (int)w.s(i, f < 2 ? j : q, f % 2 == 0 ? RFH : RBH);
            o[92 + 3 * f] = (RUB_P[r][0] - 0.8) * 5.0;
            o[93 + 3 * f] = RUB_P[r][1];
            o[94 + 3 * f] = RUB_P[r][2];
        }
        o[104] = w.s(i, j, MEN);
        o[105] = w.s(i, j, TCH);
        o[106] = w.s(i, q, MEN);
        o[107] = w.s(i, q, TCH);
    }
}

// ---------------------------------------------------------------- 句柄
struct Handle {
    int n;
    std::vector<Rng> rng;
};

World make_world(Handle* h, double* B, double* P, int64_t* I, double* T, double* S, double* F) {
    World w;
    w.n = h->n;
    w.B = B; w.P = P; w.I = I; w.T = T; w.S = S; w.F = F;
    w.rng = &h->rng;
    return w;
}

}  // namespace

// ---------------------------------------------------------------- 导出的 C 接口
PP_API int pp_version() { return 1; }

PP_API void pp_dims(int64_t* out) {
    out[0] = NI;
    out[1] = NT;
    out[2] = N_S;
    out[3] = OBS_DIM;
    out[4] = ACT_DIM;
    out[5] = HIST_DIM;
}

PP_API void* pp_create(int n, uint64_t seed) {
    Handle* h = new Handle();
    h->n = n;
    h->rng.resize(n);
    for (int i = 0; i < n; ++i) h->rng[i].seed(seed * 1000003ull + (uint64_t)i);
    return h;
}

PP_API void pp_destroy(void* hp) { delete static_cast<Handle*>(hp); }

PP_API void pp_seed(void* hp, uint64_t seed) {
    Handle* h = static_cast<Handle*>(hp);
    for (int i = 0; i < h->n; ++i) h->rng[i].seed(seed * 1000003ull + (uint64_t)i);
}

PP_API void pp_set_tape(const double* u, int64_t nu, const double* g, int64_t ng, int64_t* idx) {
    g_tape.u = nu > 0 ? u : nullptr;
    g_tape.nu = nu;
    g_tape.g = ng > 0 ? g : nullptr;
    g_tape.ng = ng;
    g_tape.idx = idx;
}

// 可调参数（numba 版的环境变量）：0 RUB_DEC[5] 1 RUB_SENS[5] 2 MEN_SIG 3 TCH_READ 4 DIVE_PW 5 ERR_SCALE 6 READ_SCALE
PP_API void pp_set_param(int which, const double* v) {
    switch (which) {
        case 0: for (int k = 0; k < N_RUB; ++k) RUB_DEC[k] = v[k]; break;
        case 1: for (int k = 0; k < N_RUB; ++k) RUB_SENS[k] = v[k]; break;
        case 2: MEN_SIG = v[0]; break;
        case 3: TCH_READ = v[0]; break;
        case 4: DIVE_PW = v[0]; break;
        case 5: ERR_SCALE = v[0]; break;
        case 6: READ_SCALE = v[0]; break;
        default: break;
    }
}

PP_API void pp_step(void* hp, double* B, double* P, int64_t* I, double* T, double* S, double* F, const double* A,
                    double* rew, uint8_t* done, int64_t* winner, int64_t* reason, int64_t* hits, double* hist,
                    int train, int nsub, double dt) {
    Handle* h = static_cast<Handle*>(hp);
    World w = make_world(h, B, P, I, T, S, F);
    Out o{rew, done, winner, reason, hits};
    int n = h->n;
    if (g_tape.u || g_tape.g) {
        for (int i = 0; i < n; ++i) step_env(w, i, A, o, hist, train != 0, nsub, dt);
        return;
    }
#pragma omp parallel for schedule(dynamic, 4)
    for (int i = 0; i < n; ++i) step_env(w, i, A, o, hist, train != 0, nsub, dt);
}

PP_API void pp_reset(void* hp, double* B, double* P, int64_t* I, double* T, double* S, double* F, int train) {
    Handle* h = static_cast<Handle*>(hp);
    World w = make_world(h, B, P, I, T, S, F);
    for (int i = 0; i < h->n; ++i) {
        for (int k = 0; k < NI; ++k) w.in(i, k) = 0;
        w.in(i, FIRST) = w.r(i).randint(0, 2);
        w.in(i, LWIN) = -1;
        w.f(i, 0) = 0.0;
        w.f(i, 1) = 0.0;
        for (int k = 0; k < 2 * N_SERVE; ++k) w.t(i, FAM0 + k) = 0.0;
        reset_serve_stats(w, i);
        reset_mind(w, i);
        start_point(w, i, train != 0);
    }
}

PP_API void pp_start_point(void* hp, int i, double* B, double* P, int64_t* I, double* T, double* S, double* F,
                           int train) {
    Handle* h = static_cast<Handle*>(hp);
    World w = make_world(h, B, P, I, T, S, F);
    start_point(w, i, train != 0);
}

PP_API void pp_random_equipment(void* hp, int i, int j, double* S, int balanced) {
    Handle* h = static_cast<Handle*>(hp);
    World w = make_world(h, nullptr, nullptr, nullptr, nullptr, S, nullptr);
    random_equipment(w, i, j, balanced != 0);
}

PP_API void pp_observe(void* hp, double* B, double* P, int64_t* I, double* T, double* S, double* F, double* O) {
    Handle* h = static_cast<Handle*>(hp);
    World w = make_world(h, B, P, I, T, S, F);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < h->n; ++i) observe_env(w, i, O);
}

// 单球推进（画面里预测轨迹等用）：用第 0 号环境的随机数
PP_API int pp_ball_substep(void* hp, double* b, double dt) {
    Handle* h = static_cast<Handle*>(hp);
    return ball_substep(b, dt, h->rng[0]);
}

PP_API int pp_serve_type_of(double a8, double a10) { return serve_type_of(a8, a10); }

// 画面层预判技术、发球进度要用的几条规则（与模拟同一份）
PP_API int pp_classify_stroke(const double* b, double pw, double top) { return classify_stroke(b, pw, top); }

PP_API int pp_tech_of(int k, int bh, double pw, double top, double side, double depth, double top_in) {
    return tech_of(k, bh != 0, pw, top, side, depth, top_in);
}

PP_API double pp_incoming_top(const double* v, const double* w) {
    return incoming_top(V3(v[0], v[1], v[2]), V3(w[0], w[1], w[2]));
}

PP_API double pp_serve_toss(int k, double a9) { return serve_toss(k, a9); }

PP_API double pp_toss_time(double toss) { return toss_time(toss); }

PP_API double pp_vmax_of(double agi, double fat) { return vmax_of(agi, fat); }

PP_API double pp_hand_off(double blend) { return hand_off(blend); }

PP_API double pp_pressure(void* hp, int i, int h, int64_t* I, double* T) {
    Handle* hd = static_cast<Handle*>(hp);
    World w = make_world(hd, nullptr, nullptr, I, T, nullptr, nullptr);
    return pressure(w, i, h);
}
