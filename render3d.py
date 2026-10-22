"""立体人物渲染：在 pygame 的 2D 画布上用透视投影画出有明暗的圆柱（四肢）、球体（头、关节）、
多面体（躯干、短裤、鞋）和有厚度的球拍。所有图元按到镜头的深度从远到近排序后绘制（画家算法）。

光照：一盏主光（斜上方）加环境光。圆柱按“侧面朝光”的一侧加亮带、背光一侧压暗；
多面体每个面按法向与光的夹角着色（兰伯特）；球体用预渲染的径向渐变贴图，高光偏向光源。
"""
import math

import numpy as np
import pygame

LIGHT = np.array([0.35, -0.45, 1.0])
LIGHT = LIGHT / np.linalg.norm(LIGHT)
AMBIENT = 0.42


def shade(col, k):
    return tuple(int(min(max(c * k, 0), 255)) for c in col)


def mix(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


class Painter:
    """收集一个场景片段（如一名球员）的图元，排序后画到画布上。"""

    def __init__(self, cam, surf):
        self.cam = cam
        self.surf = surf
        self.items = []
        self._sphere_cache = {}

    def begin(self, surf):
        self.surf = surf
        self.items = []

    def flush(self):
        for _, fn in sorted(self.items, key=lambda t: -t[0]):
            fn()
        self.items = []

    # ---------------------------------------------------------------- 工具
    def light_k(self, n):
        return AMBIENT + (1.0 - AMBIENT) * max(float(np.dot(unit(n), LIGHT)), 0.0)

    def screen_dir(self, p, d):
        a = np.array(self.cam.pt(p))
        b = np.array(self.cam.pt(np.asarray(p) + 0.05 * unit(d)))
        v = b - a
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-6 else np.array([-0.6, -0.8])

    def visible(self, center, normal):
        return float(np.dot(np.asarray(center) - self.cam.pos, normal)) < 0.0

    # ---------------------------------------------------------------- 图元
    def capsule(self, a, b, ra, rb, col, depth_bias=0.0, outline=True, seg=None):
        """seg：按这个长度切成几段、各自排序。长的一整根按中点排序时，一头在头/脖子前面、
        另一头在后面就只能整根画在上面或下面，看起来像插进身体里。"""
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(1, int(math.ceil(float(np.linalg.norm(b - a)) / seg))) if seg else 1
        mid = (a + b) / 2
        for k in range(n):
            p, q = a + (b - a) * (k / n), a + (b - a) * ((k + 1) / n)
            rp, rq = ra + (rb - ra) * (k / n), ra + (rb - ra) * ((k + 1) / n)
            # 中间的接缝不画圆头（否则一节节像串珠），明暗朝向按整根算（否则相邻两节亮面可能一左一右）
            caps = (k == 0, k == n - 1)
            self.items.append((self.cam.depth((p + q) / 2) + depth_bias,
                               lambda p=p, q=q, rp=rp, rq=rq, caps=caps:
                               self._capsule(p, q, rp, rq, col, outline, caps, mid)))

    def _capsule(self, a, b, ra, rb, col, outline, caps=(True, True), mid=None):
        cam = self.cam
        pa, pb = np.array(cam.pt(a)), np.array(cam.pt(b))
        sa, sb = max(cam.scale(a, ra), 1.0), max(cam.scale(b, rb), 1.0)
        d = pb - pa
        dn = float(np.linalg.norm(d))
        dark = shade(col, 0.5)
        if dn < 1e-3:
            pygame.draw.circle(self.surf, col, pa, max(sa, sb))
            return
        u = d / dn
        n = np.array([-u[1], u[0]])
        # 主光在圆柱截面上的投影 → 屏幕上亮的一侧
        ax = unit(b - a)
        nl = LIGHT - np.dot(LIGHT, ax) * ax
        side = float(np.linalg.norm(nl))
        sd = self.screen_dir((a + b) / 2 if mid is None else mid, nl) if side > 1e-3 else n
        sgn = 1.0 if float(np.dot(sd, n)) >= 0.0 else -1.0
        lit = AMBIENT + (1.0 - AMBIENT) * min(side * 1.1, 1.0)

        def band(t0, t1, color):
            # 一条带圆头的亮度带：截面上 [t0, t1]（-1 背光边缘 … +1 迎光边缘）
            q = [pa + n * sgn * sa * t0, pb + n * sgn * sb * t0, pb + n * sgn * sb * t1, pa + n * sgn * sa * t1]
            pygame.draw.polygon(self.surf, color, q)
            m = (t0 + t1) / 2
            h = (t1 - t0) / 2
            if caps[0]:
                pygame.draw.circle(self.surf, color, pa + n * sgn * sa * m, max(sa * h, 0.5))
            if caps[1]:
                pygame.draw.circle(self.surf, color, pb + n * sgn * sb * m, max(sb * h, 0.5))

        band(-1.0, 1.0, shade(col, 0.45 + 0.1 * lit))
        for (t0, t1), k in (((-0.8, 0.96), 0.62), ((-0.5, 0.92), 0.76), ((-0.15, 0.86), 0.88), ((0.2, 0.72), 1.0)):
            band(t0, t1, shade(col, (0.5 + 0.5 * lit) * k + 0.08))
        hi = mix(shade(col, 0.5 + 0.5 * lit + 0.08), (255, 255, 255), 0.16 * lit)
        band(0.36, 0.6, hi)

    def sphere(self, p, r, col, depth_bias=0.0, hair=None, face_dir=None):
        p = np.asarray(p, float)
        self.items.append((self.cam.depth(p) + depth_bias, lambda: self._sphere(p, r, col, hair, face_dir)))

    def _sphere_sprite(self, col, rad, lx, ly):
        key = (col, rad, lx, ly)
        spr = self._sphere_cache.get(key)
        if spr is not None:
            return spr
        size = 2 * rad + 2
        spr = pygame.Surface((size, size), pygame.SRCALPHA)
        steps = max(4, min(rad, 14))
        for k in range(steps):
            t = k / (steps - 1)
            rr = rad * (1.0 - 0.75 * t)
            off = (lx * rad * 0.42 * t, ly * rad * 0.42 * t)
            c = mix(shade(col, 0.5), mix(col, (255, 255, 255), 0.25), t ** 0.8)
            pygame.draw.circle(spr, c, (rad + 1 + off[0], rad + 1 + off[1]), max(rr, 1))
        if len(self._sphere_cache) > 400:
            self._sphere_cache.clear()
        self._sphere_cache[key] = spr
        return spr

    def _sphere(self, p, r, col, hair, face_dir):
        cam = self.cam
        cx, cy = cam.pt(p)
        rad = max(int(round(cam.scale(p, r))), 2)
        ld = self.screen_dir(p, LIGHT - np.dot(LIGHT, cam.fwd) * cam.fwd)
        lx, ly = int(round(ld[0] * 2)) / 2, int(round(ld[1] * 2)) / 2
        spr = self._sphere_sprite(col, rad, lx, ly)
        self.surf.blit(spr, (cx - rad - 1, cy - rad - 1))
        if hair is not None:
            # 头发：头顶和后脑一侧的帽状区域
            back = -unit(face_dir) if face_dir is not None else np.zeros(3)
            top = np.asarray(p) + np.array([0.0, 0.0, r * 0.35]) + back * r * 0.3
            hx, hy = cam.pt(top)
            hr = rad * 0.95
            s = pygame.Surface((2 * rad + 4, 2 * rad + 4), pygame.SRCALPHA)
            pygame.draw.circle(s, (*hair, 255), (hx - cx + rad + 2, hy - cy + rad + 2), hr)
            mask = pygame.Surface((2 * rad + 4, 2 * rad + 4), pygame.SRCALPHA)
            pygame.draw.circle(mask, (255, 255, 255, 255), (rad + 2, rad + 2), rad)
            s.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
            self.surf.blit(s, (cx - rad - 2, cy - rad - 2))

    def hexa(self, corners, col, depth_bias=0.0, edge=True, skip=(), per_face=False, subdiv=1):
        """八个角点的六面体：corners 顺序 [底面 0-3（逆时针）, 顶面 4-7（与底面一一对应）]。
        skip：不画的面（0 底面、1 顶面），用于和别的块贴在一起、本该藏在身体里的面。
        per_face：每个面按自己的中心排序（大块的躯干要这样，否则贴在胸前的东西会被画到后背上）。
        subdiv：per_face 时每个面再切成 subdiv×subdiv 小块各自排序（弯腰时后背整块的中心比头还近，会把头盖住）。"""
        cs = [np.asarray(c, float) for c in corners]
        center = sum(cs) / 8.0
        faces = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
        for fi, f in enumerate(faces):
            if fi in skip:
                continue
            pts = [cs[k] for k in f]
            fc = sum(pts) / 4.0
            nrm = np.cross(pts[1] - pts[0], pts[3] - pts[0])
            if np.dot(nrm, fc - center) < 0:
                nrm = -nrm
            if not self.visible(fc, nrm):
                continue
            k = self.light_k(nrm)
            if per_face and subdiv > 1:
                fcol = shade(col, k)
                for sub in self._split_quad(pts, subdiv):
                    self.items.append((self.cam.depth(sum(sub) / 4.0) + depth_bias,
                                       lambda sub=sub: self._poly(sub, fcol, edge, grow=True)))
                continue
            self.items.append((self.cam.depth(fc if per_face else center) + depth_bias,
                               lambda pts=pts, k=k: self._poly(pts, shade(col, k), edge)))

    @staticmethod
    def _split_quad(pts, m):
        p0, p1, p2, p3 = pts

        def at(s, t):
            return (p0 * (1 - s) + p1 * s) * (1 - t) + (p3 * (1 - s) + p2 * s) * t

        return [[at(i / m, j / m), at((i + 1) / m, j / m), at((i + 1) / m, (j + 1) / m), at(i / m, (j + 1) / m)]
                for i in range(m) for j in range(m)]

    def _poly(self, pts, col, edge, grow=False):
        sp = [self.cam.pt(p) for p in pts]
        pygame.draw.polygon(self.surf, col, sp)
        if grow:
            # 拼起来的小块之间不留缝
            pygame.draw.polygon(self.surf, col, sp, 1)
        if edge:
            pygame.draw.polygon(self.surf, shade(col, 0.7), sp, 1)

    def disc(self, c, u, e2, n, ra, rb, front_col, back_col, rim_col, thick, depth_bias=0.0, pips=(False, False)):
        """球拍拍面：以 c 为中心、u/e2 为长短轴方向的椭圆薄板，n 为正面法向；
        朝向镜头的那一面画在前，边缘画出木板厚度。pips 标记正/反面是不是颗粒胶。"""
        c = np.asarray(c, float)
        self.items.append((self.cam.depth(c) + depth_bias,
                           lambda: self._disc(c, u, e2, n, ra, rb, front_col, back_col, rim_col, thick, pips)))

    def _disc(self, c, u, e2, n, ra, rb, front_col, back_col, rim_col, thick, pips):
        ang = np.linspace(0.0, 2 * math.pi, 22, endpoint=False)
        ring = [ra * math.cos(t) * u + rb * math.sin(t) * e2 for t in ang]
        front_vis = self.visible(c, n)
        top_n = n if front_vis else -n
        col = front_col if front_vis else back_col
        dotted = pips[0] if front_vis else pips[1]
        far = [self.cam.pt(c - top_n * thick * 0.5 + r) for r in ring]
        near = [self.cam.pt(c + top_n * thick * 0.5 + r) for r in ring]
        pygame.draw.polygon(self.surf, rim_col, far)
        for k in range(len(ring)):
            q = [far[k], far[(k + 1) % len(ring)], near[(k + 1) % len(ring)], near[k]]
            pygame.draw.polygon(self.surf, rim_col, q)
        k = self.light_k(top_n)
        pygame.draw.polygon(self.surf, shade(col, 0.6 + 0.5 * k), near)
        if dotted:
            dc = shade(col, 1.5) if sum(col) < 200 else shade(col, 0.7)
            for a in np.linspace(-0.7, 0.7, 5):
                for b in np.linspace(-0.7, 0.7, 5):
                    if a * a + b * b > 0.6:
                        continue
                    p = c + top_n * thick * 0.55 + a * ra * u + b * rb * e2
                    pygame.draw.circle(self.surf, dc, self.cam.pt(p), max(1, int(self.cam.scale(p, 0.006))))
        pygame.draw.polygon(self.surf, shade(rim_col, 1.2), near, 1)
