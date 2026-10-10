#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NSS 高尔夫 · 风速 / 风向 / 横向风 / 纵向风 —— 识别库

从《Nintendo Switch Sports》高尔夫截图里识别风力信息。纯库，无命令行界面；
识别流程由 `read_wind()` 串起来，命令行入口见同目录的 `obs_live_wind_reader.py`。

约定：正前方（瞄准方向）= 0°，顺时针为正；横向风 = 风速×sin(夹角)（左/右），
纵向风 = 风速×cos(夹角)（顺/逆）。风速为 0 时不计算夹角与分量。

依赖：numpy + Pillow。风速读数靠字形模板库 `glyph_bank.npz`（不做 OCR）。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

# --------------------------------------------------------------------------
# 0. 常量与阈值（颜色用 HSV：H 0-360°，S/V 0-1）
# --------------------------------------------------------------------------

# 多色宽带掩膜：定位阶段还不知道风速，用形状打分区分箭头与同色背景。
# 每项 = (色相带, 饱和度下限, 亮度下限)
ARROW_HUES = ((165.0, 205.0, 0.45, 0.55),    # 青（天空/树/远山也偏青）
              (38.0, 80.0, 0.50, 0.70),      # 黄（场地草也偏黄，带子收窄）
              (290.0, 350.0, 0.40, 0.45))    # 品红
# 单色窄带：已知风速后按颜色选一条。实测箭头顶面：
#   1-5 青 H≈186 S≈1.00 V≈0.86 ｜ 6-10 黄 H≈59 S≈1.00 V≈0.78 ｜
#   11-15 品红 H≈322 S≈0.51 V≈0.82（浅粉，饱和度低，门槛单独放低）
# 每项 = (风速下限, 风速上限, 名称, (色相下限, 色相上限), 饱和度下限, 亮度下限)
WIND_COLORS = ((1, 5, "青", (178.0, 194.0), 0.70, 0.72),
               (6, 10, "黄", (50.0, 68.0), 0.70, 0.68),
               (11, 15, "品红", (308.0, 336.0), 0.40, 0.62))


@dataclass
class Config:
    """风盘几何的可调参数（集中原先散落在各定位函数里的魔法数字）。"""
    panel_r_frac: float = 0.0600      # 风盘半径 / 画面宽
    disc_panel_ratio: float = 0.522   # 草地圆盘半轴 / 风盘半径
    k_default: float = 0.83           # 俯视压缩系数兜底


CONFIG = Config()

# 测角超采样倍数（放大风盘裁块后再量宽度剖面）。固定 2：1080p 下放大后半径约 230px，
# 实测 ss=1/2/4 的平均误差 0.36°/0.24°/0.17°，再往上只烧 CPU（耗时按 ss² 涨）。
SUPERSAMPLE = 2

# 游戏里的风速就是 1~15 m/s（三档：1-5 青 / 6-10 黄 / 11-15 品红），0 = 无风。
# 读出来超过这个数一定是把别的白块当成数字了 —— 例如 "3m/s" 里的字形被凑成 "35"。
# 这种时候宁可当"没读出来"（下帧再读），也不能把一个错值算成 35 m/s 的风。
MAX_WIND = 15


# --------------------------------------------------------------------------
# 其余阈值
# --------------------------------------------------------------------------
KIND_GAUGE = "gauge3d"       # 3D 风盘（右手右上 / 左手左上）
DEFAULT_K = CONFIG.k_default
CYAN_H = (158.0, 218.0)      # 兼容旧判据
CYAN_S = 0.15
CYAN_V = 0.42
TOPFACE_V = 0.80             # 箭头"顶面"（明亮面）的亮度下限
GRASS_GMARGIN = 15           # 草地判据
GRASS_GMIN = 95
WHITE_V = 0.68               # 白色 UI（圆环/文字）
WHITE_S = 0.38


@dataclass
class OcrState:
    """
    字形模板库的持有者。默认实例为 `_DEFAULT_OCR`；需要隔离状态
    （多实例/多线程/单测）时自建并显式传给 load_glyph_bank / match_glyphs。
    """
    bank: dict = field(default_factory=dict)


_DEFAULT_OCR = OcrState()


# --------------------------------------------------------------------------
# 1. 基础图像工具
# --------------------------------------------------------------------------

def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """RGB(H,W,3) -> HSV，H∈[0,360)，S,V∈[0,1]。"""
    a = rgb.astype(np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = a.max(axis=2)
    mn = a.min(axis=2)
    d = mx - mn
    # 用 np.where 链代替布尔掩膜索引：结果等价但快约 1.6~2×。
    safe = np.where(d > 1e-6, d, 1.0)
    h = np.where(mx == r, (g - b) / safe % 6.0,
                 np.where(mx == g, (b - r) / safe + 2.0, (r - g) / safe + 4.0))
    h = (h * 60.0) % 360.0
    s = np.where(mx > 1e-6, d / np.where(mx > 1e-6, mx, 1.0), 0.0)
    return np.stack([h, s, mx], axis=2).astype(np.float32)


@dataclass
class Component:
    area: int
    x0: int
    y0: int
    x1: int
    y1: int
    cx: float
    cy: float
    pixels: np.ndarray | None = None

    @property
    def w(self) -> int:
        return self.x1 - self.x0 + 1

    @property
    def h(self) -> int:
        return self.y1 - self.y0 + 1


def label_components(mask: np.ndarray, keep_pixels: bool = False,
                     min_area: int = 1) -> list[Component]:
    """8 邻域连通域（行段 + 并查集，纯 numpy）。"""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return []
    h, w = m.shape
    rys, rx0, rx1 = [], [], []
    for y in range(h):
        row = m[y]
        if not row.any():
            continue
        d = np.diff(row.view(np.int8))
        st = np.flatnonzero(d == 1) + 1
        en = np.flatnonzero(d == -1) + 1
        if row[0]:
            st = np.concatenate(([0], st))
        if row[-1]:
            en = np.concatenate((en, [w]))
        rys.append(np.full(len(st), y, dtype=np.int64))
        rx0.append(st.astype(np.int64))
        rx1.append(en.astype(np.int64))
    ry, rx0, rx1 = np.concatenate(rys), np.concatenate(rx0), np.concatenate(rx1)
    n = len(ry)
    parent = np.arange(n, dtype=np.int64)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return int(a)

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra > rb:
                ra, rb = rb, ra
            parent[rb] = ra

    rows = np.unique(ry)
    for i in range(1, len(rows)):
        y_prev, y_cur = int(rows[i - 1]), int(rows[i])
        if y_cur != y_prev + 1:
            continue
        pa = np.searchsorted(ry, y_prev, "left"); pb = np.searchsorted(ry, y_prev, "right")
        ca = np.searchsorted(ry, y_cur, "left"); cb = np.searchsorted(ry, y_cur, "right")
        a0, a1 = rx0[pa:pb], rx1[pa:pb]
        b0, b1 = rx0[ca:cb], rx1[ca:cb]
        pos = np.searchsorted(a1, b0, side="left")
        for k in range(len(b0)):
            j = int(pos[k])
            while j < len(a0) and a0[j] <= b1[k]:
                union(int(ca + k), int(pa + j)); j += 1

    roots = np.array([find(i) for i in range(n)], dtype=np.int64)
    lengths = (rx1 - rx0).astype(np.float64)
    order = np.argsort(roots, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(roots[order])) + 1)
    out: list[Component] = []
    for g in groups:
        area = int(lengths[g].sum())
        if area < min_area:
            continue
        x0 = int(rx0[g].min()); x1 = int(rx1[g].max()) - 1
        y0 = int(ry[g].min()); y1 = int(ry[g].max())
        cx = float(((rx0[g] + rx1[g] - 1) / 2.0 * lengths[g]).sum() / area)
        cy = float((ry[g] * lengths[g]).sum() / area)
        pixels = None
        if keep_pixels:
            xs = np.concatenate([np.arange(a, b) for a, b in zip(rx0[g], rx1[g])])
            ys = np.repeat(ry[g], (rx1[g] - rx0[g]))
            pixels = np.stack([xs, ys], axis=1).astype(np.float64)
        out.append(Component(area, x0, y0, x1, y1, cx, cy, pixels))
    out.sort(key=lambda c: -c.area)
    return out


def fit_circle_ls(pts: np.ndarray) -> tuple[float, float, float]:
    """最小二乘圆拟合 -> (cx, cy, r)。"""
    x, y = pts[:, 0], pts[:, 1]
    A = np.stack([x, y, np.ones_like(x)], axis=1)
    sol, *_ = np.linalg.lstsq(A, x ** 2 + y ** 2, rcond=None)
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    return cx, cy, math.sqrt(max(sol[2] + cx * cx + cy * cy, 1e-9))


def fit_ellipse_axis_aligned(pts: np.ndarray, iters: int = 6) -> dict | None:
    """轴平行椭圆拟合（稳健迭代）。"""
    if len(pts) < 12:
        return None
    idx = np.arange(len(pts))
    res = None
    cx = cy = a = b = 0.0
    for _ in range(iters):
        x, y = pts[idx, 0], pts[idx, 1]
        A = np.stack([x * x, y * y, x, y, np.ones_like(x)], axis=1)
        _, _, vt = np.linalg.svd(A, full_matrices=False)
        al, be, D, E, F = vt[-1]
        if abs(al) < 1e-12 or abs(be) < 1e-12:
            return None
        cx, cy = -D / (2 * al), -E / (2 * be)
        s = al * cx * cx + be * cy * cy - F
        if abs(s) < 1e-12 or al / s <= 0 or be / s <= 0:
            return None
        a, b = 1.0 / math.sqrt(al / s), 1.0 / math.sqrt(be / s)
        rr = np.sqrt(((pts[:, 0] - cx) / a) ** 2 + ((pts[:, 1] - cy) / b) ** 2)
        res = np.abs(rr - 1.0)
        thr = max(0.01, float(np.percentile(res[idx], 60)) * 1.5)
        nxt = np.nonzero(res < thr)[0]
        if len(nxt) < 12:
            break
        idx = nxt
    return dict(cx=float(cx), cy=float(cy), a=float(a), b=float(b),
                k=float(min(a, b) / max(a, b)), n=int(len(idx)),
                rms=float(np.sqrt((res[idx] ** 2).mean())) if res is not None else 1.0,
                major_is_x=bool(a >= b))


def _bin_white(crop: Image.Image) -> np.ndarray:
    """
    近白掩膜：低饱和 + **相对亮度**阈值（用于切字形）。

    别用固定亮度阈值：风盘那圈面板是半透明的，背景亮（天空）时整块面板都变浅，
    连背景一起被当成"白"，抗锯齿边缘也被算进笔画 → 字被撑粗一圈，字形匹配从 0.94
    掉到 0.73，于是"偶尔读不出风速"。这里按本块的中位亮度做参照：只取明显更亮的部分。
    """
    a = np.asarray(crop.convert("RGB")).astype(np.int16)
    mx = a.max(axis=2); mn = a.min(axis=2)
    sat = (mx - mn) / np.maximum(mx, 1)
    bg, hi = float(np.median(mx)), float(mx.max())
    thr = max(bg + 0.35 * (hi - bg), bg + 25.0)
    return (mx > thr) & (sat < 0.40)


def arrow_mask(hsv: np.ndarray) -> np.ndarray:
    """
    三色宽带箭头掩膜（青/黄/品红）。色相带已排除草地(H≈104°)、天空(H≈206°)；
    不要再加"G 比 B 高即草地"的判据，会误杀青色箭头。
    """
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    out = np.zeros(H.shape, dtype=bool)
    for lo, hi, smin, vmin in ARROW_HUES:
        out |= (H >= lo) & (H <= hi) & (S >= smin) & (V >= vmin)
    return out


def wind_color(speed: float | None) -> tuple[float, float, float, float] | None:
    """风速 -> 该档箭头的单色窄带 (hue_lo, hue_hi, s_min, v_min)；0/超范围返回 None。"""
    if speed is None:
        return None
    s = int(round(abs(float(speed))))
    for lo, hi, _name, (h0, h1), smin, vmin in WIND_COLORS:
        if lo <= s <= hi:
            return (h0, h1, smin, vmin)
    return None


def arrow_mask_band(hsv: np.ndarray, band: tuple[float, float, float, float]) -> np.ndarray:
    """单色窄带掩膜（已知风速后使用）。"""
    h0, h1, smin, vmin = band
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (H >= h0) & (H <= h1) & (S >= smin) & (V >= vmin)


# --------------------------------------------------------------------------
# 2. 风盘定位
# --------------------------------------------------------------------------

# 风盘先验（cx/W, cy/H, r/W）—— 按 1920×1080 采集实测：
#   右手：右上角，中心 (1690,140)、半径 115；
#   左手：HUD 左右**完全对称**，风盘挪到左上角，大小位置都是镜像（别再改成缩小版）。
GAUGE_PRIOR = (0.8802, 0.1296, 0.0600)
GAUGE_PRIOR_LEFT = (1.0 - GAUGE_PRIOR[0], GAUGE_PRIOR[1], GAUGE_PRIOR[2])
GAUGE_PRIORS = (("右", GAUGE_PRIOR), ("左", GAUGE_PRIOR_LEFT))

@dataclass
class Dial:
    kind: str
    cx: float
    cy: float
    r: float
    k: float = DEFAULT_K
    anchored_by: str = ""


def _ring_coverage(pts: np.ndarray, cx: float, cy: float, nbins: int = 72) -> float:
    if len(pts) == 0:
        return 0.0
    ang = np.degrees(np.arctan2(pts[:, 0] - cx, -(pts[:, 1] - cy))) % 360.0
    return len(np.unique((ang / (360.0 / nbins)).astype(int) % nbins)) / nbins


def _in_gauge_zone(cx: float, cy: float, shape, side: str | None = None) -> bool:
    """
    风盘在画面上方**任一角**：右手在右上，左手（HUD 镜像）在左上。

    side 给定时只认那一侧 —— 玩家一局之内不会换手，锁定侧别可以彻底避免
    "在对侧误抓一个白色 UI 当风盘"（那种情况下会一直报"风速未识别"）。
    """
    h, w = shape[:2]
    if cy > 0.30 * h:
        return False
    if side == "右":
        return cx >= 0.70 * w
    if side == "左":
        return cx <= 0.30 * w
    return cx >= 0.70 * w or cx <= 0.30 * w


def _fit_ellipse_4pt(pts: np.ndarray):
    x, y = pts[:, 0], pts[:, 1]
    A = np.stack([x * x, y * y, x, y, np.ones_like(x)], axis=1)
    try:
        _, _, vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None
    al, be, D, E, F = vt[-1]
    if abs(al) < 1e-12 or abs(be) < 1e-12 or al * be <= 0:
        return None
    cx, cy = -D / (2 * al), -E / (2 * be)
    val = al * cx * cx + be * cy * cy - F
    if val == 0 or al / val <= 0 or be / val <= 0:
        return None
    return cx, cy, 1.0 / math.sqrt(al / val), 1.0 / math.sqrt(be / val)


def _fit_disc_ransac(pts: np.ndarray, tol: float, iters: int = 600,
                     seed: int = 0) -> dict | None:
    """轴平行椭圆 RANSAC（抗杂点）。"""
    if len(pts) < 20:
        return None
    rng = np.random.default_rng(seed)
    n = len(pts)
    best = None
    for _ in range(iters):
        m = _fit_ellipse_4pt(pts[rng.choice(n, size=4, replace=False)])
        if m is None:
            continue
        cx, cy, a, b = m
        if not (0.35 <= b / a <= 1.0):
            continue
        r = np.sqrt(((pts[:, 0] - cx) / a) ** 2 + ((pts[:, 1] - cy) / b) ** 2)
        d = np.abs(r - 1.0) * min(a, b)
        inl = int((d < tol).sum())
        if best is None or inl > best[0]:
            best = (inl, cx, cy, a, b)
    if best is None or best[0] < 20:
        return None
    inl, cx, cy, a, b = best
    refined = None
    for _ in range(4):
        r = np.sqrt(((pts[:, 0] - cx) / a) ** 2 + ((pts[:, 1] - cy) / b) ** 2)
        d = np.abs(r - 1.0) * min(a, b)
        sel = d < tol
        if sel.sum() < 12:
            break
        fit = fit_ellipse_axis_aligned(pts[sel], iters=3)
        if not fit:
            break
        cx, cy, a, b = fit["cx"], fit["cy"], fit["a"], fit["b"]
        rms = float(np.sqrt((((np.sqrt(((pts[sel][:, 0] - cx) / a) ** 2 +
                                      ((pts[sel][:, 1] - cy) / b) ** 2) - 1.0)
                             * min(a, b)) ** 2).mean()))
        refined = dict(cx=cx, cy=cy, a=a, b=b, k=min(a, b) / max(a, b),
                       n=int(sel.sum()), rms=rms, major_is_x=bool(a >= b))
    if refined is not None:      # 迭代精修成功：用最后一轮的内点拟合
        return refined
    return dict(cx=cx, cy=cy, a=a, b=b, k=min(a, b) / max(a, b), n=int(inl),
                rms=float(tol), major_is_x=bool(a >= b))


def _fit_disc_local(rgb: np.ndarray, cx: float, cy: float, half: int,
                    arrow: np.ndarray | None = None) -> dict | None:
    """在局部窗口里用"鲜绿草地 ∪ 箭头"轮廓拟合圆盘（得到 k）。"""
    H_, W_ = rgb.shape[:2]
    half = max(half, 60)
    x0, x1 = int(max(cx - half, 0)), int(min(cx + half, W_))
    y0, y1 = int(max(cy - half, 0)), int(min(cy + half, H_))
    if x1 - x0 < 40 or y1 - y0 < 40:
        return None
    win = rgb[y0:y1, x0:x1]
    R = win[..., 0].astype(np.int16); G = win[..., 1].astype(np.int16)
    B = win[..., 2].astype(np.int16)
    vivid = (G > R + 30) & (G > B + 50) & (G > 150)
    if arrow is not None:
        vivid = vivid | arrow[y0:y1, x0:x1]
    min_a = max(60, int(0.00012 * rgb.shape[0] * rgb.shape[1]))
    comps = label_components(vivid, keep_pixels=True, min_area=min_a)
    if not comps:
        return []
    comp = comps[0]
    px = comp.pixels
    top: dict[int, int] = {}
    for x, y in zip(px[:, 0].astype(int), px[:, 1].astype(int)):
        if x not in top or y < top[x]:
            top[x] = y
    xs = np.array(sorted(top), dtype=np.float64)
    ys = np.array([top[int(x)] for x in xs], dtype=np.float64)
    if len(xs) < 20:
        return None
    pts = np.stack([xs, ys], axis=1)
    k_trim = max(1, int(0.04 * len(pts)))
    if len(pts) - 2 * k_trim >= 20:
        pts = pts[k_trim:len(pts) - k_trim]
    fit = _fit_disc_ransac(pts, tol=max(1.5, 0.014 * max(comp.w, comp.h)), iters=600)
    if fit is None or not (0.62 <= fit["k"] <= 1.0):
        return None
    fit["cx"] += x0; fit["cy"] += y0
    return fit


def _disc_has_arrow(arrow: np.ndarray, fit: dict, min_px: int = 120) -> bool:
    """圆盘内是否含足够多的箭头像素（arrow 为已算好的三色掩膜，避免整帧重算）。"""
    a, b = fit["a"] * 1.35, fit["b"] * 1.45
    cx, cy = fit["cx"], fit["cy"]
    H_, W_ = arrow.shape
    x0, x1 = int(max(cx - a, 0)), int(min(cx + a, W_))
    y0, y1 = int(max(cy - b, 0)), int(min(cy + b, H_))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return False
    yy, xx = np.mgrid[y0:y1, x0:x1]                 # 只算圆盘外接框，不整帧
    inside = ((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2 <= 1.0
    return int((arrow[y0:y1, x0:x1] & inside).sum()) >= min_px


def _find_dial_by_arrow_color(rgb: np.ndarray, hsv: np.ndarray, arrow: np.ndarray,
                              verbose: bool = False,
                              side: str | None = None) -> Dial | None:
    """箭头颜色锚点定位；半径取先验 panel_r_frac×画面宽（比从箭头尺寸反推可靠）。"""
    img_h, img_w = rgb.shape[:2]
    am = arrow.copy()                    # 复用已算好的三色掩膜，避免整帧重算
    am[: int(0.03 * img_h), :] = False
    R = rgb[..., 0].astype(np.int16); G = rgb[..., 1].astype(np.int16)
    B = rgb[..., 2].astype(np.int16)
    vivid = (G > R + 30) & (G > B + 50) & (G > 150)
    min_a = max(40, int(0.00015 * img_w * img_h))
    comps = label_components(am, keep_pixels=True, min_area=min_a)
    cands = []
    for c in comps:
        if not _in_gauge_zone(c.cx, c.cy, rgb.shape, side):
            continue
        if c.area > 0.05 * img_h * img_w:
            continue
        d = max(c.w, c.h)
        if not (0.020 * img_w <= d <= 0.095 * img_w):
            continue
        ar = d / max(min(c.w, c.h), 1)
        if not (0.85 <= ar <= 2.8):
            continue
        if not _solid(c):
            continue
        pad = int(d * 1.1)
        x0, x1 = max(int(c.cx) - pad, 0), min(int(c.cx) + pad, img_w)
        y0, y1 = max(int(c.cy) - pad, 0), min(int(c.cy) + pad, img_h)
        if int(vivid[y0:y1, x0:x1].sum()) < 0.8 * c.area:
            continue
        cands.append((_cand_score(c, hsv), c))
    if not cands:
        return None
    cands.sort(key=lambda z: -z[0])
    c = cands[0][1]
    panel_r = CONFIG.panel_r_frac * img_w
    disc_a = panel_r * CONFIG.disc_panel_ratio
    d = Dial(kind=KIND_GAUGE, cx=float(c.cx), cy=float(c.cy) + 0.30 * disc_a,
             r=panel_r, k=DEFAULT_K, anchored_by="arrow")
    if verbose:
        print(f"[定位] 箭头锚点：团块({c.cx:.0f},{c.cy:.0f}) {c.w}×{c.h} "
              f"-> 半径用先验 {panel_r:.0f}（0.060×{img_w}）")
    return d


def _find_dial_by_disc(rgb: np.ndarray, arrow: np.ndarray,
                       verbose: bool = False,
                       side: str | None = None) -> Dial | None:
    """草地圆盘锚点：找鲜绿圆盘 -> 上沿 RANSAC 椭圆 -> 半径/中心/k。"""
    R = rgb[..., 0].astype(np.int16); G = rgb[..., 1].astype(np.int16)
    B = rgb[..., 2].astype(np.int16)
    grass = (G > R + 30) & (G > B + 50) & (G > 150)
    grass = grass | arrow
    img_h, img_w = rgb.shape[:2]
    comps = label_components(grass, keep_pixels=True, min_area=300)
    best = None
    for comp in comps[:20]:
        if comp.area < 0.0012 * img_h * img_w:
            continue
        if comp.w > 0.30 * img_w or comp.h > 0.30 * img_h:
            continue
        if not (0.5 <= comp.w / max(comp.h, 1) <= 2.0):
            continue
        px = comp.pixels
        top: dict[int, int] = {}
        for x, y in zip(px[:, 0].astype(int), px[:, 1].astype(int)):
            if x not in top or y < top[x]:
                top[x] = y
        xs = np.array(sorted(top), dtype=np.float64)
        ys = np.array([top[int(x)] for x in xs], dtype=np.float64)
        if len(xs) < 20:
            continue
        pts = np.stack([xs, ys], axis=1)
        k_trim = max(1, int(0.04 * len(pts)))
        if len(pts) - 2 * k_trim >= 20:
            pts = pts[k_trim:len(pts) - k_trim]
        fit = _fit_disc_ransac(pts, tol=max(1.5, 0.014 * max(comp.w, comp.h)), iters=600)
        if fit is None or not (0.74 <= fit["k"] <= 0.94):
            continue
        if not (0.02 * img_w <= fit["a"] <= 0.30 * img_w):
            continue
        panel_r_try = fit["a"] / CONFIG.disc_panel_ratio
        if not (0.72 * CONFIG.panel_r_frac * img_w <= panel_r_try
                <= 1.32 * CONFIG.panel_r_frac * img_w):
            continue        # 半径离先验太远：多半拟合到别的东西
        if abs(2 * fit["a"] - comp.w) / max(comp.w, 1) > 0.25:
            continue
        if not _in_gauge_zone(fit["cx"], fit["cy"], rgb.shape, side):
            continue
        if not _disc_has_arrow(arrow, fit):
            continue
        panel_r = fit["a"] / CONFIG.disc_panel_ratio
        score = 1.0 / (1.0 + fit["rms"]) - abs(fit["cx"] / img_w - 0.75) * 0.2
        if best is None or score > best[0]:
            best = (score, fit, panel_r)
    if best is None:
        return None
    _, fit, panel_r = best
    d = Dial(kind=KIND_GAUGE, cx=fit["cx"], cy=fit["cy"], r=panel_r,
             k=float(fit["k"]), anchored_by="disc")
    if verbose:
        print(f"[定位] 草地圆盘法：{2*fit['a']:.0f}×{2*fit['b']:.0f} k={fit['k']:.3f} "
              f"-> 中心=({d.cx:.0f},{d.cy:.0f}) r={d.r:.0f}")
    return d


def _find_dial_by_arrow(rgb: np.ndarray, arrow: np.ndarray,
                        verbose: bool = False,
                        side: str | None = None) -> Dial | None:
    """再兜底：箭头团块定位 + 局部草地拟合 k。"""
    img_h, img_w = rgb.shape[:2]
    min_a = max(40, int(0.00015 * img_w * img_h))
    comps = label_components(arrow, keep_pixels=True, min_area=min_a)
    for comp in comps[:6]:
        if comp.area > 0.04 * img_h * img_w:
            continue
        d0 = max(comp.w, comp.h)
        if d0 < 14 or d0 / max(min(comp.w, comp.h), 1) > 5.0:
            continue
        if not _in_gauge_zone(comp.cx, comp.cy, rgb.shape, side):
            continue
        r_prior = CONFIG.panel_r_frac * img_w
        fit = _fit_disc_local(rgb, comp.cx,
                              comp.cy + 0.3 * r_prior * CONFIG.disc_panel_ratio,
                              int(r_prior * 1.6), arrow)
        if fit is None or not (0.74 <= fit["k"] <= 0.94):
            continue
        if not (0.72 * r_prior <= fit["a"] / CONFIG.disc_panel_ratio <= 1.32 * r_prior):
            continue
        if not _in_gauge_zone(fit["cx"], fit["cy"], rgb.shape, side):
            continue
        d = Dial(kind=KIND_GAUGE, cx=fit["cx"], cy=fit["cy"],
                 r=fit["a"] / CONFIG.disc_panel_ratio,
                 k=float(fit["k"]), anchored_by="arrow+disc")
        if verbose:
            print(f"[定位] 箭头+圆盘：k={fit['k']:.3f} -> 中心=({d.cx:.0f},{d.cy:.0f})")
        return d
    return None


def _find_dial_by_prior(rgb: np.ndarray, verbose: bool = False,
                        side: str | None = None) -> Dial | None:
    """
    按固定先验定位（最快路径）：右手在右上、左手在左上，各自的先验小窗里找草地圆盘。
    圆盘在（含 0 m/s），且左右手可由此区分；找到就直接用先验的中心/半径。
    先验只用来"开窗"，找不到就返回 None，交给后面的完整搜索。
    side 给定时只试那一侧（玩家锁定左右手后就不再换）。
    """
    img_h, img_w = rgb.shape[:2]
    for name, (fx, fy, fr) in GAUGE_PRIORS:
        if side is not None and name != side:
            continue
        cx, cy, r = fx * img_w, fy * img_h, fr * img_w
        half = int(r * 1.5)
        x0, x1 = int(max(cx - half, 0)), int(min(cx + half, img_w))
        y0, y1 = int(max(cy - half, 0)), int(min(cy + half, img_h))
        if x1 - x0 < 20 or y1 - y0 < 20:
            continue
        win = rgb[y0:y1, x0:x1]
        R = win[..., 0].astype(np.int16); G = win[..., 1].astype(np.int16)
        B = win[..., 2].astype(np.int16)
        green = (G > R + 25) & (G > B + 45) & (G > 140)
        comps = [c for c in label_components(green, min_area=int(0.02 * r * r))
                 if 0.5 <= c.w / float(max(c.h, 1)) <= 1.6]      # 圆盘：略宽于高
        if not comps:
            continue
        c = max(comps, key=lambda z: z.area)
        # 圆盘在盘面中心上方 0.30·disc_a（0 m/s 也是这个布局）；
        # 质心离得太远说明窗口里是别的绿色物体 —— 不采信，交回完整搜索。
        exp_cy = cy - 0.30 * r * CONFIG.disc_panel_ratio
        if math.hypot((x0 + c.cx) - cx, (y0 + c.cy) - exp_cy) > 0.5 * r:
            continue
        d = Dial(kind=KIND_GAUGE, cx=float(cx), cy=float(cy), r=float(r),
                 k=DEFAULT_K, anchored_by=f"prior-{name}")
        if verbose:
            print(f"[定位] 先验({name})：圆盘 {c.w}×{c.h} -> 中心=({cx:.0f},{cy:.0f}) r={r:.0f}")
        return d
    return None


def find_dials(rgb: np.ndarray, hsv: np.ndarray | None = None,
               verbose: bool = False, side: str | None = None) -> list[Dial]:
    """
    找 3D 风盘：先试固定先验，失败再走白环 / 箭头锚点 / 草地圆盘搜索。
    side="右"/"左" 时只搜那一侧（调用方锁定左右手后用）。
    """
    d0 = _find_dial_by_prior(rgb, verbose, side)
    if d0 is not None:
        return [d0]
    if hsv is None:                      # 先验没命中才需要整帧 HSV
        hsv = rgb_to_hsv(rgb)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    img_h, img_w = rgb.shape[:2]
    white = (V > WHITE_V) & (S < WHITE_S)
    cyan = arrow_mask(hsv)
    dials: list[Dial] = []
    grass = None                        # 草地掩膜：只在首次需要时整帧算一次（通常 0~2 次）
    comps = label_components(white, keep_pixels=True, min_area=30)
    for comp in comps:
        if comp.area > 0.02 * img_h * img_w:
            continue
        w, h = comp.w, comp.h
        if w < 16 or h < 16 or max(w, h) > 0.35 * min(img_h, img_w):
            continue
        if max(w, h) / max(min(w, h), 1) > 1.6:
            continue
        fill = comp.area / float(w * h)
        if not (0.03 <= fill <= 0.5):
            continue
        px = comp.pixels
        cx, cy, r = comp.x0 + w / 2.0, comp.y0 + h / 2.0, max(w, h) / 2.0
        d = np.sqrt((px[:, 0] - cx) ** 2 + (px[:, 1] - cy) ** 2)
        ring_pts = px[(d > 0.6 * r) & (d < 1.15 * r)]
        if len(ring_pts) >= 24:
            cx, cy, r = fit_circle_ls(ring_pts)
        # 白环候选必须先落在"风盘该在的区域"（上方任一角；锁定左右手后只认那一侧）——
        # 原来这里没有区域检查，会画面里随便一个白圆环（小地图框、名字条）都被当成风盘。
        if not _in_gauge_zone(cx, cy, rgb.shape, side):
            continue
        if not (12 <= r <= 0.35 * min(img_h, img_w)):
            continue
        cov = _ring_coverage(ring_pts, cx, cy)
        if cov < 0.78:
            continue
        if grass is None:
            G = rgb[..., 1].astype(np.int16); R = rgb[..., 0].astype(np.int16)
            B = rgb[..., 2].astype(np.int16)
            grass = (G > R + GRASS_GMARGIN) & (G > B + GRASS_GMARGIN) & (G > GRASS_GMIN)
        rr = max(r - 2.0, r * 0.88)                     # 只在候选外接框里算"盘内"
        x0c, x1c = int(max(cx - rr, 0)), int(min(cx + rr, img_w))
        y0c, y1c = int(max(cy - rr, 0)), int(min(cy + rr, img_h))
        if x1c - x0c < 2 or y1c - y0c < 2:
            continue
        yy, xx = np.mgrid[y0c:y1c, x0c:x1c]
        inside = ((xx - cx) ** 2 + (yy - cy) ** 2) <= rr * rr
        n_cyan = int((cyan[y0c:y1c, x0c:x1c] & inside).sum())
        n_grass = int((grass[y0c:y1c, x0c:x1c] & inside).sum())
        disc_area = float(inside.sum())
        if n_cyan < 24 and n_grass < 0.08 * disc_area:
            continue
        big = r >= 0.045 * min(img_h, img_w)
        if not (big and n_grass > 0.05 * disc_area and n_grass > 200):
            if verbose:
                print(f"[跳过] 非 3D 大盘（r={r:.1f} 草地={n_grass}）")
            continue
        dials.append(Dial(kind=KIND_GAUGE, cx=cx, cy=cy, r=r,
                          k=DEFAULT_K, anchored_by="ring"))
        if verbose:
            print(f"[定位] {KIND_GAUGE} 中心=({cx:.1f},{cy:.1f}) r={r:.1f} "
                  f"环覆盖={cov:.2f} 青色={n_cyan} 草地={n_grass}")

    merged: list[Dial] = []
    for d in sorted(dials, key=lambda z: -z.r):
        if any(math.hypot(d.cx - m.cx, d.cy - m.cy) < 0.6 * max(d.r, m.r) for m in merged):
            continue
        merged.append(d)

    if not merged:
        d2 = (_find_dial_by_arrow_color(rgb, hsv, cyan, verbose, side)
              or _find_dial_by_disc(rgb, cyan, verbose, side)
              or _find_dial_by_arrow(rgb, cyan, verbose, side))
        if d2 is not None:
            merged = [d2]
    else:
        # 白环结果优先；顺便用局部圆盘把 k 校准一下
        for d in merged:
            fit = _fit_disc_local(rgb, d.cx, d.cy, int(d.r * 1.6), cyan)
            if (fit is not None and 0.80 <= fit["k"] <= 0.88
                    and fit.get("rms", 9) < 0.6 and fit.get("n", 0) >= 40
                    and 0.72 <= (fit["a"] / CONFIG.disc_panel_ratio) / max(d.r, 1) <= 1.30):
                d.k = float(fit["k"])
    return merged


# --------------------------------------------------------------------------
# 3. 风箭头方向测量
# --------------------------------------------------------------------------

@dataclass
class ArrowReading:
    phi: float
    conf: float = 0.0            # 尖端判据的置信度（0~1），低=方向可疑
    phi_samples: dict = field(default_factory=dict)
    n_pixels: int = 0
    tip: tuple = (0.0, 0.0)
    tail: tuple = (0.0, 0.0)


def _angle_mod180(v: np.ndarray) -> float:
    return math.degrees(math.atan2(v[1], v[0])) % 180.0


def _bbox_gap(a: "Component", b: "Component") -> float:
    """两个外接框之间的间距（相邻为 0，重叠为 0）。"""
    dx = max(a.x0 - b.x1, b.x0 - a.x1, 0)
    dy = max(a.y0 - b.y1, b.y0 - a.y1, 0)
    return math.hypot(dx, dy)


def _merge_fragments(comps: list["Component"], main: "Component", rr: float,
                     min_frac: float = 0.05, gap_frac: float = 0.10) -> np.ndarray:
    """
    把紧贴主体的碎块并回来。

    箭头表面有明暗渐变（顶面亮、侧面暗），单色窄带掩膜常把它切成两块
    （比如只剩头或只剩尾）。只取最大块会让宽度剖面变成半个箭头，尖端
    判据随之翻 180°，表现为"顺逆风偶尔读反"。这里把挨着主体的碎块并成
    一个完整箭头，再量方向。
    """
    keep, gap_max = [main], max(6.0, gap_frac * rr)
    changed = True
    while changed:
        changed = False
        for c in comps:
            if any(c is k for k in keep) or c.area < min_frac * main.area:
                continue
            if any(_bbox_gap(k, c) <= gap_max for k in keep):
                keep.append(c)
                changed = True
    return np.vstack([c.pixels for c in keep])


def _width_profile(P: np.ndarray, axis: np.ndarray, nb: int = 24):
    c = P.mean(axis=0)
    d = P - c
    t = d @ axis
    nrm = np.array([-axis[1], axis[0]])
    p = d @ nrm
    lo, hi = float(t.min()), float(t.max())
    edges = np.linspace(lo, hi, nb + 1)
    w = np.zeros(nb)
    for i in range(nb):
        sel = (t >= edges[i]) & (t < edges[i + 1])
        w[i] = (p[sel].max() - p[sel].min()) if sel.sum() > 1 else 0.0
    return t, w, c


def _orient_by_width_jump(P: np.ndarray, axis_mod180: float):
    """
    三判据投票判断尖端在哪一端，返回 (φ, 置信度)。

    置信度 = 加权得分 / 权重之和（0~1）。箭头斜置、被高光切成两块或只剩
    半截时，三个判据会互相矛盾，此时置信度很低 —— 调用方可以据此按住
    上一次的方向，避免"顺逆风偶尔反一下"。
    """
    a = math.radians(axis_mod180)
    axis = np.array([math.cos(a), math.sin(a)])
    _, w, _ = _width_profile(P, axis)
    nb = len(w)
    if nb < 6 or w.max() <= 0:
        return None
    wmax = float(w.max())
    i_max = int(np.argmax(w))
    mid = (nb - 1) / 2.0
    votes = []                                   # 每项 = (方向, 置信 0~1)
    d_end = w[0] - w[-1]
    if abs(d_end) > 0.08 * wmax:
        votes.append((1 if d_end < 0 else -1, min(1.0, abs(d_end) / (0.25 * wmax))))
    d_peak = mid - i_max
    if abs(d_peak) > 0.08 * nb:
        votes.append((1 if d_peak > 0 else -1, min(1.0, abs(d_peak) / (0.25 * nb))))
    jump_lo, jump_hi = float(w[1] - w[0]), float(w[-2] - w[-1])
    d_jump = jump_lo - jump_hi
    if abs(d_jump) > 0.05 * wmax:
        votes.append((1 if d_jump > 0 else -1, min(1.0, abs(d_jump) / (0.15 * wmax))))
    if not votes:
        return None
    wsum = sum(v[1] for v in votes)
    score = sum(sign * m for sign, m in votes)
    if score == 0:                               # 平票：按"最宽处偏向哪端"定，物理上最可靠
        score = 1.0 if i_max < mid else -1.0
    tip_at_lo = score > 0
    d = -axis if tip_at_lo else axis
    phi = (90.0 + math.degrees(math.atan2(d[1], d[0]))) % 360.0
    return phi, abs(score) / (wsum + 1e-9)


def _tip_tail_direction(P: np.ndarray, axis_mod180: float):
    o = _orient_by_width_jump(P, axis_mod180)
    if o is None:
        return None
    phi_axis, conf = o
    a = math.radians((phi_axis - 90.0) % 360.0)
    u = np.array([math.cos(a), math.sin(a)])
    c = P.mean(axis=0)
    t = (P - c) @ u
    lo, hi = float(t.min()), float(t.max())
    if hi - lo < 4:
        return None
    tip = P[int(np.argmax(t))]
    band = t <= lo + (hi - lo) * 0.18
    if band.sum() < 2:
        return None
    v = tip - P[band].mean(axis=0)
    if np.hypot(*v) < 4:
        return None
    return math.degrees(math.atan2(v[0], -v[1])) % 360.0, conf


def estimate_direction(P: np.ndarray):
    """估计箭头在画面上的指向 φ（0=正上，顺时针）：PCA 定轴 + 宽度剖面定尖端。"""
    if len(P) < 40:
        return 0.0, {}
    c = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    d = _tip_tail_direction(P, _angle_mod180(vt[0]))
    if d is None:
        return 0.0, {}
    return d[0], {"pca": round(d[0], 2), "conf": round(float(d[1]), 3)}


def _tip_tail_points(P: np.ndarray, phi: float):
    a = math.radians((phi - 90.0) % 360.0)
    axis = np.array([math.cos(a), math.sin(a)])
    c = P.mean(axis=0)
    t = (P - c) @ axis
    lo, hi = float(t.min()), float(t.max())
    tip = P[t.argmax()]
    band = t <= lo + (hi - lo) * 0.15
    tail = P[band].mean(axis=0) if band.sum() >= 1 else c
    return tuple(float(v) for v in tip), tuple(float(v) for v in tail)


def measure_arrow(rgb: np.ndarray, hsv: np.ndarray, dial: Dial,
                  supersample: int = 6,
                  band: tuple[float, float, float, float] | None = None) -> ArrowReading | None:
    """
    测量箭头在画面上的指向 φ（0=正上，顺时针）。
    band 给定时用单色窄带掩膜取最大团块；band=None 时用多色宽带 + 形状打分（定位兜底）。
    """
    img = Image.fromarray(rgb)
    pad = dial.r * 0.35
    box = (int(max(dial.cx - dial.r - pad, 0)), int(max(dial.cy - dial.r - pad, 0)),
           int(min(dial.cx + dial.r + pad, rgb.shape[1])),
           int(min(dial.cy + dial.r + pad, rgb.shape[0])))
    if box[2] - box[0] < 10 or box[3] - box[1] < 10:
        return None
    crop = img.crop(box)
    ss = max(1, int(supersample or 4))
    big = crop.resize((crop.width * ss, crop.height * ss), Image.LANCZOS)
    arr = np.asarray(big)
    hb = rgb_to_hsv(arr)
    if band is not None:
        clip = arrow_mask_band(hb, band)
    else:
        clip = arrow_mask(hb)
    if band is None and not clip.any() and hsv is not None:
        clip = ((hb[..., 0] > CYAN_H[0]) & (hb[..., 0] < CYAN_H[1]) &
                (hb[..., 1] > CYAN_S) & (hb[..., 2] > CYAN_V))
    topf = clip & (hb[..., 2] >= max(TOPFACE_V, 0.62))
    cxs, cys = (dial.cx - box[0]) * ss, (dial.cy - box[1]) * ss
    rr = dial.r * ss * 0.97
    yy, xx = np.mgrid[0:arr.shape[0], 0:arr.shape[1]]
    inside = ((xx - cxs) ** 2 + (yy - cys) ** 2) <= rr * rr
    clip &= inside
    topf &= inside

    def pick(mask):
        comps = label_components(mask, keep_pixels=True, min_area=30)
        if not comps:
            return None
        if band is not None:
            # 已知颜色：窄带里基本只有箭头，取最大团块 + 紧贴它的碎块。
            # 不用 _solid 过滤：箭头斜置时外接框变大、填充率会掉到 0.25 以下而被误杀。
            main = max(comps, key=lambda c: c.area)
        else:
            # 多色宽带：最大的团块往往是草地圆盘，按尺寸/面积筛出箭头再按形状打分
            lim = 1.6 * rr
            cands = [c for c in comps
                     if max(c.w, c.h) <= lim and c.area >= 0.004 * math.pi * rr * rr
                     and _solid(c)]
            main = max(cands, key=lambda c: _cand_score(c, hb)) if cands else None
        if main is None or main.area < 0.008 * math.pi * rr * rr:
            return None
        pixels = _merge_fragments(comps, main, rr) if band is not None else main.pixels
        return pixels / ss + np.array([box[0], box[1]])

    P_sil = pick(clip)
    if P_sil is None or len(P_sil) < 60:
        return None
    if band is not None:
        # 窄带的饱和度/亮度门槛已把侧面、背景滤掉，直接用即可
        P_use = P_sil
    else:
        P_top = pick(topf)
        use_top = dial.kind == KIND_GAUGE
        if P_top is None or len(P_top) < 60:
            P_top, use_top = P_sil, False
        P_use = P_top if use_top else P_sil
    phi, samples = estimate_direction(P_use)
    if not samples:
        return None
    tip, tail = _tip_tail_points(P_use, phi)
    return ArrowReading(phi=phi, conf=float(samples.get("conf", 1.0)),
                        phi_samples=samples,
                        n_pixels=int(round(len(P_use) / (ss * ss))),  # 换算回原图尺度
                        tip=tip, tail=tail)


# --------------------------------------------------------------------------
# 4. 风速文字定位辅助
# --------------------------------------------------------------------------

def _solid(c: "Component", fill_min: float = 0.25) -> bool:
    """团块是否"实心"（空心环/弥散天空的填充率很低）。"""
    return c.area >= fill_min * float(max(c.w * c.h, 1))


def _cand_score(c: "Component", hsv: np.ndarray) -> float:
    """箭头候选打分 = 饱和度中位数 × 实心度（真箭头 S 高、填充率高；空心环/天空/远山偏低）。"""
    px = c.pixels
    if px is None or len(px) == 0:
        return 0.0
    S = hsv[..., 1]
    smed = float(np.median(S[px[:, 1].astype(int), px[:, 0].astype(int)]))
    fill = c.area / float(max(c.w * c.h, 1))
    return smed * min(fill, 1.0)


def _column_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """按列投影把白色掩膜切成连续的字符块区间 [(x0, x1), ...]。"""
    cols = mask.any(axis=0)
    runs, start = [], None
    for i, v in enumerate(cols):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i)); start = None
    if start is not None:
        runs.append((start, len(cols)))
    return runs


def _digit_runs(crop: Image.Image, height_ratio: float,
                max_digits: int) -> list[tuple[int, int, int]]:
    """
    切出"数字字形"的列区间，返回 [(x0, x1, 行跨度), ...]。
    太矮的块（分隔线/噪点）跳过；遇到又宽又矮的块（多半是 "m"）就停 —— 数字在 m/s 之前。
    """
    m = _bin_white(crop)
    if m.sum() < 20:
        return []
    runs = _column_runs(m)
    if not runs:
        return []
    heights = [int(m[:, a:b].any(axis=1).sum()) for a, b in runs]
    hmax = max(heights)
    if hmax == 0:
        return []
    out: list[tuple[int, int, int]] = []
    for (a, b), h in zip(runs, heights):
        if h < height_ratio * hmax:
            continue
        if (b - a) > 1.15 * h:
            break
        out.append((a, b, h))
        if len(out) >= max_digits:
            break
    return out


def _find_text_lines(rgb: np.ndarray, dial: Dial,
                     max_lines: int = 3) -> list[tuple[int, int, int, int]]:
    """
    在风盘下方区域找"白色文字行"（排除细长的分隔线），返回若干候选文本框
    （左手/右手完全对称，文字与圆环的相对位置一致，按半径比例带即可）。
    """
    img_h, img_w = rgb.shape[:2]
    r = dial.r
    x0 = int(max(dial.cx - 1.15 * r, 0)); x1 = int(min(dial.cx + 1.15 * r, img_w))
    # "Nm/s" 落在圆盘下方一条窄带里（cy+0.10r ~ 1.05r）；再往下是"高低差/返回"图标行
    y0 = int(max(dial.cy + 0.10 * r, 0))
    y1 = int(min(dial.cy + 1.05 * r, img_h))
    if x1 - x0 < 20 or y1 - y0 < 12:
        return []
    # 只在这条窄带里算 HSV（整帧 HSV 约 70ms）
    hsv = rgb_to_hsv(rgb[y0:y1, x0:x1])
    m = (hsv[..., 2] > 0.55) & (hsv[..., 1] < 0.45)
    # 面积下限 0.0035·r²：排掉面板网点小斑点，又不误杀小分辨率下的真文字
    min_area = max(8, int(0.0035 * r * r))
    comps = [c for c in label_components(m, keep_pixels=False, min_area=min_area)
             if 0.10 * r <= c.h <= 0.75 * r and c.w >= 2
             and 0.12 <= c.w / float(c.h) <= 2.8]          # 排除细长分隔线/圆环弧
    if not comps:
        return None
    # 按 y 分组成"行"，取总面积最大的一行（真实文字行的面积远大于面板网点斑点）
    comps.sort(key=lambda c: c.cy)
    lines: list[list] = []
    for c in comps:
        placed = False
        for ln in lines:
            yc = sum(z.cy for z in ln) / len(ln)
            if abs(c.cy - yc) < 0.6 * max(c.h, 6):
                ln.append(c)
                placed = True
                break
        if not placed:
            lines.append([c])
    # 真文字由多个字块组成（数字 + m + / + s）；单个杂斑（如过场横幅）直接判掉
    lines = [ln for ln in lines
             if len(ln) >= 2 or (len(ln) == 1 and ln[0].w >= 0.35 * r)]
    if not lines:
        return []
    # 按面积从大到小给候选 —— 最大的那块未必是数字行（左手/其它布局下会变），
    # 所以交给调用方依次试，别只赌第一名。
    lines.sort(key=lambda ln: -sum(z.area for z in ln))
    boxes = []
    for line in lines[:max_lines]:
        X0 = min(c.x0 for c in line); X1 = max(c.x1 for c in line)
        Y0 = min(c.y0 for c in line); Y1 = max(c.y1 for c in line)
        anchor = max(line, key=lambda c: c.area)
        pad = max(4, int(0.12 * anchor.h))
        boxes.append((max(X0 + x0 - pad, 0), max(Y0 + y0 - pad, 0),
                      min(X1 + x0 + pad, img_w), min(Y1 + y0 + pad, img_h)))
    return boxes


def ocr_speed(rgb: np.ndarray, dial: Dial, state: OcrState | None = None) -> dict:
    """识别风盘附近的 "Nm/s" 数字：切出文字行，用字形模板库匹配。

    返回 dict(value, raw, conf, box, ok)；模板库为空、不够有把握、或读出的数值
    超出游戏范围（0~MAX_WIND m/s）时 value 为 None。

    reject 字段记着"本来读成了多少"，专供排查：读成 35 m/s 这种明显不可能的，
    既不算数，也得留下证据（调用方会因此存一帧图）。
    """
    state = state or _DEFAULT_OCR
    res = dict(value=None, raw="", conf=0.0, box=None, ok=False, reject=None)
    # 候选文字行按面积从大到小依次试，第一个匹配上的就是风速
    for roi in _find_text_lines(rgb, dial):
        x0, y0, x1, y1 = roi
        res["box"] = [x0, y0, x1, y1]
        crop = Image.fromarray(rgb[y0:y1, x0:x1])
        tm = match_glyphs(crop, state=state, accept=lambda v: 0 <= v <= MAX_WIND)
        if tm is None:
            bad = match_glyphs(crop, state=state)      # 只是一探：记下"读成了什么"供排查
            if bad is not None:
                res["reject"] = dict(value=int(bad[0]), conf=float(bad[1]), box=res["box"])
            continue
        val = int(tm[0])
        # match_glyphs 已保证每字形 ≥0.72 且与次高有 0.05 区分度，这里只兜一个绝对下限
        if 0 <= val <= MAX_WIND and tm[1] >= 0.75:
            res.update(value=val, raw=f"tpl:{tm[0]}", conf=tm[1], ok=True)
            return res
    return res



# --------------------------------------------------------------------------
# 字形模板库：模板由离线脚本构建，运行时只读匹配、不自学。
# --------------------------------------------------------------------------

_GLYPH_SIZE = (32, 48)


def _glyph_norm(crop: Image.Image) -> np.ndarray | None:
    """把单个字形归一化成固定大小的二值图（去边、等比缩放、居中）。"""
    m = _bin_white(crop)
    ys, xs = np.nonzero(m)
    if len(ys) < 12:
        return None
    sub = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    im = Image.fromarray((sub * 255).astype(np.uint8)).resize(_GLYPH_SIZE, Image.BILINEAR)
    return (np.asarray(im) > 110).astype(np.float32)


def _glyph_sim(a: np.ndarray, b: np.ndarray) -> float:
    """两个归一化字形的相似度（IoU + 归一化相关，取平均）。"""
    inter = np.logical_and(a > 0.5, b > 0.5).sum()
    union = np.logical_or(a > 0.5, b > 0.5).sum()
    iou = inter / union if union else 0.0
    aa, bb = a.ravel() - a.mean(), b.ravel() - b.mean()
    denom = (np.linalg.norm(aa) * np.linalg.norm(bb))
    ncc = float(aa @ bb / denom) if denom > 1e-6 else 0.0
    return 0.5 * iou + 0.5 * max(ncc, 0.0)


def load_glyph_bank(path: str, state: OcrState | None = None) -> int:
    """从 npz 载入字形模板库（默认写入 _DEFAULT_OCR），返回模板总数。"""
    state = state or _DEFAULT_OCR
    state.bank = {}
    if not path or not os.path.isfile(path):
        return 0
    try:
        z = np.load(path, allow_pickle=False)
        for k in z.files:
            arr = z[k]
            state.bank[k] = [arr[i] for i in range(arr.shape[0])]
    except Exception:
        state.bank = {}
    return sum(len(v) for v in state.bank.values())


def glyph_bank_digits(state: OcrState | None = None) -> str:
    """模板库已覆盖的数字字符（升序），供调用方打印/校验。"""
    return "".join(sorted((state or _DEFAULT_OCR).bank))


def _glyph_windows(crop: Image.Image, max_digits: int = 2) -> list[list[tuple[int, int]]]:
    """
    列出"可能是数字"的列区间组合，供上层逐个试。

    原来只从最左边开始取、遇到宽块（m）就 break —— 数字框里一旦混进别的白块
    （圆环残影、面板高光），第一个块就是杂块，直接 break 掉一个数字都拿不到，
    表现就是"风速识别不出"（识别耗时只有几毫秒）。这里改成：先按高度/宽度筛出
    像字形的窄块，再枚举"任意起点 + 1~2 个连续块"的组合，2 位优先、从左到右。
    """
    m = _bin_white(crop)
    if m.sum() < 20:
        return []
    runs = _column_runs(m)
    if not runs:
        return []
    heights = [int(m[:, a:b].any(axis=1).sum()) for a, b in runs]
    hmax = max(heights)
    if hmax == 0:
        return []
    keep = [(a, b) for (a, b), h in zip(runs, heights)
            if h >= 0.45 * hmax and (b - a) <= 1.15 * max(h, 1)]
    out: list[list[tuple[int, int]]] = []
    for n in range(min(max_digits, 2), 0, -1):        # 先试 2 位，再试 1 位
        for i in range(len(keep) - n + 1):
            out.append(keep[i:i + n])
    return out


def match_glyphs(crop: Image.Image,
                 state: OcrState | None = None,
                 accept=None) -> tuple[str, float] | None:
    """
    用模板库识别数字部分。返回 (数字串, 置信度)；库为空或匹配太弱返回 None。

    逐个尝试候选列区间组合，**第一个全部字形都匹配上的**就是答案（2 位优先于 1 位，
    这样 "13m/s" 会给出 13 而不是 1）。

    accept 是可选的数值校验：不通过就换下一组窗口再试。用来挡掉"3m/s 被凑成 35"
    这种 —— 2 位数先试，凑出来的值超出游戏范围就直接作废，回头还能认出正确的那一位。
    """
    bank = (state or _DEFAULT_OCR).bank
    if not bank:
        return None
    for win in _glyph_windows(crop):
        out, worst = "", 1.0
        for a, b in win:
            g = _glyph_norm(crop.crop((a, 0, b, crop.height)))
            if g is None:
                break
            best, second = 0.0, 0.0
            digit = ""
            for d, temps in bank.items():
                for t in temps:
                    sc = _glyph_sim(g, t)
                    if sc > best:
                        second = max(second, best)
                        best, digit = sc, d
                    else:
                        second = max(second, sc)
            if best < 0.72 or (best - second) < 0.05:
                break                       # 这个字形不像数字（m / s / 杂块）→ 换下一组
            out += digit
            worst = min(worst, best)
        else:
            if accept is None or accept(int(out)):
                return out, worst           # 整组都匹配上、数值也合理，才算数
    return None


# --------------------------------------------------------------------------
# 5. 角度换算与风力计算
# --------------------------------------------------------------------------

COMPASS16 = ["正前", "右前", "右前", "右前", "正右", "右后", "右后", "右后",
             "正后", "左后", "左后", "左后", "正左", "左前", "左前", "左前"]


def compass_name(theta: float) -> str:
    """16 方位中文名（0=正前，顺时针）。"""
    idx = int(round((theta % 360.0) / 22.5)) % 16
    return COMPASS16[idx]


def signed_angle(theta: float) -> float:
    """0-360 转成 (-180,180]，正=偏右，负=偏左。"""
    a = (theta + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


def apparent_to_true(phi: float, k: float) -> float:
    """
    画面夹角 φ -> 真实夹角 θ（俯视压缩反投影）：θ = atan2(sinφ·k, cosφ)，
    按象限处理保证 0-360 连续。
    """
    p = math.radians(phi % 360.0)
    x = math.sin(p) * k          # 反投影后的平面坐标
    y = -math.cos(p)
    return math.degrees(math.atan2(x, -y)) % 360.0


def wind_metrics(speed: float | None, theta: float) -> dict:
    """
    把风矢量按"正前方基准"做正交分解：

        横向 = 风速 × sin(夹角)   正 = 吹向右侧
        纵向 = 风速 × cos(夹角)   正 = 顺风
    """
    t = math.radians(theta)
    cos_v = math.cos(t)
    sin_v = math.sin(t)
    out = dict(
        theta_deg=round(theta, 1),
        theta_signed_deg=round(signed_angle(theta), 1),
        side="右" if 0 < theta < 180 else ("左" if theta > 180 else "—"),
        compass=compass_name(theta),
        cos=round(cos_v, 4),
        sin=round(sin_v, 4),
    )
    if speed is not None:
        out["speed"] = speed
        lateral = speed * sin_v          # 横向（左右）
        along = speed * cos_v            # 纵向（顺逆）
        lat2, lon2 = round(lateral, 2), round(along, 2)
        out["lateral"], out["longitudinal"] = lat2, lon2
        out["lateral_abs"], out["longitudinal_abs"] = abs(lat2), abs(lon2)
        # 方向只看"显示出来的数值"：不是 0.00 就标真实方向
        # （早先用"小于 4% 风速算无风"，会把 0.45 这种明显非零的值也吞成"无侧风"）
        out["lateral_dir"] = ("无侧风" if lat2 == 0
                              else ("向右" if lat2 > 0 else "向左"))
        out["longitudinal_dir"] = ("无顺逆风" if lon2 == 0
                                   else ("顺风" if lon2 > 0 else "逆风"))
        out["lateral_text"] = f"{abs(lat2):.2f} m/s {out['lateral_dir']}"
        out["longitudinal_text"] = f"{abs(lon2):.2f} m/s {out['longitudinal_dir']}"
    return out


# --------------------------------------------------------------------------
# 6. 顶层分析流程
# --------------------------------------------------------------------------

@dataclass
class FrameReading:
    """一次「读风速 → 测箭头 → 分解风」的结果（风盘定位/档位由调用方提供）。"""
    ok: bool = False
    no_wind: bool = False
    speed: float | None = None
    message: str = ""
    dial: Dial | None = None
    arrow: ArrowReading | None = None
    phi: float | None = None
    k: float | None = None
    theta: float | None = None
    wind: dict = field(default_factory=dict)
    ocr: dict | None = None


def read_wind(rgb: np.ndarray, dial: Dial, *, speed: float | None = None,
              supersample: int | None = None,
              hsv: np.ndarray | None = None,
              ocr_state: OcrState | None = None,
              do_ocr: bool = True) -> FrameReading:
    """
    实时识别的核心：读风速（模板匹配，或调用方给定）→ 为 0 短路 →
    测箭头 φ → 反投影 θ → 分解风。风盘需由调用方先定位好。

    do_ocr=False 表示本次不读风速（沿用调用方给的 speed）。
    """
    fr = FrameReading(dial=dial)
    if speed is None and do_ocr:
        ocr = ocr_speed(rgb, dial, state=ocr_state)
        fr.ocr = ocr
        if ocr.get("value") is not None:
            speed = float(ocr["value"])
    if speed is None:
        fr.message = "风速未识别（等待下一次读取）"
        return fr
    fr.speed = float(speed)
    if abs(fr.speed) < 1e-9:
        fr.ok, fr.no_wind, fr.speed = True, True, 0.0
        fr.message = "风速 0 m/s —— 无风，按规则跳过计算"
        return fr
    ss = max(1, int(supersample or SUPERSAMPLE))
    band = wind_color(fr.speed)          # 先有风速，才知道箭头该是什么颜色
    arrow = measure_arrow(rgb, hsv, dial, supersample=ss, band=band)
    if arrow is None:
        fr.message = ("风速对应的彩色箭头未找到（可能数字读错或不在瞄准界面）"
                      if band is not None else "风盘内未找到风箭头")
        return fr
    k = dial.k
    theta = apparent_to_true(arrow.phi, k)
    fr.ok, fr.arrow, fr.phi, fr.k, fr.theta = True, arrow, arrow.phi, k, theta
    fr.wind = wind_metrics(fr.speed, theta)
    return fr
