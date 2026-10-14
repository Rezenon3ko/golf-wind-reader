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
    k_default: float = 0.82           # 俯视压缩系数兜底（= 草地圆盘自身的压缩比，实测见下）


CONFIG = Config()

# 测角超采样倍数（放大风盘裁块后再量宽度剖面）。固定 2：
#   200 帧可信实测（同批静止帧、前后半段均值只差 0.10° 证明场景没动）：
#       ss=2 → σ 0.297°（离群帧 0/200）    ss=3 → σ 0.253°（离群帧 0/200）
#   即 ×3 只换来 15% 的单帧噪声下降，而它多花 34ms/帧（单核 ~7%）；
#   更关键的是这 15% 会被后面的 EMA 平滑（输出 σ ≈ 输入×0.147）和 0.7° 显示死区完全吃掉，
#   最终显示上看不出任何区别 —— 所以取 2，把 CPU 留给取帧/OCR（慢机器上刷新率更稳）。
SUPERSAMPLE = 2

# 游戏里的风速就是 1~15 m/s（三档：1-5 青 / 6-10 黄 / 11-15 品红），0 = 无风。
# 读出来超过这个数一定是把别的白块当成数字了 —— 例如 "3m/s" 里的字形被凑成 "35"。
# 这种时候宁可当"没读出来"（下帧再读），也不能把一个错值算成 35 m/s 的风。
MAX_WIND = 15


# --------------------------------------------------------------------------
# 其余阈值
# --------------------------------------------------------------------------
KIND_GAUGE = "gauge3d"       # 3D 风盘（右手右上 / 左手左上）
# 俯视压缩系数：盘面那个草地圆是真正的地面圆，直接量它的椭圆就是 k。**只用边界点**拟合
# （用整块内部点做代数拟合会明显偏小：同一帧上内部点给 0.79、边界点给 0.820）。实测 9 m/s
# 那张：边界点 a=58.4 b=47.9 → k=0.820（rms 0.006）；上下沿都取 → 0.829。所以取 0.82。
# 注意箭头"顶面"比盘面高一点，理论上压缩略重一点点（<1%），小到看不出，不改。
DEFAULT_K = CONFIG.k_default
CYAN_H = (158.0, 218.0)      # 兼容旧判据
CYAN_S = 0.15
CYAN_V = 0.42
# 下面两个是 1.1.20 之前"按亮度切箭头顶面"用的阈值，那条路已停用（见 measure_arrow）；
# 留着是为了对照历史日志，不要在新代码里引用。
TOPFACE_V = 0.80             # 箭头"顶面"（明亮面）的亮度下限（只剩 band=None 的兜底路在用）
TOPFACE_MIN_FRAC = 0.25      # （停用）切出的顶面至少要占整块剪影这么多
TOPFACE_MAX_DEV = 35.0       # （停用）顶面与整块剪影的轴(mod 180)容差
GLYPH_MIN_AREA_FRAC = 0.15   # 文字行里小于最大团块这个比例的碎墨直接丢（背景斜刺、面板网点、毛边）
DISC_MASK_PAD = 1.45         # 箭头只在草地圆盘上；按固定版式圈盘面时外扩这么多。
# ↑ 为什么是 1.45：箭头会**探出草地圆盘**（实测箭头尖超出盘沿 0.15r 左右），
#   掩膜太小会把尖端切掉 —— 尖端一丢，"哪头是尖"就靠残缺口径抖，实测同一静止
#   画面 θ 在 341~351° 之间跳（极差 9.95°）。放宽到 1.45 后同一批帧 θ 极差 0.48°。
#   注意固定版式的盘心比实测盘心低约 0.08r，所以这里要比"实测几何"那条路更宽一点。
DISC_MASK_PAD_FIT = 1.25     # 有实测盘面几何时外扩这么多（箭头尖会探出盘沿，实测几何不会连带背景）
DISC_COVER_FRAC = 0.50       # 预期圆盘椭圆里的草地覆盖率下限（定位兜底判据，见 _find_dial_by_prior）
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


def _morph_shift(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """按 (dy,dx) 平移掩膜，空出来的地方补 False（不环绕 —— 环绕会把左边的墨搬到右边）。"""
    h, w = mask.shape
    out = np.zeros_like(mask)
    sy, dy_ = (slice(dy, h), slice(0, h - dy)) if dy >= 0 else (slice(0, h + dy), slice(-dy, h))
    sx, dx_ = (slice(dx, w), slice(0, w - dx)) if dx >= 0 else (slice(0, w + dx), slice(-dx, w))
    if sy.stop > sy.start and sx.stop > sx.start:
        out[dy_, dx_] = mask[sy, sx]
    return out


def _morph(mask: np.ndarray, r: int, grow: bool) -> np.ndarray:
    """方形结构元的膨胀（grow=True）/腐蚀（grow=False），半径 r。"""
    out = np.zeros_like(mask) if grow else mask.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            sh = _morph_shift(mask, dy, dx)
            if grow:
                out |= sh
            else:
                out &= sh
    return out


def _open_mask(mask: np.ndarray, r: int = 1) -> np.ndarray:
    """
    开运算（先腐蚀后膨胀）：去掉 ≤2r px 宽的细线、毛刺、抗锯齿桥，粗笔画基本不动。

    风盘那圈面板是半透明的 —— 背景亮（草地/树/面板分隔线）会漏进"白字掩膜"，把数字
    咬出一根斜刺，还会把粘连的两个字形连在一起。实测一张真帧：'9' 被斜刺污染后字形
    相似度从 0.91 掉到 0.43（直接读不出风速）；开运算后回到 0.91，还顺手把粘连切开。
    """
    if r <= 0:
        return mask
    er = _morph(mask, r, grow=False)
    return _morph(er, r, grow=True)


def _drop_small_blobs(mask: np.ndarray, frac: float = GLYPH_MIN_AREA_FRAC) -> np.ndarray:
    """丢掉比最大团块小太多的碎墨（背景漏进来的斜刺/网点/毛边碎片）。"""
    comps = label_components(mask, keep_pixels=False, min_area=2)
    if len(comps) < 2:
        return mask
    biggest = max(c.area for c in comps)
    keep = np.zeros_like(mask)
    for c in comps:
        if c.area >= frac * biggest:
            keep[c.y0:c.y1 + 1, c.x0:c.x1 + 1] |= mask[c.y0:c.y1 + 1, c.x0:c.x1 + 1]
    return keep


def _clean_white_mask(mask: np.ndarray) -> np.ndarray:
    """白掩膜去噪：开运算 + 丢碎墨。切字块、定行基准、认字形都用这一份，口径保持一致。"""
    return _drop_small_blobs(_open_mask(mask))


def _hole_count(mask: np.ndarray) -> int:
    """
    数字的"洞"数：8→2，0/6/9→1，1/2/3/4/5/7→0。

    拓扑是认数字最稳的一条线索 —— 8 和 6 的像素相似度高达 0.79（10 个模板里最像的一对），
    光靠 IoU 分不开；但"两个洞"和"一个洞"永远不会错。
    """
    comps = label_components(~mask, keep_pixels=False, min_area=3)
    h, w = mask.shape
    return sum(1 for c in comps
               if c.x0 > 0 and c.y0 > 0 and c.x1 < w - 1 and c.y1 < h - 1)


_HOLES_CACHE: dict[int, int] = {}


def _template_holes(t: np.ndarray) -> int:
    """
    模板的洞数，按**内容**缓存（一帧里要被问几百次，重算太亏）。

    别按 id() 缓存：模板数组被回收后地址会被别的数组复用，缓存就串味了 —— 实测换一套
    模板库（等价于换 OcrState）时，8 的洞数会被套到 6 的模板上，数字直接读不出来。
    """
    key = t.tobytes()
    n = _HOLES_CACHE.get(key)
    if n is None:
        n = _hole_count(t > 0.5)
        _HOLES_CACHE[key] = n
    return n


def _disc_ellipse(dial: Dial, shape: tuple[int, int], ss: float = 1.0,
                  box: tuple[int, int, int, int] = (0, 0, 0, 0),
                  pad: float = DISC_MASK_PAD) -> np.ndarray:
    """
    预期草地圆盘的椭圆掩膜（固定 UI 版式）：盘心在风盘中心上方 0.30·r·(半轴/r)，
    半轴 = 0.522·r / 0.522·r·k，再外扩 pad 倍。

    箭头永远躺在这个圆盘上 —— 用它把掩膜圈在盘面里，背景（树/草地从半透明面板里
    透出来的黄绿像素）就进不来了。这是"背景有树时读不出风向"的主因。
    """
    a = CONFIG.disc_panel_ratio * dial.r * pad * ss
    b = a * dial.k
    # 注意：这里**故意**仍用固定版式（不换成实测盘面几何）。实测过：掩膜范围一挪，
    # 某些帧的尖端判据就会翻 180°（4 号帧翻过 188°），而 1.1.11 的固定范围在实机上
    # 本来就准到 1° 以内 —— 别为了"更精确的中心"把已验证的行为搞坏。
    # 实测盘面几何（dial.disc）只用来写日志/校准 k，不参与裁剪。
    cx = (dial.cx - box[0]) * ss
    cy = (dial.cy - 0.30 * dial.r * CONFIG.disc_panel_ratio - box[1]) * ss
    out = np.zeros(shape, dtype=bool)
    x0, x1 = int(max(cx - a, 0)), int(min(cx + a + 1, shape[1]))
    y0, y1 = int(max(cy - b, 0)), int(min(cy + b + 1, shape[0]))
    if x1 <= x0 or y1 <= y0:
        return out
    # 只在外接框里算椭圆（整幅 mgrid 一帧要 20ms+，这里只要 ~1ms）
    yy, xx = np.mgrid[y0:y1, x0:x1]
    out[y0:y1, x0:x1] = (((xx - cx) / max(a, 1e-6)) ** 2
                         + ((yy - cy) / max(b, 1e-6)) ** 2) <= 1.0
    return out


def _disc_turf_coverage(rgb: np.ndarray, cx: float, cy: float, r: float, k: float) -> float:
    """预期圆盘椭圆里"草地"占的比例：定位兜底用（背景有树时，最大绿团块的几何检查会失手）。"""
    a = CONFIG.disc_panel_ratio * r
    b = a * k
    cy_d = cy - 0.30 * r * CONFIG.disc_panel_ratio
    h, w = rgb.shape[:2]
    half_w, half_h = int(a * 1.1), int(b * 1.1)
    x0, x1 = int(max(cx - half_w, 0)), int(min(cx + half_w + 1, w))
    y0, y1 = int(max(cy_d - half_h, 0)), int(min(cy_d + half_h + 1, h))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return 0.0
    yy, xx = np.mgrid[y0:y1, x0:x1]
    inside = ((xx - cx) / a) ** 2 + ((yy - cy_d) / b) ** 2 <= 1.0
    grass = _grass_mask(rgb[y0:y1, x0:x1], 25, 45, 140)
    n = int(inside.sum())
    return float((grass & inside).sum()) / n if n else 0.0


def _disc_geometry(rgb: np.ndarray, dial: Dial) -> tuple | None:
    """
    实测盘面（草地圆）的椭圆：返回 (cx, cy, 半长轴, 半短轴)，失败返回 None。

    只取**亮饱和绿**（盘面）而不是"最大绿块"：面板半透明，背景树/草地会漏进来，但它们被
    面板压暗、或偏蓝，落在门槛外。逐列取上下沿 + 稳健（剔离群）拟合轴平行椭圆，箭头啃出的
    缺口会变成离群点被丢掉（用外接框会被缺口带偏，同机位两帧能差 0.2）。
    """
    H, W = rgb.shape[:2]
    r = dial.r
    x0 = int(max(dial.cx - 1.45 * r, 0)); x1 = int(min(dial.cx + 1.45 * r, W))
    y0 = int(max(dial.cy - 1.45 * r, 0)); y1 = int(min(dial.cy + 1.45 * r, H))
    if x1 - x0 < 40 or y1 - y0 < 40:
        return None
    hs = rgb_to_hsv(rgb[y0:y1, x0:x1])          # 只算盘面这一小块（整幅 70ms，这块 ~4ms）
    vivid = ((hs[..., 0] >= 70.0) & (hs[..., 0] <= 115.0)
             & (hs[..., 1] >= 0.42) & (hs[..., 2] >= 0.50))
    comps = label_components(vivid, min_area=max(120, int(0.05 * r * r)))
    if not comps:
        return None
    c = max(comps, key=lambda z: z.area)
    m = np.zeros_like(vivid)
    m[c.y0:c.y1 + 1, c.x0:c.x1 + 1] = vivid[c.y0:c.y1 + 1, c.x0:c.x1 + 1]
    for cc in label_components(~m, keep_pixels=True, min_area=1):    # 填洞（箭头盖住的部分）
        px = np.asarray(cc.pixels, dtype=int)
        if cc.x0 > 0 and cc.y0 > 0 and cc.x1 < m.shape[1] - 1 and cc.y1 < m.shape[0] - 1:
            m[px[:, 1], px[:, 0]] = True
    ys, xs = np.nonzero(m)
    if len(xs) < 0.15 * (0.522 * r) ** 2 * math.pi:
        return None
    cols = np.unique(xs)
    top = np.full(cols.max() + 1, np.inf)      # 用 ±inf 初值：np.minimum.at 遇 NaN 会一直传 NaN
    bot = np.full(cols.max() + 1, -np.inf)
    np.minimum.at(top, xs, ys.astype(float))
    np.maximum.at(bot, xs, ys.astype(float))
    cols = cols[np.isfinite(top[cols]) & np.isfinite(bot[cols])]
    P = np.stack([np.concatenate([cols, cols]),
                  np.concatenate([top[cols], bot[cols]])], axis=1)
    cur, fit = P, None
    for _ in range(4):
        fit = fit_ellipse_axis_aligned(cur, iters=3)
        if fit is None or len(cur) < 20:
            break
        a, b = fit["a"], fit["b"]
        rr = np.sqrt(((P[:, 0] - fit["cx"]) / max(a, 1e-6)) ** 2
                     + ((P[:, 1] - fit["cy"]) / max(b, 1e-6)) ** 2)
        res = np.abs(rr - 1.0)
        keep = res <= np.quantile(res, 0.75)
        if keep.sum() < 20:
            break
        cur = P[keep]
    if fit is None:
        return None
    return (float(fit["cx"]) + x0, float(fit["cy"]) + y0,
            float(max(fit["a"], fit["b"])), float(min(fit["a"], fit["b"])))


def measure_disc_k(rgb: np.ndarray, dial: Dial) -> float | None:
    """
    俯视压缩比 k = 盘面椭圆的短半轴/长半轴。用常数不行：k 是相机俯仰的函数，每个洞/每个
    机位都不一样（实测同一机位 10 帧只有 ±0.003，但换机位就会变）。所以每次定位实测一次，
    结果写进 dial.k；盘面几何（中心/长短轴）也一起存进 dial.disc，供掩膜限制复用。
    """
    g = _disc_geometry(rgb, dial)
    if g is None:
        return None
    _, _, a, b = g
    k = float(b / max(a, 1e-6))
    return k if 0.55 <= k <= 1.0 else None


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


def _grass_mask(rgb: np.ndarray, r_margin: int = 30, b_margin: int = 50,
                v_min: int = 150) -> np.ndarray:
    """
    "草地绿"掩膜。定位的四条路（先验 / 箭头锚点 / 圆盘拟合 / 白环）都要用它。

    以前这式子在 5 个地方各抄了一遍，还抄出 4 套阈值 —— 现在收成一个函数；
    阈值留成参数是因为各条路历史上分别调过（先验那条略松：25/45/140，
    白环那条按整帧的绿更杂：15/15/95），改了就得重新测。
    """
    a = rgb.astype(np.int16)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    return (G > R + r_margin) & (G > B + b_margin) & (G > v_min)


def wind_range(band: tuple[float, float, float, float]) -> tuple[int, int]:
    """色带 → 该档的风速区间（1-5 青 / 6-10 黄 / 11-15 品红）。"""
    for lo, hi, _name, (h0, h1), smin, vmin in WIND_COLORS:
        if (h0, h1, smin, vmin) == tuple(band):
            return int(lo), int(hi)
    return 1, MAX_WIND


def band_name(band: tuple[float, float, float, float] | None) -> str:
    """色带 → 档位名（青 / 黄 / 品红）；认不出来的色带返回空串。日志和界面都用它。"""
    if band is None:
        return ""
    for _lo, _hi, name, (h0, h1), smin, vmin in WIND_COLORS:
        if (h0, h1, smin, vmin) == tuple(band):
            return name
    return ""


def _dial_box(dial: Dial, shape, pad_frac: float = 0.35) -> tuple[int, int, int, int]:
    """风盘（往外放一点）在画面里的裁剪框。"""
    pad = dial.r * pad_frac
    return (int(max(dial.cx - dial.r - pad, 0)), int(max(dial.cy - dial.r - pad, 0)),
            int(min(dial.cx + dial.r + pad, shape[1])),
            int(min(dial.cy + dial.r + pad, shape[0])))


def detect_arrow_band(rgb: np.ndarray, dial: Dial,
                      min_px: int = 100,
                      relax: float = 0.0) -> tuple[float, float, float, float] | None:
    """
    看风盘里那支箭头是什么颜色 ⇒ 风速在哪一档（1-5 青 / 6-10 黄 / 11-15 品红）。

    箭头是盘里最大的一块纯色，比那两三个细笔画数字可靠得多。先用它把风速框进 5 档，
    再让数字在窄范围里认 —— "6 读成 5"这种跨档误读从源头上就不会发生。
    三色像素都不够（没在瞄准界面 / 箭头被挡住）就返回 None，调用方走原来的全范围流程。

    relax>0 时把饱和/亮度门槛各降这么多（色相不动）—— 采集卡把颜色洗淡时用，见 _relax_band。
    """
    x0, y0, x1, y1 = _dial_box(dial, rgb.shape)
    if x1 - x0 < 10 or y1 - y0 < 10:
        return None
    hsv = rgb_to_hsv(rgb[y0:y1, x0:x1])
    inside = _disc_ellipse(dial, hsv.shape[:2], box=(x0, y0, x1, y1))
    best_n, best_band = 0, None
    for _lo, _hi, _name, (h0, h1), smin, vmin in WIND_COLORS:
        n = int(((hsv[..., 0] >= h0) & (hsv[..., 0] <= h1)
                 & (hsv[..., 1] >= max(0.10, smin - relax))
                 & (hsv[..., 2] >= max(0.35, vmin - relax)) & inside).sum())
        if n > best_n:
            best_n, best_band = n, (h0, h1, smin, vmin)
    return best_band if best_n >= min_px else None


def _relax_band(band: tuple[float, float, float, float],
                ds: float = 0.30, dv: float = 0.30) -> tuple[float, float, float, float]:
    """
    洗淡兜底用的放宽色带：**色相完全不动，只降饱和/亮度门槛**。

    有的采集卡（尤其采集卡→虚拟机的链路）会把颜色洗淡：色相还是青/黄/品红，
    但饱和度/亮度掉到门槛下 —— 实测把正常帧往灰里拉 50% 就复现出"风速读得出、
    箭头找不到（θ=None）"。这种情况下色相判据仍然可靠，放宽 S/V 就能救回来。
    """
    h0, h1, smin, vmin = band
    return (h0, h1, max(0.10, smin - ds), max(0.35, vmin - dv))


# --------------------------------------------------------------------------
# 2. 风盘定位
# --------------------------------------------------------------------------

# 风盘先验（cx/W, cy/H, r/W）—— 按 1920×1080 采集实测：
#   右手：右上角，中心 (1690,140)、半径 115；
#   左手：HUD 左右**完全对称**，风盘挪到左上角，大小位置都是镜像（别再改成缩小版）。
GAUGE_PRIOR = (0.8802, 0.1296, 0.0600)
GAUGE_PRIOR_LEFT = (1.0 - GAUGE_PRIOR[0], GAUGE_PRIOR[1], GAUGE_PRIOR[2])
GAUGE_PRIORS = (("右", GAUGE_PRIOR), ("左", GAUGE_PRIOR_LEFT))
# 风盘中心允许偏离先验多少（画面宽/高的比例）—— 见 _in_gauge_zone 的注释：
# 真机 11 张实测抖动 ≤0.5%W/1.2%H，这里给 6%/10%，把"小地图那一带的白色 UI"挡在外面。
GAUGE_ZONE_TOL_X = 0.06
GAUGE_ZONE_TOL_Y = 0.10

@dataclass
class Dial:
    kind: str
    cx: float
    cy: float
    r: float
    k: float = DEFAULT_K
    anchored_by: str = ""
    disc: tuple | None = None     # 实测盘面 (cx, cy, 半长轴, 半短轴)；限制/校准都用它，见 _disc_geometry


def _ring_coverage(pts: np.ndarray, cx: float, cy: float, nbins: int = 72) -> float:
    if len(pts) == 0:
        return 0.0
    ang = np.degrees(np.arctan2(pts[:, 0] - cx, -(pts[:, 1] - cy))) % 360.0
    return len(np.unique((ang / (360.0 / nbins)).astype(int) % nbins)) / nbins


def _in_gauge_zone(cx: float, cy: float, shape, side: str | None = None) -> bool:
    """
    风盘只可能在**两个固定位置**上：右手右上、左手左上（HUD 是镜像的，位置一一对应）。

    这里卡的是"**离先验有多近**"，不是"上方 + 左右 30%"那种大范围。原因：风盘是固定版式
    的 HUD，实测 11 张真帧的白环拟合中心离先验最多 0.5%W / 1.2%H —— 位置基本是死的；
    而风盘**下方**固定跟着小地图（左右布局都一样），小地图那些白色 UI（圆角框、风速徽章、
    缩放手柄）会被白环/圆盘拟合当成风盘：实测就抓到过一个 (1514,163)，离先验 9.2%W，
    掩膜只有 776px（正常 2200+），读出来的是小地图附近的东西。
    容忍度给 6%W / 10%H：比真机抖动大一个量级，又能把那种离得远的一律挡掉。

    side 给定时只认那一侧 —— 玩家一局之内不会换手，锁定侧别可以彻底避免
    "在对侧误抓一个白色 UI 当风盘"（那种情况下会一直报"风速未识别"）。

    注意调用方给的可能是**圆盘中心**（草地圆盘在风盘中心上方约 0.24r）而不是风盘中心，
    所以纵向容差要留得比横向宽。
    """
    h, w = shape[:2]
    for name, (fx, fy, _fr) in GAUGE_PRIORS:
        if side is not None and name != side:
            continue
        if (abs(cx - fx * w) <= GAUGE_ZONE_TOL_X * w
                and abs(cy - fy * h) <= GAUGE_ZONE_TOL_Y * h):
            return True
    return False


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
    vivid = _grass_mask(win)
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
    """
    圆盘内是否含足够多的箭头像素（arrow 为已算好的三色掩膜，避免整帧重算）。

    注意这和另外两处是"同一个问题的三种问法"，阈值各不同，改一处要连着看：
      · detect_arrow_band —— 三色各数像素，定风速档位（读数用）
      · measure_arrow     —— 单色窄带取最大团块、≥60px 才算有箭头（测方向用）
      · 这里              —— 拟合椭圆内宽带箭头 ≥120px 才算这块草地是风盘（定位用）
    这里故意不复用 measure_arrow：那条路要 LANCZOS 放大 + PCA，丢在定位兜底里太贵。
    """
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
    vivid = _grass_mask(rgb)
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
    grass = _grass_mask(rgb) | arrow
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

    两条判据：① 上面这套"最大绿团块"的几何检查（快）；② ①不成立时退回看**版式** ——
    预期圆盘椭圆里的草地覆盖率 + 盘里能不能看到风箭头。背景有树/草地时，树也算"绿"，
    ①的团块几何会被搅黄（表现：一直"没定位到风盘"），②的位置/形状都来自固定 UI 版式，
    更抗背景；而且要求真的看到风箭头，不会把别的界面误判成瞄准界面。
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
        green = _grass_mask(win, 25, 45, 140)     # 先验这条窗小、更干净，阈值略松
        comps = label_components(green, min_area=int(0.02 * r * r))
        if not comps:
            continue
        # 圆盘在盘面中心上方 0.30·disc_a（0 m/s 也是这个布局）；
        exp_cy = cy - 0.30 * r * CONFIG.disc_panel_ratio
        # 箭头横躺时会盖住圆盘中间，把草地切成好几块 —— 只认"单一团块"会整个漏掉
        # 这一帧（实测 6m/s 那张就是这么定位失败的）。所以把圆盘附近的绿块合起来看，
        # 离得远的绿块（背景草地/树）不算。
        near = [c for c in comps
                if math.hypot((x0 + c.cx) - cx, (y0 + c.cy) - exp_cy) <= 0.6 * r]
        if not near:
            continue
        gx0 = min(c.x0 for c in near); gx1 = max(c.x1 for c in near)
        gy0 = min(c.y0 for c in near); gy1 = max(c.y1 for c in near)
        bw, bh = gx1 - gx0, gy1 - gy0
        if not (0.6 <= bw / float(max(bh, 1)) <= 1.5):           # 圆盘：略宽于高
            continue
        if (gx1 - gx0) * (gy1 - gy0) < 0.5 * math.pi * (r * 0.5) ** 2:
            continue
        # 合起来的草地至少要有圆盘的 1/4 —— 圆盘被箭头挡掉一部分是正常的
        if sum(c.area for c in near) < 0.25 * math.pi * (bw / 2.0) * (bh / 2.0):
            continue
        c = max(near, key=lambda z: z.area)
        d = Dial(kind=KIND_GAUGE, cx=float(cx), cy=float(cy), r=float(r),
                 k=DEFAULT_K, anchored_by=f"prior-{name}")
        if verbose:
            print(f"[定位] 先验({name})：草地 {bw}×{bh}（{len(near)} 块）"
                  f" -> 中心=({cx:.0f},{cy:.0f}) r={r:.0f}")
        return d
    # ② 退路：绿块几何被背景（树/草地）搅黄时，按**固定 UI 版式**判 —— 预期圆盘椭圆
    # 里的草地覆盖率够高，且盘里确实能看到风箭头，才认。
    for name, (fx, fy, fr) in GAUGE_PRIORS:
        if side is not None and name != side:
            continue
        cx, cy, r = fx * img_w, fy * img_h, fr * img_w
        cov = _disc_turf_coverage(rgb, cx, cy, r, DEFAULT_K)
        if cov < DISC_COVER_FRAC:
            continue
        d = Dial(kind=KIND_GAUGE, cx=float(cx), cy=float(cy), r=float(r),
                 k=DEFAULT_K, anchored_by=f"layout-{name}")
        if detect_arrow_band(rgb, d, min_px=60, relax=0.30) is None:
            continue
        if verbose:
            print(f"[定位] 版式({name})：圆盘草地覆盖 {cov:.2f} + 盘内有箭头"
                  f" -> 中心=({cx:.0f},{cy:.0f}) r={r:.0f}")
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
        _calibrate_disc(rgb, d0, verbose)
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
            # 整帧的绿更杂（远景草地、树），所以这条路的阈值单独放得松
            grass = _grass_mask(rgb, GRASS_GMARGIN, GRASS_GMARGIN, GRASS_GMIN)
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
    # 风盘整个都在画面里（贴边被裁掉一半的圆环不是风盘 —— 之前会把画面边缘的
    # 白色 UI 当成风盘，读出来一串乱七八糟的数字）
    good = [d for d in merged
            if (d.cx - d.r >= -0.05 * d.r and d.cx + d.r <= img_w + 0.05 * d.r
                and d.cy - d.r >= -0.05 * d.r and d.cy + d.r <= img_h + 0.05 * d.r)]
    # 统一校准俯视压缩比 k：用盘面实测值，不再吃写死的常数（见 measure_disc_k 注释）。
    for d in good:
        _calibrate_disc(rgb, d, verbose)
    return good


def _calibrate_disc(rgb: np.ndarray, dial: Dial, verbose: bool = False) -> None:
    """实测盘面椭圆：k 写进 dial.k，几何（中心/长短轴）写进 dial.disc 供限制复用。"""
    g = _disc_geometry(rgb, dial)
    if g is None:
        return
    _, _, a, b = g
    k = float(b / max(a, 1e-6))
    if not (0.55 <= k <= 1.0):
        return
    if verbose and abs(k - dial.k) > 0.005:
        print(f"[定位] 盘面实测 k={k:.3f}（原 {dial.k:.3f}） 中心=({g[0]:.0f},{g[1]:.0f})")
    dial.disc, dial.k = g, k


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
    source: str = ""             # 这次用了哪张掩膜（顶面/剪影）—— 排查"读数偏南/偏北"要看它


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
    两判据投票判断尖端在哪一端，返回 (φ, 置信度)。

    置信度 = 加权得分 / 权重之和（0~1）。箭头斜置、被高光切成两块或只剩
    半截时，两个判据会互相矛盾，此时置信度很低 —— 调用方可以据此按住
    上一次的方向，避免"顺逆风偶尔反一下"。
    """
    a = math.radians(axis_mod180)
    axis = np.array([math.cos(a), math.sin(a)])
    t, w, c = _width_profile(P, axis)
    nb = len(w)
    if nb < 6 or w.max() <= 0:
        return None
    wmax = float(w.max())
    i_max = int(np.argmax(w))
    mid = (nb - 1) / 2.0
    votes = []                                   # 每项 = (方向, 置信 0~1)
    # 判据一：两端外 10% 处的**宽度包络**谁窄谁就是尖。
    # 为什么不用"首/末单个 bin 的宽度"（旧写法）：箭头尾端是一段平边，但侧壁会把平边
    # 两角往下拖出一小截，最后那一两个 bin 里往往只剩这点角，宽度从 ~0.4·w 掉到 ~0.2·w，
    # 于是"越往尾越宽"的判据被顶反 —— 合成箭头实测 45° 这类斜向会整块判反 180°。
    # 取外端 10% 的包络（而不是首末 bin）就把这种稀疏 bin 的噪声平均掉了。
    nrm = np.array([-axis[1], axis[0]])
    p = (P - c) @ nrm
    lo_t, hi_t = float(t.min()), float(t.max())
    span = max(hi_t - lo_t, 1e-9)
    lo_sel, hi_sel = t <= lo_t + 0.10 * span, t >= hi_t - 0.10 * span
    w_lo = float(p[lo_sel].max() - p[lo_sel].min()) if lo_sel.sum() > 1 else 0.0
    w_hi = float(p[hi_sel].max() - p[hi_sel].min()) if hi_sel.sum() > 1 else 0.0
    if w_lo > 0 and w_hi > 0 and abs(w_lo - w_hi) > 0.06 * wmax:
        votes.append((1 if w_lo < w_hi else -1,
                      min(1.0, abs(w_lo - w_hi) / max(w_lo, w_hi))))
    # 判据二：最宽处在哪端 —— 箭头头部（宽）在尖端那一侧。
    d_peak = mid - i_max
    if abs(d_peak) > 0.08 * nb:
        votes.append((1 if d_peak > 0 else -1, min(1.0, abs(d_peak) / (0.25 * nb))))
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
    d = _tip_tail_direction(P, _principal_axis(P))
    if d is None:
        return 0.0, {}
    return d[0], {"pca": round(d[0], 2), "conf": round(float(d[1]), 3)}


def _principal_axis(P: np.ndarray) -> float:
    """
    点集主轴（mod 180，角度制）。2×2 协方差闭式解，和 SVD 首奇异向量等价但快一个量级 ——
    箭头有几千到几万点，原来每帧要跑两次 SVD，是 measure_arrow 的主要开销。
    """
    X = P - P.mean(axis=0)
    xx = float((X[:, 0] * X[:, 0]).sum())
    yy = float((X[:, 1] * X[:, 1]).sum())
    xy = float((X[:, 0] * X[:, 1]).sum())
    return math.degrees(0.5 * math.atan2(2.0 * xy, xx - yy)) % 180.0


def estimate_direction_ground(P: np.ndarray, k: float):
    """
    在"去压缩"后的盘面平面上估方向：先把屏幕纵坐标除以 k（undo 俯视压缩），再走同一套
    PCA + 宽度剖面定尖端。返回的角直接是**盘面角 θ**（0=正前，顺时针），不是画面角 φ。

    为什么尖端投票必须换到去压缩平面：屏幕空间里"垂直于轴的宽度"混进了长度方向 —— 箭头
    斜置时头部的宽度会被压扁，合成箭头实测 30°/150°/210°/330° 会把尖端判反（差 168°）。
    同一批样本换到去压缩平面后 12/12 正确，且宽度剖面就是箭头真实的头/杆宽度比。
    """
    if len(P) < 40:
        return 0.0, {}
    Q = np.stack([P[:, 0], P[:, 1] / max(k, 1e-6)], axis=1)
    d = _tip_tail_direction(Q, _principal_axis(Q))
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


def _ang_gap(a: float, b: float) -> float:
    """两个方向之间的最小夹角（度，0~180）。"""
    return abs(((a - b + 180.0) % 360.0) - 180.0)


def _otsu_threshold(values: np.ndarray, nbins: int = 64) -> float:
    """
    一维 Otsu：把这组亮度切成"亮/暗"两类，返回分界亮度。

    用来自适应地切出箭头顶面 —— 顶上"亮面 vs 侧壁"的**相对**亮度差在采集卡洗淡画面后
    基本不变，而写死的绝对亮度门槛（TOPFACE_V=0.80）会被洗淡一路带崩。

    （1.1.20 起 measure_arrow 不再切顶面 —— 真机箭头表面有横向条纹，见那里的注释；
    这个函数暂时没有调用方，留着是因为"洗淡画面下用相对阈值"的思路本身没问题。）
    """
    vals = np.asarray(values, dtype=np.float64)
    if vals.size < 32:
        return float(np.median(vals)) if vals.size else 0.0
    hist, edges = np.histogram(vals, bins=nbins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = float(hist.sum())
    if total <= 0:
        return 0.0
    centre = np.arange(nbins) + 0.5
    omega = np.cumsum(hist)                       # 前 k 档像素数
    mu = np.cumsum(hist * centre)                 # 前 k 档亮度和
    mu_t = mu[-1]
    denom = omega * (total - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        between = np.where(denom > 0, (mu_t * omega - mu * total) ** 2 / denom, 0.0)
    k = int(np.argmax(between)) if between.size else 0
    k = max(0, min(k, nbins - 2))
    return float((edges[k] + edges[k + 1]) / 2.0)


def _pick_elongated(mask: np.ndarray, rr: float,
                    min_elong: float = 1.5, min_solid: float = 0.0):
    """
    通用掩膜的取块：按形状打分（细长 × 实心 × 面积），返回该块的像素（超采样坐标）。

    为什么需要：色带（色相）是采集链最容易改坏的东西（洗淡、偏色、数字读错拿了别的档也
    会拿错色带）。"盘面内 + 高饱和 + 不是盘面草绿"这个判据不依赖具体色相，箭头一定落进来，
    剩下的就是把背景的圆斑/弥散块剔掉 —— 箭头是**细长**的，草斑和面板残留都不细长。
    """
    comps = label_components(mask, keep_pixels=True,
                             min_area=max(30, int(0.004 * math.pi * rr * rr)))
    best, best_score = None, 0.0
    for c in comps:
        if c.area < 0.006 * math.pi * rr * rr:
            continue
        pts = np.asarray(c.pixels, dtype=float)
        X = pts - pts.mean(axis=0)
        w = np.linalg.eigvalsh(X.T @ X / max(len(X), 1))
        if w[0] <= 1e-9:
            continue
        elong = float(w[1] / w[0])
        if elong < min_elong:                 # 圆斑/弥散块（草地、树影、面板残留）直接排除
            continue
        solid = c.area / float(max(c.w * c.h, 1))
        if solid < min_solid:
            continue
        # 打分 = 面积 × 实心 × 细长（细长封顶 4）。
        # 为什么细长要封顶、而且要乘面积：原先用 `elong * solid * log1p(area)`，实测在真帧上
        # 会被**一根细条**（elong 33、只有 456px，像是盘沿高光/面板分隔线）压过**真正的箭头**
        # （elong 8.4、2336px）—— 因为两者 log 面积差不多，而细长那项差了三倍。方向就这么飘到
        # 180° 外去了。细长已经用 min_elong 当硬门槛筛过，再往大里给权重没意义；"盘里最大的一块
        # 纯色就是箭头"这条前提才该主导，所以让**面积**直接进分母。
        score = min(elong, 4.0) * min(solid, 1.0) * float(c.area)
        if score > best_score:
            best_score, best = score, c
    return None if best is None else np.asarray(best.pixels, dtype=float)


def measure_arrow_generic(rgb: np.ndarray, dial: Dial, k: float | None = None,
                          supersample: int = 2) -> ArrowReading | None:
    """
    兜底测角：**完全不看色相**，在草地圆盘范围内找"高饱和、且不是盘面草绿"的细长块。

    什么时候用：紧色带和放宽色带都找不到箭头（"风速读得出、θ=None"）。常见原因是采集链
    把色相/饱和度改了、或风速数字读错导致拿了别的档的色带。这个判据不依赖具体色相，
    只要求"盘面上那块明显细长的饱和区域"——箭头必然满足，背景草斑/面板残留不满足。
    """
    img = Image.fromarray(rgb)
    box = _dial_box(dial, rgb.shape)
    ss = max(1, int(supersample or 2))
    crop = img.crop(box)
    big = crop.resize((crop.width * ss, crop.height * ss), Image.LANCZOS)
    hb = rgb_to_hsv(np.asarray(big))
    cxs, cys = (dial.cx - box[0]) * ss, (dial.cy - box[1]) * ss
    rr = dial.r * ss * 0.97
    yy, xx = np.mgrid[0:hb.shape[0], 0:hb.shape[1]]
    inside = ((xx - cxs) ** 2 + (yy - cys) ** 2) <= rr * rr
    on_disc = _disc_ellipse(dial, hb.shape[:2], ss=ss, box=box)
    grassish = ((hb[..., 0] >= 70.0) & (hb[..., 0] <= 115.0) & (hb[..., 1] >= 0.45))
    # 饱和门槛 0.60（不是 0.30）：原来 0.30 会把**天空**一起圈进来 —— 半透明面板后面的天空
    # 色相 ~201、S 中位 0.40~0.45，和箭头连成一大坨，`_pick_elongated` 判它不细长就整条放弃。
    # 实测箭头 S 中位 = 1.00（100% 的箭头像素 S>0.6），天空只有 2~12% 超过 0.6，门槛提到 0.6
    # 正好分开。唯一例外是 11-15 那档品红箭头（游戏画得偏粉，S 只有 ~0.51），单独留条口子。
    strong = ((hb[..., 1] >= 0.60)
              | ((hb[..., 0] >= 300.0) & (hb[..., 0] <= 345.0) & (hb[..., 1] >= 0.40)))
    generic = inside & on_disc & strong & (hb[..., 2] >= 0.45) & ~grassish
    P = _pick_elongated(generic, rr, min_elong=2.0, min_solid=0.35)
    if P is None or len(P) < 60 * ss * ss:
        return None
    P = P / ss + np.array([box[0], box[1]])
    theta, samples = estimate_direction_ground(P, dial.k if k is None else k)
    if not samples:
        return None
    phi = true_to_apparent(theta, dial.k if k is None else k)
    tip, tail = _tip_tail_points(P, phi)
    return ArrowReading(phi=phi, conf=float(samples.get("conf", 1.0)), phi_samples=samples,
                        n_pixels=int(round(len(P) / (ss * ss))), tip=tip, tail=tail,
                        source="通用掩膜")


def measure_arrow(rgb: np.ndarray, hsv: np.ndarray, dial: Dial,
                  supersample: int = 6,
                  band: tuple[float, float, float, float] | None = None) -> ArrowReading | None:
    """
    测量箭头在画面上的指向 φ（0=正上，顺时针）。
    band 给定时用单色窄带掩膜取最大团块；band=None 时用多色宽带 + 形状打分（定位兜底）。

    1.1.20 起，band 给定的正常路**直接用整块窄带剪影**定方向（不再按亮度切"顶面"）：
    本作箭头表面带横向条纹，全局亮度阈值切出来的是横带不是顶面，方向随之乱抖；
    整块剪影只用"箭头自己的颜色 + 形状"，既不依赖背景亮度，帧间也稳（见下面 pass 处注释）。
    """
    img = Image.fromarray(rgb)
    box = _dial_box(dial, rgb.shape)          # 和 detect_arrow_band 用同一个裁剪框
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
    # 只在草地圆盘上找箭头：半透明面板会把背景（树/草地）的黄绿像素漏进来，不圈住就会
    # 把背景当箭头（表现就是"背景有树时读不出风向"）。
    on_disc = _disc_ellipse(dial, arr.shape[:2], ss=ss, box=box)
    clip &= inside & on_disc
    topf &= inside & on_disc

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
    phi, samples = estimate_direction(P_sil)          # 兜底估计：整块剪影（屏幕空间）
    P_use = P_sil
    src = "剪影"
    if band is not None:
        # 1.1.20：已知颜色这条路**直接用整块剪影**，不再用 Otsu 按亮度切"亮顶面"。
        #
        # 为什么废掉切顶面：本作箭头是**上下两截不同亮度**（侧壁暗、顶面亮），但赛道上的
        # 真实箭头表面还带**横向条纹**（垂直长轴的明暗带）—— 按全局亮度阈值切，切出来的是
        # 一条条横带而不是那张平顶面。这个条纹掩膜的 PCA 轴随帧抖，实测同一静止画面
        # θ 在 347~355° 之间跳（极差 7.89°），而整块剪影只抖 0.48°。
        # 侧壁确实会把轴往画面下方拽，但实测把剪影逐列底边往上削 2~8px（模拟去掉侧壁）
        # θ 只动 0.15° —— 这个机位上侧壁对方向的影响远小于切顶面引入的抖动。
        # （合成箭头那个"切顶面才准"的结论建立在顶面是均匀色的前提上，真机不成立。）
        pass
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
                        tip=tip, tail=tail, source=src)


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
    # 先开运算：面板是半透明的，背景亮（草地/树/面板分隔线）会漏进白掩膜，既会把文字框
    # 撑高（0.18r~0.50r 的高度检查直接判掉整行），也会把相邻字形连成一块。
    m = _open_mask((hsv[..., 2] > 0.55) & (hsv[..., 1] < 0.45))
    # 面积下限 0.0035·r²：排掉面板网点小斑点，又不误杀小分辨率下的真文字
    min_area = max(8, int(0.0035 * r * r))
    comps = [c for c in label_components(m, keep_pixels=False, min_area=min_area)
             if 0.10 * r <= c.h <= 0.75 * r and c.w >= 2
             and 0.12 <= c.w / float(c.h) <= 2.8]          # 排除细长分隔线/圆环弧
    if not comps:
        return []          # 必须返回空列表：调用方是 for...in，返回 None 会 TypeError
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
        box = (max(X0 + x0 - pad, 0), max(Y0 + y0 - pad, 0),
               min(X1 + x0 + pad, img_w), min(Y1 + y0 + pad, img_h))
        # "Nm/s" 那行的高度是固定的（约 0.3·r）。太高的框多半是把别的东西圈进来了 ——
        # 例如小地图上的地形/徽章，读出来的"数字"根本不是风速。
        bh = box[3] - box[1]
        if 0.18 * r <= bh <= 0.50 * r:
            boxes.append(box)
    return boxes


def ocr_speed(rgb: np.ndarray, dial: Dial, state: OcrState | None = None,
              allow=None, best_effort: bool = False,
              boxes: list | None = None) -> dict:
    """识别风盘附近的 "Nm/s" 数字：切出文字行，用字形模板库匹配。

    返回 dict(value, raw, conf, box, ok)；模板库为空、不够有把握、或读出的数值
    超出游戏范围（0~MAX_WIND m/s）时 value 为 None。

    reject 字段记着"本来读成了多少"，专供排查：读成 35 m/s 这种明显不可能的，
    既不算数，也得留下证据（调用方会因此存一帧图）。

    allow 给定时只用它筛数值 —— 调用方已经知道箭头颜色时，可以把风速框死在那条
    颜色对应的区间里（箭头是大块纯色，比细笔画数字可靠得多）。

    boxes 给定时直接用这批文字行候选（一场帧里候选框跟风速无关，算一次就够；
    read_wind 一帧可能读好几轮，传进来能省掉重复的整块搜索）。
    """
    state = state or _DEFAULT_OCR
    ok_val = allow or (lambda v: 0 <= v <= MAX_WIND)
    res = dict(value=None, raw="", conf=0.0, box=None, ok=False, reject=None)
    # 候选文字行按面积从大到小依次试，第一个匹配上的就是风速
    for roi in (boxes if boxes is not None else _find_text_lines(rgb, dial)):
        x0, y0, x1, y1 = roi
        res["box"] = [x0, y0, x1, y1]
        crop = Image.fromarray(rgb[y0:y1, x0:x1])
        tm = match_glyphs(crop, state=state, accept=ok_val, best_effort=best_effort)
        if tm is None:
            bad = match_glyphs(crop, state=state)      # 只是一探：记下"读成了什么"供排查
            if bad is not None:
                res["reject"] = dict(value=int(bad[0]), conf=float(bad[1]), box=res["box"])
            continue
        val = int(tm[0])
        # match_glyphs 已保证"绝对分 ≥0.60 且与次高差 ≥0.10"，这里只兜游戏范围的上下限
        if ok_val(val) and tm[1] >= (0.45 if best_effort else 0.60):
            res.update(value=val, raw=f"tpl:{tm[0]}", conf=tm[1], ok=True)
            return res
    return res



# --------------------------------------------------------------------------
# 字形模板库：模板由离线脚本构建，运行时只读匹配、不自学。
# --------------------------------------------------------------------------

_GLYPH_SIZE = (32, 48)


def _line_band(mask: np.ndarray) -> tuple[int, int]:
    """
    文字行的公共"上下沿"（行基准）。

    用整行的墨迹行剖面取主体（掐掉抗锯齿毛边）—— 这一行里 `m/s` 永远存在、
    字高永远一样，拿它当基准比让每个字自己围包围盒稳得多。
    """
    rows = mask.sum(axis=1)
    if rows.max() <= 0:
        return 0, mask.shape[0]
    idx = np.nonzero(rows >= max(1.0, 0.25 * float(rows.max())))[0]
    return int(idx[0]), int(idx[-1]) + 1


def _glyph_norm(crop: Image.Image,
                band: tuple[int, int] | None = None) -> np.ndarray | None:
    """
    把单个字形归一化成固定大小的二值图（等比缩放）。

    band 是文字行的公共上下沿（见 _line_band）。**优先信它**，而不是字形自己的包围盒：
    抗锯齿多出来的那一两行/一两列会把细笔画（比如 "1"）整个压扁，分数就掉一大截。
    给了 band 就按它定高度；水平方向仍按本字形自己的墨迹取满 —— 宽度正是区分数字的
    依据，不能掐（"1" 顶上那个小撇子墨很少，一掐就只剩一根竖杠了）。
    """
    # 二值化必须用"本字形这块"自己的阈值（_bin_white 是按裁块中位亮度定阈的），
    # 拿整行算好的掩膜切下来会换一套口径，和模板库的建库口径对不上 —— 别省这一步。
    m = _bin_white(crop)
    h = m.shape[0]
    y0, y1 = band if band else (0, h)
    y0, y1 = max(0, min(int(y0), h - 1)), max(1, min(int(y1), h))
    sub = m[y0:y1]
    ys, xs = np.nonzero(sub)
    if len(ys) < 12:
        return None
    sub = sub[:, xs.min():xs.max() + 1]
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


def _split_wide_runs(mask: np.ndarray, runs: list[tuple[int, int]],
                     kth: int = 0) -> list[tuple[int, int]]:
    """
    把"过宽的列块"从第 kth 细的腰处拆开（kth=0 就是最细的那一列）。

    "1" 紧挨着 "m" 时会被并成一整块：这块的高度按高的那个（数字）算、宽高比超标，
    于是数字整个从候选里消失，只剩 "/" 和 "s" —— 表现就是 1m/s 读成 5。
    粘连处通常只有几个抗锯齿像素，是块内墨迹最少的列，从那儿下刀基本能分开。
    但"最细的列"未必正好是接缝（m 的竖笔之间也可能更细），所以调用方会多切几个位置，
    切错了没关系 —— 切出来的块不像数字，匹配那关自然会被刷掉。

    **只拆最左边那一块，而且只在这块"宽得不像单个字形"时才拆**：
    数字永远在最左边，粘连（1+m、13 之类）只可能发生在那里。拆别的块纯属惹祸 ——
    实测把 "6m/s" 的 m 拆开，碎片能凑成 "5"（0.77）而真数字反而被挤掉。
    """
    out: list[tuple[int, int]] = []
    for idx, (a, b) in enumerate(runs):
        w = b - a
        h = int(mask[:, a:b].any(axis=1).sum())
        # 不是最左边那块、或宽度还在"单个字形"范围内 → 原样保留
        if idx != 0 or w < 12 or w <= 1.15 * max(h, 1):
            out.append((a, b))
            continue
        col = mask[:, a:b].sum(axis=0)
        lo, hi = max(1, int(0.15 * w)), min(w - 1, int(0.85 * w))
        if hi <= lo:
            out.append((a, b))
            continue
        order = sorted(range(lo, hi), key=lambda i: int(col[i]))
        k = order[min(kth, len(order) - 1)]
        # 只认"真正的缝"：数字内部的细腰（比如 4 的横杠那一列）也常常是全块最细的列，
        # 从那儿下刀会把 4 切成"竖杠 + 残块"→ 竖杠被认成 1。缝的墨量要明显低于全块中位数。
        med = float(np.median(col[lo:hi]))
        if col[k] > max(2.0, 0.35 * med):
            out.append((a, b))
            continue
        out.append((a, a + k))
        out.append((a + k, b))
    return out


def _run_variants(mask: np.ndarray, runs: list[tuple[int, int]],
                  cuts: int = 3) -> list[list[tuple[int, int]]]:
    """列块变体：原样一份，再加上"在几个不同的细腰处下刀"的若干份。"""
    out = [runs]
    for k in range(cuts):
        v = _split_wide_runs(mask, runs, kth=k)
        if v != runs and v not in out:
            out.append(v)
    return out


def _glyph_windows(crop: Image.Image, max_digits: int = 2,
                   two_start: int = 3, one_start: int = 2,
                   mask: np.ndarray | None = None
                   ) -> list[list[tuple[int, int]]]:
    """
    列出"可能是数字"的列区间组合，供上层逐个试。

    风速数字**永远在 "m/s" 前面**，所以只在最左边几个列块里找（留点余量给混进
    数字框的杂块）。这条约束把 "m"、"/"、"s" 排除在候选之外 —— 之前是"任意起点"
    全枚举，"s" 会被认成 5：于是 1m/s 读成 5、3m/s 读成 35。
    （1 位数只在前 one_start 个块里找：粘连时 "s" 会顶到前面来，得卡死。）

    另外再补一份"把过宽粘连块拆开"的列块（见 _split_wide_runs）作为第二组候选。
    mask 给定时直接拿来用（省一次二值化）。
    """
    m = mask if mask is not None else _bin_white(crop)
    if m.sum() < 20:
        return []
    runs = _column_runs(m)
    if not runs:
        return []
    out: list[list[tuple[int, int]]] = []
    for rs in _run_variants(m, runs):
        for n, start_cap in ((min(max_digits, 2), two_start), (1, one_start)):
            head = rs[:start_cap]
            for i in range(max(0, len(head) - n + 1)):
                win = list(head[i:i + n])
                if len(win) == n and win not in out:
                    out.append(win)
    return out


def match_glyphs(crop: Image.Image,
                 state: OcrState | None = None,
                 accept=None,
                 best_effort: bool = False) -> tuple[str, float] | None:
    """
    用模板库识别数字部分。返回 (数字串, 置信度)；库为空或匹配太弱返回 None。

    逐个尝试候选列区间组合，**第一个全部字形都匹配上的**就是答案（2 位优先于 1 位，
    这样 "13m/s" 会给出 13 而不是 1）。

    accept 是可选的数值校验：不通过就换下一组窗口再试。用来挡掉"3m/s 被凑成 35"
    这种 —— 2 位数先试，凑出来的值超出游戏范围就直接作废，回头还能认出正确的那一位。

    best_effort=True 时，如果一组都没通过上面的严格判据，就退一步：**只在 accept
    允许的数字里**挑分最高的那个（门槛也放低）。用在"箭头颜色已经告诉我们风速在哪一档"
    的时候 —— 例如黄色箭头 ⇒ 只能是 6~10，那么 5 就被排除在外，剩下的候选里 6 通常
    就是最高分，能把"6 读成 5"救回来。
    """
    # state 是空的（或压根没加载过库）时回落到默认库 —— 否则整条链会静默地"永远读不出风速"
    bank = ((state or _DEFAULT_OCR).bank) or _DEFAULT_OCR.bank
    if not bank:
        return None
    # 整个裁块只二值化这一次（往下传，不再逐层重算），并先清一遍：开运算去掉背景漏进来的
    # 细刺/毛边、顺手把粘连的两个字形分开，再丢掉远小于最大团块的碎墨（否则一根斜刺会
    # 变成一个"列块"，被当成 '1' —— 实测真帧里 '9' 左边就有这么一根）。
    m = _clean_white_mask(_bin_white(crop))
    band = _line_band(m)                     # 行基准：所有字形共用同一套上下沿
    fallback: tuple[float, str] | None = None
    normed: dict[tuple[int, int], np.ndarray | None] = {}   # 同一个列块只归一化一次
    holes_of: dict[tuple[int, int], int] = {}               # 洞数同理（label_components 不便宜）
    for win in _glyph_windows(crop, mask=m):
        out, worst = "", 1.0
        alt_out, alt_worst = "", 1.0
        for a, b in win:
            key = (int(a), int(b))
            if key not in normed:            # 各候选窗口大量重叠，同一块别算两遍
                normed[key] = _glyph_norm(crop.crop((a, 0, b, crop.height)), band)
                holes_of[key] = -1           # 惰性算：真要用到这一步才数洞
            g = normed[key]
            if g is None:
                break
            if holes_of[key] < 0:
                holes_of[key] = _hole_count(g > 0.5)
            best, second, digit = 0.0, 0.0, ""
            best_ok, digit_ok = 0.0, ""
            # 拓扑约束：字形**看得出洞**时，洞数必须和模板一致。8 是唯一有两个洞的数字，
            # 而 8 和 6 的像素相似度高达 0.79（全库最像的一对）—— 只靠 IoU 会互相认错。
            # 只在输入确实有洞时才卡（洞=0 也可能是圈被抗锯齿/压缩咬断了，不能反过来定死）。
            # 洞数不符的模板直接不进 best/second，否则 8 的那点差距会被 6 抢走。
            holes_g = holes_of[key]
            for d, temps in bank.items():
                allowed = accept is None or accept(int(d))
                for t in temps:
                    if holes_g and _template_holes(t) != holes_g:
                        continue
                    sc = _glyph_sim(g, t)
                    if sc > best:
                        second = max(second, best)
                        best, digit = sc, d
                    else:
                        second = max(second, sc)
                    if allowed and sc > best_ok:
                        best_ok, digit_ok = sc, d
            if best_ok > 0:
                alt_out += digit_ok
                alt_worst = min(alt_worst, best_ok)
            # 判据是"差距"，不是绝对分：
            #   "1" 这种细笔画的字，绝对相似度天生上不去（归一化后差几个像素，IoU 掉一大截，
            #   实测干净切出来的 "1" 也只有 0.66），但它跟第二名的差距极大（0.66 vs 0.45）。
            #   反过来 "/"、"s" 这些杂块是"谁都不像"：最高 0.70、第二名 0.69，差距只有 0.01。
            #   所以：绝对分兜个底 + 差距要求大，两个都要满足。
            # 但"绝对分很高"本身就是强证据，这时差距可以放宽：实测真图里的 "6" 是
            # 0.820 / 次 0.731（差 0.088），和 "8" 差一口气就差 0.01 没够着 0.10，
            # 结果真数字被判掉、反而让杂块钻空子 —— 所以 0.80 以上放宽到 0.06。
            if best < 0.60 or (best - second < 0.06 and best < 0.80):
                break                       # 这个字形不像数字（m / s / 杂块）→ 换下一组
            out += digit
            worst = min(worst, best)
        else:
            if accept is None or accept(int(out)):
                return out, worst           # 整组都匹配上、数值也合理，才算数
        if best_effort and len(alt_out) == len(win) and alt_worst >= 0.45:
            if fallback is None or alt_worst > fallback[0]:
                fallback = (alt_worst, alt_out)
    if fallback is not None:
        return fallback[1], fallback[0]
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


def true_to_apparent(theta: float, k: float) -> float:
    """真实夹角 θ -> 画面夹角 φ（apparent_to_true 的逆）：φ = atan2(sinθ, k·cosθ)。"""
    t = math.radians(theta % 360.0)
    return math.degrees(math.atan2(math.sin(t), k * math.cos(t))) % 360.0


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
# 6. 洞口力度（小地图）
# --------------------------------------------------------------------------
# 要算的东西：**洞口在瞄准线的第几格**（力度）。同一洞风速固定，但"这个洞要几力"
# 每杆都在变，所以是实时算的。
#
# 游戏小地图上本来就把要素画全了（实测 8/9 两张真帧）：
#   · 玩家 = 一个黄色小三角（还有个更大的黄箭头是右下角风向徽章里的，要排除）
#   · 瞄准线 = 青色分段线，上面有白色刻度点，**相邻两点间距 = 1.0 力**
#   · 线末端一个黄点 = 4.0 力（满力）
#   · 洞口 = 旗杆根部那个黑椭圆的中心（红旗面是品红三角，黑椭圆在它正下方）
# 于是：  力度 = 4.0 × |玩家→洞口| / (4 × 刻度间距)
#
# 刻度间距每帧现量（实测两帧 54.8 / 67.2px 不一样 —— 瞄准线长度随杆变化，写死必错）。
MINIMAP_W_FRAC = 0.193      # 小地图面板宽 / 画面宽（面板在风盘正下方、与风盘同轴心）
MINIMAP_H_FRAC = 0.431      # 面板高 / 画面高
MINIMAP_TOP_FRAC = 0.319    # 面板顶边 / 画面高
HOLE_MAX_POWER = 4.0        # 满力（所有杆都是 4.0）
HOLE_UNIT_RANGE = (25.0, 130.0)   # "1.0 力"对应像素长度的合理区间，超出当误检
HOLE_MAX_OUT = 20.0         # 只当误检门槛用：真人打得远、又拿了短杆时 9~15 力都正常
HOLE_DY = 26.0              # 洞口相对**旗面底边**的纵向偏移（球洞图标是固定图案，实测 ±2px）


def _minimap_box(shape, fx: float) -> tuple[int, int, int, int]:
    """小地图面板的裁剪框：水平中心与风盘同轴（右手 0.8802W / 左手 0.1198W）。"""
    h, w = shape[:2]
    cx = fx * w
    hw = MINIMAP_W_FRAC * w / 2.0
    y0 = MINIMAP_TOP_FRAC * h
    return (int(max(cx - hw, 0)), int(max(y0, 0)),
            int(min(cx + hw, w)), int(min(y0 + MINIMAP_H_FRAC * h, h)))


def read_hole_power(rgb: np.ndarray, side: str | None = None) -> dict | None:
    """
    算"洞口位于多少力处"：小地图上量玩家→洞口的距离，用瞄准线刻度换算成力度。

    side="右"/"左" 时只看那一侧的小地图（小地图在风盘正下方，左右手镜像）。
    side=None 时两侧都试 —— 推杆界面没有风盘、调用方还不知道左右手，就得靠这一步自己找。
    返回 dict(power, unit, dist_px, player, hole, n_dots, side, box)，找不到返回 None。
    """
    for name, (fx, _fy, _fr) in GAUGE_PRIORS:
        if side is not None and name != side:
            continue
        box = _minimap_box(rgb.shape, fx)
        got = _hole_power_in(rgb, box)
        if got is not None:
            got["side"] = name
            return got
    return None


def _hole_power_in(rgb: np.ndarray, box: tuple[int, int, int, int]) -> dict | None:
    """在给定的小地图裁剪框里量一次（见 read_hole_power 的注释）。"""
    x0, y0, x1, y1 = box
    if x1 - x0 < 80 or y1 - y0 < 80:
        return None
    sub = rgb[y0:y1, x0:x1]
    hs = rgb_to_hsv(sub)
    H, S, V = hs[..., 0], hs[..., 1], hs[..., 2]

    # ① 瞄准线（青色分段）→ 主轴方向 + 靠玩家那一端。
    # 掩膜要紧：小地图上还有**水/河**（H≈206、V≈0.64）和玩家头像的青圈（S 只有 0.5 左右），
    # 放宽会把它们一起圈进来、主轴直接歪掉（实测过：3.png 的"线起点"被拽到 323 去）。
    # 实测瞄准线本体 H≈181~184、S≈0.77~0.87、V≈0.90~0.93，且**每一段都是细长条**
    # （elong 26~124），所以再加一条形状过滤：细长 + 够长。
    comps = []
    for c in label_components((H >= 176.0) & (H <= 196.0) & (S >= 0.65) & (V >= 0.85),
                              keep_pixels=True, min_area=30):
        # 形状判据必须用 **PCA 细长度**，不能用外接框长宽比：瞄准线常常是斜的，
        # 斜向线段的包围盒接近正方（实测 8.png 的段是 31×35），按 bbox 判会全部误杀 ——
        # 之前 5/7/8 三帧量不出来就是这个原因。细长度对斜线免疫（线段 ~26~124，圆斑 ~2）。
        pts = np.asarray(c.pixels, dtype=float)
        X = pts - pts.mean(axis=0)
        ev = np.linalg.eigvalsh(X.T @ X / max(len(X), 1))
        if ev[0] > 1e-9 and ev[1] / ev[0] >= 8.0:
            comps.append(c)
    if not comps:
        return None
    L = np.vstack([np.asarray(c.pixels, dtype=float) for c in comps])
    c0 = L.mean(axis=0)
    _ew, vecs = np.linalg.eigh(np.cov((L - c0).T))
    axis = vecs[:, -1]
    if axis[1] < 0:
        axis = -axis                     # 让它指向画面下方（玩家那一侧）
    perp = np.array([-axis[1], axis[0]])
    t = (L - c0) @ axis
    origin = c0 + axis * float(t.max())

    # ② 白刻度点 → 1.0 力 = 相邻两点间距（这就是每帧现量的那个 s）
    dots = []
    for c in label_components((V >= 0.85) & (S <= 0.20), keep_pixels=True, min_area=16):
        if c.area > 340 or not (5 <= c.w <= 20) or abs(c.w - c.h) > 5:
            continue                     # 太大/太扁的都不是刻度点（面板边框、文字）
        q = np.array([c.pixels[:, 0].mean(), c.pixels[:, 1].mean()])
        if abs(float((q - c0) @ perp)) > 9.0:
            continue                     # 不在线上
        dots.append((float((q - c0) @ axis), q))
    if len(dots) < 2:
        return None                      # 量不出刻度就没法换算力度
    dots.sort(key=lambda z: -z[0])
    unit = float(np.median(np.diff([-a for a, _ in dots])))
    if not (HOLE_UNIT_RANGE[0] <= unit <= HOLE_UNIT_RANGE[1]):
        return None

    # ③ 玩家 = 离瞄准线起点最近的**小**黄块（那个三角）。
    #    要排掉两个坑：右下角风向徽章里的黄箭头更大（面积 >300），线上 4.0 那个黄点虽小
    #    但离起点远（一个刻度以上）。
    player = None
    best_d = None
    for c in label_components((H >= 40.0) & (H <= 64.0) & (S >= 0.60) & (V >= 0.70),
                              keep_pixels=True, min_area=60):
        if c.area >= 300:
            continue
        q = np.array([c.pixels[:, 0].mean(), c.pixels[:, 1].mean()])
        d = float(np.hypot(*(q - origin)))
        if d > 2.5 * unit:                # 线从球画起、三角标在人身上，两者本来就有间隔；
                                          # 2.5 个刻度足够松，又能把线末端那个 4.0 黄点挡在外面
            continue
        if best_d is None or d < best_d:
            best_d, player = d, q
    if player is None:
        return None

    # ④ 旗子（品红三角）→ 洞口（旗杆根部的黑椭圆中心）。
    #    直接找"黑椭圆"不靠谱，两个坑都踩过：
    #      · "旗子附近最大的暗块"—— 旗子上方的地图暗色地形/水面更大；
    #      · "旗面下方那扇窗里最下面的暗像素"—— 旗子在小地图**边缘**时，旗面下方紧邻的
    #        就是地图暗色地形，会取到那底下（实测 10-08/10-09 两张截图就这么错的：洞口
    #        偏 60px，力度从 ~4.0 变成 3.06）。
    #    而黑椭圆自己还会被瞄准线/红弧切成好几块。
    #    所以改用**旗杆的 x**：旗杆是旗面正下方那根白色细线，实测 4 帧它的 x 与真洞口
    #    差 ≤2px；纵向用固定偏移（球洞图标是固定图案，"旗面底边往下 26px"在 4 帧上只差
    #    ±2px）。最后再用附近的小黑块（有就用）微调一下。
    flags = label_components((H >= 300.0) & (H <= 348.0) & (S >= 0.50) & (V >= 0.70),
                             keep_pixels=True, min_area=100)
    if not flags:
        return None
    fc = max(flags, key=lambda c: c.area)
    zone = np.zeros(V.shape, dtype=bool)
    yb0, yb1 = int(fc.y1) + 1, min(int(fc.y1) + 50, V.shape[0])
    for xa, xb in ((fc.x0 - 10, fc.x0 + 8), (fc.x1 - 8, fc.x1 + 10)):
        zone[yb0:yb1, max(int(xa), 0):min(int(xb), V.shape[1])] = True
    py, px = np.nonzero(zone & (V >= 0.70) & (S <= 0.45))
    hole_x = float(px.mean()) if len(px) >= 6 else float(fc.x0) - 2.0
    hole = np.array([hole_x, float(fc.y1) + HOLE_DY])
    refine = None                               # 附近的小黑块 = 黑椭圆，有就拿它微调
    for c in label_components(V <= 0.45, keep_pixels=True, min_area=10):
        if c.area > 900 or not (0.35 <= c.w / float(max(c.h, 1)) <= 3.0):
            continue
        if c.area < 0.30 * c.w * c.h:
            continue
        q = np.array([c.pixels[:, 0].mean(), c.pixels[:, 1].mean()])
        if abs(float(q[0]) - hole_x) > 16 or abs(float(q[1]) - hole[1]) > 14:
            continue
        if refine is None or c.area > refine[0]:
            refine = (c.area, q)
    if refine is not None:
        hole = refine[1]

    # ⑤ 力度
    dist = float(np.hypot(*(hole - player)))
    power = HOLE_MAX_POWER * dist / (4.0 * unit)
    if not (0.0 < power <= HOLE_MAX_OUT):
        return None
    return dict(power=round(power, 2), unit=round(unit, 1), dist_px=round(dist, 1),
                n_dots=len(dots), box=box,
                player=(round(float(player[0] + x0), 1), round(float(player[1] + y0), 1)),
                hole=(round(float(hole[0] + x0), 1), round(float(hole[1] + y0), 1)))


# --------------------------------------------------------------------------
# 7. 顶层分析流程
# --------------------------------------------------------------------------

@dataclass
class FrameReading:
    """一次「读风速 → 测箭头 → 分解风」的结果（风盘定位/档位由调用方提供）。"""
    ok: bool = False
    no_wind: bool = False
    speed: float | None = None
    speed_note: str = ""               # ""=确定；"?"=数字只是勉强过线，显示时带问号
    speed_range: tuple | None = None   # 只认出颜色档、数字读不出：给区间（如 (6,10)），分量算不了
    cands: list = field(default_factory=list)   # 数字候选表 [(值, 分数, 是否严格过关, 档名)]
    message: str = ""
    dial: Dial | None = None
    arrow: ArrowReading | None = None
    phi: float | None = None
    k: float | None = None
    theta: float | None = None
    wind: dict = field(default_factory=dict)
    ocr: dict | None = None
    arrow_band: tuple | None = None    # 箭头**实际量到**的颜色档（不是数字推出来的那个档）
    # ↑ 用途：风速锁定的"安全网"要拿它跟锁定档比 —— 箭头颜色是量出来的，比细笔画数字可信；
    #   通用掩膜那条路量不出颜色，这里是 None（表示"这拍不知道箭头什么颜色"）。


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
    ss = max(1, int(supersample or SUPERSAMPLE))
    weak = False                                 # 数字只是"勉强过线"（best_effort）读出来的
    band_hint = None                             # 箭头自己的颜色档（读数字用，也是区间显示的来源）
    band_strict = False                          # 严格档（不放宽）认出了箭头颜色 —— "盘里真有箭头"的判据
    boxes: list | None = None                    # 文字行候选框（一帧只搜一次，多轮复用）
    if speed is None and do_ocr:
        boxes = _find_text_lines(rgb, dial)          # 一帧只搜一次文字行候选，下面几轮复用
        # ① 先看箭头颜色 ⇒ 风速在哪一档，再让数字在这个窄范围里认。
        #    箭头是盘里最大的一块纯色，比细笔画数字可靠；这样"6 读成 5"这种
        #    跨档误读从源头上就不会发生（5 直接不在候选里）。
        band_hint = detect_arrow_band(rgb, dial)
        band_strict = band_hint is not None
        if band_hint is None:                       # 颜色被采集卡洗淡时，放宽 S/V 再认一次
            band_hint = detect_arrow_band(rgb, dial, min_px=60, relax=0.30)
        if band_hint is not None:
            lo, hi = wind_range(band_hint)

            def in_band(v, lo=lo, hi=hi):
                return lo <= v <= hi

            ocr = ocr_speed(rgb, dial, state=ocr_state, allow=in_band, boxes=boxes)
            strong = ocr.get("value") is not None   # 严格判据（绝对分 ≥0.60 且与次高差 ≥0.10）过关
            if not strong:                          # 严格没过 → 在这一档里挑最像的
                ocr = ocr_speed(rgb, dial, state=ocr_state, allow=in_band,
                                best_effort=True, boxes=boxes)
            fr.ocr = ocr
            if ocr.get("value") is not None:
                weak = not strong                   # 勉强过线 → 读数要带问号显示（见 speed_note）
                fr.cands.append([int(ocr["value"]), float(ocr.get("conf") or 0.0),
                                 bool(strong), band_name(band_hint)])
                speed = float(ocr["value"])
        if speed is None:                           # 没看出颜色 / 这一档读不出 → 老的全区间读法
            ocr = ocr_speed(rgb, dial, state=ocr_state, boxes=boxes)
            if fr.ocr is None or ocr.get("value") is not None:
                fr.ocr = ocr
            if ocr.get("value") is not None:
                fr.cands.append([int(ocr["value"]), float(ocr.get("conf") or 0.0), True, ""])
                speed = float(ocr["value"])
        if speed is None:
            # 还是读不出：把**另外两档**也各读一次，只为凑出"到底在每一档里读成了什么"的候选表
            # （日志/界面要显示它）。放在失败路里，正常帧一次都不会多跑。
            for lo2, hi2, name2, (h0, h1), smin, vmin in WIND_COLORS:
                if band_hint is not None and (h0, h1, smin, vmin) == tuple(band_hint):
                    continue                        # 这一档上面刚试过

                def in2(v, lo2=lo2, hi2=hi2):
                    return lo2 <= v <= hi2

                r2 = ocr_speed(rgb, dial, state=ocr_state, allow=in2, boxes=boxes)
                strong2 = r2.get("value") is not None
                if not strong2:
                    r2 = ocr_speed(rgb, dial, state=ocr_state, allow=in2,
                                   best_effort=True, boxes=boxes)
                if r2.get("value") is not None:     # 没严格过关的会带 "~" 标记出现在候选表里
                    fr.cands.append([int(r2["value"]), float(r2.get("conf") or 0.0),
                                     bool(strong2), name2])
    fr.speed = None if speed is None else float(speed)
    fr.speed_note = "?" if weak else ""
    pre_arrow = None                      # 0 m/s 那一步如果确认"其实有风"，直接把量到的箭头带下来
    if fr.speed is not None and abs(fr.speed) < 1e-9:
        # 读到 0 ≠ 真的无风：**无风时游戏不画箭头**，所以"盘里有没有箭头"才是判据。
        # 0 和 8 是全库最像的一对（IoU 0.659），把 8 认成 0 完全可能，而一旦按"无风"短路，
        # 方向也跟着丢了（"本来能识别的项被另一项拖死"）。
        # 判据用**严格档**（不放宽）：
        #   · 真无风帧实测：严格档 = None（干净），而"放宽档"会给个青档假阳性、
        #     连不看色相的通用掩膜都会在空盘上找到 1290px 的假箭头 —— 那两个都不能当判据。
        #   · 有风帧：严格档能认出箭头颜色（三档都是 1~15），所以读到 0 必然是误读。
        if not band_strict:
            fr.ok, fr.no_wind, fr.speed = True, True, 0.0
            fr.message = "风速 0 m/s —— 无风，按规则跳过计算"
            return fr
        # 有箭头 ⇒ 这个 0 是误读：按"风速未读出、方向照给"处理（band 沿用严格档那个颜色）
        fr.speed = None
    # 数字读出来 → 用数字对应的颜色；数字没读出但箭头颜色认出来了 → 用箭头自己的颜色
    band = wind_color(fr.speed) if fr.speed is not None else band_hint
    arrow = pre_arrow if pre_arrow is not None else measure_arrow(rgb, hsv, dial, supersample=ss, band=band)
    if arrow is None and band is not None:
        # 采集卡把颜色洗淡时（色相还在、S/V 不达标），先用放宽门槛的同一档色带再试一次，
        # 再谈"是不是另一种颜色"。实测这就是"风速读得出、箭头找不到"的主因。
        arrow = measure_arrow(rgb, hsv, dial, supersample=ss, band=_relax_band(band))
    mismatch = False
    # 箭头实际量到的颜色档：正常路/放宽路就是 band，颜色对不上改走别档时在下面被改写，
    # 通用掩膜（不看色相）那条路量不出颜色，保持 None。
    arrow_band = band if arrow is not None else None
    if arrow is None and band is not None and do_ocr:
        # 这一档的箭头没找到 → 探一下盘里是不是"另一种颜色"的箭头。
        # 箭头是盘里最大的一块纯色，比那两三个细笔画数字可靠得多：如果它是别的颜色，
        # 几乎可以断定**数字读错了** —— 例如 6m/s 读成 5，于是拿 1-5 的青带去抓
        # 黄色箭头，当然抓不到，窗口就报"彩色箭头未找到"。这时按该颜色对应的风速
        # 区间把数字重读一遍，数字和颜色就自洽了。
        for lo, hi, _name, (h0, h1), smin, vmin in WIND_COLORS:
            full = (h0, h1, smin, vmin)          # 和 wind_color() 返回的是同一种四元组
            if full == band:                     # 这条刚试过，跳过
                continue
            alt = measure_arrow(rgb, hsv, dial, supersample=ss, band=full)
            if alt is None:
                continue
            mismatch = True
            ocr2 = ocr_speed(rgb, dial, state=ocr_state, boxes=boxes,
                             allow=lambda v, lo=lo, hi=hi: lo <= v <= hi,
                             best_effort=True)
            arrow, arrow_band = alt, full        # 箭头是**量**出来的，比数字可信，先认它
            if ocr2.get("value") is not None:
                fr.ocr = ocr2
                fr.speed = float(ocr2["value"])
                fr.speed_note = "?"                 # 这条是"跟箭头颜色对不上、换一档重读"来的，存疑
                fr.cands.append([int(ocr2["value"]), float(ocr2.get("conf") or 0.0),
                                 False, band_name(full)])
                band = wind_color(fr.speed) or full
            else:
                # 箭头颜色明确落在这一档，但数字在这一档里读不出来 —— 那刚才读到的那个数字
                # 就是**自相矛盾**的：品红箭头 ⇒ 只可能是 11~15，绝不可能读出 1。
                # 以前这里保留那个数字、箭头换成"通用掩膜"，结果窗口显示 1 m/s 而游戏是 13。
                # 现在丢掉那个数字，风速降级成区间显示（方向照给），也不能给一个跟箭头打架的数。
                # band 也必须改成它 —— 后面 speed_range 是按 band 取的（漏了这步就会出现
                # "区间显示 1-5、箭头却是黄档"这种自相矛盾）。
                fr.speed, fr.speed_note, band = None, "", full
            break
    if arrow is None:
        # 最后一招：完全不看色相，在盘面里找"高饱和、非草绿、明显细长"的那块（见函数注释）。
        # **不管 band 有没有都要试**：1.1.30 之前这里写了 `if band is not None`，于是
        # "数字和颜色都认不出"（band=None）的帧连方向也一起丢了 —— 而方向恰恰不依赖它们俩。
        arrow = measure_arrow_generic(rgb, dial, supersample=ss)
    if arrow is None:
        if mismatch:
            fr.message = "箭头颜色与风速数字对不上（数字读错？）"
        elif band is not None:
            fr.message = "风速对应的彩色箭头未找到（可能数字读错或不在瞄准界面）"
        else:
            fr.message = "风盘内未找到风箭头"
        return fr
    k = dial.k
    theta = apparent_to_true(arrow.phi, k)
    fr.ok, fr.arrow, fr.phi, fr.k, fr.theta = True, arrow, arrow.phi, k, theta
    fr.arrow_band = arrow_band
    if fr.speed is None:
        fr.wind = wind_metrics(None, theta)
        if band is not None:
            # 数字读不出、但箭头颜色认出来了：风速**降级成区间**显示，方向照样给（分量算不了）。
            # 这就是"数字置信度低/读不出"时窗口该有的样子：`11-15 m/s` + 风向箭头。
            fr.speed_range = wind_range(band)
            fr.message = (f"数字未读出：风速按 {band_name(band)}档 "
                          f"{fr.speed_range[0]}-{fr.speed_range[1]} m/s 显示，方向照给")
        else:
            # 连颜色都没认出来，但箭头还是被"不看色相"的掩膜找到了 —— 有方向就先把方向给出去，
            # 风速显示成 `? m/s`（窗口本来就有这个分支，以前永远走不到）。
            fr.message = "风速与箭头颜色都没读出：只给方向（风速显示 ? m/s）"
    else:
        fr.wind = wind_metrics(fr.speed, theta)
    return fr
