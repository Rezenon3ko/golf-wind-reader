#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重建字形模板库 `glyph_bank.npz`（风速数字 0-9）。

这个库以前只能靠一次性脚本手工搭出来 —— 库里那 10 个模板到底是从哪几帧、按什么口径
切出来的，没人能复现。这个脚本把整条链固定下来：

    整帧截图 ──统一裁切(风盘)──> 文字行 ──列块切分──> 数字字形 ──归一化(32×48)──> 模板

用法：
    python3 build_glyph_bank.py                  # 用 字形素材/ 里的 0.png..9.png
    python3 build_glyph_bank.py --samples 字形素材/多帧样本
                                                 # 用 N/N-1.png 多帧中位数投票
    python3 build_glyph_bank.py --materials 别的目录
    python3 build_glyph_bank.py --no-write       # 只跑自检，不覆盖 glyph_bank.npz

素材有两种命名：
  * 单样本：`N.png`，N 就是那一帧显示的风速（0~9）；
  * 多样本：`多帧样本/N/N-1.png`、`N-2.png`……，同数字取多帧中位数投票。

脚本会**强制自检**：重建之后，
每一张素材都必须能被自己的新库读回它自己的数字，否则直接报错退出（不让坏库落盘）。

顺带产出统一裁切图 `字形素材/裁切/N.png` + `_meta.json`（记录每张图的风盘几何，
这样以后哪怕原始整帧截图删了，也能只用裁切图重建库）。

关于"每字几个模板"：库格式支持一个数字挂多个模板，但实测**加模板没用** ——
把每个字形做细/粗/±1 像素平移的形变当额外模板后，抗退化能力反而略降
（模糊 1.0px 那档从 10/10 掉到 9/10）。所以这里默认每个数字只放 1 个模板，
要加就得先拿 build_glyph_bank.py --self-test 之外的真实退化样本验证。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wind_reader as W          # noqa: E402

DIGITS = "0123456789"
META_NAME = "_meta.json"


def _dial_of_frame(rgb: np.ndarray) -> W.Dial:
    """整帧截图：走正常的定位（固定先验那一步就够，素材都是瞄准界面）。"""
    dials = W.find_dials(rgb, W.rgb_to_hsv(rgb))
    if not dials:
        raise SystemExit("✗ 这张素材里没定位到风盘（不是瞄准界面？）")
    return dials[0]


def _dial_from_meta(meta: dict, crop_shape) -> W.Dial:
    """裁切图：几何从 _meta.json 读回来（裁切图里没有整帧版式，定位函数用不了）。"""
    return W.Dial(kind=W.KIND_GAUGE, cx=meta["cx"] - meta["box"][0],
                  cy=meta["cy"] - meta["box"][1], r=meta["r"], k=meta.get("k", W.DEFAULT_K),
                  disc=tuple(meta["disc"]) if meta.get("disc") else None)


def digit_glyph(rgb: np.ndarray, dial: W.Dial):
    """
    从一帧里切出**风速数字**那一个字形的归一化模板（32×48）。

    切的是"文字行最左边那一块"：`Nm/s` 里数字永远在最左，所以最左列块就是它
    （0~9 都是个位数）。返回 (模板, 文字框, 全部列块)。
    """
    lines = W._find_text_lines(rgb, dial)
    if not lines:
        raise SystemExit("✗ 找不到 'Nm/s' 文字行")
    x0, y0, x1, y1 = lines[0]
    crop = Image.fromarray(rgb[y0:y1, x0:x1])
    m = W._clean_white_mask(W._bin_white(crop))
    band = W._line_band(m)
    runs = W._column_runs(m)
    if not runs:
        raise SystemExit("✗ 文字行里没切出列块")
    a, b = runs[0]
    g = W._glyph_norm(crop.crop((a, 0, b, crop.height)), band)
    if g is None:
        raise SystemExit("✗ 最左列块归一化失败（墨太少）")
    return g, (x0, y0, x1, y1), runs


def collect(materials: str, crops_dir: str | None):
    """读素材 → 统一裁切（并落盘）→ 提取数字模板。返回 {数字: [模板]} 和统计。"""
    bank: dict[str, list] = {}
    stats = []
    os.makedirs(crops_dir, exist_ok=True) if crops_dir else None
    meta_all = {}
    for ch in DIGITS:
        raw = os.path.join(materials, f"{ch}.png")
        cut = os.path.join(crops_dir, f"{ch}.png") if crops_dir else None
        if os.path.exists(raw):
            # 整帧：定位 → 统一裁切（落盘）→ 字形仍在**整帧坐标**上提取
            full = np.asarray(Image.open(raw).convert("RGB"))
            dial = _dial_of_frame(full)
            box = W._dial_box(dial, full.shape)
            geom = dict(cx=dial.cx, cy=dial.cy, r=dial.r, k=dial.k, box=list(box),
                        disc=list(dial.disc) if dial.disc else None)
            if crops_dir:
                Image.fromarray(full[box[1]:box[3], box[0]:box[2]]).save(cut)
                meta_all[ch] = geom
            src = full
        elif cut and os.path.exists(cut):
            # 只有裁切图：几何从 _meta.json 还原（裁切图的坐标系 = 裁切框内坐标）
            src = np.asarray(Image.open(cut).convert("RGB"))
            meta_path = os.path.join(crops_dir, META_NAME)
            if not os.path.exists(meta_path):
                raise SystemExit(f"✗ 只有裁切图却找不到 {META_NAME}，没法还原风盘几何")
            geom = json.load(open(meta_path, encoding="utf-8"))[ch]
            dial = _dial_from_meta(geom, src.shape[:2])
        else:
            raise SystemExit(f"✗ 缺素材：{raw}（或 {cut}）")
        g, line, runs = digit_glyph(src, dial)
        bank.setdefault(ch, []).append(g)
        stats.append((ch, line, runs, (dial.cx, dial.cy, dial.r)))
    if crops_dir and meta_all:
        with open(os.path.join(crops_dir, META_NAME), "w", encoding="utf-8") as fh:
            json.dump(meta_all, fh, ensure_ascii=False, indent=1)
    return bank, stats


def _sample_files(materials: str, ch: str) -> list[str]:
    """单样本 N.png 或多帧样本 N/N-*.png。"""
    direct = os.path.join(materials, f"{ch}.png")
    if os.path.exists(direct):
        return [direct]
    return sorted(glob.glob(os.path.join(materials, ch, "*.png")))


def collect_multi(materials: str, crops_dir: str | None):
    """从多帧整帧样本提取，每个数字做二值中位数投票。"""
    bank: dict[str, list] = {}
    stats = []
    meta_all = {}
    for ch in DIGITS:
        paths = _sample_files(materials, ch)
        if not paths:
            raise SystemExit(f"✗ 缺多帧素材：{os.path.join(materials, ch)}/*.png")
        glyphs = []
        for i, path in enumerate(paths, 1):
            full = np.asarray(Image.open(path).convert("RGB"))
            dial = _dial_of_frame(full)
            box = W._dial_box(dial, full.shape)
            if crops_dir:
                out_dir = os.path.join(crops_dir, ch)
                os.makedirs(out_dir, exist_ok=True)
                Image.fromarray(full[box[1]:box[3], box[0]:box[2]]).save(
                    os.path.join(out_dir, f"{ch}-{i}.png"))
                meta_all[f"{ch}-{i}"] = dict(
                    cx=dial.cx, cy=dial.cy, r=dial.r, k=dial.k,
                    box=list(box),
                    disc=list(dial.disc) if dial.disc else None)
            glyph, line, runs = digit_glyph(full, dial)
            glyphs.append(glyph)
            stats.append((ch, os.path.basename(path), line, runs,
                          (dial.cx, dial.cy, dial.r)))
        merged = np.median(np.stack(glyphs), axis=0)
        bank[ch] = [(merged >= 0.5).astype(np.float32)]
    if crops_dir and meta_all:
        os.makedirs(crops_dir, exist_ok=True)
        with open(os.path.join(crops_dir, "_meta_multi.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(meta_all, fh, ensure_ascii=False, indent=1)
    return bank, stats


def self_test(bank: dict, materials: str, crops_dir: str | None) -> int:
    """自检：拿新库去认每张素材，必须 10/10 读回自己的数字。"""
    st = W.OcrState()
    st.bank = bank
    bad = []
    for ch in DIGITS:
        raw = os.path.join(materials, f"{ch}.png")
        cut = os.path.join(crops_dir, f"{ch}.png") if crops_dir else None
        if os.path.exists(raw):
            full = np.asarray(Image.open(raw).convert("RGB"))
            dial = _dial_of_frame(full)
        else:
            rgb = np.asarray(Image.open(cut).convert("RGB"))
            geom = json.load(open(os.path.join(crops_dir, META_NAME), encoding="utf-8"))[ch]
            dial = _dial_from_meta(geom, rgb.shape[:2])
            full = rgb
        lines = W._find_text_lines(full, dial)
        got = None
        for box in lines:                     # 跟运行时一样：候选文字框依次试
            x0, y0, x1, y1 = box
            tm = W.match_glyphs(Image.fromarray(full[y0:y1, x0:x1]), state=st)
            if tm is not None:
                got = tm[0]
                break
        if got != ch:
            bad.append((ch, got))
    return bad


def self_test_multi(bank: dict, materials: str, crops_dir: str | None) -> int:
    """多帧自检：每个数字的每一张样本都要被共识库读回自己的数字。"""
    st = W.OcrState()
    st.bank = bank
    bad = []
    for ch in DIGITS:
        for path in _sample_files(materials, ch):
            full = np.asarray(Image.open(path).convert("RGB"))
            dial = _dial_of_frame(full)
            got = None
            for box in W._find_text_lines(full, dial):
                x0, y0, x1, y1 = box
                tm = W.match_glyphs(
                    Image.fromarray(full[y0:y1, x0:x1]), state=st)
                if tm is not None:
                    got = tm[0]
                    break
            if got != ch:
                bad.append((os.path.basename(path), ch, got))
    return bad


def confusion(bank: dict) -> None:
    """打印两两相似度最高（最容易认错）的几对，改库/改切分时看它。"""
    pairs = []
    for i, a in enumerate(DIGITS):
        for b in DIGITS[i + 1:]:
            pairs.append((W._glyph_sim(bank[a][0], bank[b][0]), a, b))
    pairs.sort(reverse=True)
    print("  最像的 6 对（越像越危险）：" +
          "  ".join(f"{a}-{b}:{v:.3f}" for v, a, b in pairs[:6]))
    print("  参考：6-8 那对全靠'洞数拓扑'分开（8 两个洞 / 6 一个洞）；1-4 只有 0.29，"
          "运行时认错基本是**切分切歪**，不是模板不像。")


def main(argv=None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="重建风速数字字形库")
    ap.add_argument("--materials", default=os.path.join(here, "字形素材"))
    ap.add_argument("--samples", default=None,
                    help="多帧样本根目录（例如 字形素材/多帧样本）")
    ap.add_argument("--crops", default=None,
                    help="统一裁切图的输出目录（默认按单样本/多帧模式选择）")
    ap.add_argument("--out", default=os.path.join(here, "glyph_bank.npz"))
    ap.add_argument("--no-write", action="store_true", help="只跑自检，不覆盖已有的库")
    ap.add_argument("--no-crops", action="store_true", help="不落盘裁切图")
    args = ap.parse_args(argv)

    default_crops = os.path.join(
        here, "字形素材", "多帧裁切" if args.samples else "裁切")
    crops = None if args.no_crops else (args.crops or default_crops)
    if args.samples:
        bank, stats = collect_multi(args.samples, crops)
        print(f"[1/3] 多帧提取完成：{len(bank)} 个数字，共 {len(stats)} 张样本")
        for ch, name, line, runs, geo in stats:
            print(f"      {ch}: {name} 风盘({geo[0]:.0f},{geo[1]:.0f}) "
                  f"r={geo[2]:.0f} 文字框{line} 列块{len(runs)}个")
        bad = self_test_multi(bank, args.samples, crops)
    else:
        bank, stats = collect(args.materials, crops)
        print(f"[1/3] 提取完成：{len(bank)} 个数字，每字 {len(bank['0'])} 个模板")
        for ch, line, runs, geo in stats:
            print(f"      {ch}: 风盘({geo[0]:.0f},{geo[1]:.0f}) r={geo[2]:.0f}  "
                  f"文字框{line}  列块{len(runs)}个 首块={runs[0]} 宽={runs[0][1]-runs[0][0]}")
        bad = self_test(bank, args.materials, crops)
    if bad:
        print("✗ 自检没通过：", bad)
        print("  库没写盘 —— 先把这几张素材的切分/字形看明白再重建。")
        return 1
    print("[2/3] 自检通过：每张素材都读回自己的数字")
    confusion(bank)

    if args.no_write:
        print("[3/3] --no-write：没有覆盖 glyph_bank.npz")
        return 0
    # 必须用 savez_compressed：二值模板压缩后很小，库要跟着 exe 走
    np.savez_compressed(args.out, **{k: np.asarray(v, dtype=np.float32) for k, v in bank.items()})
    n = W.load_glyph_bank(args.out)
    print(f"[3/3] 已写入 {args.out}（载入校验：{n} 个数字 {W.glyph_bank_digits()}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
