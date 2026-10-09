#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NSS 高尔夫风向 —— 桌面悬浮窗（Tkinter）

用一个置顶的小窗口显示识别结果，不用盯着命令行。

    python3 overlay.py                    # 固定每秒识别 1 次
    python3 overlay.py --obs-source "视频采集设备"
    python3 overlay.py --alpha 0.5        # 更透明（默认 0.5）

操作：**拖动窗口**移动；**右键 → 退出**。

依赖和命令行版一样（numpy / Pillow / obsws-python），另加 Tkinter（Python 自带）。
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
from types import SimpleNamespace

# 打包成无控制台程序时 stdout/stderr 是 None，print 会报错 —— 兜一下
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import obs_live_wind_reader as O          # 复用取帧（ObsWebSocketSource）与识别（Recognizer）


# --------------------------------------------------------------------------
# 外观
# --------------------------------------------------------------------------
BG = "#15171c"
FG = "#f2f4f7"
DIM = "#8b93a1"
ERR = "#ff6b6b"
EDGE = "#2e3440"          # 圆角描边（比底色略亮一点，边界更清楚）
RADIUS = 12               # 圆角半径
EDGE_PAD = 14             # 内容到窗口边缘的留白（给圆角让位，四周一致）
CHROMA = "#0b0c10"        # Windows/Linux 的透明色键（几乎不会出现在内容里的颜色）
SYSTEM_TRANSPARENT = "systemTransparent"     # macOS 的透明背景色名
# 风向箭头：按真实夹角 θ（0=正前，顺时针）取 8 向
ARROWS8 = "↑↗→↘↓↙←↖"
LAT_ARROW = {"向右": "→", "向左": "←"}          # 横向分量（左右）
LON_ARROW = {"顺风": "↑", "逆风": "↓"}          # 纵向分量（顺逆）
# 风速档位配色：和游戏里风盘箭头的三种颜色一致（1-5 青 / 6-10 黄 / 11-15 品红）
# 三行同色：方向已经由箭头（↑↗→… / →← / ↑↓）表达，颜色统一用来标风速档位
SPEED_COLORS = ((5, "#2fd6e8"), (10, "#ffd23f"), (15, "#ff6ec7"))


def speed_color(speed: float) -> str:
    """风速 → 档位颜色（超出表格/无风时用中性色）。"""
    for top, col in SPEED_COLORS:
        if speed <= top:
            return col if speed > 0 else DIM
    return FG


def wind_arrow(theta: float) -> str:
    """真实夹角 θ → 8 向箭头（箭头指向 = 风吹去的方向）"""
    return ARROWS8[int((theta % 360.0) / 45.0 + 0.5) % 8]


def ui_font(size: int, bold: bool = True):
    family = "PingFang SC" if sys.platform == "darwin" else "Microsoft YaHei"
    return (family, size, "bold" if bold else "normal")


HEAD_SIZE = 24                 # 两行同一个字号：第一行 箭头+风速+夹角，第二行 横向/纵向分量
GAP = 10                       # 同一行里两项之间的间距（两行共用一个值）
# 量窗口尺寸用的"最长一行"样本（定死尺寸，文字变长也不会跑出屏幕）
SAMPLES = (("↘  15 m/s", "+179.9°", "→ 13.55", "↑ 13.55"),   # 正常结果的最宽一拍
           ("·  0 m/s（无风）", "", "", ""))                   # 无风提示


# --------------------------------------------------------------------------
# 后台线程：取帧 → 识别 → 丢进队列
# --------------------------------------------------------------------------
class Reader(threading.Thread):
    def __init__(self, out: queue.Queue, stop: threading.Event):
        super().__init__(daemon=True)
        self.out, self.stop = out, stop

    def run(self) -> None:
        src = rec = None
        while not self.stop.is_set():
            if src is None:                       # （重）连 OBS + 建识别器
                try:
                    src = O.make_source()
                    rec = O.Recognizer(SimpleNamespace(verbose=False))
                    self.out.put({"kind": "status", "text": "已连接，等待画面…"})
                except Exception as exc:
                    self.out.put({"kind": "error", "text": f"连接 OBS 失败：{exc}"})
                    self.stop.wait(3.0)           # 3 秒后重试（先开悬浮窗、后开 OBS 也行）
                    continue
            t0 = time.time()
            rgb = src.read()
            if rgb is None:
                self.out.put({"kind": "status", "text": src.status_message()})
                if src.dead:                      # 连续失败：丢掉连接，下轮重连（顺便重新挑源）
                    src = None
                self.stop.wait(0.3)
                continue
            try:
                st = rec.recognize(rgb)
            except Exception as exc:
                self.out.put({"kind": "error", "text": f"识别出错：{exc}"})
                src = None                        # 丢掉连接，下轮重连
                self.stop.wait(1.0)
                continue
            st["kind"] = "result"
            self.out.put(st)
            rest = 1.0 / O.FPS - (time.time() - t0)     # 固定 1 次/秒
            if rest > 0:
                self.stop.wait(rest)


# --------------------------------------------------------------------------
# 悬浮窗
# --------------------------------------------------------------------------
class Overlay:
    def __init__(self, args):
        self.args = args
        self.q: queue.Queue = queue.Queue()
        self.stop = threading.Event()
        self.ok = False

        self.root = tk.Tk()
        self.root.title("风向")
        self.root.overrideredirect(True)              # 无边框
        self.root.attributes("-topmost", True)        # 置顶
        try:
            self.root.attributes("-alpha", args.alpha)  # 半透明
        except Exception:
            pass
        self.transparent = self._enable_transparency()

        # 圆角背景画在 Canvas 上（圆角外用真透明，露出桌面），内容用 create_window 居中放进去
        self.canvas = tk.Canvas(self.root, highlightthickness=0, bd=0,
                                bg=SYSTEM_TRANSPARENT if self.transparent else BG)
        self.canvas.pack(fill="both", expand=True)
        self._bg = None

        body = tk.Frame(self.canvas, bg=BG)      # 只是内容容器，尺寸跟着两行文字走
        # 第一行：风向箭头 + 风速（按档位着色）+ 夹角，整行水平居中
        head = tk.Frame(body, bg=BG)
        head.pack()
        self.l_wind = tk.Label(head, text="·  --", font=ui_font(HEAD_SIZE), bg=BG, fg=FG)
        self.l_wind.pack(side="left")
        self.l_ang = tk.Label(head, text="", font=ui_font(HEAD_SIZE), bg=BG, fg=FG)
        self.l_ang.pack(side="left", padx=(GAP, 0))

        # 第二行：横向 + 纵向，排法和第一行一样 —— 两项并排 + 同间距，整组居中
        row = tk.Frame(body, bg=BG)
        row.pack(pady=(2, 0))
        self.l_lat = tk.Label(row, text="", font=ui_font(HEAD_SIZE), bg=BG, fg=DIM)
        self.l_lat.pack(side="left")
        self.l_lon = tk.Label(row, text="", font=ui_font(HEAD_SIZE), bg=BG, fg=DIM)
        self.l_lon.pack(side="left", padx=(GAP, 0))

        # 尺寸：内容自然大小 + 四周留白（定死，文字变长也不会跑出屏幕）
        self.win_w = self.win_h = 0
        labs = (self.l_wind, self.l_ang, self.l_lat, self.l_lon)
        for sample in SAMPLES:
            for lab, txt in zip(labs, sample):
                lab.config(text=txt)
            self.root.update_idletasks()
            self.win_w = max(self.win_w, body.winfo_reqwidth())
            self.win_h = max(self.win_h, body.winfo_reqheight())
        self.win_w = max(self.win_w + 2 * EDGE_PAD, 200)     # 兜底
        self.win_h = max(self.win_h + 2 * EDGE_PAD, 60)
        self.text_px = max(80, self.win_w - 2 * EDGE_PAD - 6)   # 第一行提示可用的像素宽

        # 圆角底图 + 内容居中
        if self.transparent:      # 透明可用才画圆角；不支持（少数 Tk）就退回原来的直角实心
            self._bg = self._rounded_bg(self.win_w, self.win_h)
            self.canvas.create_image(0, 0, anchor="nw", image=self._bg)
        self.canvas.create_window(self.win_w / 2, self.win_h / 2,
                                  window=body, anchor="center")
        self.l_wind.config(text="·  --")
        self.l_ang.config(text="")
        self.l_lat.config(text="")
        self.l_lon.config(text="")

        widgets = (self.root, self.canvas, body, head, self.l_wind, self.l_ang,
                   row, self.l_lat, self.l_lon)
        for w in widgets:                             # 拖动
            w.bind("<Button-1>", self._press)
            w.bind("<B1-Motion>", self._drag)
        self.menu = tk.Menu(self.root, tearoff=0)
        self.menu.add_command(label="退出", command=self.quit)
        for w in widgets:                             # 右键（macOS 用 Button-2）
            w.bind("<Button-3>", self._popup)
            w.bind("<Button-2>", self._popup)

        self._place()
        Reader(self.q, self.stop).start()
        self.root.after(80, self._poll)

    # ---- 圆角外观 ----
    def _enable_transparency(self) -> bool:
        """让窗口背景真透明（圆角外露出桌面）。macOS 用 systemTransparent，Windows 用色键；
        都不支持（少数 Linux/旧 Tk）就退回直角实心，功能照常。"""
        try:
            if sys.platform == "darwin":
                self.root.configure(bg=SYSTEM_TRANSPARENT)
                self.root.attributes("-transparent", True)
                return True
            if sys.platform.startswith("win"):
                self.root.configure(bg=CHROMA)
                self.root.attributes("-transparentcolor", CHROMA)
                return True
        except Exception:
            pass
        self.root.configure(bg=BG)
        return False

    def _rounded_bg(self, w: int, h: int):
        """画一张圆角矩形背景（4× 超采样抗锯齿）并转成 Tk 图片对象。"""
        ss = 4
        img = Image.new("RGBA", (w * ss, h * ss), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle([0, 0, w * ss - 1, h * ss - 1], radius=RADIUS * ss,
                               fill=self._rgb(BG), outline=self._rgb(EDGE), width=ss)
        return ImageTk.PhotoImage(img.resize((w, h), Image.LANCZOS))

    @staticmethod
    def _rgb(hex_color: str) -> tuple[int, int, int, int]:
        h = hex_color.lstrip("#")
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 255)

    # ---- 窗口位置 ----
    def _place(self) -> None:
        """定死两行尺寸 + 默认贴屏幕右上角；无论屏幕多小都保证整窗可见（拖窗口可挪）。"""
        w, h = self.win_w, self.win_h
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        x = max(0, sw - w - 40)      # 贴右上角，留 40px 边距（游戏风盘也在右上，习惯一致）
        y = max(0, min(120, sh - h))
        self.root.geometry(f"{w}x{h}+{int(x)}+{int(y)}")

    # ---- 拖动 ----
    def _press(self, e) -> None:
        self._dx, self._dy = e.x_root - self.root.winfo_x(), e.y_root - self.root.winfo_y()

    def _drag(self, e) -> None:
        self.root.geometry(f"+{e.x_root - self._dx}+{e.y_root - self._dy}")

    def _popup(self, e) -> None:
        try:
            self.menu.tk_popup(e.x_root, e.y_root)
        finally:
            self.menu.grab_release()

    # ---- 主线程轮询队列 ----
    def _poll(self) -> None:
        latest = None
        try:
            while True:
                st = self.q.get_nowait()
                latest = st
        except queue.Empty:
            pass
        if latest is not None:
            self._render(latest)
        self.root.after(80, self._poll)

    def _render(self, st: dict) -> None:
        kind = st.get("kind")
        if kind in ("error", "status"):             # 异常/状态直接顶到第一行
            self.ok = False
            self._line1(st["text"], ERR if kind == "error" else DIM)
            return

        if st.get("no_wind"):
            self.ok = True
            self._line1("·  0 m/s（无风）", DIM)
        elif not st.get("ok"):
            self.ok = False
            self._line1(st.get("message", "识别中…"), DIM)
            return
        else:
            self.ok = True
            w = st["wind"]
            # 大行：风向箭头 + 总风速 + 夹角（夹角中性色）
            # 小行：横向 / 纵向分量
            # 数字部分统一用当前风速档位的颜色
            ang = w.get("theta_signed_deg", w["theta_deg"])
            col = speed_color(w["speed"])
            self.l_wind.config(text=f"{wind_arrow(w['theta_deg'])}  {w['speed']:g} m/s", fg=col)
            self.l_ang.config(text=f"{ang:+.1f}°")
            # 第二行沿用第一行的排法：箭头 + 两个空格 + 数值
            self.l_lat.config(text=f"{LAT_ARROW.get(w['lateral_dir'], '·')}  {w['lateral_abs']:.2f}",
                              fg=col)
            self.l_lon.config(text=f"{LON_ARROW.get(w['longitudinal_dir'], '·')}  {w['longitudinal_abs']:.2f}",
                              fg=col)

    def _line1(self, text: str, color: str) -> None:
        """把整句提示写进第一行（过长按窗口宽度截断），其余行清空。"""
        self.l_wind.config(text=self._fit(text), fg=color)
        self.l_ang.config(text="")
        self.l_lat.config(text="")
        self.l_lon.config(text="")

    def _fit(self, text: str) -> str:
        """按像素宽度估算法截断提示（中文 1.05em、西文 0.6em），避免顶出窗口。"""
        used, out = 0.0, []
        for ch in text:
            used += HEAD_SIZE * (1.05 if ord(ch) > 0x2E80 else 0.6)
            if used > self.text_px:
                return "".join(out) + "…"
            out.append(ch)
        return text

    def quit(self) -> None:
        self.stop.set()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


# --------------------------------------------------------------------------
# 命令行
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NSS 高尔夫风向 —— 桌面悬浮窗")
    p.add_argument("--alpha", type=float, default=0.5, help="窗口不透明度 0~1，默认 0.5")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    Overlay(args).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
