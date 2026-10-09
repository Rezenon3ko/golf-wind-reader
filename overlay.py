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
import tkinter.font as tkfont
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
PAD = 0                        # 左右留白；0 = 窗口宽度就是第一行的宽度（第二行字小，顶不出窗口）
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


HOLD_BAD = 3           # 连续这么多次"没识别出来"才把提示刷到窗口（中间保持上一帧，避免闪烁）


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
    def __init__(self, out: queue.Queue, stop: threading.Event, verbose: bool = False):
        super().__init__(daemon=True)
        self.out, self.stop = out, stop
        self.verbose = verbose
        self.bad = 0                 # 连续"没识别出来"的次数
        self.have_good = False       # 是否出过至少一次好结果

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
            t_read = time.time() - t0
            if rgb is None:
                self.out.put({"kind": "status", "text": src.status_message()})
                if src.dead:                      # 连续失败：丢掉连接，下轮重连（顺便重新挑源）
                    src = None
                self.stop.wait(0.3)
                continue
            try:
                t1 = time.time()
                st = rec.recognize(rgb)
                t_rec = time.time() - t1
            except Exception as exc:
                self.out.put({"kind": "error", "text": f"识别出错：{exc}"})
                src = None                        # 丢掉连接，下轮重连
                self.stop.wait(1.0)
                continue
            if self.verbose:                      # 诊断：每帧耗时写日志，定位"慢在哪一段"
                tq, td = getattr(src, "last_timing", (0.0, 0.0))
                self._log(f"取帧 {t_read*1000:.0f}ms（请求 {tq*1000:.0f} / 解码 {td*1000:.0f}）"
                          f" | 识别 {t_rec*1000:.0f}ms | 风速 {st.get('speed')} "
                          f"θ={st.get('theta') and round(st['theta'], 1)}")
            # 偶发单帧没识别出来（转身/遮挡/动画中）先按住上一帧，别让窗口闪一下"未识别"；
            # 连续 HOLD_BAD 次都失败才把原因显示出来。还没出过好结果时（刚启动、没在瞄准界面）立刻显示。
            if st.get("ok"):
                self.bad, self.have_good = 0, True
            else:
                self.bad += 1
                if self.have_good and self.bad < HOLD_BAD:
                    rest = 1.0 / O.FPS - (time.time() - t0)
                    if rest > 0:
                        self.stop.wait(rest)
                    continue
            st["kind"] = "result"
            self.out.put(st)
            rest = 1.0 / O.FPS - (time.time() - t0)     # 固定 1 次/秒
            if rest > 0:
                self.stop.wait(rest)

    @staticmethod
    def _log(line: str) -> None:
        """诊断日志写到用户目录，方便把内容发回来（--verbose 才写）。"""
        try:
            path = os.path.join(os.path.expanduser("~"), "nssr-overlay.log")
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(f"{time.strftime('%H:%M:%S')}  {line}\n")
        except Exception:
            pass


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
        self.root.configure(bg=BG)
        # Tk 的字号单位是「点」：24pt 在屏幕上约等于 32px，
        # 所以截断提示文案必须按真实宽度量（font.measure），靠估算会算少 1/3。
        fam, size, weight = ui_font(HEAD_SIZE)
        self.font = tkfont.Font(family=fam, size=size, weight=weight)

        # 第一行：风向箭头 + 风速（按档位着色）+ 夹角，整行水平居中
        head = tk.Frame(self.root, bg=BG)
        head.pack(padx=PAD, pady=(6, 0))
        self.l_wind = tk.Label(head, text="·  --", font=self.font, bg=BG, fg=FG)
        self.l_wind.pack(side="left")
        self.l_ang = tk.Label(head, text="", font=self.font, bg=BG, fg=FG)
        self.l_ang.pack(side="left", padx=(GAP, 0))

        # 第二行：横向 + 纵向，排法和第一行一样 —— 两项并排 + 同间距，整组居中
        row = tk.Frame(self.root, bg=BG)
        row.pack(padx=PAD, pady=(0, 8))
        self.l_lat = tk.Label(row, text="", font=self.font, bg=BG, fg=DIM)
        self.l_lat.pack(side="left")
        self.l_lon = tk.Label(row, text="", font=self.font, bg=BG, fg=DIM)
        self.l_lon.pack(side="left", padx=(GAP, 0))

        # 宽度以第一行为准（第一行居中，左右各留 PAD）；高度取两行的自然高度。
        # 尺寸定死，文字变长也不会跑出屏幕。
        self.win_w = self.win_h = 0
        labs = (self.l_wind, self.l_ang, self.l_lat, self.l_lon)
        for sample in SAMPLES:
            for lab, txt in zip(labs, sample):
                lab.config(text=txt)
            self.root.update_idletasks()
            # 宽度以第一行为准；第二行万一更宽也只补到不叠字为止
            self.win_w = max(self.win_w, head.winfo_reqwidth() + 2 * PAD,
                             row.winfo_reqwidth() + 2 * PAD)
            self.win_h = max(self.win_h, self.root.winfo_reqheight())
        self.win_w, self.win_h = max(self.win_w, 200), max(self.win_h, 60)   # 兜底
        self.text_px = max(80, self.win_w - 2 * PAD - 6)   # 第一行提示可用的像素宽
        self.l_wind.config(text="·  --")
        self.l_ang.config(text="")
        self.l_lat.config(text="")
        self.l_lon.config(text="")

        widgets = (self.root, head, self.l_wind, self.l_ang,
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
        Reader(self.q, self.stop, verbose=getattr(args, "verbose", False)).start()
        self.root.after(80, self._poll)

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
        """用真实字体宽度截断提示（Tk 字号单位是点，估算必偏小、会顶出窗口）。"""
        limit, font = self.text_px, self.font
        if font.measure(text) <= limit:
            return text
        lo, hi = 0, len(text)                      # 二分找最长能放下的前缀
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if font.measure(text[:mid] + "…") <= limit:
                lo = mid
            else:
                hi = mid - 1
        return text[:lo] + "…"

    def quit(self) -> None:
        """
        退出。

        Windows 上如果直接在菜单回调里 destroy 窗口，右键菜单那个框有时不会被系统擦掉，
        就"残影"留在屏幕上（窗口是半透明/置顶的 layered window，更容易出这毛病）。
        所以：先收起菜单 → 关掉置顶 → 把窗口藏起来（让系统重画那块区域）→
        等事件队列空一轮之后再真正销毁。
        """
        self.stop.set()
        try:
            self.menu.unpost()            # 收起菜单
            self.menu.destroy()           # 连菜单窗口一起销毁 —— 不销毁它就可能在屏幕上留个"退出"小框
            self.root.attributes("-topmost", False)
            self.root.withdraw()
            self.root.update_idletasks()
        except Exception:
            pass
        # 再等 120ms（Windows 的菜单淡出动画要一会儿），然后真正销毁
        self.root.after(120, self.root.destroy)

    def run(self) -> None:
        self.root.mainloop()


# --------------------------------------------------------------------------
# 命令行
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NSS 高尔夫风向 —— 桌面悬浮窗")
    p.add_argument("--alpha", type=float, default=0.5, help="窗口不透明度 0~1，默认 0.5")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="诊断：把每帧「取帧/识别」耗时写进 ~/nssr-overlay.log")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    Overlay(args).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
