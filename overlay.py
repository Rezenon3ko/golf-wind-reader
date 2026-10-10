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
from PIL import Image
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


HOLD_BAD = 3          # 连续这么多帧没识别出来…
HOLD_SEC = 2.5        # …且距上次成功超过这么多秒，才把提示刷到窗口。
                      # 两个条件都要满足：单帧失败一律按住（慢机器一帧 2~3 秒，只看时间会一失败就弹提示）


def ui_font(size: int, bold: bool = True):
    family = "PingFang SC" if sys.platform == "darwin" else "Microsoft YaHei"
    return (family, size, "bold" if bold else "normal")


HEAD_SIZE = 24                 # 正常显示的字号（两行统一）
ERROR_SIZE = HEAD_SIZE - 6     # 提示/报错的字号：小一号，Windows 上 24pt 太大、长句会被切
ERR_MIN_SIZE = 12              # 提示字号的**下限**：放不下就逐级缩小，缩到这么小为止（再小就难认了）
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
        self.have_good = False       # 是否出过至少一次好结果
        self.last_ok = 0.0           # 上一次识别成功的时间
        self.bad = 0                 # 连续没识别出来的帧数（要和 last_ok 一起判断才提示）

    def run(self) -> None:
        src = rec = None
        backoff = 3.0                             # 连不上时的重试间隔（失败就翻倍，最长 30 秒）
        self._log(f"启动：{sys.platform} / Python {sys.version.split()[0]} / "
                  f"{O.FPS:g} 次每秒 / 日志 {os.path.join(os.path.expanduser('~'), 'nssr-overlay.log')}")
        while not self.stop.is_set():
            if src is None:                       # （重）连 OBS + 建识别器
                if rec is None:                   # 识别器只建一次：它跟 OBS 无关，
                    try:                          # 先建它，出问题就不会留下一截连上的 OBS 会话
                        rec = O.Recognizer(SimpleNamespace(verbose=False))
                    except Exception as exc:      # 本地环境问题（多半是字形库/依赖），照实说
                        self._report(f"初始化失败：{exc}")
                        self.stop.wait(backoff)
                        backoff = min(backoff * 2.0, 30.0)
                        continue
                try:
                    src = O.make_source()
                    backoff = 3.0                 # 连上了，退避重置
                    self._log(f"已连接：{src.note}")
                    self.out.put({"kind": "status", "text": "已连接，等待画面…"})
                except Exception as exc:
                    full = f"连接 OBS 失败：{exc}"
                    print("\n" + full)                      # 终端里给完整信息（双击启动时无终端，无妨）
                    self._log(full)
                    self.out.put({"kind": "error", "text": self._conn_hint(full)})
                    self.stop.wait(backoff)       # 3→6→12→24→30 秒：别一秒一次地往 OBS 灌会话
                    backoff = min(backoff * 2.0, 30.0)
                    continue
            t0 = time.time()
            try:
                alive = self._one_frame(src, rec, t0)
            except Exception as exc:              # 兜底：任何意外都不能让取帧线程死掉（窗口会冻住）
                self._report(f"识别线程异常：{exc}")
                self.stop.wait(1.0)
                alive = True
            if alive is False:                    # 连接/源已判死 → 先关上再丢，下轮重连
                self._drop(src)
                src = None
                self.stop.wait(backoff)           # 重连前也等一下（同样退避），别秒连秒断
                backoff = min(backoff * 2.0, 30.0)
            rest = 1.0 / O.FPS - (time.time() - t0)
            if rest > 0:
                self.stop.wait(rest)

    @staticmethod
    def _conn_hint(full: str) -> str:
        """
        窗口只放一句**能照做的动作**，不放那一大段英文报错（报错全文照样打到终端/日志）。

        之前把"端口探针 + 分步清单"整段塞进窗口，300 多字符怎么排版都显示不全。
        认得出原因的，给一句照做的动作；认不出的，把**真实原因**照实显示（别再一律说
        "OBS 没开"，那正是"OBS 明明连着、程序却说连不上"这种误导的来源）。
        """
        low = full.lower()
        if ("password" in low) or ("auth" in low):
            return "连不上 OBS：请取消勾选「启用身份验证」再试"
        if ("10061" in full) or ("refused" in low) or ("完全连不上" in full):
            return "连不上 OBS：确认 OBS 已开、并启用了 WebSocket 服务器"
        first = full.splitlines()[0].replace("连接 OBS 失败：", "").strip()
        return Reader._brief(first, 46)

    def _report(self, msg: str) -> None:
        """把一条错误同时送到三处：终端、日志、窗口（窗口只放摘要）。"""
        print("\n" + msg)
        self._log(msg)
        self.out.put({"kind": "error", "text": self._brief(msg)})

    @staticmethod
    def _brief(text: str, limit: int = 60) -> str:
        """
        窗口只放得下一行摘要：取第一行、压掉换行和多余空格、太长就截断加「…」。

        原因：库里的报错为了终端排错写得很详细（端口探针 + 分步清单 ≈ 300 字符），
        直接塞进小窗口怎么排版都不够 —— 详细内容走终端/日志，窗口只留最关键那句。
        """
        one = " ".join(str(text).split())
        return one if len(one) <= limit else one[:limit] + "…"

    @staticmethod
    def _drop(src) -> None:
        """
        丢掉连接前先关掉它。

        之前直接 `src = None` 就不管了 —— 底层 WebSocket 要等 GC 才关，
        OBS 的「已连接的 WebSocket 会话」里就会堆一堆僵尸会话（每个挂几十秒）。
        """
        try:
            src.close()
        except Exception:
            pass

    def _one_frame(self, src, rec, t0: float) -> bool:
        """跑一帧：取帧 → 识别 → 决定要不要把结果丢给窗口。返回 False 表示该重连。"""
        rgb = src.read()
        t_read = time.time() - t0
        if rgb is None:
            msg = src.status_message()
            self._log(msg)                        # 取帧失败是最容易出问题的一环，一律记进日志
            self.out.put({"kind": "status", "text": self._brief(msg)})
            return not src.dead                   # 连续取帧失败 → 交给调用方重连
        try:
            t1 = time.time()
            st = rec.recognize(rgb)
            t_rec = time.time() - t1
        except Exception as exc:
            self._report(f"识别出错：{exc}")
            return False                          # 丢掉连接，下轮重连
        if self.verbose:                          # 诊断：每帧耗时写日志，定位"慢在哪一段"
            tq, td = getattr(src, "last_timing", (0.0, 0.0))
            d = st.get("dial")
            where = f"风盘({d.cx:.0f},{d.cy:.0f}) r={d.r:.0f}" if d else "没定位到风盘"
            ocr = st.get("ocr") or {}
            if ocr.get("box"):
                where += f" 数字框{ocr['box']}"
            self._log(f"取帧 {t_read*1000:.0f}ms（请求 {tq*1000:.0f} / 解码 {td*1000:.0f}）"
                      f" | 识别 {t_rec*1000:.0f}ms | 风速 {st.get('speed')} "
                      f"θ={st.get('theta') and round(st['theta'], 1)} | {where}")
        if self.verbose and st.get("speed") is None:
            self._dump(rgb, st)                   # 诊断：把读不出风速的帧存下来（最多 3 张）
        if st.get("ok"):
            self.have_good, self.last_ok, self.bad = True, time.time(), 0
        else:
            self.bad += 1
            # 单帧失败一律按住：要"连续 ≥HOLD_BAD 帧"且"距上次成功 ≥HOLD_SEC 秒"才把提示刷上窗口
            # （慢机器一帧 2~3 秒，只看时间会一失败就弹"未识别"）
            if self.have_good and (self.bad < HOLD_BAD
                                   or time.time() - self.last_ok < HOLD_SEC):
                return True
        st["kind"] = "result"
        self.out.put(st)
        return True

    _DUMPS = 0

    def _dump(self, rgb, st: dict) -> None:
        """诊断（-v）：把"读不出风速"的那一帧和数字框存成 PNG，最多 3 张，方便离线定位。"""
        if Reader._DUMPS >= 3:
            return
        Reader._DUMPS += 1
        try:
            home = os.path.expanduser("~")
            p = os.path.join(home, f"nssr-fail-{Reader._DUMPS}.png")
            Image.fromarray(rgb).save(p)
            box = (st.get("ocr") or {}).get("box")
            if box:
                x0, y0, x1, y1 = box
                Image.fromarray(rgb[y0:y1, x0:x1]).resize(
                    ((x1 - x0) * 4, (y1 - y0) * 4), Image.NEAREST).save(
                    os.path.join(home, f"nssr-fail-{Reader._DUMPS}-box.png"))
            self._log(f"已存诊断图 {p}（数字框 {box}）")
        except Exception:
            pass

    @staticmethod
    def _log(line: str) -> None:
        """
        写诊断日志到用户目录（**总是写**：双击启动没有终端，出问题只能靠这个日志查）。

        每帧的耗时明细仍然只在 --verbose 时写，平时只有启动/连接/报错这几行，文件很小。
        超过 2MB 就滚动一次，别无限长。
        """
        try:
            path = os.path.join(os.path.expanduser("~"), "nssr-overlay.log")
            if os.path.exists(path) and os.path.getsize(path) > 2 * 1024 * 1024:
                os.replace(path, path + ".1")      # 保留上一轮，日志总共最多 ~4MB
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
        self._alpha = args.alpha
        try:
            self.root.attributes("-alpha", args.alpha)  # 半透明
        except Exception:
            pass
        self.root.configure(bg=BG)
        # Tk 的字号单位是「点」：24pt 在屏幕上约等于 32px，
        # 所以截断提示文案必须按真实宽度量（font.measure），靠估算会算少 1/3。
        self._fonts: dict[int, tkfont.Font] = {}      # 字号 → Font（提示要按需缩小，缓存起来别每次新建）
        self.font = self._font_at(HEAD_SIZE)
        self.font_small = self._font_at(ERROR_SIZE)

        # 第一行：风向箭头 + 风速（按档位着色）+ 夹角，整行水平居中
        self.head = tk.Frame(self.root, bg=BG)
        self.head.pack(padx=PAD, pady=(6, 0))
        self.l_wind = tk.Label(self.head, text="·  --", font=self.font, bg=BG, fg=FG)
        self.l_wind.pack(side="left")
        self.l_ang = tk.Label(self.head, text="", font=self.font, bg=BG, fg=FG)
        self.l_ang.pack(side="left", padx=(GAP, 0))

        # 第二行：横向 + 纵向，排法和第一行一样 —— 两项并排 + 同间距，整组居中
        self.row = tk.Frame(self.root, bg=BG)
        self.row.pack(padx=PAD, pady=(0, 8))
        self.l_lat = tk.Label(self.row, text="", font=self.font, bg=BG, fg=DIM)
        self.l_lat.pack(side="left")
        self.l_lon = tk.Label(self.row, text="", font=self.font, bg=BG, fg=DIM)
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
            self.win_w = max(self.win_w, self.head.winfo_reqwidth() + 2 * PAD,
                             self.row.winfo_reqwidth() + 2 * PAD)
            self.win_h = max(self.win_h, self.root.winfo_reqheight())
        self.win_w, self.win_h = max(self.win_w, 200), max(self.win_h, 60)   # 兜底
        self.text_px = max(80, self.win_w - 2 * PAD - 6)   # 第一行提示可用的像素宽
        self.cur_w = self.win_w                          # 当前窗口宽度（提示时会临时加宽）
        self.l_wind.config(text="·  --")
        self.l_ang.config(text="")
        self.l_lat.config(text="")
        self.l_lon.config(text="")

        widgets = (self.root, self.head, self.l_wind, self.l_ang,
                   self.row, self.l_lat, self.l_lon)
        for w in widgets:                             # 拖动
            w.bind("<Button-1>", self._press)
            w.bind("<B1-Motion>", self._drag)
        self.menu = tk.Menu(self.root, tearoff=0)
        self.menu.add_command(label="退出", command=self.quit)
        self._menu_at: tuple[int, int, int, int] | None = None   # 菜单弹出时占的屏幕区域（退出时要擦掉）
        self._quitting = False
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
        self.cur_w = w
        self.root.geometry(f"{w}x{h}+{int(x)}+{int(y)}")

    # ---- 拖动 ----
    def _press(self, e) -> None:
        self._dx, self._dy = e.x_root - self.root.winfo_x(), e.y_root - self.root.winfo_y()

    def _drag(self, e) -> None:
        self.root.geometry(f"+{e.x_root - self._dx}+{e.y_root - self._dy}")

    def _popup(self, e) -> None:
        # 记下菜单占的屏幕区域：Windows 上菜单被销毁后那块常常不会被重画，留个白框。
        self._menu_at = (e.x_root, e.y_root,
                         self.menu.winfo_reqwidth(), self.menu.winfo_reqheight())
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
            self._resize(self.win_w)          # 回到固定宽度
            self.l_ang.pack_configure(padx=(GAP, 0))
            self.l_lon.pack_configure(padx=(GAP, 0))
            w = st["wind"]
            # 大行：风向箭头 + 总风速 + 夹角（夹角中性色）
            # 小行：横向 / 纵向分量
            # 数字部分统一用当前风速档位的颜色
            ang = w.get("theta_signed_deg", w["theta_deg"])
            col = speed_color(w["speed"])
            self.l_wind.config(text=f"{wind_arrow(w['theta_deg'])}  {w['speed']:g} m/s",
                               fg=col, font=self.font)
            self.l_ang.config(text=f"{ang:+.1f}°", font=self.font)
            # 第二行沿用第一行的排法：箭头 + 两个空格 + 数值
            # 颜色各自按**自己的数值**取档位（横向 7 → 黄、纵向 12 → 品红），不是跟着总风速
            self.l_lat.config(text=f"{LAT_ARROW.get(w['lateral_dir'], '·')}  {w['lateral_abs']:.2f}",
                              fg=speed_color(w["lateral_abs"]), font=self.font)
            self.l_lon.config(text=f"{LON_ARROW.get(w['longitudinal_dir'], '·')}  {w['longitudinal_abs']:.2f}",
                              fg=speed_color(w["longitudinal_abs"]), font=self.font)

    def _line1(self, text: str, color: str) -> None:
        """
        提示/报错：目标是**整句都看得见**，所以先缩字号、再考虑加宽，最后才截断。
        """
        font, line1, line2, w = self._fit_error(text)
        self.l_wind.config(text=line1, fg=color, font=font)
        self.l_ang.config(text="")
        self.l_ang.pack_configure(padx=0)                 # 夹角为空时不占位，才会真正居中
        self.l_lat.config(text=line2, fg=color, font=font)
        self.l_lon.config(text="")
        self.l_lat.pack_configure(padx=0)
        self.l_lon.pack_configure(padx=0)
        self._resize(w)

    def _fit_error(self, text: str) -> tuple[tkfont.Font, str, str, int]:
        """
        给提示挑一套「字号 + 折行 + 窗口宽」：
        ① 先在原窗口宽度里，用 18pt 折两行 —— 放得下就原样（宽度不变）；
        ② 放不下就逐级缩小字号（18→16→14→12）再折，能塞下就不动窗口；
        ③ 缩到下限还放不下，才把窗口加宽到刚好容下（最多占屏幕 90%），中间仍不做截断。

        之所以要这一串：Windows 上 18pt 比 macOS 大一圈，长报错在原宽度里必然顶出去；
        单靠拆分或单靠加宽都治不了，配合着来才能"整句可见"。
        """
        base = self.win_w
        for size in range(ERROR_SIZE, ERR_MIN_SIZE - 1, -2):
            width = base - 2 * PAD - 8
            if self._fits2(text, size, width):
                font = self._font_at(size)
                line1, line2 = self._two_lines(text, font, width)
                return font, line1, line2, base
        font = self._font_at(ERR_MIN_SIZE)
        cap = max(base, int(self.root.winfo_screenwidth() * 0.90))
        width = cap - 2 * PAD - 8
        line1, line2 = self._two_lines(text, font, width)
        if font.measure(line2) > width:                   # 连屏幕 90% 都放不下 → 只能截断
            line2 = self._fit(line2, width, font)
        need = max(font.measure(line1), font.measure(line2)) + 2 * PAD + 8
        return font, line1, line2, int(min(cap, max(base, need)))

    def _font_at(self, size: int) -> tkfont.Font:
        """取某个字号的字体（缓存）：提示放不下时要一级一级缩小，别每次都新建 Font 对象。"""
        f = self._fonts.get(size)
        if f is None:
            fam, _, weight = ui_font(size)
            f = tkfont.Font(family=fam, size=size, weight=weight)
            self._fonts[size] = f
        return f

    def _fits2(self, text: str, size: int, width: int) -> bool:
        """text 在 size 号字下，能否放进「两行 × width」里。"""
        font = self._font_at(size)
        if font.measure(text) <= width:
            return True
        line1, line2 = self._two_lines(text, font, width)
        return font.measure(line1) <= width and font.measure(line2) <= width

    def _two_lines(self, text: str, font, width: int) -> tuple[str, str]:
        """
        折两行：在候选断点里挑「最长那行最短」的一个。
        候选 = 句子中间的自然分隔符 + 按宽度硬折的位置（硬折是兜底，两者都会参与比谁更均衡）。
        只按分隔符断容易切出「一行很短、一行超宽」的畸形，所以这里改成按均衡度选。
        """
        if font.measure(text) <= width:
            return text, ""
        mid = len(text) // 2
        cands = [i for i, ch in enumerate(text)
                 if ch in "（(：:，,、；; " and abs(i - mid) <= 10]
        cands.append(len(self._take(text, font, width)) or 1)   # 按宽度硬折的位置
        best, cost = cands[-1], None
        for i in cands:
            if 0 < i < len(text):
                c = max(font.measure(text[:i]), font.measure(text[i:]))
                if cost is None or c < cost:
                    best, cost = i, c
        return text[:best], text[best:]

    def _take(self, text: str, font, width: int) -> str:
        """能塞进 width 的最长前缀（不加省略号，二分找）。"""
        if font.measure(text) <= width:
            return text
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if font.measure(text[:mid]) <= width:
                lo = mid
            else:
                hi = mid - 1
        return text[:lo]

    def _resize(self, w: int) -> None:
        """改窗口宽度：贴住原来的右边缘（不跑出屏幕），高度不变。"""
        sw = self.root.winfo_screenwidth()
        w = int(min(max(w, self.win_w), sw - 20))
        x = max(0, min(self.root.winfo_x() + self.cur_w - w, sw - w))
        self.cur_w = w
        self.root.geometry(f"{w}x{self.win_h}+{int(x)}+{self.root.winfo_y()}")

    def _fit(self, text: str, limit: int | None = None, font=None) -> str:
        """用真实字体宽度截断提示（Tk 字号单位是点，估算必偏小、会顶出窗口）。"""
        limit, font = (self.text_px if limit is None else limit), (font or self.font)
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
        就"白框"留在屏幕上（窗口是半透明/置顶的 layered window，更容易出这毛病）。
        所以先把菜单收起来、关掉置顶、把窗口藏起来，等事件真正处理完再销毁；
        菜单刚消失、窗口刚销毁这两个时刻各擦一次那块屏幕 —— 这才是清残影的关键一步。
        """
        if self._quitting:
            return
        self._quitting = True
        self.stop.set()
        rects = self._dead_rects()
        try:
            self.menu.unpost()            # 收起菜单
            self.root.update()            # 让"收起菜单"真正发生（update_idletasks 不处理它）
            self.menu.destroy()           # 连菜单窗口一起销毁 —— 不销毁它就可能在屏幕上留个"退出"小框
            self.root.attributes("-topmost", False)
            self.root.withdraw()
            self.root.update()
        except Exception:
            pass
        self._erase(rects)                # 先把菜单那块擦一遍（这时菜单刚没，最可能出现白框）
        # 再等 150ms（Windows 的菜单淡出动画要一会儿），然后真正销毁
        self.root.after(150, self._finish)

    def _finish(self) -> None:
        """真正退出前的最后一件事：销毁窗口 + 擦掉桌面残影。"""
        rects = self._dead_rects()
        try:
            self.root.destroy()
        except Exception:
            pass
        self._erase(rects)

    def _dead_rects(self) -> list[tuple[int, int, int, int]]:
        """
        退出时可能留残影的两块区域（屏幕坐标）：弹出的右键菜单 + 悬浮窗本身。
        菜单四周再多留 10px —— Tk 可能把菜单挪了点位置，或者带一圈边/阴影，宁可多擦一点。
        """
        raw = []
        if self._menu_at:
            raw.append(self._menu_at)
        try:
            raw.append((self.root.winfo_rootx(), self.root.winfo_rooty(), self.cur_w, self.win_h))
        except Exception:
            pass
        m = 10
        return [(x - m, y - m, w + 2 * m, h + 2 * m) for x, y, w, h in raw]

    @staticmethod
    def _erase(rects: list[tuple[int, int, int, int]]) -> None:
        """
        Windows 专属兜底：让这几块区域**下面的窗口**重画一遍，擦掉被销毁的菜单/窗口留下的白框。

        Tk 的 layered 窗口（-alpha）加上弹出菜单，销毁后那块区域常常"没人重画"，
        残影就这么贴在屏幕上。这里对着区域的几个点问系统"这块是谁的窗口"，
        让它把整窗失效重画 —— 不管残影是谁留下的，那块都会被重画掉。
        """
        if sys.platform != "win32":
            return
        try:
            import ctypes
            from ctypes import wintypes

            class POINT(ctypes.Structure):
                _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]

            u = ctypes.windll.user32
            u.WindowFromPoint.restype = wintypes.HWND
            u.WindowFromPoint.argtypes = [POINT]
            u.GetAncestor.restype = wintypes.HWND
            u.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
            u.InvalidateRect.argtypes = [wintypes.HWND, ctypes.c_void_p, wintypes.BOOL]
            GA_ROOT = 2
            targets = set()
            for x, y, w, h in rects:
                if w <= 0 or h <= 0:
                    continue
                for px, py in ((x + w // 2, y + h // 2), (x + 2, y + 2),
                               (x + w - 3, y + h - 3), (x + w // 2, y + 2),
                               (x + w // 2, y + h - 3)):
                    hw = u.WindowFromPoint(POINT(int(px), int(py)))
                    if hw:
                        targets.add(u.GetAncestor(hw, GA_ROOT) or hw)
            for hw in (targets or {0}):       # 什么都没找到就当桌面
                u.InvalidateRect(hw, None, True)     # 整窗失效 → 由它自己重画（含被白框盖住的那块）
        except Exception:
            pass

    def run(self) -> None:
        self.root.mainloop()


# --------------------------------------------------------------------------
# 命令行
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NSS 高尔夫风向 —— 桌面悬浮窗")
    p.add_argument("--alpha", type=float, default=0.5, help="窗口不透明度 0~1，默认 0.5")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="诊断：每帧「取帧/识别」耗时也写进 ~/nssr-overlay.log（报错本来就写）")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    Overlay(args).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
