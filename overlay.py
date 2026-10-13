#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NSS 高尔夫风向 —— 桌面悬浮窗（Tkinter）

用一个置顶的小窗口显示识别结果，不用盯着命令行。

    python3 overlay.py                    # 固定每秒识别 1 次
    python3 overlay.py --alpha 0.5        # 更透明（默认 0.5）

操作：**拖动窗口**移动；**右键 → 退出**。
第三行是「横风 × 系数 = 结果」：点输入框填系数，结果随识别结果实时变（横风未知时显示 —）。
日志和诊断图放在程序旁边的「NSSR日志」文件夹里（exe / .app 同级）：
nssr-overlay.log、nssr-fail-*.jpg。

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


def _safe_stdio() -> None:
    """
    让 print 永远不因**编码**抛异常。

    Windows 上 stdout 用的是本地代码页（中文系统 GBK、英文系统 cp1252），
    编不出的字符（中文、⚠、✗…）会直接抛 UnicodeEncodeError。打包成无控制台程序后
    stdout 被接到 devnull，同样按本地代码页编码 —— 一样会抛。

    这个坑很阴：抛的地方常常正好是"刚连上 OBS、正在打印源名"那一刻，
    于是连接被丢在半路、OBS 里留下僵尸会话，程序却报"连不上"。
    所以这里先把 stdout/stderr 变成"编不出就替换成 ?"的宽松模式。
    """
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None:                            # 无控制台（打包后的窗口程序）
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8", errors="replace"))
            continue
        try:
            stream.reconfigure(errors="replace")      # 保留原编码，编不出的字符换成 ?，不再抛
        except Exception:
            pass


_safe_stdio()


def _program_dir() -> str:
    """
    程序所在文件夹 —— 日志和诊断图就放这儿，玩家最容易找（就在 exe / .app 旁边）。

    打包后 sys.executable 是 exe 本体；macOS 的 .app 里它是
    xxx.app/Contents/MacOS/xxx，得往上找到 .app 再取它**所在**的文件夹
    （往 .app 里面写会破坏签名，也可能没权限）。源码运行时就是本文件所在目录。
    """
    if getattr(sys, "frozen", False):
        exe = os.path.abspath(sys.executable)
        head = os.path.dirname(exe)
        while head and head != os.path.dirname(head):
            if head.endswith(".app"):
                return os.path.dirname(head)
            head = os.path.dirname(head)
        return os.path.dirname(exe)
    return os.path.dirname(os.path.abspath(__file__))


_ART_DIR: str | None = None
LOG_DIR_NAME = "NSSR日志"      # 日志/诊断图放在程序旁边的这个文件夹（不再直接摊在 exe 那层）


def _artifact_dir() -> str:
    """
    存日志/诊断图的目录：**程序旁边新建的 NSSR日志 文件夹**（exe / .app 同级，
    这样 exe 那层不会被日志和截图弄乱）。

    建不出来或写不进去（Program Files、只读盘…）就逐级回退：
    程序所在文件夹 → 用户主目录 —— 宁可变位置，也不能因为写不进去而丢掉日志。
    """
    global _ART_DIR
    if _ART_DIR is None:
        base = _program_dir()
        for d in (os.path.join(base, LOG_DIR_NAME), base, os.path.expanduser("~")):
            try:
                os.makedirs(d, exist_ok=True)
                probe = os.path.join(d, ".nssr-write-test")
                with open(probe, "w") as fh:
                    fh.write("")
                os.remove(probe)
                _ART_DIR = d
                break
            except Exception:
                continue
        if _ART_DIR is None:
            _ART_DIR = os.path.expanduser("~")
    return _ART_DIR


def _migrate_old_artifacts() -> None:
    """旧版本把日志/截图直接摊在程序目录里 —— 搬家到 NSSR日志 文件夹（只挪不删）。"""
    d, base = _artifact_dir(), _program_dir()
    if d == base:
        return
    try:
        names = os.listdir(base)
    except Exception:
        return
    for n in names:
        if not (n.startswith("nssr-overlay.log") or n.startswith("nssr-fail-")):
            continue
        src, dst = os.path.join(base, n), os.path.join(d, n)
        try:
            if not os.path.exists(dst):
                os.replace(src, dst)
        except Exception:
            pass


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import obs_live_wind_reader as O          # 复用取帧（ObsWebSocketSource）与识别（Recognizer）
from version import __version__ as APP_VERSION   # 版本号唯一来源（打包脚本读的是同一个文件）


# --------------------------------------------------------------------------
# 外观
# --------------------------------------------------------------------------
BG = "#15171c"
FG = "#f2f4f7"
DIM = "#8b93a1"
ERR = "#ff6b6b"
ENT_BG = "#0e1014"     # 系数输入框的底：比窗口底色再深一点，看着像个"槽"
WARN = "#ffd166"       # 提示色（"风速拿不准，点一个"）
ACCENT = "#7fd1ff"     # 按钮按下去的底色
PICK_SIZE = 16         # 备选风速弹窗里面的字号（比主行小一号，别喧宾夺主）
PICK_PAD = 8
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

# ---- 占盘上限：日志 + 诊断图总共不超过 5MB ----
LOG_ROLL = 512 * 1024             # 日志写到 512KB 就滚动一次（保留上一轮 → 日志最多 1MB）
LOG_LINE_MAX = 320                # 单行兜底：再长也截断（正常行 ~240 字符，基本截不到）
DUMP_BUDGET = 2 * 1024 * 1024     # 所有诊断图加起来不超过 2MB（超了按时间删旧的）
DUMP_MAX_W = 1280                 # 诊断图缩到最宽 1280：够看上下文，体积只有几十~几百 KB
DUMP_Q = 80                       # 整帧存 JPEG（有损但够看）；数字框另存无损 PNG（那才是要抠字形的地方）


def ui_font(size: int, bold: bool = True):
    family = "PingFang SC" if sys.platform == "darwin" else "Microsoft YaHei"
    return (family, size, "bold" if bold else "normal")


HEAD_SIZE = 24                 # 正常显示的字号（两行统一）
ERROR_SIZE = HEAD_SIZE - 6     # 提示/报错的字号：小一号，Windows 上 24pt 太大、长句会被切
ERR_MIN_SIZE = 12              # 提示字号的**下限**：放不下就逐级缩小，缩到这么小为止（再小就难认了）
GAP = 10                       # 同一行里两项之间的间距（两行共用一个值）
# 整窗缩放（长宽比固定）：字号、间距、输入框宽度一起按同一比例走，所以比例不会变
MIN_SCALE = 0.5
MAX_SCALE = 4.0
SCALE_STEP = 1.1               # 滚轮/菜单每次缩放比例
# 风速锁定的标记（跟在风速后面，如 `↑  7 m/s ◆`）。换符号只改这一个字符串。
# 注意：**不能用 🔒 这类 emoji** —— 它们是非 BMP 字符（U+1F512），Windows 的
# Microsoft YaHei 里没有，Tk 8.6 也不会自动回退到 Segoe UI Emoji，结果是方框/豆腐块。
# 只能用 BMP 里、中文字体本来就带的那种符号：※ ● ◆ ■ ▲ ★ ◇ □ △ ☆ 都稳妥。
LOCK_MARK = "※"
# 量窗口尺寸用的"最长一行"样本（定死尺寸，文字变长也不会跑出屏幕）。
# 箭头字形用 "→"：八个方向箭头里 → ↗ ← ↙ 都是最宽的（实测 24px，↓ ↑ 只有 18px），
# 拿它量才不会被"换了方向就顶出窗口"咬到（1.1.20 实测过：用 ↘ 量，换成 → 会差 6px）。
SAMPLES = (("→  15 m/s", "+179.9°", "→ 13.55", "↑ 13.55"),   # 正常结果的最宽一拍
           ("→  11-15 m/s", "+179.9°", "", ""),               # 只认出颜色档：三段区间里最长的 11-15
           ("→  15 m/s " + LOCK_MARK, "+179.9°", "→ 13.55", "↑ 13.55"),  # 锁定：数字后面多个标记
           ("·  0 m/s（无风）", "", "", ""))                   # 无风提示


def _num(v, fmt: str = "g", dash: str = "—") -> str:
    """
    把"可能是 None / 根本不存在"的数值安全地格式化成字符串。

    为什么需要：日志行里混着各种来源的字段（风速、θ、置信度…），只要有一个是 None，
    f"{v:g}" 就会抛 TypeError。而这条日志是在**识别线程**里拼的 —— 一抛异常整帧白干，
    用户看到的就是"识别线程异常：unsupported format string passed to NoneType.__format__"。
    日志只是旁路，绝不能反过来把识别搞崩，所以所有数值都走这里。
    """
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return dash


# --------------------------------------------------------------------------
# 后台线程：取帧 → 识别 → 丢进队列
# --------------------------------------------------------------------------
class Reader(threading.Thread):
    def __init__(self, out: queue.Queue, stop: threading.Event):
        super().__init__(daemon=True)
        self.out, self.stop = out, stop
        self.rec = None              # 建好之后挂在这里：悬浮窗上的"选风速"按钮要往里递玩家点的数
        self._last_msg, self._last_msg_t = "", 0.0   # 同样的报错 10 秒内只报一次，别刷屏
        self.have_good = False       # 是否出过至少一次好结果
        self.last_ok = 0.0           # 上一次识别成功的时间
        self.bad = 0                 # 连续没识别出来的帧数（要和 last_ok 一起判断才提示）
        self.next_beat = 0.0         # 下一帧"体检"行写日志的时间（每 10 秒一条，别刷爆日志）

    def run(self) -> None:
        src = rec = None
        backoff = 3.0                             # 连不上时的重试间隔（失败就翻倍，最长 30 秒）
        _migrate_old_artifacts()                  # 旧版摊在 exe 那层的日志/截图先收进子文件夹
        self._log(f"启动：v{APP_VERSION} / {sys.platform} / Python {sys.version.split()[0]} / "
                  f"{O.FPS:g} 次每秒 / 日志 {os.path.join(_artifact_dir(), 'nssr-overlay.log')}")
        self._prune_dumps()                       # 旧诊断图超预算先清掉（含旧版本留下的 .png）
        while not self.stop.is_set():
            if src is None:                       # （重）连 OBS + 建识别器
                if rec is None:                   # 识别器只建一次：它跟 OBS 无关，
                    try:                          # 先建它，出问题就不会留下一截连上的 OBS 会话
                        rec = O.Recognizer(SimpleNamespace(verbose=False))
                        self.rec = rec            # ← 给 UI 线程用（见 Overlay._choose）
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
        """
        把一条错误同时送到三处：终端、日志、窗口（窗口只放摘要）。

        同一条错误 10 秒内只报一次：一帧一次的重复报错会把日志刷爆，
        而窗口上显示的本来就是那句，不用反复重画。
        """
        now = time.time()
        if msg == self._last_msg and now - self._last_msg_t < 10.0:
            return
        self._last_msg, self._last_msg_t = msg, now
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
            # 识别出错是**本地算法**的问题，跟 OBS 连接无关：丢掉定位缓存下帧重找就行。
            # 早先这里 return False 会连 OBS 一起扔掉 → 每 3 秒"已连接"一次，日志刷屏还一直不好用。
            try:
                rec.reset()
            except Exception:
                pass
            return True
        now = time.time()
        if now >= self.next_beat:                 # 每 10 秒记一条"体检"行：够定位慢在哪一段，又不刷屏
            self.next_beat = now + 10.0
            # 整段拼装都包在 try 里：日志是旁路，任何字段缺失/类型不对都不许把识别线程搞崩
            # （1.1.27 之前这里 f"{st.get('speed'):g}" 在"锁着但这一拍没读出风速"的帧上会抛
            #   TypeError: unsupported format string passed to NoneType.__format__，整帧白干）。
            try:
                tq, td = getattr(src, "last_timing", (0.0, 0.0))
                d = st.get("dial")
                where = f"风盘({d.cx:.0f},{d.cy:.0f}) r={d.r:.0f}" if d else "没定位到风盘"
                if d is not None:
                    # k 是每帧实测的俯视压缩比；掩膜取哪张决定读数有没有侧壁偏置
                    arrow = st.get("arrow")
                    where += f" k={_num(d.k, '.3f')}"
                    if arrow is not None and getattr(arrow, "source", ""):
                        where += f" 掩膜={arrow.source}({arrow.n_pixels}px)"
                    # 判定：带 "?" = 数字勉强过线；"(区间)" = 只认出颜色档
                    sp_val, rng = st.get("speed"), st.get("speed_range")
                    if rng:
                        sp_txt = f"{rng[0]}-{rng[1]}(区间)"
                    elif sp_val is None:
                        sp_txt = "未读出"
                    else:
                        sp_txt = f"{_num(sp_val)}{st.get('speed_note') or ''}"
                    where += f" 判定={sp_txt}"
                    # 候选表：只在"不止一个候选"或"有没严格过关的"时才写 —— 单个强候选跟
                    # 判定= 完全重复，写出来只是把每行撑长
                    cands = st.get("cands") or []
                    if len(cands) > 1 or any(not c[2] for c in cands):
                        where += " 候选=" + "/".join(
                            f"{v}{'' if strong else '~'}({_num(s, '.2f')},{n or '无档'})"
                            for v, s, strong, n in cands[:4])
                    where += " 锁定=" + (_num(sp_val) if st.get("speed_locked") else "未")
                    if st.get("lock_mismatch"):
                        where += f" 独立={st['lock_mismatch']}"
                ocr = st.get("ocr") or {}
                if ocr.get("box"):
                    where += f" 数字框{ocr['box']}"
                self._log(f"取帧 {t_read*1000:.0f}ms（请求 {tq*1000:.0f} / 解码 {td*1000:.0f}）"
                          f" | 识别 {t_rec*1000:.0f}ms | 风速 {_num(st.get('speed'))} "
                          f"θ={_num(st.get('theta'), '.1f')} | {where}")
            except Exception as exc:              # 兜底：日志行拼不出来也不能影响识别
                self._log(f"体检行拼装失败：{exc!r}")
        if st.get("lock_event"):                  # 锁定/解锁只在状态切换时各写一行
            self._log(f"风速锁定：{st['lock_event']}")
        rej = (st.get("ocr") or {}).get("reject")
        if rej:                                   # 读成 35 m/s 这种不可能的 → 存帧留证并记日志
            self._log(f"读数越界已丢弃：读成 {rej['value']} m/s"
                      f"（置信 {rej['conf']:.2f}，框 {rej['box']}）")
            self._dump(rgb, st)
        elif st.get("dial") is not None and not st.get("ok"):
            # 定位到了风盘却读不出风速 —— 这才值得存帧。没定位到（没在瞄准界面）不存，
            # 否则 3 个名额会被一堆废帧占满。
            # ↑ 1.1.22：这里原先把 not st.get("ok") 漏了，于是"开跑前 6 个定位到风盘的帧"
            #   就把名额全占满（哪怕那几帧识别得好好的），真出问题时反而没图可看。
            self._dump(rgb, st)
        elif st.get("lock_mismatch"):
            # 锁着某个风速，但这一拍独立量出来的**箭头颜色档**跟锁定档打架 ——
            # 这正好是"锁定错了 / 换洞了 / 认错颜色"的现场证据，存帧供离线查。
            self._log(f"锁定与独立读数不一致：{st['lock_mismatch']}")
            self._dump(rgb, st)
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

    @staticmethod
    def _dump(rgb, st: dict) -> None:
        """
        把"定位到风盘但读不出来"的那一帧存下来（最多 6 组），方便离线定位。

        凡是"风速读不出"（数字没认出来）和"箭头找不到"（θ=None）都会走到这里 ——
        这两种正是最难离线复现的失败，多存几张才够排查。

        整帧缩到 1280 宽存 JPEG（体积小，看上下文够用）；数字框单独放大 4× 存 PNG，
        因为要抠字形的是它，不能有 JPEG 噪点。
        """
        if Reader._DUMPS >= 6:
            return
        Reader._DUMPS += 1
        try:
            d = _artifact_dir()
            p = os.path.join(d, f"nssr-fail-{Reader._DUMPS}.jpg")
            im = Image.fromarray(rgb)
            if im.width > DUMP_MAX_W:
                im = im.resize((DUMP_MAX_W, max(1, int(im.height * DUMP_MAX_W / im.width))),
                               Image.BILINEAR)
            im.save(p, quality=DUMP_Q)
            box = (st.get("ocr") or {}).get("box")
            if box:
                x0, y0, x1, y1 = box
                Image.fromarray(rgb[y0:y1, x0:x1]).resize(
                    ((x1 - x0) * 4, (y1 - y0) * 4), Image.NEAREST).save(
                    os.path.join(d, f"nssr-fail-{Reader._DUMPS}-box.png"))
            Reader._log(f"已存诊断图 {p}（数字框 {box}）")
        except Exception:
            pass

    @staticmethod
    def _log(line: str) -> None:
        """
        写诊断日志（**总是写**：双击启动没有终端，出问题只能靠这个日志查）。

        位置在程序所在文件夹（exe / .app 旁边）；写不进去才退回用户主目录。
        内容：启动、连接、报错、越界读数，外加每 10 秒一条耗时"体检"行。
        写到 LOG_ROLL 就滚动一次、只留上一轮（.log.1），所以日志最多 2×LOG_ROLL。
        """
        try:
            path = os.path.join(_artifact_dir(), "nssr-overlay.log")
            if len(line) > LOG_LINE_MAX:              # 单行兜底：任何字段异常都不该写出一条巨行
                line = line[:LOG_LINE_MAX] + f"…（原 {len(line)} 字符）"
            if os.path.exists(path) and os.path.getsize(path) > LOG_ROLL:
                try:
                    os.replace(path, path + ".1")    # 只留上一轮：日志总共最多 2×LOG_ROLL
                except OSError:
                    # Windows 上日志被记事本之类的程序打开着时 os.replace 会失败 ——
                    # 以前这里直接吞掉异常，结果**日志从那以后就再也不滚动了**（无限长大）。
                    # 退一步：原地清空。宁可丢掉旧内容，也不能让日志无上限地涨。
                    with open(path, "w", encoding="utf-8"):
                        pass
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(f"{time.strftime('%H:%M:%S')}  {line}\n")
        except Exception:
            pass

    @staticmethod
    def _prune_dumps() -> None:
        """
        诊断图总量超过 DUMP_BUDGET 就按时间删旧的（顺手清掉旧版本留下的 .png）。

        这样"日志 ≤2MB + 诊断图 ≤2MB"，整个程序在磁盘上最多 5MB 以内。
        """
        d = _artifact_dir()
        try:
            names = [n for n in os.listdir(d) if n.startswith("nssr-fail-")]
        except Exception:
            return
        items = []
        total = 0
        for n in names:
            p = os.path.join(d, n)
            try:
                st = os.stat(p)
            except Exception:
                continue
            items.append((st.st_mtime, st.st_size, p))
            total += st.st_size
        for _, size, p in sorted(items):            # mtime 最早的先删
            if total <= DUMP_BUDGET:
                break
            try:
                os.remove(p)
                total -= size
            except Exception:
                pass


# --------------------------------------------------------------------------
# 悬浮窗
# --------------------------------------------------------------------------
class Overlay:
    COEF_DIGITS = 4              # 系数最多 4 位数字：第一位整数 + 最多 3 位小数（1.999 够用）

    def __init__(self, args):
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
        self._fonts: dict[int, tkfont.Font] = {}      # 字号 → Font（提示要按需缩小，缓存起来别每次新建）
        self.font = self._font_at(HEAD_SIZE)

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

        # 第三行：横风 × 系数 = 结果。系数自己填（输入框），结果实时算。
        self.calc = tk.Frame(self.root, bg=BG)
        self.calc.pack(padx=PAD, pady=(0, 8))
        self.l_mul = tk.Label(self.calc, text="*", font=self.font, bg=BG, fg=DIM)
        self.l_mul.pack(side="left")
        self.coef = tk.StringVar(value="1")
        self.ent = tk.Entry(self.calc, textvariable=self.coef, width=4, justify="center",
                            font=self.font, bg=ENT_BG, fg=FG, insertbackground=FG,
                            relief="flat", bd=0, highlightthickness=0,
                            validate="key",
                            validatecommand=(self.root.register(self._coef_ok), "%P"))
        self.ent.pack(side="left", padx=(GAP, 0))
        self.l_eq = tk.Label(self.calc, text="=", font=self.font, bg=BG, fg=DIM)
        self.l_eq.pack(side="left", padx=(GAP, 0))
        self.l_prod = tk.Label(self.calc, text="—", font=self.font, bg=BG, fg=DIM)
        self.l_prod.pack(side="left", padx=(GAP, 0))
        self.lat_now: float | None = None              # 当前横风（绝对值），来自识别结果
        self.coef.trace_add("write", lambda *_: self._calc())     # 改系数立刻重算
        self.ent.bind("<Button-1>", self._focus_ent)              # 无边框窗口要主动要焦点才能打字
        self.ent.bind("<FocusIn>", lambda e: self.ent.selection_range(0, "end"))
        self.ent.bind("<Return>", lambda e: self.root.focus_set())

        # 宽度以第一行为准（第一行居中，左右各留 PAD）；高度取各行的自然高度。
        # 尺寸定死，文字变长也不会跑出屏幕。
        self.win_w = self.win_h = 0
        labs = (self.l_wind, self.l_ang, self.l_lat, self.l_lon)
        for sample in SAMPLES:
            for lab, txt in zip(labs, sample):
                lab.config(text=txt)
            self.root.update_idletasks()
            # 宽度以第一行为准；第二/三行万一更宽也只补到不叠字为止
            self.win_w = max(self.win_w, self.head.winfo_reqwidth() + 2 * PAD,
                             self.row.winfo_reqwidth() + 2 * PAD,
                             self.calc.winfo_reqwidth() + 2 * PAD)
            self.win_h = max(self.win_h, self.root.winfo_reqheight())
        self.win_w, self.win_h = max(self.win_w, 200), max(self.win_h, 60)   # 兜底
        self.text_px = max(80, self.win_w - 2 * PAD - 6)   # 第一行提示可用的像素宽
        self.cur_w = self.win_w                          # 当前窗口宽度（提示时会临时加宽）
        # 缩放基准：窗口/字号/间距都按 base_* × scale 走，长宽比自然不变
        self.base_w, self.base_h = self.win_w, self.win_h
        self.scale = 1.0
        self.pad, self.gap = PAD, GAP
        self.cur_h = self.win_h
        self.l_wind.config(text="·  --")
        self.l_ang.config(text="")
        self.l_lat.config(text="")
        self.l_lon.config(text="")

        widgets = (self.root, self.head, self.l_wind, self.l_ang,
                   self.row, self.l_lat, self.l_lon,
                   self.calc, self.l_mul, self.l_eq, self.l_prod)
        self._dx = self._dy = None
        self._grip_from = None                        # 拖左下角把手时的起始状态（见 _press/_drag）
        for w in widgets:                             # 拖动
            w.bind("<Button-1>", self._press)
            w.bind("<B1-Motion>", self._drag)
        self.menu = tk.Menu(self.root, tearoff=0)
        self.menu.add_command(label=f"NSSR风向识别 v{APP_VERSION}", state="disabled")
        self.menu.add_separator()
        self.menu.add_command(label="放大", command=lambda: self._zoom(SCALE_STEP))
        self.menu.add_command(label="缩小", command=lambda: self._zoom(1.0 / SCALE_STEP))
        self.menu.add_command(label="恢复默认大小", command=lambda: self._set_scale(1.0))
        self.menu.add_separator()
        self.menu.add_command(label="退出", command=self.quit)
        self._menu_at: tuple[int, int, int, int] | None = None   # 菜单弹出时占的屏幕区域（退出时要擦掉）
        self._quitting = False
        for w in widgets:                             # 右键（macOS 用 Button-2）
            w.bind("<Button-3>", self._popup)
            w.bind("<Button-2>", self._popup)
        for seq in ("<Button-3>", "<Button-2>"):      # 输入框也给右键菜单，但不参与拖动
            self.ent.bind(seq, self._popup)

        # 左下角缩放把手（无边框窗口没有系统边框，得自己给一个）+ 滚轮缩放。
        # 放左下角是因为窗口默认贴屏幕右上角：拖这个角时右上角钉住，最符合手感。
        # 事件不用单独绑：Tk 的 bindtags 会把子控件的事件也送到顶层窗口，统一在
        # _press/_drag/_release 里按 e.widget 分流（否则"拖窗口"和"缩放"会同时启动，互相打架）。
        # 图形用 "◣"：直角正好坐在左下角、斜边朝窗口内 —— 角落把手就该这么画（别改成 ◢）。
        # 要调的是**光标**方向：
        # 光标按平台挑：Windows 认 size_ne_sw（↙↗ 双向缩放，最贴切）；macOS 不认这个名字，
        # 退到 X11 名的 bottom_left_corner（↙ 单箭头）；都不行才是系统默认的 sizing（↖↘）。
        self.grip = tk.Label(self.root, text="◣", font=self._font_at(10), bg=BG, fg=DIM)
        for cur in ("size_ne_sw", "bottom_left_corner", "sizing"):
            try:
                self.grip.config(cursor=cur)
                break
            except Exception:
                continue
        self.grip.place(x=0, rely=1.0, anchor="sw")
        for w in (self.root, self.grip):
            w.bind("<ButtonRelease-1>", self._release)
        for w in (self.root, self.head, self.row, self.calc,
                  self.l_wind, self.l_ang, self.l_lat, self.l_lon,
                  self.l_mul, self.l_eq, self.l_prod):
            w.bind("<MouseWheel>", self._wheel)       # Windows / macOS
            w.bind("<Button-4>", self._wheel)         # Linux 上滚
            w.bind("<Button-5>", self._wheel)         # Linux 下滚

        self._place()
        # 备选风速的弹窗（"锁定前拿不准就问玩家"用）。懒建：不用的时候一个窗口都不多。
        self.pick: tk.Toplevel | None = None
        self._pick_key: tuple | None = None       # 当前弹窗显示的是哪一组备选（变了就重建）
        self._alpha = args.alpha
        self.reader = Reader(self.q, self.stop)
        self.reader.start()
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
        if e.widget is self.grip:                     # 按在缩放把手上 → 这次是缩放，不是拖窗口
            self._dx = self._dy = None
            self._grip_from = (e.x_root, e.y_root,
                               self.root.winfo_x() + self.cur_w, self.root.winfo_y())
            try:
                self.grip.grab_set()                  # 抓指针：拖出把手范围也不丢事件
            except Exception:
                pass
            return
        if e.widget is self.ent:                      # 点输入框是要打字/选字，别把窗口拖走
            self._dx = self._dy = None
            return
        self._dx, self._dy = e.x_root - self.root.winfo_x(), e.y_root - self.root.winfo_y()

    def _drag(self, e) -> None:
        if self._grip_from is not None:               # 正在拖把手 → 缩放，别挪窗口
            self._grip_drag(e)
            return
        if self._dx is None:                          # 刚才按在输入框上，不当拖动
            return
        self.root.geometry(f"+{e.x_root - self._dx}+{e.y_root - self._dy}")

    def _release(self, e) -> None:
        if self._grip_from is not None:
            self._grip_from = None
            try:
                self.grip.grab_release()
            except Exception:
                pass
    def _focus_ent(self, e=None) -> None:
        """无边框置顶窗口不会自动拿键盘焦点：点一下输入框要主动要焦点，否则打不了字。"""
        try:
            self.root.after(1, self.ent.focus_force)
        except Exception:
            pass

    # ---- 整窗缩放（长宽比固定：字号/间距/窗口一起按同一比例走）----
    def _zoom(self, factor: float) -> None:
        self._set_scale(self.scale * float(factor))

    def _wheel(self, e) -> None:
        delta = getattr(e, "delta", 0) or (120 if getattr(e, "num", 0) == 4 else -120)
        self._zoom(SCALE_STEP if delta > 0 else 1.0 / SCALE_STEP)

    def _grip_drag(self, e) -> None:
        """
        拖左下角把手缩放：**钉住右上角**（窗口默认就贴屏幕右上角），
        目标宽度 = 右边缘 − 光标 x，目标高度 = 光标 y − 上边缘，取较大的一维定比例。
        """
        if not getattr(self, "_grip_from", None):
            return
        _, _, right, top = self._grip_from
        w = max(40.0, right - e.x_root)
        h = max(24.0, e.y_root - top)
        self._set_scale(max(w / self.base_w, h / self.base_h))

    def _set_scale(self, s: float, anchor: str = "right") -> None:
        """
        把整窗按 s 倍缩放：**字号、内边距、输入框宽度、窗口尺寸一起乘 s**。

        一起乘的好处是长宽比天然不变（不用事后纠正），字也不会相对窗口变胖变瘦。
        anchor="right" 时贴住右边缘（滚轮/菜单缩放时不会往左跑），"topleft" 时钉住左上角
        （拖右下角把手时才符合手感）。
        """
        s = max(MIN_SCALE, min(MAX_SCALE, float(s)))
        if abs(s - self.scale) < 1e-3:
            return
        self.scale = s
        font = self._font_at(max(8, int(round(HEAD_SIZE * s))))
        self.font = font
        for w in (self.l_wind, self.l_ang, self.l_lat, self.l_lon,
                  self.l_mul, self.l_eq, self.l_prod, self.ent):
            w.config(font=font)
        self.pad = max(0, int(round(PAD * s)))     # PAD 默认就是 0（窗口宽 = 第一行宽）
        self.gap = max(2, int(round(GAP * s)))
        self.head.pack_configure(padx=self.pad, pady=(max(1, int(round(6 * s))), 0))
        self.row.pack_configure(padx=self.pad, pady=(0, max(1, int(round(8 * s)))))
        self.calc.pack_configure(padx=self.pad, pady=(0, max(1, int(round(8 * s)))))
        self.l_ang.pack_configure(padx=(self.gap, 0))
        self.l_lon.pack_configure(padx=(self.gap, 0))
        self.l_mul.pack_configure(padx=(self.gap, 0))
        self.l_eq.pack_configure(padx=(self.gap, 0))
        self.l_prod.pack_configure(padx=(self.gap, 0))
        self.ent.pack_configure(padx=(self.gap, 0))
        # 注意：**不要**在这里改输入框的 width。Tk 的 Entry(width=N) 单位是"字符数"，
        # 跟着字体走；字体已经按 s 放大了一次，再乘一次 N 就会按 s² 涨（实测把 "=" 和
        # 结果挤出窗口）。保持 4 个字符宽，像素宽度自然随字体等比变化。
        try:
            # 把手也随字号长，但封顶：放太大就成了个巨大的三角，难看
            self.grip.config(font=self._font_at(min(16, max(8, int(round(10 * s))))))
        except Exception:
            pass
        self.win_w = max(120, int(round(self.base_w * s)))
        self.win_h = max(40, int(round(self.base_h * s)))
        self.text_px = max(60, self.win_w - 2 * self.pad - 6)
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        x = self.root.winfo_x()
        y = self.root.winfo_y()
        if anchor == "right":                       # 贴住右边缘，缩放时窗口不会往左跑
            x = self.root.winfo_x() + self.cur_w - self.win_w
        x = max(0, min(int(x), max(0, sw - self.win_w)))
        y = max(0, min(int(y), max(0, sh - self.win_h)))
        self.cur_w, self.cur_h = self.win_w, self.win_h
        self.root.geometry(f"{self.win_w}x{self.win_h}+{x}+{y}")

    def _calc(self) -> None:
        """
        算「横风 × 系数」。

        系数框**只收数字**，且约定：**第一位是整数、剩下的都当小数**（见 _coef_value）。
        这样不用打小数点，数字小键盘点几下就完事；空框或还没识别到横风时显示 —。
        """
        k = self._coef_value()
        if k is None or self.lat_now is None:
            self.l_prod.config(text="—", fg=DIM)
            return
        v = abs(self.lat_now) * k
        self.l_prod.config(text=f"{v:.2f}", fg=speed_color(abs(v)))

    def _coef_value(self) -> float | None:
        """系数框里的数字 → 数值：第一位整数、其余小数（115 → 1.15、85 → 0.85、8 → 8）。"""
        s = self.coef.get().strip()
        if not s or not s.isdigit():
            return None
        return int(s) / (10 ** (len(s) - 1))

    def _coef_ok(self, proposed: str) -> bool:
        """输入校验：只放数字，最多 COEF_DIGITS 位（第一位整数 + 后面几位小数）。"""
        return proposed == "" or (proposed.isdigit() and len(proposed) <= self.COEF_DIGITS)

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
            self._choice_popup(None)                # 报错/状态时把备选弹窗收掉（别挂着误导人）
            self._line1(st["text"], ERR if kind == "error" else DIM)
            return

        self._choice_popup(st.get("choice_options") if st.get("need_choice") else None)
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
            self.l_ang.pack_configure(padx=(self.gap, 0))
            self.l_lon.pack_configure(padx=(self.gap, 0))
            w = st["wind"]
            # 大行：风向箭头 + 总风速 + 夹角（夹角中性色）
            # 小行：横向 / 纵向分量
            # 数字部分统一用当前风速档位的颜色
            ang = w.get("theta_signed_deg", w["theta_deg"])
            # 低置信度的显示语义（数字识别不确定时怎么显示）：
            #   "7? m/s"           数字读出来了，但只是勉强过线 → 带问号提醒这拍存疑
            #   "6-10 m/s"         只认出箭头的颜色档、数字没读出 → 给区间，方向照样显示
            #   "? m/s（风速未知）"  连颜色档都没有 → 只给方向
            sp, rng, note = st.get("speed"), st.get("speed_range"), st.get("speed_note") or ""
            if sp is None and rng:
                sp_txt, col = f"{rng[0]}-{rng[1]} m/s", DIM
            elif sp is None:
                # 连颜色档都没认出来（正常流程走不到这里，留着当兜底）：只给方向、速度打问号。
                # 不写"（风速未知）"那种长句 —— 窗口宽度是按"最长一行"定死的，长句会被切掉，
                # 反而连方向都看不清；详情走日志/终端。
                sp_txt, col = "? m/s", DIM
            else:
                sp_txt = f"{sp:g}{'?' if note else ''} m/s"
                if st.get("speed_locked"):
                    sp_txt += f" {LOCK_MARK}"   # 这一洞风速已锁定：不再跟着单帧数字误读跳
                col = speed_color(sp)
            self.l_wind.config(text=f"{wind_arrow(w['theta_deg'])}  {sp_txt}",
                               fg=col, font=self.font)
            self.l_ang.config(text=f"{ang:+.1f}°", font=self.font)
            if "lateral_abs" in w:
                # 第二行沿用第一行的排法：箭头 + 两个空格 + 数值
                # 颜色各自按**自己的数值**取档位（横向 7 → 黄、纵向 12 → 品红），不是跟着总风速
                self.l_lat.config(text=f"{LAT_ARROW.get(w['lateral_dir'], '·')}  {w['lateral_abs']:.2f}",
                                  fg=speed_color(w["lateral_abs"]), font=self.font)
                self.l_lon.config(text=f"{LON_ARROW.get(w['longitudinal_dir'], '·')}  {w['longitudinal_abs']:.2f}",
                                  fg=speed_color(w["longitudinal_abs"]), font=self.font)
                self.lat_now = w["lateral_abs"]       # 横风 → 给下面的乘法器用
            else:
                # 风速只有一个区间 → 分量算不了。留空位，不能显示成 0.00（会被当成真值）
                self.l_lat.config(text="—", fg=DIM, font=self.font)
                self.l_lon.config(text="", fg=DIM, font=self.font)
                self.lat_now = None
            self._calc()

    # ---- 锁定前"拿不准就问玩家"：备选风速的小弹窗 ----
    def _choice_popup(self, options) -> None:
        """
        显示 / 更新 / 收起"选风速"的弹窗；options=None 表示收起。

        为什么单独开一个 Toplevel，而不是往主窗口里加一行：主窗口的尺寸是按"最长一行"定死的，
        还带左下角把手拖拽缩放和滚轮缩放，往里塞一行动态内容很容易把那些**已经验证过的**行为
        搞坏（尺寸一抖动，缩放和贴边就跟着歪）。弹窗平时不存在，只有需要玩家决策时才冒出来。
        """
        opts = options or []
        key = tuple((int(v), round(float(c), 2)) for v, c, _n in opts)
        if not key:
            if self.pick is not None:
                self.pick.withdraw()
            self._pick_key = None
            return
        if self.pick is None:
            self._build_pick()
        if key != self._pick_key:                     # 备选变了 → 重建按钮（同一个备选就不闪）
            for w in self._pick_btns:
                w.destroy()
            self._pick_btns = []
            for v, conf, n in opts:
                b = tk.Button(self._pick_row, text=str(int(v)), width=2,
                              font=self._font_at(PICK_SIZE), bg=ENT_BG, fg=FG,
                              activebackground=ACCENT, activeforeground=BG,
                              relief="flat", bd=0, highlightthickness=0,
                              cursor="hand2",
                              command=lambda v=v: self._choose(v))
                b.pack(side="left", padx=(0, 6))
                self._pick_btns.append(b)
            self._pick_key = key
        # 弹窗里的分数只写日志（按钮上只放数字，玩家该按游戏画面来选，不是按我们的分数）
        self.pick.deiconify()
        self.pick.lift()
        try:
            self.pick.attributes("-topmost", True)
        except Exception:
            pass
        self._place_pick()

    def _build_pick(self) -> None:
        p = tk.Toplevel(self.root)
        p.overrideredirect(True)                      # 无边框，跟主窗口一个风格
        p.configure(bg=BG)
        try:
            p.attributes("-topmost", True)
            p.attributes("-alpha", max(0.88, min(1.0, float(self._alpha) + 0.34)))
        except Exception:
            pass
        tk.Label(p, text="风速拿不准，点一个：", font=self._font_at(PICK_SIZE),
                 bg=BG, fg=WARN).pack(padx=PICK_PAD, pady=(PICK_PAD, 4))
        self._pick_row = tk.Frame(p, bg=BG)
        self._pick_row.pack(padx=PICK_PAD, pady=(0, PICK_PAD))
        self._pick_btns: list = []
        self.pick = p
        p.withdraw()

    def _place_pick(self) -> None:
        """贴在主窗口正下方；下面放不下就摆到主窗口上方（始终整窗可见）。"""
        self.pick.update_idletasks()
        w, h = self.pick.winfo_reqwidth(), self.pick.winfo_reqheight()
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        x = self.root.winfo_x()
        y = self.root.winfo_y() + self.cur_h + 4
        if y + h > sh - 8:
            y = max(0, self.root.winfo_y() - h - 4)
        x = max(0, min(x, sw - w - 8))
        self.pick.geometry(f"{w}x{h}+{int(x)}+{int(y)}")

    def _choose(self, value) -> None:
        """玩家点了某个备选风速 → 递给识别线程，并立刻收起弹窗（手感上"点了就生效"）。"""
        rec = getattr(self.reader, "rec", None)
        if rec is not None:
            try:
                rec.choose(value)
                Reader._log(f"玩家选了风速 {int(value)} m/s")   # 日志是 Reader 的静态方法
            except Exception as exc:
                Reader._log(f"备选风速递交失败：{exc!r}")
        self._choice_popup(None)

    def _line1(self, text: str, color: str) -> None:
        """
        提示/报错：目标是**整句都看得见**，所以先缩字号、再考虑加宽，最后才截断。
        """
        font, line1, line2, w = self._fit_error(text)
        self.lat_now = None                           # 提示状态下横风是未知的 → 乘法结果显示 —
        self._calc()
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
        step = max(1, int(round(2 * self.scale)))
        for size in range(max(9, int(round(ERROR_SIZE * self.scale))),
                          max(9, int(round(ERR_MIN_SIZE * self.scale))) - 1, -step):
            width = base - 2 * self.pad - 8
            if self._fits2(text, size, width):
                font = self._font_at(size)
                line1, line2 = self._two_lines(text, font, width)
                return font, line1, line2, base
        font = self._font_at(max(9, int(round(ERR_MIN_SIZE * self.scale))))
        cap = max(base, int(self.root.winfo_screenwidth() * 0.90))
        width = cap - 2 * self.pad - 8
        line1, line2 = self._two_lines(text, font, width)
        if font.measure(line2) > width:                   # 连屏幕 90% 都放不下 → 只能截断
            line2 = self._fit(line2, width, font)
        need = max(font.measure(line1), font.measure(line2)) + 2 * self.pad + 8
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
        self.cur_w, self.cur_h = w, self.win_h
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
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    Overlay(args).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
