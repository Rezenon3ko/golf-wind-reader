#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OBS 直播风向识别（obs-websocket 抓源，默认 4 次/秒）

先在 OBS 里启用 WebSocket 服务器（OBS 28+ 默认已启用），再运行：

    python3 obs_live_wind_reader.py                # 挂机识别
    python3 obs_live_wind_reader.py -t             # 抓一帧识别一次，打印详情后退出
    python3 obs_live_wind_reader.py --obs-source "视频采集设备"

取帧走 obs-websocket 的 GetSourceScreenshot（1920×1080 JPEG q90，可用
NSSR_JPG_QUALITY 覆盖）：
不依赖摄像头驱动、不占摄像头，也不需要"启动虚拟摄像机"。

依赖：numpy + Pillow + obsws-python。
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import socket
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wind_reader as W


def _safe_stdio() -> None:
    """
    让本文件的 print 永远不因**编码**抛异常（Windows 中文系统 GBK 编不出 ⚠ ✗，
    英文系统 cp1252 连中文都编不出）。

    这不是纸上谈兵：抛的地方常常正好是"刚连上 OBS、正在打印源名"那一刻，
    连接被丢在半路、OBS 里留下僵尸会话，程序却报"连不上"。
    """
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8", errors="replace"))
            continue
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass


_safe_stdio()

FPS = 4.0                          # 识别频率（次/秒）—— 单拍耗时决定上限，显示侧不受影响
# ↑ 2026-10-15 从 2 → 4：JPEG 质量切到 90 之后单拍只要 ~200ms（取帧 133ms + 识别 69ms），
#   4 次/秒 = 250ms 预算，还有 ~50ms 余量。好处是 **3 秒锁定窗口里能收 ~12 拍**（原来 ~6 拍），
#   投票更稳、也更容易在"拿不准要问玩家"之前就分出高下。
#   **显示侧完全没动**：悬浮窗仍旧每 80ms 刷一次（`Overlay._poll`），只是每拍拿到的状态新了。
#   角度的 EMA 时间常数、跳变阈值、各处"连续 N 拍"的判据都按 _frames() 等价换算，
#   所以看起来不该比以前更抖。上限：OBS 抓图 4~5 次/秒没压力，再往上通常被它的截屏路径卡住。
#   想改：NSSR_FPS=2（回老节奏）或 3/5/6（夹在 1~6）。
_fps_env = os.environ.get("NSSR_FPS", "").strip()
if _fps_env:
    try:
        FPS = max(1.0, min(6.0, float(_fps_env)))
    except ValueError:
        pass                           # 写错就按默认 4 走，别让程序起不来


def _frames(seconds: float) -> int:
    """秒 → 帧数：所有"连续 N 拍"的阈值都按秒等价换算，调 FPS 不用重新调参。"""
    return max(1, int(round(seconds * FPS)))


QUALITY = 90                       # OBS 抓图 JPEG 质量（可用环境变量 NSSR_JPG_QUALITY 覆盖）
# 取值依据（2026-10-15 实机实测，1920×1080 采集卡源）：
#   q100 请求 395ms + 解码 22ms = 整拍 459ms，体积 1773KB
#   q90  请求 124ms + 解码  9ms = 整拍 178ms，体积  418KB   ← 2.6 倍，体积 -76%
#   识别部分三档完全一样（都是 43ms，质量只影响传输与解码）。
#   精度：静态 20 帧 + 转视角 60 帧，q90 定位/校验 80/80、风速零错、φ 相对 q100 中位偏 2.3°、
#   **0 次 180° 反向**；q95 反而丢了 1 帧。历史上 q80 曾统计出"反向多 3/4"，所以停在 90 不再往下。
QUALITY_ENV = "NSSR_JPG_QUALITY"   # 想回退/再压：NSSR_JPG_QUALITY=100（或 80）启动即可
try:
    _q_env = int(os.environ.get(QUALITY_ENV, "").strip() or QUALITY)
    QUALITY = max(50, min(100, _q_env))
except ValueError:
    pass                           # 环境变量写错了就按默认 90 走，别让程序起不来
OBS_HOST = "127.0.0.1"             # OBS WebSocket 地址（端口/密码读 OBS 自己的配置）
REQ_TIMEOUT = 3                    # 单次请求超时（秒）—— 只影响"一次取帧最多等多久"，不是重连间隔。
# ↑ 5 → 3（2026-10-15）：OBS 卡住时，"3 次连续失败 → 重连"的判定从 ~15.6s 缩到 ~9.6s。
#   余量仍然很大：实测同机 q90 取帧 108~133ms、q100 也才 383~405ms（7~25 倍余量）。
WIDTH, HEIGHT = 1920, 1080         # 抓图分辨率（相对先验定位 / 超采样都按这个标定）

# 采集卡类输入源的 inputKind（OBS 的类型标识，与界面语言无关）
CAPTURE_INPUT_KINDS = (
    "dshow_input", "dshow_input_beta",            # Windows 视频采集设备
    "av_capture_input_v2", "av_capture_input",    # macOS 视频采集设备
    "v4l2_input",                                 # Linux 视频采集设备
    "decklink-input", "blackmagic_device",        # SDI / HDMI 采集卡
)
# 认不出类型时的源名兜底（多语言）
DEFAULT_SOURCES = ("视频采集设备", "采集卡设备", "Video Capture Device", "視訊擷取裝置")


def _glyph_bank_path() -> str:
    """
    字形模板库的位置。源码运行时就在本文件旁边；打成 .app（PyInstaller）后，
    数据文件被放在解包目录（sys._MEIPASS）或 Resources 里 —— 逐处找一遍。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (here, getattr(sys, "_MEIPASS", ""), os.path.join(here, os.pardir, "Resources")):
        if d and os.path.exists(os.path.join(d, "glyph_bank.npz")):
            return os.path.join(d, "glyph_bank.npz")
    return os.path.join(here, "glyph_bank.npz")      # 都没有：返回源码位置，由调用方报错


GLYPH_BANK = _glyph_bank_path()
RELOCATE_S = 1.5                   # 找不到风盘时的重试间隔（秒）
RECONNECT_FAILS = 3                # 连续取帧失败这么多次就重连（顺便重新挑源）
RELOCATE_BAD = _frames(3.0)        # 连续 3 秒读不出来就丢掉缓存的定位，重新找一次（原 2 次/秒 × 6 拍）
HYST = 0.05                        # 横向/纵向 |值| 小于它就沿用上一次方向（0 附近不抖）

# ---- 风速锁定：同一洞的风速是固定的，锁住它就不会因为某一拍数字读错而乱跳 ----
WIND_MEASURE_S = 3.0               # 看到风向盘后先正常测这么久，再从这几拍里挑"置信度最高"的锁住。
# ↑ 4 次/秒 ⇒ 大约 12 拍样本（2 次/秒时只有 6 拍；调 FPS 不用改这个数，窗口本来就是按秒算的）。
#   （原先 5 秒：玩家从"摆好姿势"到出手常常不到 5 秒，锁还没
#   定下来就打完了；而且窗口越长，越容易把"某一拍读错"也编进票里。）
LOCK_MIN_VOTES = max(2, _frames(0.5))   # 至少要有这么多拍读到同一个值才允许锁 —— 单拍不锁。
# 按"≥0.5 秒的一致证据"折算：2 次/秒 → 2 拍（跟以前一样）、4 次/秒 → 2 拍、6 次/秒 → 3 拍。
# ↑ 天大的坑：锁定判据是"票数×平均置信度"，于是**只有一拍**也能锁。实测 10-14 那局：
#   风盘还没出现时误抓的假风盘某一拍读成 1 m/s，3 秒后就把它锁死了，之后一直显示错误风速。
# ---- 早锁：不等满 WIND_MEASURE_S，只要"票够多 + 跨度够 + 一边倒"就立刻锁 ----
# 动机：4 次/秒下满窗口要等 3 秒（12 拍才收齐），而玩家从"摆好姿势"到出手常常不到 3 秒。
# 但"多收几拍"本身买不到新信息（画面基本不动，多出来的样本主要在压噪声），所以真正的收益是
# **早点定下来**：4 拍一致就锁 ≈ 1 秒，而不是硬等 3 秒。下面四条同时满足才早锁：
LOCK_EARLY_VOTES = 4        # ① 至少 4 拍读到同一个值（比"1 拍就锁"的老 bug 严 4 倍）
LOCK_EARLY_SPAN_S = 0.6     # ② 这 4 票得横跨 ≥0.6 秒（防"一瞬间 4 连帧"；高帧率下才起作用）
LOCK_EARLY_CONF = 0.82      # ③ 平均置信度 ≥0.82
# ↑ 定这个数是因为实测分两簇：正常读（1 位/2 位数）落在 0.83~0.99（3 张真机帧 0.97/0.91/0.83），
#   而"勉强过线"的弱读（日志里带 `?` 的、走 best_effort 的）实测只有 0.61~0.72。
#   取 0.82 正好把弱读挡在门外，又不至于让 0.83 那种正常 2 位数白等 3 秒。
LOCK_EARLY_RATIO = 0.80     # ④ 第一名占前两名置信度之和 ≥0.80（两个值咬得紧就不早锁，等满窗口）
MEASURE_GAP_S = 2.5                # 风盘消失超过这么久 → 这一轮测的票作废，下次出现重新测。
WIND_UNLOCK_S = 30.0               # 风向盘消失超过这么久（时限，暂定）→ 判定进入下一洞 → 解锁重新测
LOCK_BREAK_RUN = _frames(3.0)      # 锁定期间"独立读数连续 3 秒落在另一档" → 认为换洞了（安全网）
# ---- 锁定前"拿不准就问人" ----
# 测完（WIND_MEASURE_S），如果读数不够一边倒，就先别锁，把前几名当**备选**弹出来让玩家点一个。
# 判据两条（满足任一条就问）：
#   ① 第一名的平均置信度 < CHOICE_CONF          —— 谁都不像；
#   ② 第一名占前两名"置信度之和"的比例 < CHOICE_RATIO —— 两个值咬得很紧。
CHOICE_CONF = 0.80                 # 平均置信度下限（正常帧 0.87~1.00，根本不会触发）
CHOICE_RATIO = 0.65                # 第一名在"前两名总票"里的占比下限
CHOICE_MAX = 5                     # 最多给几个备选（一排按钮放得下；"数字读不出"那档正好 5 个）
CHOICE_TIMEOUT_S = 5.0             # 备选弹出 5 秒还没人点 → 不再自己猜：换成"全风档候选"（1~15）
# ↑ 原来是"20 秒没人点就按最高分自动锁"。按"风速绝不能错、拿不准就让玩家选"的原则改成：
#   5 秒后把全部 15 个可能值（按青/黄/品红三档分行）都列出来，由玩家点一个 —— 程序不替玩家
#   拍板。玩家点了就锁，不点就一直挂着（换洞/风盘消失会自然收起）。
# ---- 锁定值与独立读数持续打架时的判据（同色档内部也算，见 _update_lock 的已锁分支）----
LOCK_VALUE_RUN = _frames(3.0)      # 连续 3 秒读到"另一个值"（同档也算）→ 弹候选让玩家裁决
LOCK_VALUE_CONF = 0.82             # 只认高置信度的独立读数当证据（正常读 0.83~0.99、弱读 0.61~0.72）
FLIP_CONF = 0.6                    # 方向置信度低于此值时才做反 180° 保护
FLIP_RUN = _frames(1.5)            # 连续 1.5 秒都反向才认为真的转向了（原 2 次/秒 × 3 拍）
SMOOTH_A = 1.0 - (1.0 - 0.25) ** (2.0 / FPS)
# ↑ 角度指数平滑系数：按"2 次/秒时代的 0.25"换算（4 次/秒 → 0.134），时间常数不变。
#   越小越稳、越大越跟手。
                                   # 实测扫描（81 拍静止数据）：0.35→0.25 让显示改动量降到 1/3，
                                   # 微调首次响应只慢 0.2 秒；再降到 0.2 稳定性不变但更钝，故取 0.25
SMOOTH_JUMP = 5.0 * (2.0 / FPS)    # 单拍变化超过它：连着两拍 = 真在转（直通），只一拍 = 测歪了（丢弃）
# ↑ 同样按时间等价换算（4 次/秒 → 2.5°）：跟"每秒转多少度"这个真实阈值保持一致，
#   否则帧率一高，"几度的正常抖动"会被误判成"在转视角"而直通。
TURN_RUN = _frames(1.0)            # 连续 1 秒都在大角度变化 → 认定真在转视角（见 _steady_theta ①）
SMOOTH_STEP = 0.7                  # 显示死区：EMA 偏离当前显示值超过它才更新 —— 静止时显示纹丝不动


# ==========================================================================
# 一、取帧：obs-websocket 抓源
# ==========================================================================

def _obs_ws_config_paths() -> list[str]:
    """obs-websocket 配置文件的候选位置（标准安装 + 便携模式 + 老式路径）。"""
    home = os.path.expanduser("~")
    rel = os.path.join("plugin_config", "obs-websocket", "config.json")
    if sys.platform == "darwin":
        base = os.path.join(home, "Library", "Application Support")
    elif sys.platform.startswith("win"):
        base = os.environ.get("APPDATA") or os.path.join(home, "AppData", "Roaming")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(home, ".config")
    return [
        os.path.join(base, "obs-studio", rel),                 # 标准安装
        os.path.join(os.getcwd(), "config", "obs-studio", rel),  # 便携模式（程序目录下）
        os.path.join(home, "obs-studio", rel),                  # 便携模式（用户目录下）
    ]


def _obs_ws_defaults() -> dict:
    """
    读 OBS 自己的 obs-websocket 配置（地址 / 端口 / 密码）。

    开了「身份验证」时，客户端必须带对密码才能连 —— 我们在这里自动读出来。
    密码在配置里是 base64；顺手兼容几个键名（server_password / auth_password / password）。
    另外支持用环境变量覆盖：NSSR_OBS_HOST / NSSR_OBS_PORT / NSSR_OBS_PASSWORD
    （读不到配置文件时，玩家设一个环境变量就能连，不用改代码）。
    """
    cfg = {"host": os.environ.get("NSSR_OBS_HOST") or "127.0.0.1",
           "port": 4455, "password": "", "from": "默认值"}
    for path in _obs_ws_config_paths():
        try:
            with open(path, encoding="utf-8") as fh:
                j = json.load(fh)
        except Exception:
            continue
        try:
            cfg["port"] = int(j.get("server_port") or cfg["port"])
        except Exception:
            pass
        raw = str(j.get("server_password") or j.get("auth_password")
                  or j.get("password") or "")
        if raw:
            try:
                cfg["password"] = base64.b64decode(raw, validate=True).decode("utf-8")
            except Exception:
                cfg["password"] = raw          # 已经是明文
        cfg["from"] = path
        break
    if os.environ.get("NSSR_OBS_PORT"):
        try:
            cfg["port"] = int(os.environ["NSSR_OBS_PORT"])
            cfg["from"] = "环境变量 NSSR_OBS_PORT"
        except Exception:
            pass
    if os.environ.get("NSSR_OBS_PASSWORD"):
        cfg["password"] = os.environ["NSSR_OBS_PASSWORD"]
    return cfg


def _probe_port(host: str, port: int) -> str:
    """连不上时探一下端口，给一句人话原因（区分"没服务" vs "密码不对"）。"""
    try:
        with socket.create_connection((host, port), timeout=1.5):
            return (f"TCP {host}:{port} 能连上，但 WebSocket 握手失败 → 多半是密码/身份验证；"
                    f"OBS → 工具 → WebSocket 服务器设置里取消「启用身份验证」并点「应用」")
    except Exception:
        return (f"TCP {host}:{port} 完全连不上 → OBS 没启动、没勾「启用 WebSocket 服务器」，"
                f"或端口不是 {port}（改过端口就设环境变量 NSSR_OBS_PORT=新端口）")


def _allow_local_direct() -> None:
    """
    把本机地址加进 no_proxy，保证连 OBS 不走代理。

    很多玩家机器上挂着代理（Clash / V2Ray 等），有的会把 http_proxy 设成全局环境变量；
    底层 websocket-client 会照单全收，于是连 ws://127.0.0.1:4455 也被塞进代理 ——
    结果是：OBS 那边看到"有客户端连进来了"，客户端这边却在握手/识别阶段报错，
    表现就是"明明连上了却一直报连不上"。本机直连既更快也不会被代理搅和。
    """
    for key in ("no_proxy", "NO_PROXY"):
        cur = [x for x in os.environ.get(key, "").replace(" ", "").split(",") if x]
        for host in ("127.0.0.1", "localhost", "::1"):
            if host not in cur:
                cur.append(host)
        os.environ[key] = ",".join(cur)


def _connect_client(host: str, port: int, password: str):
    """
    建 obs-websocket 客户端。失败就原样抛异常，但**先把半截 WebSocket 关掉**。

    ReqClient 在「TCP 已连上、但握手/身份识别没走完」时抛异常，那时句柄还在它肚子里，
    我们拿不到、也就关不掉 —— OBS 的「已连接的 WebSocket 会话」里就会挂一个永远不走的
    僵尸连接（正是"OBS 显示已连接、程序却说连不上"这种情况的典型现场）。
    """
    import obsws_python as obs

    class _Client(obs.ReqClient):
        def __init__(self, **kw):
            try:
                super().__init__(**kw)
            except Exception:
                try:
                    self.base_client.ws.close()
                except Exception:
                    pass
                raise

    return _Client(host=host, port=port, password=password, timeout=REQ_TIMEOUT)


class ObsWebSocketSource:
    """通过 obs-websocket 抓 OBS 某个源的画面（JPEG 压缩换速度）。

    不依赖虚拟摄像头驱动；OBS 28+ 的 WebSocket 服务器默认已启用。
    依赖：pip install obsws-python
    """

    name = "obs"

    def __init__(self, source: str | None = None, host: str = OBS_HOST, port: int | None = None,
                 password: str | None = None, width: int = WIDTH, height: int = HEIGHT,
                 fmt: str = "jpg", quality: int = QUALITY):
        self.want_source = source                    # 调用方指定的源名（None = 自动认）
        self.host, self.port, self.password = host, port, password
        self.width, self.height, self.fmt, self.quality = width, height, fmt, quality
        self.last_error = ""                         # 最近一次取帧失败的原因
        self.fails = 0                               # 连续失败次数
        self.dead = False                            # 连接/源已经救不回来，该重连
        self.note = ""                               # 连上后的说明（源名/分辨率/配置来源），供日志用
        self.last_timing: tuple[float, float] = (0.0, 0.0)   # (请求, 解码) 秒，诊断用
        self.cli = None                              # 连接句柄（close() 会用到）
        self._connect()

    def _connect(self) -> None:
        """连 OBS 的 WebSocket 并挑源。失败直接抛异常，由调用方决定重试节奏。"""
        cfg = _obs_ws_defaults()                     # 没显式给就用 OBS 配置里的
        port = cfg["port"] if self.port is None else self.port
        password = cfg["password"] if self.password is None else self.password
        cfg_from = cfg.get("from", "默认值")
        self.close()                                 # 重连前先关掉旧会话，别在 OBS 里留僵尸连接
        _allow_local_direct()                        # 本机连接绕开代理
        try:
            self.cli = _connect_client(self.host, port, password)
        except ImportError as exc:
            raise RuntimeError(
                "websocket 取帧需要 obsws-python：pip install obsws-python") from exc
        except Exception as exc:
            why = str(exc)
            low = why.lower()
            hint = ""
            if ("password" in low) or ("auth" in low):
                hint = ("\n  提示：OBS 开着「身份验证」，但没能自动读到密码。两种解法：\n"
                        "        1) OBS → 工具 → WebSocket 服务器设置 → 取消勾选「启用身份验证」\n"
                        "           （只监听本机，够安全）；\n"
                        "        2) 或设环境变量 NSSR_OBS_PASSWORD=你的密码 后再启动。")
            elif ("refused" in low or "10061" in low or "timed out" in low
                  or "timeout" in low or "unreachable" in low or "no route" in low
                  or "operation not permitted" in low):
                hint = (f"\n  提示：{self.host}:{port} 上没有 OBS 在听（跟身份认证无关）。依次确认：\n"
                        "        1) OBS 正在运行；\n"
                        "        2) OBS → 工具 → WebSocket 服务器设置 → 勾选「启用 WebSocket 服务器」；\n"
                        f"        3) 程序和 OBS 在同一台机器上（虚拟机/宿主机算两台；"
                        f"换机器就设 NSSR_OBS_HOST=那台的IP）。")
            raise RuntimeError(f"连不上 OBS WebSocket（{self.host}:{port}）：{why}\n"
                               f"  {_probe_port(self.host, port)}{hint}") from exc
        # 连上之后的挑源/打印，万一再出任何岔子（例如 Windows 上打印源名时编码炸了），
        # 也要把连接关掉再抛 —— 否则 OBS 里就留一个"已连接"的僵尸会话，
        # 表现正是"OBS 显示连着、程序却说连不上"。
        try:
            inputs = self._inputs()
            self.source = self._pick_source(self.want_source, inputs)
            kind = next((i["kind"] for i in inputs if i["name"] == self.source), "")
            self.note = (f"OBS 源「{self.source}」{f'({kind})' if kind else ''} -> "
                         f"{self.width}×{self.height} {self.fmt.upper()} q={self.quality}"
                         f"（{self.host}:{port}，{'已带密码' if password else '无密码'}"
                         f"，配置来自 {cfg_from}）")
            print(f"[取帧] {self.note}")
        except Exception:
            self.close()
            raise
        self.fails, self.dead = 0, False

    @staticmethod
    def _pick_source(source: str | None, inputs: list[dict]) -> str:
        """
        挑源名，优先级：
          1) 传进来的 source（库里/命令行 --obs-source 可显式指定；默认 None 自动识别）
          2) 按 inputKind 自动找采集设备（与界面语言无关）
          3) 常见源名（多语言兜底）
          4) 只有一个源 / 第一个源
        """
        names = [i["name"] for i in inputs]
        if source:                                   # 用户显式给了
            if not names or source in names:
                return source
            if len(names) == 1:                      # 只有一个源就直接用它
                print(f"[提示] 没有「{source}」，改用唯一的源「{names[0]}」")
                return names[0]
            print(f"[提示] OBS 里没有名为「{source}」的源，现有：")
            for n in names:
                print(f"        - {n}")
            return source                            # 保持原样，让取帧报错
        if not inputs:                               # 列不出源：先用第一个候选
            return DEFAULT_SOURCES[0]
        caps = [i for i in inputs
                if any(i["kind"].startswith(k) for k in CAPTURE_INPUT_KINDS)]
        if caps:                                     # 认出来了（不看名字也行）
            if len(caps) > 1:                        # 多个采集设备：优先常见名
                for n in DEFAULT_SOURCES:
                    if n in {i["name"] for i in caps}:
                        return n
                print("[提示] 发现多个采集设备，取第一个：" +
                      "、".join(i["name"] for i in caps))
            return caps[0]["name"]
        for cand in DEFAULT_SOURCES:
            if cand in names:
                return cand
        print(f"[提示] 没认出采集设备，改用第一个源「{names[0]}」，现有：")
        for n in names:
            print(f"        - {n}")
        return names[0]

    def _inputs(self) -> list[dict]:
        """OBS 的输入源列表 [{name, kind}]；kind 是 inputKind，与界面语言无关。"""
        try:
            out = []
            for i in self.cli.get_input_list().inputs:
                n = i.get("inputName", "")
                if n:
                    out.append({"name": n, "kind": i.get("inputKind", "")})
            return out
        except Exception:
            return []

    def read(self):
        """
        取一帧。失败返回 None，并把原因记在 last_error；连续失败到 RECONNECT_FAILS
        就把 dead 置 True —— 调用方看到 dead 应该重建连接（顺便重新挑源）。
        """
        t0 = time.time()
        try:
            r = self.cli.get_source_screenshot(
                self.source, self.fmt, self.width, self.height, self.quality)
        except Exception as exc:
            self._fail(f"{exc}")
            return None
        t_req = time.time() - t0
        try:
            data = getattr(r, "image_data", "") or ""
            if not isinstance(data, str):
                raise TypeError(f"image_data 类型异常：{type(data).__name__}")
            if data.startswith("data:"):
                data = data.split(",", 1)[-1]
            t1 = time.time()
            with Image.open(io.BytesIO(base64.b64decode(data))) as im:
                arr = np.asarray(im.convert("RGB"))
        except Exception:
            self._fail("画面解码失败（JPEG 数据异常）")
            return None
        t_dec = time.time() - t1
        self.last_timing = (t_req, t_dec)      # 诊断用（--verbose 写进日志）
        # 这一拍成功了 → 失败计数归零。**必须清零**：RECONNECT_FAILS 的语义是"**连续**失败"，
        # 以前只在建连时清过，于是"一局里零散失败 3 次"（中间隔着几十分钟正常识别）也会被判死重连，
        # 表现就是莫名其妙卡一下、风速重测。
        self.fails = 0
        return arr

    def _fail(self, why: str) -> None:
        self.last_error = why
        self.fails += 1
        if self.fails >= RECONNECT_FAILS:
            self.dead = True                 # 交给调用方重连；重连会重新挑源
        print(f"\n⚠ 取帧失败（第 {self.fails} 次）：{why}")
        time.sleep(0.2)

    def status_message(self) -> str:
        """给悬浮窗/命令行看的一句人话（带 OBS 原话，便于排障）。"""
        why = (self.last_error or "OBS 没给出原因").replace("\n", " ")
        return f"取不到画面：{why}"

    def close(self) -> None:
        """关掉 WebSocket 会话（重连/退出时都要关，否则 OBS 会话列表里会堆僵尸连接）。"""
        cli, self.cli = self.cli, None
        try:
            cli.disconnect()
        except Exception:
            pass


def make_source(source: str | None = None):
    """按约定建取帧器：源名默认自动识别，地址/端口/密码读 OBS 配置（可用环境变量覆盖）。"""
    cfg = _obs_ws_defaults()
    return ObsWebSocketSource(source=source, host=cfg["host"], port=cfg["port"],
                              password=cfg["password"],
                              width=WIDTH, height=HEIGHT, quality=QUALITY)


# ==========================================================================
# 二、识别状态机（定位一次 + 抽帧识别 + 平滑 + 风速优先）
# ==========================================================================

class Recognizer:
    def __init__(self, args):
        self.args = args
        self.dial: W.Dial | None = None
        self.shape: tuple[int, int] | None = None
        self.last_locate_t = 0.0
        self.no_wind = False
        self.side: str | None = None      # 锁定后的左右手（"右"/"左"），首次定位成功时确定
        self.bad = 0                      # 连续读不出来的次数
        self.last_theta: float | None = None
        self.flip_run = 0                # 连续"反向"次数
        self.last_dir: dict[str, str] = {}   # 横向/纵向上一次的方向（做迟滞用）
        self.ema: float | None = None    # 角度指数平滑值（压抖动）
        self.shown: float | None = None  # 当前显示的角度（带死区，静止时不动）
        self.turn_run = 0                # 连续几拍大幅变化（≥2 判定为真在转视角）
        # ---- 风速锁定（同一洞风速固定）----
        self.locked: float | None = None  # 锁定的风速；None = 未锁（正在测）
        self.votes: dict[int, list] = {}               # 本轮测量：读数 -> [出现次数, 置信度之和]
        self.vote_span: dict[int, list] = {}            # 读数 -> [第一次、最近一次读到它的时刻]（早锁用）
        self.vote_band_ok: dict[int, int] = {}          # 读数 -> "这一拍箭头颜色档认得出"的票数
        self.measure_t0: float | None = None           # 本轮测量的起点（第一次看到风向盘的时刻）
        self.dial_t: float | None = None               # 上一次看到风向盘的时刻（用来算"消失多久"）
        self.break_run = 0                             # 独立读数落在"另一档"的连续拍数（安全网）
        self.value_run = 0                             # 独立读数落在"同一档但另一个值"的连续拍数
        self.value_seen: int | None = None             # 那个"另一个值"（弹候选用）
        self.lock_event = ""                           # 状态切换时留一句话，交给日志（每帧清空）
        self.lock_mismatch = ""                        # 锁定值与这一拍独立读数打架时的说明（每帧清空）
        self._choice: int | None = None                # 玩家点的备选风速（UI 线程写、识别线程读）
        self.choice_options: list | None = None        # 弹出备选时的按钮内容 [[值, 平均置信度, 拍数], ...]
        self.choice_hint = ""                          # 弹窗标题里的那句（"风速拿不准" / "数字没读出"）
        self.ask_since: float | None = None            # 备选弹出来的时刻（超时自动锁用）
        self.band_asked = False                        # "数字读不出"那组备选已经问过且被晾着 → 别再弹
        self.choice_all = False                        # 弹窗是否已从"前几名"展开成"全风档候选"
        self.ocr = W.OcrState()          # 字形模板库状态独立
        n = W.load_glyph_bank(GLYPH_BANK, state=self.ocr)
        if not n:
            # 没库就一定读不出风速 —— 直接报出来（窗口会显示"初始化失败：…"），
            # 别让它静默地一直显示"风速未识别"，那种没人查得出来。
            raise RuntimeError(f"字形模板库没载入（文件缺失或损坏）：{GLYPH_BANK}")
        print(f"[字形] 已载入模板库 {GLYPH_BANK}（{n} 个模板，数字 "
              f"{W.glyph_bank_digits(state=self.ocr)}）")

    # ---- 定位 ----
    def locate(self, rgb) -> bool:
        # 定位只负责找风盘；**锁左右手放在"真读出结果"之后**（见 recognize）——
        # 只看几何的话，一次假风盘就会把侧别锁死在对侧，之后一直读不出来。
        dials = W.find_dials(rgb, verbose=self.args.verbose, side=self.side)
        if not dials:
            return False
        d = dials[0]
        self.dial = d
        self.shape = rgb.shape[:2]
        if self.args.verbose:
            print(f"\n[定位] 风盘 中心=({d.cx:.0f},{d.cy:.0f}) r={d.r:.0f} k={d.k:.3f}"
                  f" 侧别={'右' if d.cx >= rgb.shape[1] / 2.0 else '左'}")
        return True

    def reset(self) -> None:
        """丢掉缓存的定位（保留左右手锁定），下帧重新找 —— 识别出错时用。"""
        self.dial, self.shape, self.bad = None, None, 0
        self._reset_angle_state()

    def _reset_angle_state(self) -> None:
        """清空跟风盘位置/洞相关的角度平滑与迟滞状态。"""
        self.last_theta = None
        self.flip_run = 0
        self.turn_run = 0
        self.ema = self.shown = None
        self.last_dir = {}

    # ---- 风速锁定：同一洞风速固定，锁住它就不会被单帧数字误读带跑 ----
    def _update_lock(self, fr) -> None:
        """
        风速锁定状态机（只在"这一帧确实看到风向盘"时调一次）。

        怎么锁：看到风向盘后先**正常测 WIND_MEASURE_S 秒**，把这几秒里的读数按**值**归并，
        取"置信度之和"最高的那个值锁住 —— 也就是 票数 × 平均置信度。
        另外还有一条**早锁**：满窗口之前，只要"票够多（4 拍）+ 跨度够（0.6s）+ 一边倒
        （平均置信度 ≥0.82、占前两名 ≥0.80）"就立刻锁 —— 4 次/秒下约 1 秒就定，而不是干等 3 秒
        （见 LOCK_EARLY_* 与 _early_lock_ok）。

        为什么不是"单拍置信度最高"：单拍高分可能正好是误读。按"和"排就好：读得又多又像的
        那个值赢。
        为什么不是"纯平均置信度"：纯平均会让**只出现一次的高分误读**压过**连续出现 8 次的
        正确值**（0.95 > 0.86），所以次数必须带进去算。

        锁定期间**仍然每帧独立读一遍数字**（不省这一步），但只当安全网用：
          · 首选拿 **fr.arrow_band（箭头实际量到的颜色档）**跟锁定档比 —— 箭头是一大块纯色，
            比两三个细笔画数字可信得多；连续 LOCK_BREAK_RUN 拍落在另一个档 ⇒ 多半换洞了。
          · 箭头是"通用掩膜"量的（量不出颜色）时，才退回用数字推出来的档。
        """
        now = time.time()
        if self._choice is not None:
            # 玩家刚点了备选（可能是"拿不准"、也可能是"锁定值与读数打架"那张弹窗）——
            # 一律照玩家的选择锁，并把弹窗收掉。放在最前面，锁定状态下也认。
            self.locked, self.break_run, self.value_run = float(self._choice), 0, 0
            self.value_seen = None
            self.lock_event = f"风速按玩家的选择锁定 {self.locked:g} m/s"
            self._clear_choice()
            return
        if self.locked is None:
            if self.measure_t0 is None:
                self.measure_t0 = now                 # 第一次看到风盘 → 本轮测量开始
            if fr.speed is not None:                  # 没读出数字的这一拍不进票（只有区间）
                conf = float((fr.ocr or {}).get("conf") or 0.0)
                val = int(round(fr.speed))
                slot = self.votes.setdefault(val, [0, 0.0])
                slot[0] += 1
                slot[1] += conf
                span = self.vote_span.setdefault(val, [now, now])   # 早锁要判"这批票横跨多久"
                span[1] = now
                if fr.arrow_band is not None:         # 这一拍箭头颜色档认得出 → 记一票"有档"
                    self.vote_band_ok[val] = self.vote_band_ok.get(val, 0) + 1
            if self.votes:
                ranked = sorted(self.votes.items(), key=lambda kv: (-kv[1][1], -kv[1][0]))
                v, (n, s) = ranked[0]
                if self.choice_options is not None:
                    # 备选弹窗一旦出现，只能由玩家点击或超时展开来决定；
                    # 后续帧即使票数变得一边倒，也不能绕过玩家直接锁定。
                    self._expand_choice_if_timeout(now)
                    return
                if n < LOCK_MIN_VOTES:
                    # 只有孤零零一拍读到它 → 先不锁，继续收票（真在瞄准界面的话下一拍就来第二票）。
                    # 单拍读错是最常见的"锁错风速"，而锁错之后整洞都在显示错值，代价太大。
                    return
                full = now - self.measure_t0 >= WIND_MEASURE_S
                if not full and not self._early_lock_ok(ranked, now):
                    return                            # 还没满窗口、也不够一边倒 → 继续收票
                if full and (self._too_close_to_call(ranked)
                             or self.vote_band_ok.get(v, 0) * 2 < n):
                    # 拿不准 → 把备选弹给玩家，等他点；期间继续收票（也继续显示）。
                    # 两条触发条件：
                    #   · 前两名咬得紧（_too_close_to_call）；
                    #   · 这个值的票**过半都没认出箭头颜色档** —— 数字失去了"颜色档交叉校验"
                    #     这个独立证据，等于裸读，按"绝不能错、拿不准让玩家选"一律不自动锁。
                    #     （实测日志里确实有 `掩膜=通用掩膜` 的帧。）
                    # 只有"满窗口"才问人：早锁那条本来就要求一边倒 + 票都有颜色档（见 _early_lock_ok）。
                    if self.choice_options is None:
                        self.choice_options = [[int(k), round(c[1] / max(c[0], 1), 2), int(c[0])]
                                               for k, c in ranked[:CHOICE_MAX]]
                        self.ask_since = now
                        self.choice_hint = ("风速拿不准" if self._too_close_to_call(ranked)
                                            else "箭头颜色档认不出，风速请确认")
                    self._expand_choice_if_timeout(now)
                    return                                # 玩家没点 → 不锁，继续显示待确认状态
                self.locked, self.break_run = float(v), 0
                self._clear_choice()
                losers = "、".join(f"{k}({c[0]}拍/Σ{c[1]:.2f})"
                                   for k, c in sorted(self.votes.items(),
                                                      key=lambda kv: -kv[1][1]) if k != v)
                if not self.lock_event:
                    if full:
                        self.lock_event = (f"风速锁定 {v} m/s（前 {WIND_MEASURE_S:g}s 里 "
                                           f"{n} 拍读到它、置信度和 {s:.2f} 最高"
                                           + (f"；其它 {losers}" if losers else "") + "）")
                    else:
                        sp = self.vote_span.get(v, [now, now])
                        self.lock_event = (f"风速早锁 {v} m/s（{n} 拍一致、跨度 "
                                           f"{sp[1] - sp[0]:.1f}s、平均置信度 {s / max(n, 1):.2f}、"
                                           f"占前两名 {s / max(s + (ranked[1][1][1] if len(ranked) > 1 else 0.0), 1e-9):.2f}"
                                           + (f"；其它 {losers}" if losers else "") + "）")
            elif fr.speed_range and now - (self.measure_t0 or now) >= WIND_MEASURE_S:
                # 数字**一直读不出**、但箭头颜色认得出（风速只给了个区间）——以前这条路锁永远锁不上，
                # 区间会一直挂着。现在把这一档的数字全列出来让玩家点一个（6-10 就列 6 7 8 9 10）。
                if self.choice_options is None and not self.band_asked:
                    lo, hi = fr.speed_range
                    self.choice_options = [[v, 0.0, 0] for v in range(lo, hi + 1)][:CHOICE_MAX]
                    self.ask_since = now
                    self.choice_hint = f"数字没读出（{W.band_name(fr.arrow_band)}档）"
                if self.choice_options is not None:
                    # 5 秒没人点 → 不再"猜一档就收摊"，改成把全部 15 个可能值列出来（按三档分行），
                    # 由玩家点一个 —— 区间本身也只是一个估计，玩家可能知道真实风速。
                    self._expand_choice_if_timeout(now)
            return
        # ---- 已锁定：这一拍的独立读数只当安全网 ----
        band_lock = W.wind_color(self.locked)
        band_now = fr.arrow_band                        # 首选：箭头实际量到的颜色档
        if band_now is None and fr.speed is not None:
            band_now = W.wind_color(fr.speed)           # 退路：通用掩膜量不出颜色时用数字推
        self.lock_mismatch = ""
        if band_now is not None and band_now != band_lock:
            self.break_run += 1
            self.lock_mismatch = (f"锁定 {self.locked:g} m/s（{W.band_name(band_lock)}档），"
                                  f"但这一拍的箭头是 {W.band_name(band_now)}档"
                                  f"（数字读出 {fr.speed if fr.speed is not None else '—'}）")
            if self.break_run >= LOCK_BREAK_RUN:
                self.lock_event = (f"独立读数连续 {self.break_run} 拍落在另一档"
                                   f"（锁定 {self.locked:g} → 现在箭头是 "
                                   f"{W.band_name(band_now)}档）"
                                   f"⇒ 判定换洞，解锁重测")
                self._unlock()
        else:
            self.break_run = 0                        # 同档、或这拍没读出来 → 清零
            # 同档但**值**一直不一样（例如锁了 7、之后稳定读到 8）：跨档那条安全网永远不触发，
            # 显示会一直停在旧值（实测 40 拍都不动）。按"绝不能错、拿不准让玩家选"的原则：
            # 不自己改、也不硬扛 —— 连续 3 秒读到另一个**高置信度**的值就弹候选让玩家裁决。
            if (fr.arrow_band is not None and fr.speed is not None
                    and abs(fr.speed - self.locked) > 1e-9
                    and float((fr.ocr or {}).get("conf") or 0.0) >= LOCK_VALUE_CONF):
                self.value_run += 1
                self.value_seen = int(round(fr.speed))
                self.lock_mismatch = (f"锁定 {self.locked:g} m/s，但连续 {self.value_run} 拍"
                                      f"独立读到 {self.value_seen} m/s（同档）")
                if self.value_run >= LOCK_VALUE_RUN and self.choice_options is None:
                    self.choice_options = [[int(self.locked), 0.0, 0], [self.value_seen, 0.0, 0]]
                    self.ask_since = now
                    self.choice_hint = (f"锁定 {self.locked:g}，读数一直 "
                                        f"{self.value_seen} —— 选哪个？")
            else:
                self.value_run, self.value_seen = 0, None

    def _expand_choice_if_timeout(self, now: float) -> None:
        """
        备选弹窗挂了 CHOICE_TIMEOUT_S 秒还没人点 → 不再自己猜，换成"全风档候选"：
        把 1~15 全部列出来（悬浮窗那边按青/黄/品红三档分三行、颜色跟着档位），玩家点一个即可。

        为什么要这样：原先是"超时按最高分自动锁"，但"拿不准"本身说明证据不足 ——
        按"风速绝不能错"的原则，宁可由玩家点，也不替玩家拍板。
        """
        if self.choice_options is None or self.choice_all:
            return
        if now - (self.ask_since or now) < CHOICE_TIMEOUT_S:
            return
        self.choice_options = [[v, 0.0, 0] for v in range(1, W.MAX_WIND + 1)]
        self.choice_all = True
        self.choice_hint = "拿不准，全部档位都列出来了"

    def _early_lock_ok(self, ranked: list, now: float) -> bool:
        """
        够不够"早锁"（不等满 WIND_MEASURE_S 就定下来）—— 四条同时满足：
          ① 第一名 ≥ LOCK_EARLY_VOTES 拍（4 拍，比"1 拍就锁"的老 bug 严得多）
          ② 这 4 票横跨 ≥ LOCK_EARLY_SPAN_S（0.6s；帧率越高越重要，防止一瞬间连出 4 帧）
          ③ 平均置信度 ≥ LOCK_EARLY_CONF（0.82）—— 弱读（best_effort，带 ? 的）自然被挡住
          ④ 第一名占前两名置信度之和 ≥ LOCK_EARLY_RATIO（0.80）—— 咬得紧就等满窗口再问玩家
        为什么敢比满窗口那条更早定：画面在瞄准界面里基本不动，"多等 2 秒"买不到新信息
        （多出来的样本主要在压噪声）；而**系统性**读错（例如 15 被截成 1）不靠票数防，
        靠箭头颜色档交叉校验 + match_glyphs 的截断保护防。所以 4 拍一致 + 高置信度足够。
        """
        (v, (n, s)) = ranked[0]
        if n < LOCK_EARLY_VOTES:
            return False
        if self.vote_band_ok.get(v, 0) < LOCK_EARLY_VOTES:
            # ⑤ 这 4 票必须**每一拍都认出了箭头颜色档** —— 没有颜色档交叉校验的裸读
            #    不许早锁（否则一旦采集链把颜色洗坏，就没人能拦住一个错误数字）。
            return False
        span = self.vote_span.get(v)
        if not span or span[1] - span[0] < LOCK_EARLY_SPAN_S:
            return False
        if s / max(n, 1) < LOCK_EARLY_CONF:
            return False
        second = ranked[1][1][1] if len(ranked) > 1 else 0.0
        return s / max(s + second, 1e-9) >= LOCK_EARLY_RATIO

    @staticmethod
    def _too_close_to_call(ranked: list) -> bool:
        """
        读数够不够一边倒？不够就让玩家来定（见 CHOICE_CONF / CHOICE_RATIO 的注释）。
        ranked 是 [(值, [拍数, 置信度之和]), ...]，已按置信度之和降序。
        """
        if not ranked:
            return True
        (top_v, (top_n, top_sum)) = ranked[0]
        top_avg = top_sum / max(top_n, 1)
        if top_avg < CHOICE_CONF:
            return True
        if len(ranked) < 2:
            return False                              # 单候选，但绝对置信度已经够高
        second_sum = ranked[1][1][1]
        return top_sum / max(top_sum + second_sum, 1e-9) < CHOICE_RATIO

    def choose(self, value) -> None:
        """
        玩家在悬浮窗上点了一个备选风速（**UI 线程**调用）。

        只写一个属性：读它的识别线程下一拍就会用上。整型/None 的赋值在 CPython 里是原子的，
        这里不需要锁（也刻意不引入锁 —— 识别线程一拍几百毫秒，锁在这个场景只会添乱）。
        """
        self._choice = int(round(float(value)))

    def _clear_choice(self) -> None:
        self._choice = None
        self.choice_options = None
        self.choice_hint = ""
        self.ask_since = None
        self.choice_all = False            # "全风档候选"是否已经展开（同一个弹窗只展开一次）

    def _unlock(self) -> None:
        """解锁：回到"重新测量"状态，等风向盘下次出现再锁一次。"""
        self.locked = None
        self.votes = {}
        self.vote_span = {}
        self.vote_band_ok = {}
        self.measure_t0 = None
        self.break_run = 0
        self.value_run = 0
        self.value_seen = None
        self.lock_mismatch = ""
        self.band_asked = False
        self._clear_choice()
        self._reset_angle_state()

    def _wind_gone(self) -> None:
        """
        这一拍没看到风向盘（在果岭推杆、球在飞、过场动画…）。

        规矩：风向盘消失**超过时限** WIND_UNLOCK_S（暂定 30 秒）就当这一洞打完了
        —— 一洞要所有玩家都打完才进下一洞，那个间隔通常远大于 30 秒 —— 于是解锁风速，
        等下次看见风向盘重新测。没超时限的就是"同一洞里的短暂消失"（挥杆/走位），锁定保持。
        """
        now = time.time()
        if self.dial_t is None:
            self.dial_t = now
            return
        gone = now - self.dial_t
        if self.locked is None and gone > MEASURE_GAP_S and (self.votes or self.measure_t0):
            # 这一轮测量被打断了（挥杆 / 走位 / 切界面）→ 票作废、重新开始量。
            # 不清的话，"6 秒前孤零零一拍读错的 1 m/s"会跟回来后新看到的帧拼在一起，
            # 凑够票数被锁死（实测就是这么锁错整洞的）。
            self.votes, self.vote_span, self.vote_band_ok, self.measure_t0 = {}, {}, {}, None
            self.value_run, self.value_seen = 0, None
            self._clear_choice()
        if gone <= WIND_UNLOCK_S:
            return
        self.dial_t = now                             # 重置，免得后面每拍都重复判一次
        if self.locked is not None:
            self.lock_event = (f"风向盘消失 {gone:.0f}s（>{WIND_UNLOCK_S:g}s）"
                               f"⇒ 判定进入下一洞，解锁风速")
        self._unlock()

    def recognize(self, rgb) -> dict:
        t0 = time.time()
        self.lock_event = ""                      # 每帧清空：状态切换的那一句话只报一次
        self.lock_mismatch = ""                    # 同上：锁定值与这一拍打架的说明
        h, w = rgb.shape[:2]
        if self.dial is None or self.shape != (h, w):
            if time.time() - self.last_locate_t < RELOCATE_S:
                self._wind_gone()
                return {"ok": False, "message": "未找到风盘",
                        "speed_locked": self.locked is not None, "lock_event": self.lock_event}
            self.last_locate_t = time.time()
            if not self.locate(rgb):
                self._wind_gone()
                return {"ok": False, "message": "未找到风盘",
                        "speed_locked": self.locked is not None, "lock_event": self.lock_event}

        d = self.dial
        assert d is not None

        # 缓存里的定位会过期：球一飞出去 / 走位 / 过场动画时，风盘的位置上变成天空、树、
        # 草地，那会儿再按缓存去"读数字"，就会把背景里随便一块白字当成风速读出来（实测
        # 就这么锁过一个假风速）。所以每帧先确认风盘**真的还在**（白环闭合 + 盘内有 Nm/s
        # 数字行，实测 ~10ms）。不在了就当"风盘消失"处理，绝不拿过期定位出结果。
        if not W.gauge_verified(rgb, d):
            self.bad += 1
            if self.bad >= RELOCATE_BAD:          # 连续几帧都对不上 → 丢掉缓存重新找
                self.dial, self.side, self.bad = None, None, 0
                self.ema = self.shown = None
                if self.args.verbose:
                    print("\n[定位] 风盘对不上 → 丢弃缓存，重新定位")
            self._wind_gone()
            return {"ok": False, "message": "未找到风盘",
                    "speed_locked": self.locked is not None, "lock_event": self.lock_event}

        # 风盘本身已经通过校验。即使这一拍没找到箭头/没读出数字，也不能把
        # 这个可见的风盘算成“消失”，否则会把测量投票提前清掉。
        self.dial_t = time.time()
        # 每帧都重新读风速（模板匹配只要 ~4ms，读准了才知道箭头该是什么颜色）
        fr = W.read_wind(rgb, d, ocr_state=self.ocr)
        # fr.ok 且找到箭头就算"这一帧读到了"：只认出颜色档、数字没读出的帧方向照样显示，
        # 不该因为"数字读不出"就去反复重新定位（重定位会清掉角度平滑，显示反而更抖）。
        if fr.ok:
            self.bad = 0
            if self.side is None and fr.speed is not None:   # 真读出风速了才锁侧别
                self.side = "右" if d.cx >= w / 2.0 else "左"
        else:
            self.bad += 1
            if self.bad >= RELOCATE_BAD:      # 一直读不出来：多半是定位错了（或界面切走又切回）
                self.dial, self.side, self.bad = None, None, 0   # 清缓存，下帧重新定位+重判左右
                self.ema = self.shown = None
                if self.args.verbose:
                    print("\n[定位] 连续读不出来 → 丢弃缓存，重新定位")
            # 读不出来也可能是"风盘已经不在了"（在果岭推杆 / 球在飞）→ 走消失计时
            self._wind_gone()
        if fr.ocr is not None and fr.ocr.get("value") is None and self.args.verbose:
            print("\n[风读] 未读出风速（等待下一次读取）")
        # 只有"风速、区间、方向**全都没有**"才算整帧失败 —— 只要箭头找到了（哪怕是"不看色相"
        # 的通用掩膜量出来的），方向就该照给，风速显示 ? m/s 也比整条丢掉强。
        if fr.speed is None and not fr.speed_range and fr.arrow is None:
            return {"ok": False, "message": "风速未识别（等待下一次读取）",
                    "dial": d, "ocr": fr.ocr,
                    "speed_locked": self.locked is not None, "lock_event": self.lock_event}
        if fr.no_wind:
            self.no_wind = True
            if self.locked is not None:
                # 锁着 7 m/s 却读到 0 m/s：同一洞风速不会变，所以这基本只有两种可能 ——
                # 真换洞了（新洞无风），或者数字被读成了 0。两种情况都该解锁重测。
                self.lock_event = (f"读到 0 m/s（盘里没有风向箭头）⇒ 解锁风速"
                                   f"（原先锁的是 {self.locked:g}）")
                self._unlock()
            return {"ok": True, "no_wind": True, "speed": 0.0, "dial": d,
                    "message": fr.message,
                    "speed_locked": False, "lock_event": self.lock_event}
        if not fr.ok:
            return {"ok": False, "message": fr.message, "speed": fr.speed, "dial": d,
                    "speed_locked": self.locked is not None, "lock_event": self.lock_event}

        # ---- 这一帧确实看到了风向盘：先更新锁定状态，再用锁定值覆盖风速 ----
        self._update_lock(fr)
        if self.locked is not None:
            # 锁上了：风速用锁定值（数字这一拍读成什么、读没读出来都不影响显示），
            # 方向照常每帧实测 —— θ 是"相对瞄准方向"的夹角，本来就会跟着转身变化。
            fr.speed, fr.speed_note, fr.speed_range = self.locked, "", None
        elif self.choice_options:
            # 拿不准、正等玩家从备选里点一个：显示先钉在**第一候选**上（带 "?"），
            # 不显示这一拍的原始读数 —— 否则玩家看按钮的同时数字还在乱跳。
            fr.speed = float(self.choice_options[0][0])
            fr.speed_note, fr.speed_range = "?", None

        # 反 180° 保护 → 静止取平均（抖动抑制）
        theta = self._flip_guard(fr.theta, fr.arrow.conf if fr.arrow else 1.0)
        theta = self._steady_theta(theta)

        wind = W.wind_metrics(fr.speed, theta)
        self._hold_dir(wind)
        return {"ok": True, "speed": fr.speed, "phi": fr.phi, "k": fr.k, "theta": theta,
                "wind": wind, "dial": d, "arrow": fr.arrow,
                # 低置信度的显示语义：speed_note="?" 表示数字只是勉强过线（显示成 "7? m/s"）；
                # speed_range 表示只认出颜色档（显示成 "6-10 m/s"，分量算不了）；cands 供日志排查
                "speed_note": fr.speed_note, "speed_range": fr.speed_range,
                "cands": fr.cands,
                "speed_locked": self.locked is not None, "lock_event": self.lock_event,
                "lock_mismatch": self.lock_mismatch,
                "arrow_band": fr.arrow_band,
                # 只要有待确认的备选就弹 —— 锁定时也会弹（"锁定值与读数一直打架"那种情况），
                # 所以别再拿 locked is None 来挡。
                "need_choice": bool(self.choice_options),
                "choice_options": self.choice_options or [],
                "choice_hint": self.choice_hint,
                "latency_ms": (time.time() - t0) * 1000.0}

    # ---- 抖动抑制：静止时取平均并定住 ----
    def _steady_theta(self, theta: float) -> float:
        """
        抖动抑制：实机测静止时角度每拍抖 0.6~2.0°（横风经系数放大后就看得出在跳），
        但手柄微调方向时每拍只挪零点几度 —— 两者必须分开对待。

        做法（三步，按优先级）：
          ① **快转直通**：连续两拍都变化 > SMOOTH_JUMP → 认定真的在转视角，输出原始值，
             不平滑（保证跟手）；"连着几拍"按 TURN_RUN（≈1 秒，随 FPS 换算）算，不是写死两拍。
          ② **单拍大跳丢弃**：只有一拍变化很大 → 多半是这一帧箭头测歪了，丢掉它，
             保持上一拍的显示（免得显示被一帧噪声带跳）；
          ③ **EMA + 显示死区**：正常情况用指数平均压噪声（σ 从 ~0.5° 压到 ~0.2°），
             再要求"EMA 偏离显示值 > SMOOTH_STEP 才更新显示" —— 于是静止时显示纹丝不动，
             而微调造成的是**持续偏移**，EMA 会跟过去、显示每隔一小会儿挪一格，能跟上。
        """
        if self.ema is None:
            self.ema = self.shown = theta
            return theta
        d = self._ang_diff(theta, self.ema)
        self.turn_run = self.turn_run + 1 if d > SMOOTH_JUMP else 0
        if self.turn_run >= TURN_RUN:                 # ① 真的在转 → 直通
            self.ema = self.shown = theta
            return theta
        if d > SMOOTH_JUMP:                          # ② 单拍跳变 → 丢弃这一拍
            return self.shown
        dd = ((theta - self.ema + 180.0) % 360.0) - 180.0
        self.ema = (self.ema + SMOOTH_A * dd) % 360.0
        if self._ang_diff(self.ema, self.shown) > SMOOTH_STEP:   # ③ 死区：挪够了才改显示
            self.shown = self.ema
        return self.shown

    @staticmethod
    def _ang_diff(a: float, b: float) -> float:
        """两个角度之间的最小夹角（度，0~180）。"""
        return abs(((a - b + 180.0) % 360.0) - 180.0)

    @staticmethod
    def _circ_median(deg: list[float]) -> float:
        """
        圆周中位数：以圆周均值为参考取偏差的中位数。

        比均值抗单帧跳变 —— 某一拍测歪 8°，均值会被拽走 1~2°，中位数几乎不动。
        """
        m = Recognizer._circ_mean(deg)
        devs = sorted(((d - m + 180.0) % 360.0) - 180.0 for d in deg)
        return (m + devs[len(devs) // 2]) % 360.0

    @staticmethod
    def _circ_mean(deg: list[float]) -> float:
        """角度平均走圆周：179° 和 -179° 的平均是 180°，不是 0°。"""
        x = sum(math.cos(math.radians(d)) for d in deg)
        y = sum(math.sin(math.radians(d)) for d in deg)
        return math.degrees(math.atan2(y, x))

    @staticmethod
    def _ang_span(deg: list[float]) -> float:
        """一组角度相对其圆心的最大跨度（度）。"""
        m = Recognizer._circ_mean(deg)
        devs = [((d - m + 180.0) % 360.0) - 180.0 for d in deg]
        return max(devs) - min(devs)

    def _hold_dir(self, wind: dict) -> None:
        """
        方向迟滞：|分量| < HYST 时沿用上一次的方向。

        θ 压在 0°/90°/180° 附近时，横向或纵向会在 ±0.0x 之间来回变号；数字照实显示，
        但箭头不再跟着抖 —— 只有 |值| 真的超过 HYST 才翻转。
        """
        for abs_key, dir_key, txt_key in (("lateral_abs", "lateral_dir", "lateral_text"),
                                          ("longitudinal_abs", "longitudinal_dir", "longitudinal_text")):
            val = wind.get(abs_key)
            if val is None:
                continue
            prev = self.last_dir.get(dir_key)
            if val >= HYST or prev is None:      # 明确变化 / 还没有历史 → 采用当前方向
                prev = wind[dir_key]
            else:                                # 抖动区 → 沿用上一次
                wind[dir_key] = prev
            self.last_dir[dir_key] = prev
            wind[txt_key] = f"{val:.2f} m/s {wind[dir_key]}"

    def _flip_guard(self, theta: float, conf: float) -> float:
        """
        反 180° 保护：本轮方向置信度低、又跟上一帧几乎反向时，先按住上一帧的
        方向（连续 FLIP_RUN 次都反才认），免得"顺逆风偶尔读反一下"。
        """
        prev, out = self.last_theta, theta
        if prev is not None and conf < FLIP_CONF:
            if abs((theta - prev + 180.0) % 360.0 - 180.0) > 120.0:
                self.flip_run += 1
                if self.flip_run < FLIP_RUN:
                    out = (theta + 180.0) % 360.0
                self.last_theta = out
                return out
        self.flip_run = 0
        self.last_theta = out
        return out


# ==========================================================================
# 三、输出
# ==========================================================================

def _num(v, fmt: str = "g", dash: str = "—") -> str:
    """安全格式化"可能是 None"的数值。终端/日志是旁路，不该因为一个字段缺失就抛异常。"""
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return dash


def _deg(v) -> str:
    """角度：有值给 "351.6°"，没有就给一个 "—"（不留孤零零的度数符号）。"""
    return "—" if v is None else f"{float(v):.1f}°"


def line_for(st: dict) -> str:
    if st.get("no_wind"):
        return "风速 0 m/s —— 无风，跳过计算（等风速变化）"
    if not st.get("ok"):
        sp = st.get("speed")
        tail = f"（风速 {sp:g} m/s）" if sp is not None else ""
        return f"{st.get('message', '识别中…')}{tail}"
    w = st["wind"]
    rng = st.get("speed_range")
    if st.get("speed") is None and rng:
        return (f"风速 {rng[0]}-{rng[1]} m/s（数字未读出，按箭头颜色档显示）| "
                f"夹角 {_deg(w.get('theta_deg'))} | 横向/纵向算不了")
    if "lateral_abs" not in w:                     # 风速没定就没有分量，别在这里 KeyError
        return f"夹角 {_deg(w.get('theta_deg'))} | 横向/纵向算不了（风速未定）"
    return (f"风 {_num(w.get('speed'))} m/s | 夹角 {_deg(w.get('theta_deg'))} | "
            f"横向 {w['lateral_abs']:.2f} {w['lateral_dir']} | "
            f"纵向 {w['longitudinal_abs']:.2f} {w['longitudinal_dir']}")


def print_test_report(st: dict, rgb) -> None:
    print("\n" + "=" * 60)
    print("单次识别测试（-t / --test）")
    print("=" * 60)
    print(f"帧尺寸      : {rgb.shape[1]}×{rgb.shape[0]}")
    d = st.get("dial")
    if d is not None:
        print(f"风盘        : 中心=({d.cx:.0f},{d.cy:.0f})  半径={d.r:.0f}px  k={d.k:.3f}")
    if st.get("no_wind"):
        print("结果        : 风速 0 m/s —— 无风，按规则不计算夹角/横向/纵向")
    elif not st.get("ok"):
        print(f"结果        : {st.get('message')}")
        sp = st.get("speed")
        if sp is not None:
            print(f"风速        : {sp:g} m/s")
        ocr = st.get("ocr") or {}
        if ocr.get("raw"):
            print(f"读数字      : {ocr['raw']}  选框={ocr.get('box')}")
    else:
        w = st["wind"]
        sp, rng, note = st.get("speed"), st.get("speed_range"), st.get("speed_note") or ""
        if sp is None and rng:
            # 只认出颜色档：风速降级成区间（方向照给，分量算不了）
            print(f"风速        : {rng[0]}-{rng[1]} m/s（数字未读出，按箭头颜色档显示）")
        elif sp is None:
            print("风速        : 未读出（方向照给）")
        else:
            print(f"风速        : {_num(sp)}{'?' if note else ''} m/s")
        if st.get("cands"):
            print("数字候选    : " + "  ".join(
                f"{v}{'' if strong else '~'}({s:.2f},{n or '无档'})"
                for v, s, strong, n in st["cands"][:5]))
        print(f"画面夹角 φ  : {st['phi']:.1f}°   （俯视压缩 k={st['k']:.3f}）")
        print(f"风向夹角 θ  : {_deg(w.get('theta_deg'))}  {w.get('compass', '—')}"
              f"   （cos={_num(w.get('cos'), '.4f')}  sin={_num(w.get('sin'), '.4f')}）")
        if "lateral_abs" in w:
            print(f"横向风速    : {w['lateral_abs']:.2f} m/s {w['lateral_dir']}")
            print(f"纵向风速    : {w['longitudinal_abs']:.2f} m/s {w['longitudinal_dir']}")
        else:
            print("横向/纵向   : 风速未定（只有颜色档）→ 算不了")
        print(f"耗时        : {st.get('latency_ms', 0):.0f} ms")
    if not st.get("ok") and st.get("dial") is None:
        print("\n提示：请让游戏停在「瞄准 / 击球前」的画面（右上/左上角有风盘），再跑一次。")
    print("=" * 60)


# ==========================================================================
# 四、命令行
# ==========================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="OBS 直播风向识别（obs-websocket 抓源，按 FPS 识别）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例：\n"
               "  python3 obs_live_wind_reader.py                # 挂机识别\n"
               "  python3 obs_live_wind_reader.py -t             # 抓一帧做一次识别\n"
               "  python3 obs_live_wind_reader.py --obs-source \"视频采集设备\"\n")
    p.add_argument("-t", "--test", action="store_true",
                   help="测试模式：抓一帧做一次识别，打印详情后退出")
    p.add_argument("-v", "--verbose", action="store_true", help="打印定位、耗时等细节")
    p.add_argument("--obs-source", default=None,
                   help="OBS 源名（默认按采集设备类型自动识别）")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    try:
        src = make_source(args.obs_source)
    except Exception as exc:
        print(f"✗ 取帧失败：{exc}")
        print("  排查：OBS → 工具 → WebSocket 服务器设置 → 勾选「启用 WebSocket 服务器」")
        print("        （端口/密码自动读 OBS 配置；源可自动识别，或用 --obs-source 指定）")
        return 2

    rec = Recognizer(args)

    # ---------- 测试模式：抓一帧，识别一次 ----------
    if args.test:
        print("[测试] 抓取一帧…")
        rgb = None
        for _ in range(5):
            rgb = src.read()
            if rgb is not None:
                break
            time.sleep(0.3)
        src.close()
        if rgb is None:
            print(f"✗ 抓不到画面：{src.last_error or 'OBS 没给出原因'}")
            print("  排查：OBS 里该源在不在当前场景、有没有输出画面")
            return 2
        st = rec.recognize(rgb)
        print_test_report(st, rgb)
        return 0 if st.get("ok") else 3

    # ---------- 实时循环：按 FPS 识别 ----------
    min_dt = 1.0 / FPS
    print(f"[识别] 按 {FPS:g} 次/秒  （Ctrl-C 退出）")
    frames = reads = ok_frames = 0
    t_start = last_print = last_rec = time.time()
    st: dict = {"ok": False, "message": "启动中…"}
    try:
        while True:
            # 限速放在取帧之前：否则没到点的帧也要白读一次，挤占识别预算
            now = time.time()
            if min_dt:
                wait = last_rec + min_dt - now
                if wait > 0:
                    time.sleep(min(wait, 0.02))
                    continue
            last_rec = now          # 立刻打点：取帧时间也算进本帧预算
            rgb = src.read()
            if rgb is None:
                if src.dead:        # 连接死了 / 源变了 → 重新连（顺便重新挑源）
                    src.close()
                    while True:
                        try:
                            src = make_source(args.obs_source)
                            break
                        except Exception as exc:
                            print(f"\n⚠ 重连失败：{exc}（3 秒后再试，Ctrl-C 退出）")
                            time.sleep(3.0)
                    continue
                time.sleep(0.02)
                continue
            reads += 1
            frames += 1
            st = rec.recognize(rgb)
            if args.verbose:
                print(f"\n[帧{frames}] 耗时={st.get('latency_ms', 0):.0f}ms")
            if st.get("ok"):
                ok_frames += 1
            if now - last_print >= 0.2:
                last_print = now
                fps = frames / max(now - t_start, 1e-6)
                sys.stdout.write(f"\r[{fps:5.1f} 次/秒] {line_for(st)}" + " " * 6)
                sys.stdout.flush()
    except KeyboardInterrupt:
        print()
    finally:
        src.close()
        dt = max(time.time() - t_start, 1e-6)
        print(f"\n[汇总] 取帧 {reads} 帧 / 识别 {frames} 次（成功 {ok_frames} 次）/ "
              f"平均 {frames / dt:.1f} 次每秒 / 用时 {dt:.1f}s")
        sys.stdout.flush()          # 重定向到文件时也要能立刻看到汇总
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
