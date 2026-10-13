#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OBS 直播风向识别（obs-websocket 抓源，固定 1 次/秒）

先在 OBS 里启用 WebSocket 服务器（OBS 28+ 默认已启用），再运行：

    python3 obs_live_wind_reader.py                # 挂机识别
    python3 obs_live_wind_reader.py -t             # 抓一帧识别一次，打印详情后退出
    python3 obs_live_wind_reader.py --obs-source "视频采集设备"

取帧走 obs-websocket 的 GetSourceScreenshot（1920×1080 JPEG q100）：
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

FPS = 2.0                          # 识别频率（次/秒）；越高越跟手、CPU 线性增加（每帧约 70ms）
QUALITY = 100                      # OBS 抓图 JPEG 质量 —— 固定 100（实测 80→100 少 3/4 的偶发反向）
OBS_HOST = "127.0.0.1"             # OBS WebSocket 地址（端口/密码读 OBS 自己的配置）
REQ_TIMEOUT = 5                    # 单次请求超时（秒）；只是兜底，别设太小（慢机器单次取帧可能要 2~3 秒）
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
RELOCATE_BAD = 6                   # 连续这么多帧读不出来就丢掉缓存的定位，重新找一次
HYST = 0.05                        # 横向/纵向 |值| 小于它就沿用上一次方向（0 附近不抖）

# ---- 风速锁定：同一洞的风速是固定的，锁住它就不会因为某一拍数字读错而乱跳 ----
WIND_MEASURE_S = 5.0               # 看到风向盘后先正常测这么久，再从这几拍里挑"置信度最高"的锁住。
# ↑ 2 次/秒 ⇒ 大约 10 拍样本。（试过压到 3 秒想过"有人瞄得快"这一层，结论是 5 秒的样本量
#   值得留 —— 真正该解决"瞄得快"的办法是"某一拍置信度特别高就立刻锁"，而不是缩小窗口。）
WIND_UNLOCK_S = 30.0               # 风向盘消失超过这么久（时限，暂定）→ 判定进入下一洞 → 解锁重新测
LOCK_BREAK_RUN = 6                 # 锁定期间"独立读数连续落在另一档"这么多拍 → 认为换洞了（安全网）
FLIP_CONF = 0.6                    # 方向置信度低于此值时才做反 180° 保护
FLIP_RUN = 3                       # 连续这么多次"反向"才认为真的转向了
SMOOTH_A = 0.25                    # 角度指数平滑系数：越小越稳、越大越跟手
                                   # 实测扫描（81 拍静止数据）：0.35→0.25 让显示改动量降到 1/3，
                                   # 微调首次响应只慢 0.2 秒；再降到 0.2 稳定性不变但更钝，故取 0.25
SMOOTH_JUMP = 5.0                  # 单拍变化超过它：连着两拍 = 真在转（直通），只一拍 = 测歪了（丢弃）
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
          1) 传进来的 source（库里可显式指定；命令行版固定传 None）
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
        data = getattr(r, "image_data", "") or ""
        if data.startswith("data:"):
            data = data.split(",", 1)[-1]
        try:
            t1 = time.time()
            with Image.open(io.BytesIO(base64.b64decode(data))) as im:
                arr = np.asarray(im.convert("RGB"))
        except Exception:
            self._fail("画面解码失败（JPEG 数据异常）")
            return None
        t_dec = time.time() - t1
        self.last_timing = (t_req, t_dec)      # 诊断用（--verbose 写进日志）
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


def make_source():
    """按默认约定建取帧器：源名自动识别、地址/端口/密码读 OBS 配置（可用环境变量覆盖）。"""
    cfg = _obs_ws_defaults()
    return ObsWebSocketSource(source=None, host=cfg["host"], port=cfg["port"],
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
        self.best: tuple[float, float] | None = None   # 本轮"正常测量"里置信度最高的一拍 (conf, speed)
        self.measure_t0: float | None = None           # 本轮测量的起点（第一次看到风向盘的时刻）
        self.dial_t: float | None = None               # 上一次看到风向盘的时刻（用来算"消失多久"）
        self.break_run = 0                             # 独立读数落在"另一档"的连续拍数（安全网）
        self.lock_event = ""                           # 状态切换时留一句话，交给日志（每帧清空）
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
        self.ema = self.shown = None

    # ---- 风速锁定：同一洞风速固定，锁住它就不会被单帧数字误读带跑 ----
    def _update_lock(self, fr) -> None:
        """
        风速锁定状态机（只在"这一帧确实看到风向盘"时调一次）。

        怎么锁：看到风向盘后先**正常测 WIND_MEASURE_S 秒**，把这几拍里**置信度最高**的
        那个读数定下来 —— 同一洞风速固定，"读得最像的那一拍"就是对这一洞最可信的估计。

        锁定期间**仍然每帧独立读一遍数字**（不省这一步），但只当安全网用：独立读数连续
        LOCK_BREAK_RUN 拍都落在**另一个颜色档**（箭头颜色都变了）⇒ 多半已经换洞、只是
        风向盘没消失够久 ⇒ 解锁重测。只比"档"不比具体数值：数字误读一般发生在同一档内
        （6/7/8 都是黄档），不该拿它去推翻锁定；真换洞时箭头颜色往往也跟着换。
        """
        now = time.time()
        if self.locked is None:
            if self.measure_t0 is None:
                self.measure_t0 = now                 # 第一次看到风盘 → 本轮测量开始
            conf = float((fr.ocr or {}).get("conf") or 0.0)
            if fr.speed is not None and (self.best is None or conf > self.best[0]):
                self.best = (conf, float(fr.speed))
            if self.best is not None and now - self.measure_t0 >= WIND_MEASURE_S:
                self.locked, self.break_run = self.best[1], 0
                self.lock_event = (f"风速锁定 {self.locked:g} m/s"
                                   f"（前 {WIND_MEASURE_S:g}s 置信度最高的一拍，"
                                   f"{self.best[0]:.2f}）")
            return
        band_lock = W.wind_color(self.locked)
        band_now = W.wind_color(fr.speed) if fr.speed is not None else None
        if band_now is not None and band_now != band_lock:
            self.break_run += 1
            if self.break_run >= LOCK_BREAK_RUN:
                self.lock_event = (f"独立读数连续 {self.break_run} 拍落在另一档"
                                   f"（锁定 {self.locked:g} → 现在读 {fr.speed:g}）"
                                   f"⇒ 判定换洞，解锁重测")
                self._unlock()
        else:
            self.break_run = 0                        # 同档、或这拍没读出来 → 清零

    def _unlock(self) -> None:
        """解锁：回到"重新测量"状态，等风向盘下次出现再锁一次。"""
        self.locked = None
        self.best = None
        self.measure_t0 = None
        self.break_run = 0

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
        if fr.speed is None and not fr.speed_range:
            return {"ok": False, "message": "风速未识别（等待下一次读取）",
                    "dial": d, "ocr": fr.ocr,
                    "speed_locked": self.locked is not None, "lock_event": self.lock_event}
        if fr.no_wind:
            self.no_wind = True
            self.dial_t = time.time()             # 0 m/s 也是"风盘在"：别把它算成消失
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
        self.dial_t = time.time()
        self._update_lock(fr)
        if self.locked is not None:
            # 锁上了：风速用锁定值（数字这一拍读成什么、读没读出来都不影响显示），
            # 方向照常每帧实测 —— θ 是"相对瞄准方向"的夹角，本来就会跟着转身变化。
            fr.speed, fr.speed_note, fr.speed_range = self.locked, "", None

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
                "latency_ms": (time.time() - t0) * 1000.0}

    # ---- 抖动抑制：静止时取平均并定住 ----
    def _steady_theta(self, theta: float) -> float:
        """
        抖动抑制：实机测静止时角度每拍抖 0.6~2.0°（横风经系数放大后就看得出在跳），
        但手柄微调方向时每拍只挪零点几度 —— 两者必须分开对待。

        做法（三步，按优先级）：
          ① **快转直通**：连续两拍都变化 > SMOOTH_JUMP → 认定真的在转视角，输出原始值，
             不平滑（保证跟手）；
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
        if self.turn_run >= 2:                       # ① 真的在转 → 直通
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
                f"夹角 {w['theta_deg']:.1f}° | 横向/纵向算不了")
    if "lateral_abs" not in w:                     # 风速没定就没有分量，别在这里 KeyError
        return f"夹角 {w['theta_deg']:.1f}° | 横向/纵向算不了（风速未定）"
    return (f"风 {_num(w.get('speed'))} m/s | 夹角 {w['theta_deg']:.1f}° | "
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
        print(f"风向夹角 θ  : {w['theta_deg']:.1f}°  {w['compass']}"
              f"   （cos={w['cos']:.4f}  sin={w['sin']:.4f}）")
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
        description="OBS 直播风向识别（obs-websocket 抓源，固定 1 次/秒）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例：\n"
               "  python3 obs_live_wind_reader.py                # 挂机识别\n"
               "  python3 obs_live_wind_reader.py -t             # 抓一帧做一次识别\n")
    p.add_argument("-t", "--test", action="store_true",
                   help="测试模式：抓一帧做一次识别，打印详情后退出")
    p.add_argument("-v", "--verbose", action="store_true", help="打印定位、耗时等细节")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    try:
        src = make_source()
    except Exception as exc:
        print(f"✗ 取帧失败：{exc}")
        print("  排查：OBS → 工具 → WebSocket 服务器设置 → 勾选「启用 WebSocket 服务器」")
        print("        （端口/密码自动读 OBS 配置；源按类型自动识别，不用指定名字）")
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

    # ---------- 实时循环：固定 1 次/秒 ----------
    min_dt = 1.0 / FPS
    print(f"[识别] 固定 {FPS:g} 次/秒  （Ctrl-C 退出）")
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
                            src = make_source()
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
