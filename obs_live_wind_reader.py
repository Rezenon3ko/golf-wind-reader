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
FLIP_CONF = 0.6                    # 方向置信度低于此值时才做反 180° 保护
FLIP_RUN = 3                       # 连续这么多次"反向"才认为真的转向了
STEADY_DEG = 2.5                   # 最近几拍角度都在这个跨度内 → 判定画面静止（实测静止抖动 0.6~1.4°）
STEADY_MIN = 3                     # 至少攒这么多拍才开始锁定
STEADY_N = 5                       # 参与平均的最近读数个数（2 次/秒 → 约 2.5 秒）
RELEASE_DEG = 3.0                  # 离锁定值超过它、且连续 RELEASE_RUN 拍 → 解锁（真的转了）
RELEASE_RUN = 2                    # 连续两拍才解锁：单拍跳一下不放开，免得显示又抖起来


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
        self.recent: list[float] = []    # 最近几拍的真实夹角（做"静止取平均"用）
        self.held: float | None = None   # 静止时"定住"的输出角度
        self.out_run = 0                 # 连续几拍明显偏离锁定值（达到 RELEASE_RUN 才解锁）
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
        self.recent.clear()

    def recognize(self, rgb) -> dict:
        t0 = time.time()
        h, w = rgb.shape[:2]
        if self.dial is None or self.shape != (h, w):
            if time.time() - self.last_locate_t < RELOCATE_S:
                return {"ok": False, "message": "未找到风盘"}
            self.last_locate_t = time.time()
            if not self.locate(rgb):
                return {"ok": False, "message": "未找到风盘"}

        d = self.dial
        assert d is not None

        # 每帧都重新读风速（模板匹配只要 ~4ms，读准了才知道箭头该是什么颜色）
        fr = W.read_wind(rgb, d, ocr_state=self.ocr)
        if fr.ok and fr.speed is not None:
            self.bad = 0
            if self.side is None:                 # 真读出结果了才锁侧别，之后只搜这一侧
                self.side = "右" if d.cx >= w / 2.0 else "左"
        else:
            self.bad += 1
            if self.bad >= RELOCATE_BAD:      # 一直读不出来：多半是定位错了（或界面切走又切回）
                self.dial, self.side, self.bad = None, None, 0   # 清缓存，下帧重新定位+重判左右
                self.recent.clear()
                if self.args.verbose:
                    print("\n[定位] 连续读不出来 → 丢弃缓存，重新定位")
        if fr.ocr is not None and fr.ocr.get("value") is None and self.args.verbose:
            print("\n[风读] 未读出风速（等待下一次读取）")
        if fr.speed is None:
            return {"ok": False, "message": "风速未识别（等待下一次读取）",
                    "dial": d, "ocr": fr.ocr}
        if fr.no_wind:
            self.no_wind = True
            return {"ok": True, "no_wind": True, "speed": 0.0, "dial": d,
                    "message": fr.message}
        if not fr.ok:
            return {"ok": False, "message": fr.message, "speed": fr.speed, "dial": d}

        # 反 180° 保护 → 静止取平均（抖动抑制）
        theta = self._flip_guard(fr.theta, fr.arrow.conf if fr.arrow else 1.0)
        theta = self._steady_theta(theta)

        wind = W.wind_metrics(fr.speed, theta)
        self._hold_dir(wind)
        return {"ok": True, "speed": fr.speed, "phi": fr.phi, "k": fr.k, "theta": theta,
                "wind": wind, "dial": d, "arrow": fr.arrow,
                "latency_ms": (time.time() - t0) * 1000.0}

    # ---- 抖动抑制：静止时取平均并定住 ----
    def _steady_theta(self, theta: float) -> float:
        """
        画面不动时，风向箭头每拍仍有 0.6~1.4° 的测量噪声（实机测的），横风经系数
        放大后就能看出在跳。这里做**"锁定-保持"**：

          · 未锁定：最近 STEADY_N 拍角度跨度 ≤ STEADY_DEG → 取这几拍的圆周平均**锁定**；
          · 已锁定：输出**一直不动**（这才是"定住"），忽略期间所有小抖动；
          · 解锁条件很严格：**连续 RELEASE_RUN 拍**都离锁定值超过 RELEASE_DEG 才放开 ——
            单拍跳一下不理会，免得显示又跟着抖。转视角时每拍动好几度，两拍就解锁，
            所以照样跟手（解锁后立刻用原始值显示）。
        """
        if self.held is not None:
            if self._ang_diff(theta, self.held) > RELEASE_DEG:
                self.out_run += 1
                if self.out_run >= RELEASE_RUN:          # 连续两拍明显偏开 → 真的转了
                    self.held, self.out_run, self.recent = None, 0, []
            else:
                self.out_run = 0
            if self.held is not None:
                return self.held                         # 锁定期间输出纹丝不动
        self.recent.append(theta)
        if len(self.recent) > STEADY_N:
            self.recent.pop(0)
        if len(self.recent) >= STEADY_MIN and self._ang_span(self.recent) <= STEADY_DEG:
            self.held = self._circ_mean(self.recent) % 360.0   # 与原始 θ 一样用 0~360 口径
            self.out_run = 0
            return self.held
        return theta

    @staticmethod
    def _ang_diff(a: float, b: float) -> float:
        """两个角度之间的最小夹角（度，0~180）。"""
        return abs(((a - b + 180.0) % 360.0) - 180.0)

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

def line_for(st: dict) -> str:
    if st.get("no_wind"):
        return "风速 0 m/s —— 无风，跳过计算（等风速变化）"
    if not st.get("ok"):
        sp = st.get("speed")
        tail = f"（风速 {sp:g} m/s）" if sp is not None else ""
        return f"{st.get('message', '识别中…')}{tail}"
    w = st["wind"]
    return (f"风 {w['speed']:g} m/s | 夹角 {w['theta_deg']:.1f}° | "
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
        print(f"风速        : {w['speed']:g} m/s")
        print(f"画面夹角 φ  : {st['phi']:.1f}°   （俯视压缩 k={st['k']:.3f}）")
        print(f"风向夹角 θ  : {w['theta_deg']:.1f}°  {w['compass']}"
              f"   （cos={w['cos']:.4f}  sin={w['sin']:.4f}）")
        print(f"横向风速    : {w['lateral_abs']:.2f} m/s {w['lateral_dir']}")
        print(f"纵向风速    : {w['longitudinal_abs']:.2f} m/s {w['longitudinal_dir']}")
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
