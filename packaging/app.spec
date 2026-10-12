# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller 打包配置（macOS 独立应用）。用 ./packaging/build_app.sh 调用。

要改什么直接改这里：
  * 版本号        → 改项目根目录的 version.py（一处生效：Info.plist + 程序界面/日志）
  * 应用图标      → 放 packaging/app.icns（没有就用 PyInstaller 默认图标）
  * 精简/排除模块 → excludes
"""
import os

APP_NAME = "NSSR风向识别"
BUNDLE_ID = "com.rezenon3ko.nssr-wind-reader"

ROOT = os.path.abspath(os.path.join(SPECPATH, os.pardir))     # 项目根目录

# 版本号从 version.py 读（和程序界面显示的是同一个来源）
_ver_ns: dict = {}
with open(os.path.join(ROOT, "version.py"), encoding="utf-8") as _fh:
    exec(_fh.read(), _ver_ns)
APP_VERSION = _ver_ns["__version__"]

ENTRY = os.path.join(ROOT, "overlay.py")                      # 入口：悬浮窗
GLYPHS = os.path.join(ROOT, "glyph_bank.npz")                 # 字形模板库（运行时必需）
ICON = os.path.join(SPECPATH, "app.icns")
if not os.path.exists(ICON):
    ICON = None

a = Analysis(
    [ENTRY],
    pathex=[ROOT],
    binaries=[],
    datas=[(GLYPHS, ".")],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=["matplotlib", "scipy", "pandas", "IPython", "pytest", "tkinter.test"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    strip=False,
    upx=False,
    console=False,                       # 不弹终端（= 命令行的 --windowed）
    icon=ICON,
)

coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name=APP_NAME)

app = BUNDLE(
    coll,
    name=f"{APP_NAME}.app",
    icon=ICON,
    bundle_identifier=BUNDLE_ID,
    version=APP_VERSION,
    info_plist={
        "CFBundleShortVersionString": APP_VERSION,   # Finder 里显示的版本号
        "CFBundleVersion": APP_VERSION,              # 构建号
        "CFBundleDisplayName": APP_NAME,
        "NSHighResolutionCapable": True,
    },
)
