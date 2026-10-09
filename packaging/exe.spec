# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller 打包配置（Windows x64 单文件 exe）。用 packaging/build_exe.bat 调用，
也可以直接 `pyinstaller --noconfirm --clean packaging/exe.spec`。

要改什么直接改这里：
  * 应用名称 → APP_NAME（产物 dist\\<名称>.exe）
  * 图标     → 放 packaging/app.ico（没有就用 PyInstaller 默认图标）
  * 想改成"文件夹版"（启动更快）：把 EXE(...) 换成 EXE + COLLECT 两段，见文件末尾注释
"""
import os

APP_NAME = "NSSR风向识别"

ROOT = os.path.abspath(os.path.join(SPECPATH, os.pardir))     # 项目根目录
ENTRY = os.path.join(ROOT, "overlay.py")                      # 入口：悬浮窗
GLYPHS = os.path.join(ROOT, "glyph_bank.npz")                 # 字形模板库（运行时必需）
ICON = os.path.join(SPECPATH, "app.ico")
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

# 单文件：把 binaries/datas 都塞进 exe（启动时会解压到 %TEMP%，约 2~4 秒）
exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [],
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,             # 不弹黑框（报错显示在悬浮窗第一行）
    disable_windowed_traceback=False,
    icon=ICON,
)

# 想要"文件夹版"（启动快、免解压）就改成下面这样，并把上面 EXE 的 a.binaries/a.datas 去掉：
#   exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name=APP_NAME, console=False, icon=ICON)
#   coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name=APP_NAME)
