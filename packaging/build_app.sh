#!/bin/bash
# 把悬浮窗打包成 macOS 应用「NSSR风向识别.app」——玩家不用装 Python，双击即用。
#
#   ./packaging/build_app.sh            # 打 arm64（Apple 芯片，用本机 Python 跑）
#   ./packaging/build_app.sh --dmg      # 顺手再包一个 .dmg 方便分发
#
# 注意：PyInstaller 不能跨架构，产物只对**当前机器的架构**有效（本机 arm64）。
# 要给 Intel Mac 用，得在 Intel 机器上跑一遍同样的脚本。
set -euo pipefail

cd "$(dirname "$0")/.."                 # 回到项目根目录
APP_NAME="NSSR风向识别"
VENV=".venv-build"
MAKE_DMG=0
[ "${1:-}" = "--dmg" ] && MAKE_DMG=1

say() { printf '%s\n' "$*"; }

say "[1/4] 准备打包环境（${VENV}，第一次会装 pyinstaller，几十 MB）"
[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --quiet --upgrade pip
"$VENV/bin/python" -m pip install --quiet pyinstaller numpy Pillow obsws-python
"$VENV/bin/python" -c 'import PyInstaller, numpy, PIL, obsws_python; print("      环境就绪：pyinstaller", PyInstaller.__version__)'
say "      Python: $("$VENV/bin/python" -V) / 架构 $(uname -m)"

say "[2/4] 打包（窗口程序，不弹终端；字形库一起带上）"
rm -rf build dist
export PYINSTALLER_CONFIG_DIR="$PWD/build/pyi-cache"   # 缓存留在项目里，别污染 ~/Library

# 版本号 / 图标 / 排除模块都在 packaging/app.spec 里（图标：packaging/app.icns，
# 金色高尔夫球，素材来自 ~/NSS/nss-golf-tournament/public/favicon.svg）
"$VENV/bin/pyinstaller" --noconfirm --clean packaging/app.spec

say "[3/4] 去掉隔离属性（本机自测用；发给别人时对方仍需右键→打开一次）"
[ -d "dist/$APP_NAME.app" ] || { say "✗ 没生成 .app，请把上面的报错发我"; exit 1; }
xattr -cr "dist/$APP_NAME.app" 2>/dev/null || true
rm -rf "dist/$APP_NAME"          # PyInstaller 还会留一份 onedir 中间产物，.app 里已有副本，删掉省空间

if [ "$MAKE_DMG" = 1 ]; then
    say "[4/4] 打包 dmg"
    rm -f "dist/$APP_NAME.dmg"
    hdiutil create -volname "$APP_NAME" -srcfolder "dist/$APP_NAME.app" \
        -ov -format UDZO "dist/$APP_NAME.dmg" >/dev/null
    say "      产物：dist/$APP_NAME.dmg（$(du -h "dist/$APP_NAME.dmg" | cut -f1)）"
else
    say "[4/4] 跳过 dmg（要的话加 --dmg）"
fi

say ""
say "✓ 产物：dist/$APP_NAME.app（$(du -sh "dist/$APP_NAME.app" | cut -f1)）"
say "  自测：open \"dist/$APP_NAME.app\""
say "  首次发给别人：对方要右键图标 →「打开」→ 再点「打开」，或跑一次："
say "      xattr -dr com.apple.quarantine /Applications/$APP_NAME.app"
