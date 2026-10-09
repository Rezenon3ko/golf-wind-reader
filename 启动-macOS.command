#!/bin/bash
# NSS 高尔夫风向识别 —— macOS 一键启动
# 双击本文件即可运行（会自动连本机 OBS，源和端口都自动识别）
cd "$(dirname "$0")" || exit 1

say() { printf '%s\n' "$*"; }

# 找一个可用的 python3
PY=""
for c in python3 /opt/homebrew/bin/python3 /usr/local/bin/python3 /usr/bin/python3; do
    if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
    say "✗ 没找到 python3。"
    say "  安装：https://www.python.org/downloads/  或  brew install python"
    say ""
    read -n 1 -s -r -p "按任意键关闭…"
    exit 1
fi
VER="$("$PY" -V 2>&1)"
say "使用 ${PY}（${VER}）"

# 缺依赖就补。只要 numpy / Pillow / obsws-python —— 都有通用 wheel，秒装。
MISS=""
"$PY" -c 'import numpy' >/dev/null 2>&1 || MISS="$MISS numpy"
"$PY" -c 'import PIL'   >/dev/null 2>&1 || MISS="$MISS Pillow"
"$PY" -c 'import obsws_python' >/dev/null 2>&1 || MISS="$MISS obsws-python"
if [ -n "$MISS" ]; then
    say "安装缺少的依赖：$MISS"
    if ! "$PY" -m pip install --quiet $MISS; then
        say "✗ 依赖安装失败，请手动执行：$PY -m pip install$MISS"
        read -n 1 -s -r -p "按任意键关闭…"
        exit 1
    fi
fi

say ""
say "提示：走 obs-websocket 抓 OBS 的源（OBS 28+ 默认已启用 WebSocket）。"
say "      Ctrl-C 退出；OBS 源和端口都是自动识别的。"
say ""

"$PY" obs_live_wind_reader.py "$@"
CODE=$?
if [ "$CODE" -ne 0 ]; then
    say ""
    say "退出码 ${CODE}。"
    read -n 1 -s -r -p "按任意键关闭…"
fi
