# NSS 高尔夫风向识别

从《Nintendo Switch Sports》高尔夫画面的**右上角风盘**里，实时读出**风速**和**风向**，
算出**横向风 / 纵向风**，一行文字输出。画面来源是 **OBS**（通过 obs-websocket 抓源）。

约定：

* **正前方（瞄准方向）= 0°**，顺时针为正（偏右）
* **横向风** = 风速 × sin(夹角) → 标注 **向左 / 向右**
* **纵向风** = 风速 × cos(夹角) → 标注 **顺风 / 逆风**
* **风速优先**：风速为 0 时直接判定"无风"，不计算夹角与分量

```
            ↑ 正前方（瞄准方向）
            |
            |  θ = 风向夹角
            | /
            |/  风矢量（箭头指向 = 风吹去的方向）
    ────────┼────────→ 右
```

## 依赖

三个包，都有通用 wheel（Windows / macOS / Linux，含 ARM），直接装：

```bash
pip install numpy Pillow obsws-python
```

| 包 | 用途 |
|---|---|
| `numpy` | 图像计算 |
| `Pillow` | 裁剪 / 缩放 / 二值化 |
| `obsws-python` | 通过 obs-websocket 抓 OBS 的画面 |

风速读数用自带的字形模板库 `glyph_bank.npz`，**不需要装 OCR**。

## 部署与使用

### 0. 需要哪些文件

运行时只要这 4 个（其余是开发用的，见文末「文件说明」）：

```
wind_reader.py            识别库
obs_live_wind_reader.py   命令行入口
glyph_bank.npz            字形模板库
启动-Windows.bat  /  启动-macOS.command     一键启动脚本
```

### 1. 装 Python

* **Windows**：到 https://www.python.org/downloads/windows/ 下 **`Windows installer (64-bit)`**
  （不要选 ARM64 那个），安装第一屏务必勾上 **`Add python.exe to PATH`**。
* **macOS**：系统自带 `python3` 就行；没有就 `brew install python`。

装完开个终端验证：

```bash
python --version        # Windows
python3 --version       # macOS
```

### 2. 装依赖

```bash
pip install numpy Pillow obsws-python
```

国内网络慢就加镜像：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy Pillow obsws-python
```

### 3. 让 OBS 打开 WebSocket

OBS → **工具 → WebSocket 服务器设置 → 勾选「启用 WebSocket 服务器」**。
**OBS 28 及以上默认就是开的**，一般不用动（端口和密码脚本会自动读 OBS 配置）。

源名、分辨率都不用管：脚本会按**源类型**（采集设备）自动找到你的采集卡，
跟 OBS 界面是中文、英文还是日文都没关系；认不出类型时退回常见源名/第一个源，
并把 OBS 里的源名全部打印出来（方便你回 OBS 里看是不是加错了源）。

### 4. 启动

**推荐：双击启动脚本**（会自动补装缺的依赖）

| 系统 | 双击 | 说明 |
|---|---|---|
| Windows | `启动-Windows.bat` | 出错会停住窗口，方便看提示 |
| macOS | `启动-macOS.command` | 首次若提示"无法打开"，右键 → 打开 |
| macOS（免装 Python） | `dist/NSSR风向识别.app` | 悬浮窗版，已打包好，见下面「打包成 .app」 |

**或者用命令行**：

```bash
python obs_live_wind_reader.py        # 默认每秒识别 2 次
python obs_live_wind_reader.py -t     # 抓一帧做一次识别，打印详情后退出（建议先用它确认）
```

### 5. 看输出

让游戏停在**瞄准界面**（右上角有风盘），窗口会原地刷新一行：

```
[  1.0 次/秒] 风 13 m/s | 夹角 30.6° | 横向 6.62 向右 | 纵向 11.19 顺风
```

读法：**风速 13 m/s，风从正前方偏右 30.6° 吹来，往右偏 6.62，顺风 11.19**。

按 `Ctrl-C` 退出。

### 6.（可选）用悬浮窗代替命令行

不想盯着命令行的话，用 `overlay.py`——一个置顶的小窗口：

```bash
python overlay.py                    # 固定每秒识别 2 次
python overlay.py --alpha 0.5        # 更透明（默认 0.5）
```

* 窗口**无边框 + 置顶 + 半透明 + 固定两行大小**（宽度正好是第一行的宽度，左右不留白）；
  **拖动**移动，**右键 → 退出**（默认贴屏幕右上角，尺寸算好不会跑出屏幕）。
* 显示是**箭头 + 风速**（箭头指向 = 风吹去的方向，↑ 就是顺风）：

  ```
  ↗  13 m/s  +30.6°     ← 第一行：风向箭头 + 风速 + 夹角（+ 偏右 / − 偏左）
  →  6.62   ↑  11.19    ← 第二行：横向 / 纵向分量，字号和排法与第一行一致，整组居中
  ```

  **两行的数字部分统一按风速档位染色**（和游戏里风盘箭头的三种颜色一致）：
  1~5 m/s 青、6~10 m/s 黄、11~15 m/s 品红；方向由箭头本身表达
  （`↗` 风向、`→/←` 左右、`↑/↓` 顺逆），夹角保持中性色。
  识别频率固定 2 次/秒（和命令行版一样，代码里写死），不做多拍平滑，转视角跟手。
  横向 / 纵向的**方向箭头**带迟滞：|值| < 0.05 时沿用上一次方向，θ 压在 0°/90°/180° 附近时箭头不抖，数字照实显示。
* 识别异常（连不上 OBS、不在瞄准界面、箭头没找到…）时**直接在第一行显示原因**，
  字号与正常结果一致；连不上 OBS **不会崩**，每 3 秒自动重连——所以先开悬浮窗、
  后开 OBS 也行。

### 7.（可选）打包成 macOS 应用「NSSR风向识别.app」

想直接发给**没装 Python** 的玩家，就用 PyInstaller 把悬浮窗打成独立 `.app`
（Python 解释器 + numpy/Pillow/obsws-python + Tk 8.6 全部塞进去，约 47MB）：

```bash
./packaging/build_app.sh          # 产物 dist/NSSR风向识别.app
./packaging/build_app.sh --dmg    # 顺手再包一个 dmg 方便分发
```

* 双击 `dist/NSSR风向识别.app` 即用；脚本会自建 `.venv-build` 打包环境（首次装 PyInstaller）。
* **版本号 / 图标 / 排除模块**都在 `packaging/app.spec` 里：
  * 版本号 = `APP_VERSION`（当前 `1.0.0`，写进 `Info.plist`，Finder「显示简介」能看到）
  * 图标 = `packaging/app.icns`（金色高尔夫球，素材取自隔壁 `~/NSS/nss-golf-tournament/public/favicon.svg`；
    换图标：新图渲染成 1024px PNG → `sips` 生成各尺寸 → `iconutil -c icns` 覆盖即可）
* **架构**：PyInstaller 不能跨架构，产物只对当前机器架构有效（本机是 arm64）。
  要给 Intel Mac，就得在 Intel 机器上跑一遍同样的脚本。
* **首次打开**：没签名的 app 会被 Gatekeeper 拦（"来自身份不明的开发者"），
  右键图标 → **打开** → 再点"打开"即可；批量的话让对方跑一次：
  `xattr -dr com.apple.quarantine /Applications/NSSR风向识别.app`。
  想彻底没提示就得花钱签名+公证（Developer ID）。
* 打包版的报错显示在悬浮窗第一行；要看细节还是用命令行版 `-t -v`。
* 改完代码重新打包就再跑一次脚本（`.venv-build` 会复用，第二次约 20 秒）。

### 8.（可选）打包成 Windows exe

想要一个**玩家不用装 Python**的 Windows 版，就用 PyInstaller 打成单文件 `NSSR风向识别.exe`
（约 40~50MB，含 Python + numpy/Pillow/obsws-python + Tk）：

**方式一：在任意 x64 Windows 上双击脚本**（云电脑 / 实体机都行）

```
packaging\build_exe.bat        # 自建 .venv-build-win → 装依赖 → 打包 → dist\NSSR风向识别.exe
```

* 需要 **64 位 Python 3.10~3.12**（python.org 安装包即可，装的时候勾上 Add to PATH）。
* **PyInstaller 不能交叉编译**：exe 的架构 = 这台机器的架构。ARM64 Windows（比如 M 芯片 Mac 的
  Parallels）打出来的是 ARM64 exe，给 Intel/AMD 玩家用不了；反之亦然。
* 首次启动要解压约 2~4 秒（单文件 exe 的固有代价）；想快就在 `packaging/exe.spec` 里
  用它末尾注释那两行换成"文件夹版"。

**方式二：让 GitHub 免费帮你打**（本机不用装任何东西）

仓库里已经放了 `.github/workflows/build-windows-exe.yml`：推到 GitHub 后自动在
`windows-latest`（x64）上打包，产物在 Actions 页面 → 该次运行 → Artifacts 里下载。
公开仓库完全免费。

> 打包版报错显示在悬浮窗第一行；要看细节还是用命令行版 `-t -v`。

## 常见问题

| 现象 | 处理 |
|---|---|
| `连不上 OBS WebSocket` | OBS 没开 WebSocket（见第 3 步）。端口/密码是脚本自动读 OBS 配置的，一般不用管 |
| 认不出采集设备 / 选错了源 | 脚本会把 OBS 里所有源名打印出来；回 OBS 看「源」列表，只留采集卡那一个即可 |
| `未找到右上角风盘` | 游戏没停在瞄准界面（右上角没有那个风盘） |
| `风速未识别` | 风盘上的数字没读出来；加 `-t -v` 看细节 |
| 识别到的是别的画面 | 回 OBS 把多余的源（摄像头、窗口捕获…）删掉，只留采集卡 |

## 工作原理（简述）

1. **风盘定位**：位置固定（右手右上、左手左上，左右镜像）。先在固定先验小窗
   （`cx=0.880W, cy=0.130H, r=0.060W`，实测 1080p 中心 (1690,140)、半径 115）里确认
   草地圆盘，命中就直接用先验中心/半径（约 1ms）；未命中再退回"白环 + 草地圆盘 /
   箭头颜色锚点"的完整搜索。
2. **箭头指向**：PCA 定轴（箭头的杆又细又长，第一主成分即轴线）+ 宽度剖面
   判尖端方向（哪一端更窄、最大宽度靠近那一端就是尖）。
3. **俯视压缩 k**：优先用盘面草地椭圆的半轴比（白环路径标定，限 0.80~0.88），
   否则用先验 0.83。不用箭头几何反解——实测它有 θ 相关偏差，反而更差。
4. **风速优先**：用字形模板匹配读数字（两位数由 0-9 单字形拼接），不做 OCR。
   **有了风速再选箭头颜色**（1-5 青 / 6-10 黄 / 11-15 品红），用该色的单色窄带掩膜测箭头
   ——窄带把天空等同色背景排掉。数字与箭头颜色不符时不输出（等待重读）。

## 运行频率与抓图质量

**固定 2 次/秒，不开放调节**（识别频率、JPEG 质量都写死在代码里）：

* 2 次/秒是为了跟手：快速转视角时 1 次/秒会明显慢半拍。每帧约 70~90ms，
  2 次/秒约占单核 15~20%；嫌费 CPU 就把 `obs_live_wind_reader.py` 里的 `FPS` 改回 `1.0`，
  想更跟手就调到 `3.0`。
* **不做角度平滑**（早期版本做过）：多拍平均会让显示滞后 1~2 秒，正是「转了但数字半天不跟」的元凶。
  防抖改由「反 180° 保护」负责——低置信度的翻转会按住上一帧，不会来回跳。
* JPEG 质量固定 **100**：实测 80→100 能把"偶发反向"的概率降到约 1/4，
  代价是每帧多 20ms 左右、单帧 352KB→1.6MB（本机回环，无感）。
* 每帧都真测一次（不做静止帧复用，延迟最低）；定位走固定先验，不到 1ms。

抓图分辨率也固定 **1920×1080**（超采样和先验定位都按这个标定的，换分辨率会掉精度）。

## 常用参数

| 参数 | 说明 |
|---|---|
| `-t` / `--test` | 抓一帧做一次识别并打印详情后退出 |
| `-v` / `--verbose` | 调试输出：风盘定位过程、每帧耗时、读风速失败等 |

取帧那边的参数（源名 / 地址 / 端口 / 密码 / 分辨率）**已经全部自动化**，没有开关：
源按类型自动认，端口密码读 OBS 配置，分辨率固定 1920×1080，JPEG q100，2 次/秒。

## 结果字段

`wind_metrics(speed, theta)` 返回的字典（实时识别内部用的就是它）：

| 字段 | 含义 |
|---|---|
| `speed` / `theta_deg` / `compass` | 风速 / 夹角 / 方位文字（右前、右后、左前、左后、正前…） |
| `cos` / `sin` | 夹角余弦 / 正弦 |
| `lateral_abs` / `lateral_dir` | 横向风速绝对值 / 向左·向右（非 0 就标方向；接口层带 0.05 迟滞） |
| `longitudinal_abs` / `longitudinal_dir` | 纵向风速绝对值 / 顺风·逆风 |
| `lateral_text` / `longitudinal_text` | 现成文字，如 `7.18 m/s 向右` |
| `no_wind` / `no_speed` | 无风 / 风速未识别（此时不计算分量） |

## 库接口

`wind_reader.py` 是纯识别库，内部流程由 `read_wind()` 串起：
定位风盘 → 读风速（字形模板匹配）→ 按风速选箭头颜色 → 测夹角 → 分解横向/纵向风。

| 分类 | 函数 / 类 | 说明 |
|---|---|---|
| 核心 | `read_wind(rgb, dial, speed=None, supersample=None, hsv=None, ocr_state=None, do_ocr=True)` | → `FrameReading`（先读风速再选色；`supersample` 默认走 `SUPERSAMPLE=2`） |
| 定位 | `find_dials(rgb, hsv=None)` | → `list[Dial]`（先验优先，未命中再完整搜索） |
| 箭头 | `measure_arrow(rgb, hsv, dial, supersample=6, band=None)` | → `ArrowReading` |
| 风速 | `ocr_speed(rgb, dial, state=None)` | → `dict(value, raw, conf, box, ok)`（字形模板匹配） |
| 风速（模板） | `match_glyphs(crop, state=None)` / `load_glyph_bank(path, state=None)` | 字形模板匹配 |
| 换算 | `apparent_to_true(phi, k)` / `wind_metrics(speed, theta)` / `compass_name(theta)` | 角度与分量 |
| 颜色 | `wind_color(speed)` / `arrow_mask_band(hsv, band)` / `arrow_mask(hsv)` | 风速 ↔ 箭头颜色窄带 |
| 图像工具 | `rgb_to_hsv` / `label_components` / `fit_circle_ls` / `fit_ellipse_axis_aligned` | 纯 numpy 实现 |
| 数据类 | `Dial` / `ArrowReading` / `Component` / `FrameReading` / `Config` / `OcrState` | |

## 字形模板库 `glyph_bank.npz`

风速读数靠它：运行时只读取，不自学、不写回。库里按字符 `'0'`~`'9'` 存归一化到 32×48
的二值字形，识别时把文字框切成单字形逐个比对。

当前库覆盖 **0~9**；两位数 10~15 由单字形拼接（已验证全部正确）。

## 文件说明

| 文件 | 说明 |
|---|---|
| `wind_reader.py` | 识别库（所有识别功能，无命令行界面） |
| `obs_live_wind_reader.py` | 命令行入口（obs-websocket 取帧，固定 2 次/秒、JPEG q100） |
| `overlay.py` | 桌面悬浮窗入口（可选，同样的取帧 + 识别） |
| `glyph_bank.npz` | 字形模板库（0~9），**运行时必需** |
| `启动-Windows.bat` / `启动-macOS.command` | 一键启动脚本（需要机器上有 Python） |
| `packaging/build_app.sh` | 打成 macOS `.app`（免 Python，见上面第 7 节） |
| `packaging/build_exe.bat` / `exe.spec` | 打成 Windows `.exe`（见上面第 8 节） |
| `.github/workflows/build-windows-exe.yml` | 推 GitHub 后自动打 Windows exe（免费） |
| `dist/NSSR风向识别.app` | 打包产物（双击即用；重新打包会覆盖） |
