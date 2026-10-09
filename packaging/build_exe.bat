@echo off
rem ---------------------------------------------------------------------------
rem  Build a standalone Windows exe (NSSR.exe) - players need no Python.
rem  Run by double-clicking, or:  packaging\build_exe.bat
rem  Requires: 64-bit Python 3.10-3.12 for Windows (python.org installer).
rem  Note: PyInstaller cannot cross-compile - the exe matches THIS machine's arch.
rem ---------------------------------------------------------------------------
chcp 65001 >nul
setlocal
cd /d "%~dp0.."
set "VENV=.venv-build-win"

rem ---- 1. find python -------------------------------------------------------
set "PY="
where py >nul 2>nul
if not errorlevel 1 set "PY=py -3"
if not defined PY (
    where python >nul 2>nul
    if not errorlevel 1 set "PY=python"
)
if not defined PY (
    echo [X] Python not found.
    echo     Install 64-bit Python from https://www.python.org/downloads/windows/
    echo     and tick "Add python.exe to PATH" on the first screen.
    pause
    exit /b 1
)
echo [1/4] Python:
%PY% -c "import sys,platform;print('      ',sys.version.split()[0],platform.machine(),'64bit' if sys.maxsize>2**32 else '32bit')"
%PY% -c "import sys,platform;sys.exit(0 if (platform.machine() in ('AMD64','x86_64') and sys.maxsize>2**32) else 1)"
if errorlevel 1 (
    echo [WARN] This Python is not x64 ^(ARM64 / 32-bit^).
    echo        The exe will only run on the same architecture.
)

rem ---- 2. build venv + deps ------------------------------------------------
echo [2/4] Preparing %VENV% ...
if not exist "%VENV%\Scripts\python.exe" %PY% -m venv "%VENV%"
"%VENV%\Scripts\python.exe" -m pip install --quiet --upgrade pip
"%VENV%\Scripts\python.exe" -m pip install --quiet pyinstaller numpy Pillow obsws-python
if errorlevel 1 (
    echo [X] pip install failed. Check network / proxy, then run again.
    pause
    exit /b 1
)

rem ---- 3. pyinstaller ------------------------------------------------------
echo [3/4] Building exe ...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
set "PYINSTALLER_CONFIG_DIR=%CD%\build\pyi-cache"
"%VENV%\Scripts\pyinstaller.exe" --noconfirm --clean packaging\exe.spec
if errorlevel 1 (
    echo [X] PyInstaller failed - send me the messages above.
    pause
    exit /b 1
)

rem ---- 4. done --------------------------------------------------------------
echo [4/4] Done.
echo.
set "EXE="
for %%f in ("dist\*.exe") do set "EXE=%%~ff"
if not defined EXE (
    echo [X] No exe found in dist\
    pause
    exit /b 1
)
echo    Output: %EXE%
echo    First run may take a few seconds (single-file exe unpacks itself).
echo    The app talks to OBS on this machine - OBS + WebSocket must be running.
echo.
pause
endlocal
