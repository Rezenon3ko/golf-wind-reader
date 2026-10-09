@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

rem ---- find a python (python.exe or the py launcher) ----
set "PY="
where python >nul 2>nul
if not errorlevel 1 set "PY=python"
if not defined PY (
    where py >nul 2>nul
    if not errorlevel 1 set "PY=py"
)
if not defined PY (
    echo [X] Python not found.
    echo     Install it from https://www.python.org/downloads/windows/
    echo     and tick "Add python.exe to PATH" on the first screen.
    pause
    exit /b 1
)
echo Using %PY%

rem ---- deps: numpy + Pillow + obsws-python ----
%PY% -c "import numpy,PIL,obsws_python" >nul 2>nul
if errorlevel 1 (
    echo Installing dependencies: numpy Pillow obsws-python
    %PY% -m pip install numpy Pillow obsws-python
    if errorlevel 1 (
        echo [X] pip install failed. Try manually:
        echo     %PY% -m pip install numpy Pillow obsws-python
        pause
        exit /b 1
    )
)

echo.
%PY% obs_live_wind_reader.py %*
set "RC=%errorlevel%"
if not "%RC%"=="0" (
    echo.
    echo Exit code %RC%.
    pause
)
endlocal
