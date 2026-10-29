@echo off
cd /d "%~dp0"

if not exist ".\.venv\Scripts\python.exe" (
    echo Missing virtual environment at .\.venv\Scripts\python.exe
    exit /b 1
)

if not exist ".\python_ads1299\mouse.ico" (
    echo [0/3] Generating icon...
    ".\.venv\Scripts\python.exe" ".\python_ads1299\create_mouse_icon.py"
    if errorlevel 1 (
        echo Icon generation failed
        exit /b %errorlevel%
    )
)

echo [1/3] Checking Python syntax...
".\.venv\Scripts\python.exe" -m py_compile ".\python_ads1299\main_uart_v1.2.py"
if errorlevel 1 (
    echo Python compile failed
    exit /b %errorlevel%
)

if exist ".\build" rmdir /s /q ".\build"
if exist ".\main_uart_v1.2.exe" del /q ".\main_uart_v1.2.exe"
if exist ".\main_uart_v1.2" rmdir /s /q ".\main_uart_v1.2"

echo [2/3] Building executable with PyInstaller...
".\.venv\Scripts\python.exe" -m PyInstaller --noconfirm --distpath "." --workpath ".\build" ".\python_ads1299\main_uart_v1.2.spec"
if errorlevel 1 (
    echo PyInstaller build failed
    exit /b %errorlevel%
)

echo [3/3] Build complete

set "VERSION_FILE=.\python_ads1299\VERSION"
if exist "%VERSION_FILE%" (
    for /f "usebackq delims=" %%v in ("%VERSION_FILE%") do set "CUR_VER=%%v"
) else (
    set "CUR_VER=1.2"
)

for /f "usebackq delims=" %%n in (`powershell -NoProfile -Command "('{0:F1}' -f ([double]('%CUR_VER%') + 0.1))"`) do set "NEW_VER=%%n"
>"%VERSION_FILE%" echo %NEW_VER%

for %%F in (".\python_ads1299\*.spec") do set "SPEC_BASENAME=%%~nF"
if exist ".\%SPEC_BASENAME%.exe" (
    move /Y ".\%SPEC_BASENAME%.exe" ".\main_uart_v%NEW_VER%.exe" >nul 2>nul
    if exist ".\main_uart_v%NEW_VER%.exe" (
        echo Output: .\main_uart_v%NEW_VER%.exe
    ) else (
        echo Output found but rename failed
    )
) else (
    echo Output not found
)
