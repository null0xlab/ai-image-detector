@echo off
:: Keep window open when double-clicked (re-launch in cmd /k)
if /i not "%~1"=="INTERNAL_KEEPOPEN" (
    cd /d "%~dp0"
    cmd /k call "%~f0" INTERNAL_KEEPOPEN
    exit /b
)
setlocal EnableDelayedExpansion
cd /d "%~dp0"
title AI Image Detector - Smart Launcher v7.3

:: ==============================================================
::  AI IMAGE DETECTOR - SMART LAUNCHER v7.3 (section pipelines + ELA + metadata)
:: ==============================================================

:: --- ANSI Colors ---
for /F "delims=#" %%E in ('"prompt #$E# & echo on & for %%b in (1) do rem"') do set "ESC=%%E"
set "GREEN=%ESC%[92m"
set "CYAN=%ESC%[96m"
set "YELLOW=%ESC%[93m"
set "RED=%ESC%[91m"
set "WHITE=%ESC%[97m"
set "DIM=%ESC%[90m"
set "RESET=%ESC%[0m"

:: --- Project Path (folder containing this .bat) ---
set "PROJECT=%~dp0"
if "%PROJECT:~-1%"=="\" set "PROJECT=%PROJECT:~0,-1%"

:: Initialize running flags
set "LOCAL_RUNNING=0"
set "FRONTEND_RUNNING=0"
set "TUNNEL_RUNNING=0"
set "APP_RUNNING=0"

:: ==============================================================
:: INITIAL BOOTUP
:: ==============================================================
cls
call :LOADING_BAR "Initializing AI Detector Smart Launcher..."
goto CHECK_STATUS

:: ==============================================================
:: CHECK STATUS
:: ==============================================================
:CHECK_STATUS
set "LOCAL_RUNNING=0"
set "FRONTEND_RUNNING=0"
set "TUNNEL_RUNNING=0"
set "APP_RUNNING=0"

:: Check if port 8000 is listening (Backend)
netstat -ano 2>nul | findstr /C:":8000 " | findstr "LISTENING" >nul 2>&1
if %ERRORLEVEL%==0 set "LOCAL_RUNNING=1"

:: Check if port 3000 is listening (Frontend)
netstat -ano 2>nul | findstr /C:":3000 " | findstr "LISTENING" >nul 2>&1
if %ERRORLEVEL%==0 set "FRONTEND_RUNNING=1"

:: Check for Cloudflare
tasklist 2>nul | find /I "cloudflared.exe" >nul 2>&1
if %ERRORLEVEL%==0 set "TUNNEL_RUNNING=1"

:: Calculate APP_RUNNING
if "%LOCAL_RUNNING%"=="1" set "APP_RUNNING=1"
if "%FRONTEND_RUNNING%"=="1" set "APP_RUNNING=1"

:: ==============================================================
:: MAIN MENU
:: ==============================================================
:MAIN_MENU
cls
echo.
echo %CYAN%  +==============================================================+%RESET%
echo %CYAN%  ^|                                                              ^|%RESET%
echo %CYAN%  ^|       %WHITE%AI IMAGE DETECTOR  --  SMART LAUNCHER v7.3%CYAN%           ^|%RESET%
echo %CYAN%  ^|                                                              ^|%RESET%
echo %CYAN%  +==============================================================+%RESET%
echo.
echo %DIM%  Project Path: %PROJECT%%RESET%
if exist "%PROJECT%\.venv\Scripts\activate.bat" (
    echo %DIM%  Python venv   : %GREEN%found%RESET%
) else (
    echo %DIM%  Python venv   : %YELLOW%not set up - use Option 1 or 4 to create it%RESET%
)
echo.
echo %YELLOW%  +-------------------------------------------------------------+%RESET%
echo %YELLOW%  ^|   CURRENT STATUS                                            ^|%RESET%
echo %YELLOW%  +-------------------------------------------------------------+%RESET%

if "%FRONTEND_RUNNING%"=="1" goto STATUS_APP_UP
echo %YELLOW%  ^|  Web Application :  %RED%[STOPPED]%YELLOW%                                ^|%RESET%
goto STATUS_APP_DONE
:STATUS_APP_UP
echo %YELLOW%  ^|  Web Application :  %GREEN%[RUNNING]  http://127.0.0.1:3000%YELLOW%         ^|%RESET%
:STATUS_APP_DONE

if "%LOCAL_RUNNING%"=="1" goto STATUS_BACKEND_UP
echo %YELLOW%  ^|   ^- Backend API  :  %RED%[STOPPED]%YELLOW%                                ^|%RESET%
goto STATUS_BACKEND_DONE
:STATUS_BACKEND_UP
echo %YELLOW%  ^|   ^- Backend API  :  %GREEN%[RUNNING]  http://127.0.0.1:8000%YELLOW%         ^|%RESET%
:STATUS_BACKEND_DONE

if "%TUNNEL_RUNNING%"=="1" goto STATUS_TUNNEL_UP
echo %YELLOW%  ^|  Cloudflare Tunnel: %RED%[NONE]%YELLOW%                                   ^|%RESET%
goto STATUS_TUNNEL_DONE
:STATUS_TUNNEL_UP
echo %YELLOW%  ^|  Cloudflare Tunnel: %GREEN%[ACTIVE] Shareable Link Enabled%YELLOW%        ^|%RESET%
:STATUS_TUNNEL_DONE

echo %YELLOW%  +-------------------------------------------------------------+%RESET%
echo.
echo %WHITE%  MENU OPTIONS:%RESET%
echo.

if "%APP_RUNNING%"=="1" goto MENU_APP_RUNNING
echo %GREEN%    [1] Start Web App [Unified Local Server]%RESET%
goto MENU_APP_DONE
:MENU_APP_RUNNING
echo %RED%    [1] Stop Web App [Shutdown Local Server]%RESET%
:MENU_APP_DONE

if "%TUNNEL_RUNNING%"=="1" goto MENU_TUNNEL_RUNNING
echo %CYAN%    [2] Share Publicly [Cloudflare Tunnel]%RESET%
goto MENU_TUNNEL_DONE
:MENU_TUNNEL_RUNNING
echo %RED%    [2] Stop Cloudflare Sharing Tunnel%RESET%
:MENU_TUNNEL_DONE

if "%APP_RUNNING%"=="1" goto MENU_STOP_ALL_RUNNING
if "%TUNNEL_RUNNING%"=="1" goto MENU_STOP_ALL_RUNNING
echo %DIM%    [3] Stop ALL Services [Nothing is running]%RESET%
goto MENU_STOP_ALL_DONE
:MENU_STOP_ALL_RUNNING
echo %RED%    [3] Stop ALL Services [Kills App + Tunnel]%RESET%
:MENU_STOP_ALL_DONE

echo %YELLOW%    [4] Fix/Update Dependencies [Run if facing errors]%RESET%
echo %DIM%    [5] Refresh Status%RESET%
echo %RED%    [0] Exit%RESET%
echo.

set "CHOICE="
set /p "CHOICE=  Enter your choice (0-5): "

if "%CHOICE%"=="1" goto TOGGLE_APP
if "%CHOICE%"=="2" goto TOGGLE_TUNNEL
if "%CHOICE%"=="3" goto STOP_ALL
if "%CHOICE%"=="4" goto INSTALL_DEPS
if "%CHOICE%"=="5" (
    call :LOADING_BAR "Scanning system ports..."
    goto CHECK_STATUS
)
if "%CHOICE%"=="0" goto EXIT

echo.
echo %RED%  Invalid choice! Please enter a number between 0 and 5.%RESET%
timeout /t 2 /nobreak >nul
goto MAIN_MENU

:: ==============================================================
:: [1] TOGGLE APPLICATION (BACKEND + FRONTEND)
:: ==============================================================
:TOGGLE_APP
cls
echo.
if "%APP_RUNNING%"=="1" goto STOP_APP

:START_APP
call :LOADING_BAR "Preparing to launch unified local server..."
echo.

call :ENSURE_VENV
if !ERRORLEVEL! NEQ 0 goto CHECK_STATUS

if exist "%PROJECT%\frontend\node_modules" goto FRONTEND_DEPS_OK
echo %YELLOW%  [!] node_modules not found in frontend directory!%RESET%
echo %YELLOW%  Running npm install... Please wait for the progress bar.%RESET%
cd /d "%PROJECT%\frontend"
call npm install
cd /d "%PROJECT%"
:FRONTEND_DEPS_OK

if "%LOCAL_RUNNING%"=="0" (
    echo %CYAN%  [*] Launching backend API engine...
    start "AI-Detector Backend Server" cmd /k "call .venv\Scripts\activate.bat && cd backend && ..\.venv\Scripts\python main.py"
)

if "%FRONTEND_RUNNING%"=="0" (
    echo %CYAN%  [*] Launching frontend web application...
    start "AI-Detector Frontend Dev" cmd /k "cd frontend && npm run dev"
)

echo.
echo %GREEN%  Success! The unified application has been launched.%RESET%
echo.
echo %YELLOW%  -------------------------------------------------------------+%RESET%
echo %GREEN%    OPEN THIS SINGLE LINK TO USE THE APP:                       %RESET%
echo %CYAN%    http://127.0.0.1:3000                                       %RESET%
echo %YELLOW%  -------------------------------------------------------------+%RESET%
echo %DIM%    (The frontend automatically routes all database/API calls    %RESET%
echo %DIM%     to the backend on port 8000 behind the scenes.)             %RESET%
echo.
pause
goto CHECK_STATUS

:STOP_APP
call :LOADING_BAR "Sending shutdown signals to local servers..."
for /f "tokens=5" %%p in ('netstat -ano 2^>nul ^| findstr /C:":8000 " ^| findstr "LISTENING"') do (
    taskkill /PID %%p /F >nul 2>&1
)
for /f "tokens=5" %%p in ('netstat -ano 2^>nul ^| findstr /C:":3000 " ^| findstr "LISTENING"') do (
    taskkill /PID %%p /F >nul 2>&1
)
echo %GREEN%  Servers successfully stopped!%RESET%
timeout /t 2 /nobreak >nul
goto CHECK_STATUS

:: ==============================================================
:: [2] TOGGLE CLOUDFLARE TUNNEL
:: ==============================================================
:TOGGLE_TUNNEL
cls
echo.
if "%TUNNEL_RUNNING%"=="1" goto STOP_TUNNEL

where cloudflared >nul 2>&1
if %ERRORLEVEL% NEQ 0 goto NO_CLOUDFLARED

:: Auto-start backend and frontend if they are not running
if "%LOCAL_RUNNING%"=="0" goto AUTO_START_APP
if "%FRONTEND_RUNNING%"=="0" goto AUTO_START_APP
goto APP_ALREADY_RUNNING

:AUTO_START_APP
echo %CYAN%  Web app is not running. Auto-starting both servers first...%RESET%
call :LOADING_BAR "Waking up required background processes..."
call :ENSURE_VENV
if !ERRORLEVEL! NEQ 0 goto CHECK_STATUS

if not exist "%PROJECT%\frontend\node_modules" (
    echo %YELLOW%  node_modules missing. Running npm install...%RESET%
    cd /d "!PROJECT!\frontend"
    call npm install
    cd /d "!PROJECT!"
)

if "%LOCAL_RUNNING%"=="0" (
    start "AI-Detector Backend Server" cmd /k "call .venv\Scripts\activate.bat && cd backend && ..\.venv\Scripts\python main.py"
)

if "%FRONTEND_RUNNING%"=="0" (
    start "AI-Detector Frontend Dev" cmd /k "cd frontend && npm run dev"
)

call :LOADING_BAR "Waiting for servers to initialize..."

:APP_ALREADY_RUNNING
echo %GREEN%  Tunnel will point to Frontend on http://127.0.0.1:3000%RESET%
echo %DIM%  [API requests will be proxied automatically to Backend]%RESET%
echo.
echo %WHITE%  Starting Cloudflare tunnel...%RESET%
start "Cloudflare Tunnel" cmd /k "cloudflared tunnel --url http://127.0.0.1:3000 --http-host-header 127.0.0.1:3000"
echo.
echo %GREEN%  Cloudflare Tunnel Launched! Check the new window for your public URL.%RESET%
echo  Press any key to return to main menu...
pause >nul
goto CHECK_STATUS

:STOP_TUNNEL
call :LOADING_BAR "Disconnecting active Cloudflare tunnel..."
taskkill /IM cloudflared.exe /F >nul 2>&1
echo %GREEN%  Tunnel successfully stopped!%RESET%
timeout /t 2 /nobreak >nul
goto CHECK_STATUS

:NO_CLOUDFLARED
echo %RED%  Cloudflared is not installed! Please run option [4] from the menu first.%RESET%
timeout /t 4 /nobreak >nul
goto CHECK_STATUS

:: ==============================================================
:: [3] STOP ALL SERVICES
:: ==============================================================
:STOP_ALL
cls
echo.
call :LOADING_BAR "Executing global shutdown sequence..."

for /f "tokens=5" %%p in ('netstat -ano 2^>nul ^| findstr /C:":8000 " ^| findstr "LISTENING"') do (
    taskkill /PID %%p /F >nul 2>&1
)

for /f "tokens=5" %%p in ('netstat -ano 2^>nul ^| findstr /C:":3000 " ^| findstr "LISTENING"') do (
    taskkill /PID %%p /F >nul 2>&1
)

taskkill /IM cloudflared.exe /F >nul 2>&1

echo %GREEN%  Done! All background services have been terminated.%RESET%
timeout /t 2 /nobreak >nul
goto CHECK_STATUS

:: ==============================================================
:: [4] FIX/INSTALL DEPENDENCIES
:: ==============================================================
:INSTALL_DEPS
cls
echo.
echo %GREEN%  [ AUTO SETUP ] - FIXING AND UPDATING DEPENDENCIES%RESET%
echo  ---------------------------------------------------------------
echo.
echo %CYAN%  [1/5] Checking Python version...%RESET%
python --version >nul 2>&1
if %ERRORLEVEL% EQU 0 goto PYTHON_OK
echo %YELLOW%  Python not found! Installing Python 3.12 via Winget...%RESET%
winget install --id Python.Python.3.12 -e --accept-package-agreements --accept-source-agreements
echo %RED%  IMPORTANT: Winget modifies system PATH. You MUST close this launcher and reopen it.%RESET%
pause
exit

:PYTHON_OK
for /f "delims=" %%v in ('python --version 2^>^&1') do echo %GREEN%  OK: %%v%RESET%
echo.

echo %CYAN%  [2/5] Updating PIP (Python Package Manager)...%RESET%
python -m pip install --upgrade pip
echo %GREEN%  OK: PIP check complete.%RESET%
echo.

echo %CYAN%  [3/5] Checking Node.js version...%RESET%
node --version >nul 2>&1
if %ERRORLEVEL% EQU 0 goto NODE_OK
echo %YELLOW%  Node.js not found! Installing Node.js via Winget...%RESET%
winget install --id OpenJS.NodeJS -e --accept-package-agreements --accept-source-agreements
echo %RED%  IMPORTANT: Winget modifies system PATH. You MUST close this launcher and reopen it.%RESET%
pause
exit

:NODE_OK
for /f "delims=" %%v in ('node --version 2^>^&1') do echo %GREEN%  OK: %%v%RESET%
echo.

echo %CYAN%  [4/5] Checking Cloudflare Tunnel...%RESET%
where cloudflared >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo %GREEN%  OK: Cloudflared is already installed.%RESET%
) else (
    echo %YELLOW%  Installing Cloudflare Tunnel...%RESET%
    winget install --id Cloudflare.cloudflared --force -e --accept-package-agreements --accept-source-agreements
)
echo.

echo %CYAN%  [5/5] Setting up virtual environment and downloading packages...%RESET%
:: Setup Backend Venv
echo %YELLOW%  - Setting up Backend virtual environment [.venv]...%RESET%
cd /d "%PROJECT%"
if exist ".venv\Scripts\activate.bat" goto VENV_EXISTS
echo %DIM%  Creating new virtual environment (This takes a moment)...%RESET%
python -m venv .venv
:VENV_EXISTS
call .venv\Scripts\activate.bat
cd backend
echo %YELLOW%  - Downloading Backend Python packages...%RESET%
..\.venv\Scripts\pip install -r requirements.txt
echo %YELLOW%  - Verifying section-detection dependencies (easyocr, piexif, mediapipe)...%RESET%
..\.venv\Scripts\python -c "import piexif; import easyocr; import mediapipe; print('Section detection deps OK')"
if %ERRORLEVEL% NEQ 0 (
    echo %YELLOW%  [!] Optional section deps missing - retrying targeted install...%RESET%
    ..\.venv\Scripts\pip install easyocr piexif mediapipe
    ..\.venv\Scripts\python -c "import piexif; import easyocr; import mediapipe; print('Section detection deps OK')"
)
echo %YELLOW%  - Downloading pretrained AI detector weights (Hugging Face)...%RESET%
..\.venv\Scripts\python scripts\download_models.py
if %ERRORLEVEL% NEQ 0 (
    echo %YELLOW%  [!] Model download had warnings - app may use CLIP-only fallback.%RESET%
)
cd /d "%PROJECT%"

:: Setup Frontend packages
echo.
echo %YELLOW%  - Installing Frontend NPM dependencies...%RESET%
cd /d "%PROJECT%\frontend"
call npm install
cd /d "%PROJECT%"

echo.
echo  ---------------------------------------------------------------
call :LOADING_BAR "Finalizing Setup..."
echo %GREEN%  Setup Complete! All dependencies and environments are ready.%RESET%
echo.
pause
goto CHECK_STATUS

:: ==============================================================
:: SUBROUTINES
:: ==============================================================

:ENSURE_VENV
:: Validate project root (backend must exist next to this .bat)
if not exist "%PROJECT%\backend\main.py" (
    echo %RED%  ERROR: Cannot find backend\main.py%RESET%
    echo %DIM%  Expected project folder: !PROJECT!%RESET%
    echo %YELLOW%  Place "deepfake run.bat" in the ai-image-detector folder and try again.%RESET%
    timeout /t 5 /nobreak >nul
    exit /b 1
)

cd /d "%PROJECT%"
if exist "%PROJECT%\.venv\Scripts\activate.bat" exit /b 0
if exist "%PROJECT%\venv\Scripts\activate.bat" exit /b 0

echo %YELLOW%  Virtual environment not found - creating .venv now...%RESET%
echo %DIM%  Location: %PROJECT%\.venv%RESET%
where python >nul 2>&1
if !ERRORLEVEL! NEQ 0 (
    echo %RED%  Python is not installed or not on PATH.%RESET%
    echo %YELLOW%  Run menu Option [4] to install Python and all dependencies.%RESET%
    timeout /t 5 /nobreak >nul
    exit /b 1
)

python -m venv "%PROJECT%\.venv"
if not exist "%PROJECT%\.venv\Scripts\activate.bat" (
    echo %RED%  Failed to create virtual environment.%RESET%
    echo %YELLOW%  Run Option [4] from the menu for full setup.%RESET%
    timeout /t 5 /nobreak >nul
    exit /b 1
)

echo %GREEN%  Virtual environment created.%RESET%
echo %YELLOW%  Installing backend packages (first time only, may take several minutes)...%RESET%
call "%PROJECT%\.venv\Scripts\activate.bat"
cd /d "%PROJECT%\backend"
..\.venv\Scripts\pip install -r requirements.txt
..\.venv\Scripts\python -c "import piexif; import easyocr; import mediapipe" >nul 2>&1
if !ERRORLEVEL! NEQ 0 (
    ..\.venv\Scripts\pip install easyocr piexif mediapipe
)
if !ERRORLEVEL! NEQ 0 (
    echo %RED%  pip install failed. Run Option [4] from the menu.%RESET%
    cd /d "%PROJECT%"
    exit /b 1
)
cd /d "%PROJECT%"
echo %GREEN%  Dependencies installed.%RESET%
exit /b 0

:LOADING_BAR
echo %DIM%  %~1%RESET%
<nul set /p "=%CYAN%  ["
for /L %%I in (1,1,20) do (
    <nul set /p "=%GREEN%#%RESET%"
    ping 127.0.0.1 -n 1 -w 100 >nul 2>&1
)
echo %CYAN%] Done!%RESET%
echo.
exit /b

:: ==============================================================
:: EXIT
:: ==============================================================
:EXIT
cls
echo.
call :LOADING_BAR "Shutting down launcher securely..."
echo %GREEN%  Have a great day!%RESET%
echo.
timeout /t 2 /nobreak >nul
pause
exit /b