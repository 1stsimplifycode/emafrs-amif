@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ===============================================
echo  EMAFRS / AMIF -- local demo launcher
echo ===============================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [1/5] Creating Python virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create venv. Is Python installed and on PATH?
        pause
        exit /b 1
    )
) else (
    echo [1/5] Virtual environment already exists, skipping creation.
)

echo [2/5] Installing backend dependencies (this can take a while the first time)...
call .venv\Scripts\pip.exe install -q -r requirements.txt
call .venv\Scripts\pip.exe install -q -e .
if errorlevel 1 (
    echo ERROR: Backend dependency install failed. See output above.
    pause
    exit /b 1
)

if not exist ".env" (
    if exist ".env.example" (
        echo [3/5] No .env found -- copying .env.example. Fill in real API keys there for
        echo        live data sources ^(YouTube/NewsAPI/AlphaVantage/etc^); the app still runs
        echo        without them, with those sources reported as "not configured".
        copy /y ".env.example" ".env" >nul
    )
) else (
    echo [3/5] .env already exists, leaving it as-is.
)

if not exist "frontend\node_modules" (
    echo [4/5] Installing frontend dependencies ^(npm install^)...
    pushd frontend
    call npm install
    popd
    if errorlevel 1 (
        echo ERROR: Frontend dependency install failed. See output above.
        pause
        exit /b 1
    )
) else (
    echo [4/5] Frontend dependencies already installed, skipping.
)

if not exist "frontend\.env" (
    if exist "frontend\.env.example" (
        copy /y "frontend\.env.example" "frontend\.env" >nul
    )
)

echo [5/5] Starting backend and frontend in separate windows...
start "EMAFRS Backend (FastAPI :8000)" cmd /k ".venv\Scripts\activate.bat && uvicorn emafrs.api.app:app --reload --port 8000"

timeout /t 4 /nobreak >nul

start "EMAFRS Frontend (Vite :5173)" cmd /k "cd /d "%~dp0frontend" && npm run dev"

timeout /t 5 /nobreak >nul

echo.
echo Opening dashboard in your browser...
start "" "http://localhost:5173"

echo.
echo ===============================================
echo  Backend:   http://localhost:8000  (docs at /docs)
echo  Frontend:  http://localhost:5173
echo.
echo  Two new windows were opened for the backend and frontend servers.
echo  Close those windows (or Ctrl+C inside them) to stop the demo.
echo ===============================================
echo.
pause
endlocal
