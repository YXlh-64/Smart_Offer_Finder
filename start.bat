@echo off
echo ========================================
echo   Smart Offer Finder - Startup
echo ========================================
echo.

echo [1/2] Starting Backend (FastAPI)...
echo.
start "Smart Offer Finder - Backend" cmd /k "cd /d %~dp0 && python main.py"
timeout /t 3 /nobreak >nul

echo [2/2] Starting Frontend (Vite + React)...
echo.
start "Smart Offer Finder - Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo ========================================
echo   Both servers are starting!
echo ========================================
echo   Backend:  http://localhost:8000
echo   Frontend: http://localhost:8080
echo ========================================
echo.
echo Press any key to exit this window...
pause >nul
