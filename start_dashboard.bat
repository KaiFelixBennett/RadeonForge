@echo off
REM RadeonForge — one-click LIVE dashboard demo (no GPU, no WSL, no install).
REM Runs the brand-neutral demo with simulated training so you see the full UI in ~10 s.
REM Needs only Python 3.8+ on PATH (pure stdlib). Change --port if 8765 is taken.
chcp 65001 >nul
title RadeonForge - Live Dashboard Demo (http://localhost:8765/gl)
echo.
echo   RadeonForge - live dashboard demo
echo   Opening http://localhost:8765/gl   (close this window to stop)
echo.
python "%~dp0scripts\demo_dashboard.py" --port 8765 --open
echo.
echo   Dashboard stopped.
pause
