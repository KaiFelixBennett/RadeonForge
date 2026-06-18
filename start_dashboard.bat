@echo off
setlocal
chcp 65001 >nul
title RadeonForge Dashboard (Engine-Demo) - http://localhost:8766/gl

REM ============================================================
REM  Startet die generische RadeonForge-Dashboard-Engine auf den
REM  Beispieldaten (examples/gemma4-12b-qlora). Projektneutral,
REM  reines Python-stdlib. Laeuft auf Port 8766 (parallel zum
REM  ARIA-Dashboard auf 8765 nutzbar).
REM  Vollansicht (3D + Charts + HUD):  http://localhost:8766/gl
REM ============================================================

set "PORT=8766"
set "DATA=/mnt/e/Coding/RadeonForge/examples/gemma4-12b-qlora"
set "TRACKS=/mnt/e/Coding/RadeonForge/scripts/dashboard.tracks.example.json"
echo.
echo   RadeonForge-Dashboard (Engine + Beispieldaten) auf Port %PORT%
echo   Vollbild : http://localhost:%PORT%/gl   (oeffnet automatisch)
echo.

start "" /b powershell -NoProfile -Command "Start-Sleep 5; Start-Process 'http://localhost:%PORT%/gl'"

wsl -d Ubuntu -u root -- bash -lic "pkill -f 'examples/gemma4-12b-qlora' 2>/dev/null; sleep 1; python3 /mnt/e/Coding/RadeonForge/scripts/progress_dashboard.py --data-dir %DATA% --tracks %TRACKS% --host 0.0.0.0 --port %PORT%"

echo.
echo   Dashboard beendet.
pause
