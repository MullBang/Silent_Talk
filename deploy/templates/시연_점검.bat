@echo off
rem Silent Talk demo check: server health, GPU/CPU, camera, 10 sample recordings timing, offline scan, checklist.
rem Start the demo first (the start .bat). Options go to deploy\check.py (e.g. --no-camera, --ward-runs 10).
chcp 65001 >nul
title Silent Talk 시연 점검
cd /d "%~dp0"

if not exist "%~dp0.venv-server\Scripts\python.exe" goto noinstall

"%~dp0.venv-server\Scripts\python.exe" "%~dp0deploy\check.py" %*
echo.
pause
exit /b

:noinstall
echo.
echo 아직 설치되지 않았습니다. 먼저 설치.bat을 실행하세요.
echo.
pause
exit /b 1
