@echo off
rem Silent Talk demo stop: closes only the server/screen processes started from this folder.
chcp 65001 >nul
title Silent Talk 시연 중지
cd /d "%~dp0"

if not exist "%~dp0.venv-server\Scripts\python.exe" goto noinstall

"%~dp0.venv-server\Scripts\python.exe" "%~dp0deploy\launch.py" --stop
echo.
pause
exit /b

:noinstall
echo.
echo 설치되지 않은 폴더입니다. 서버·화면 창이 떠 있으면 창을 직접 닫으세요.
echo.
pause
exit /b 1
