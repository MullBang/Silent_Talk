@echo off
rem Silent Talk demo start: server window + screen window, then patient (#ward) and nurse (#nurse) browser windows.
rem Does not start a second copy if already running. Options go to deploy\launch.py (e.g. --device cpu).
chcp 65001 >nul
title Silent Talk 시연 실행
cd /d "%~dp0"

if not exist "%~dp0.venv-server\Scripts\python.exe" goto noinstall

"%~dp0.venv-server\Scripts\python.exe" "%~dp0deploy\launch.py" %*
if errorlevel 1 goto failed
echo.
echo 이 창은 10초 뒤 닫힙니다. (서버·화면 창은 그대로 둡니다)
ping -n 11 127.0.0.1 >nul
exit /b 0

:failed
echo.
pause
exit /b 1

:noinstall
echo.
echo 아직 설치되지 않았습니다. 먼저 설치.bat을 실행하세요.
echo.
pause
exit /b 1
