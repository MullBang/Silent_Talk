@echo off
rem Silent Talk demo laptop installer. Needs internet once. Safe to run again (finished parts are skipped).
rem Options are passed to deploy\install.py, e.g. --device cpu
chcp 65001 >nul
title Silent Talk 설치
cd /d "%~dp0"

set "PYEXE="
py -3.11 -c "import sys" >nul 2>&1 && set "PYEXE=py -3.11"
if not defined PYEXE (
    python -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>&1 && set "PYEXE=python"
)
if not defined PYEXE goto nopython

%PYEXE% "%~dp0deploy\install.py" %*
set "RC=%ERRORLEVEL%"
echo.
pause
exit /b %RC%

:nopython
echo.
echo 파이썬 3.11을 찾지 못했습니다.
echo 1) 아래 주소에서 Python 3.11.9 (64-bit) 설치 파일을 받아 실행하세요.
echo    https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
echo 2) 설치 첫 화면 아래의 "Add python.exe to PATH"를 체크하고 "Install Now"를 누르세요.
echo 3) 설치가 끝나면 이 창을 닫고 설치.bat을 다시 실행하세요.
echo.
pause
exit /b 1
