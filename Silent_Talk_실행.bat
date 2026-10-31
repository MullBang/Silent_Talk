@echo off
rem Silent Talk 실행: 서버(모델)와 화면을 각각 새 창으로 켜고, 준비되면 브라우저로 녹화 화면을 연다.
rem 끄기: "Silent Talk 서버"와 "Silent Talk 화면" 창을 닫는다.
chcp 65001 >nul
title Silent Talk 실행
cd /d "%~dp0"

set "HEALTH=http://127.0.0.1:8000/health"
set "FRONT=http://127.0.0.1:5173/"
set "APP=http://localhost:5173/#record"
set "APPPATHS=SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\chrome.exe"

curl -s --max-time 2 -o nul "%HEALTH%"
if errorlevel 1 (
    echo 서버와 모델을 켭니다...
    start "Silent Talk 서버" /d "%~dp0backend" cmd /k py -3.11 -m uvicorn main:app --host 127.0.0.1 --port 8000
) else (
    echo 서버가 이미 켜져 있습니다.
)

curl -s --max-time 2 -o nul "%FRONT%"
if errorlevel 1 (
    echo 화면을 켭니다...
    start "Silent Talk 화면" /d "%~dp0frontend" cmd /k npm.cmd run dev -- --host 127.0.0.1 --port 5173 --strictPort
) else (
    echo 화면이 이미 켜져 있습니다.
)

echo 모델을 불러오는 중입니다. 잠시 기다려 주세요...
set /a TRY=0

:wait
curl -s --max-time 2 "%HEALTH%" 2>nul | findstr /c:"roi_worker\":\"ready" >nul || goto notyet
curl -s --max-time 2 "%HEALTH%" 2>nul | findstr /c:"infer_worker\":\"ready" >nul || goto notyet
curl -s --max-time 2 -o nul "%FRONT%" || goto notyet
goto ready

:notyet
set /a TRY+=1
if %TRY% geq 90 goto failed
ping -n 3 127.0.0.1 >nul
goto wait

:ready
echo 준비됐습니다. 녹화 화면을 엽니다.
reg query "HKLM\%APPPATHS%" >nul 2>&1 && goto chrome
reg query "HKCU\%APPPATHS%" >nul 2>&1 && goto chrome
start "" msedge "%APP%"
goto done

:chrome
start "" chrome "%APP%"

:done
echo.
echo 끄려면 "Silent Talk 서버"와 "Silent Talk 화면" 창을 닫으세요.
ping -n 6 127.0.0.1 >nul
exit /b 0

:failed
echo.
echo 3분이 지나도 준비되지 않았습니다. "Silent Talk 서버"와 "Silent Talk 화면" 창의 오류를 확인하세요.
pause
exit /b 1
