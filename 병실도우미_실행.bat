@echo off
rem 병실 무음 도우미 실행: 서버(모델)와 화면을 켜고(이미 켜져 있으면 그대로 둠), 준비되면 환자 창과 간호사실 창을 따로 연다.
rem 두 창은 반드시 같은 주소(http://localhost:5173)로 연다. 두 창은 BroadcastChannel로 통하는데, 주소가 같은 창끼리만 통한다.
rem 127.0.0.1로 열면 localhost 창과 연결되지 않는다. 두 화면 배치에서는 간호사 창을 프로젝터(확장 화면)로 옮기면 된다.
rem 끄기: "Silent Talk 서버"와 "Silent Talk 화면" 창을 닫는다.
chcp 65001 >nul
title 병실 무음 도우미 실행
cd /d "%~dp0"

set "HEALTH=http://127.0.0.1:8000/health"
set "FRONT=http://127.0.0.1:5173/"
set "WARD=http://localhost:5173/#ward"
set "NURSE=http://localhost:5173/#nurse"
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
echo 준비됐습니다. 환자 화면과 간호사실 화면을 엽니다.
set "BROWSER=msedge"
reg query "HKLM\%APPPATHS%" >nul 2>&1 && set "BROWSER=chrome"
reg query "HKCU\%APPPATHS%" >nul 2>&1 && set "BROWSER=chrome"
start "" %BROWSER% --new-window "%WARD%"
rem 첫 창이 브라우저를 다 띄운 뒤 두 번째 창을 연다 (같은 브라우저 프로필이어야 두 창이 통한다).
ping -n 4 127.0.0.1 >nul
start "" %BROWSER% --new-window "%NURSE%"

echo.
echo 환자 화면:   %WARD%
echo 간호사 화면: %NURSE%
echo  - 간호사 화면에서 [소리 켜기]를 한 번 눌러 주세요.
echo  - 두 화면 배치면 간호사 창을 프로젝터(확장 화면)로 옮기세요.
echo  - 카메라 없이 연습: %WARD%?practice=1
echo 끄려면 "Silent Talk 서버"와 "Silent Talk 화면" 창을 닫으세요.
ping -n 8 127.0.0.1 >nul
exit /b 0

:failed
echo.
echo 3분이 지나도 준비되지 않았습니다. "Silent Talk 서버"와 "Silent Talk 화면" 창의 오류를 확인하세요.
pause
exit /b 1
