@echo off
chcp 65001 > nul
echo ========================================================
echo    DeNoiseAI - Khởi Động Server & Tạo Tên Miền Public
echo ========================================================
echo.

echo [1/2] Đang khởi động Backend FastAPI (Port 8000)...
start /b python app/server.py

timeout /t 3 > nul

echo [2/2] Đang tạo tên miền Public HTTPS qua Cloudflare Tunnel...
"C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel --url http://localhost:8000

pause
