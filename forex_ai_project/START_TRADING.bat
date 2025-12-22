@echo off
title Forex AI Trading System
color 0A

echo ============================================================
echo            FOREX AI TRADING SYSTEM - STARTUP
echo ============================================================
echo.

cd /d "C:\Users\pc5\Desktop\ForexAI\forex_ai_project"

echo [0/2] Chiudo eventuali processi precedenti...
taskkill /F /IM python.exe >nul 2>&1
timeout /t 2 >nul

echo [1/2] Avvio Trading Agent (LIVE MODE)...
start "AI Trading Agent" cmd /k "C:\Users\pc5\AppData\Local\Programs\Python\Python313\python.exe" -m agent.trading_agent --mode live --confirm

timeout /t 5 >nul

echo [2/2] Avvio Dashboard...
start "Forex Dashboard" cmd /k "C:\Users\pc5\AppData\Local\Programs\Python\Python313\python.exe" -m streamlit run dashboard.py

echo.
echo ============================================================
echo    SISTEMA AVVIATO!
echo ============================================================
echo.
echo    Trading Agent: In esecuzione (finestra separata)
echo    Dashboard: http://localhost:8501
echo.
echo    IMPORTANTE: Non chiudere le finestre cmd!
echo    Per fermare tutto usa STOP_TRADING.bat
echo ============================================================
pause >nul
