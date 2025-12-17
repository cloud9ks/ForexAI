@echo off
title Forex AI Trading System
color 0A

echo ============================================================
echo            FOREX AI TRADING SYSTEM - STARTUP
echo ============================================================
echo.

cd /d "C:\Users\pc5\Desktop\ForexAI\forex_ai_project"

echo [1/2] Avvio Trading Agent (LIVE MODE)...
start "AI Trading Agent" cmd /k "C:\Users\pc5\AppData\Local\Programs\Python\Python313\python.exe" -m agent.trading_agent --mode live --confirm

timeout /t 3 >nul

echo [2/2] Avvio Dashboard...
start "Forex Dashboard" cmd /k "C:\Users\pc5\AppData\Local\Programs\Python\Python313\python.exe" -m streamlit run dashboard.py

echo.
echo ============================================================
echo    SISTEMA AVVIATO!
echo ============================================================
echo.
echo    Trading Agent: In esecuzione (finestra separata)
echo    Dashboard: http://localhost:8503
echo.
echo    Premi un tasto per chiudere questa finestra...
echo ============================================================
pause >nul
