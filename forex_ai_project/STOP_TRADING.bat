@echo off
title Forex AI Trading System - STOP
color 0C

echo ============================================================
echo            FOREX AI TRADING SYSTEM - SHUTDOWN
echo ============================================================
echo.

echo Chiudo Trading Agent...
taskkill /FI "WINDOWTITLE eq AI Trading Agent*" /F >nul 2>&1

echo Chiudo Dashboard...
taskkill /FI "WINDOWTITLE eq Forex Dashboard*" /F >nul 2>&1

echo Chiudo processi Python...
taskkill /IM python.exe /F >nul 2>&1

echo.
echo ============================================================
echo    SISTEMA FERMATO!
echo ============================================================
echo.
pause
