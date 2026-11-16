@echo off
title Real Cordova AI Analytics - Arranque
cd /d "%~dp0"
echo ============================================================
echo   Iniciando Real Cordova AI Analytics (Completo)
echo ============================================================
powershell -ExecutionPolicy Bypass -File .\run_daemon.ps1
pause
