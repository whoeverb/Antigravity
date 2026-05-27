@echo off
title Stock Research & Projection App Launcher
echo ===================================================
echo   Launching Stock Research & Projection App...
echo ===================================================
echo.
call "%~dp0env\Scripts\activate.bat"
streamlit run "%~dp0main.py"
pause
