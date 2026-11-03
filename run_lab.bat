@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe neural_lab\lab.py %*
if errorlevel 1 pause
