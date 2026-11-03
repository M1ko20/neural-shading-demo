@echo off
cd /d "%~dp0"
for %%d in (1 2 3 4 5 6 7) do .venv\Scripts\python.exe neural_lab\lab.py --headless 60 --demo %%d >> neural_lab\run_log.txt 2>&1
echo done >> neural_lab\run_log.txt
