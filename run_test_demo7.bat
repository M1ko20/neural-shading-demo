@echo off
cd /d "%~dp0"
echo Running demo 7 test on the GPU, this takes a few minutes...
.venv\Scripts\python.exe neural_lab\test_demo7.py > neural_lab\test_demo7_log.txt 2>&1
type neural_lab\test_demo7_log.txt
echo.
echo Finished. You can close this window.
pause
