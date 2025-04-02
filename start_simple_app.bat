@echo off
echo Starting Simple Document Evaluation Tool...
echo.

cd /d "%~dp0"
python run.py --simple

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo An error occurred while starting the application.
    echo Please check that Python and all required packages are installed.
    echo.
    pause
) 