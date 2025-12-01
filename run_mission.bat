@echo off
echo Setting up Project Armstrong Environment...
call C:\Users\deepe\anaconda3\Scripts\activate.bat armstrong_env

echo Checking for API Key...
if "%GOOGLE_API_KEY%"=="" (
    echo WARNING: GOOGLE_API_KEY is not set. The agent will run in MOCK MODE.
    echo To set it, run: set GOOGLE_API_KEY=your_key_here
    echo.
)

echo Starting Mission Control...
python main_mission.py
pause
