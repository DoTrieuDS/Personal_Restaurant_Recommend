@echo off
echo 🚀 Starting Local LoRA Restaurant Recommendation Server
echo ================================================

echo 📂 Current directory: %CD%
echo 🔧 Setting environment variables...
set SKIP_PRELOAD=1
set TF_ENABLE_ONEDNN_OPTS=0

echo 💾 Checking system resources...
for /f "tokens=4" %%i in ('wmic computersystem get TotalPhysicalMemory /value') do set "RAM=%%i"
echo    Available RAM: %RAM:~0,-9% GB

echo 🌐 Starting server on port 8000...
echo    Press Ctrl+C to stop
echo    Open restaurant_ui.html in browser after startup
echo.

python -m uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info

echo.
echo 👋 Server stopped
pause 