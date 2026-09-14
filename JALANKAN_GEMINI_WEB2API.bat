@echo off
setlocal
cd /d "%~dp0"
echo Menjalankan Gemini Web2API di http://127.0.0.1:8081 ...
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe gemini_web2api.py
) else (
  python gemini_web2api.py
)

