@echo off
cd /d "%~dp0"
if exist "venv\Scripts\python.exe" (
  "venv\Scripts\python.exe" -m unittest discover -s tests -v
  "venv\Scripts\python.exe" validate_project.py
) else (
  py -m unittest discover -s tests -v
  py validate_project.py
)
pause
