@echo off
setlocal

set BASE_DIR=C:\Users\laura\OneDrive\TESIS\ETL_LauraChacon
set ETL_DIR=%BASE_DIR%\ETL_code
set PYTHON_EXE=%BASE_DIR%\venv\Scripts\python.exe

echo BASE_DIR=%BASE_DIR%
echo ETL_DIR=%ETL_DIR%
echo PYTHON_EXE=%PYTHON_EXE%
echo.

cd /d "%ETL_DIR%"
echo Running from: %CD%
echo.

"%PYTHON_EXE%" -m src.etl.main_etl

echo.
echo Exit code: %ERRORLEVEL%
pause
endlocal
