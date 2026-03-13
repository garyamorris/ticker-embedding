@echo off
setlocal

set "ROOT=%~dp0"
set "PYTHON=%ROOT%.venv\Scripts\python.exe"
set "APP=%ROOT%app.py"

pushd "%ROOT%" >nul

if not exist "%PYTHON%" (
    echo Missing virtual environment interpreter at "%PYTHON%". Create .venv and install requirements first.
    popd >nul
    exit /b 1
)

if not exist "%APP%" (
    echo Could not find app.py at "%APP%".
    popd >nul
    exit /b 1
)

"%PYTHON%" -m streamlit run "%APP%" %*
set "EXITCODE=%ERRORLEVEL%"
popd >nul
exit /b %EXITCODE%
