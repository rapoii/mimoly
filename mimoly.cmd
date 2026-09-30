@echo off
REM mimoly-cli launcher for Windows CMD / PowerShell / Git Bash
REM Forwards all args to the Python wrapper without modifying upstream CLI.

setlocal
set "SCRIPT_DIR=%~dp0"
python "%SCRIPT_DIR%mimoly-cli" %*
exit /b %ERRORLEVEL%
