@echo off
setlocal enabledelayedexpansion

:: =============================================================================
:: run.bat — MCC-Amplify-v2 Pipeline Launcher (Windows / WSL bridged)
::
:: Usage:
::   script\run.bat                         Show help
::   script\run.bat --revit                 Build + start Revit C# service
::   script\run.bat --revit-stop            Stop the Revit service
::   script\run.bat --check                 Verify Revit + service status
::   script\run.bat --pipeline <pdf_path>   Run full pipeline via WSL Python
::   script\run.bat --frontend              Start Vite frontend via WSL
::
:: On the same Windows machine running Revit, this script:
::   1. Builds and launches RevitService.exe (C# Add-in bridge)
::   2. Verifies the service is healthy on port 5000
::   3. Optionally triggers the Python pipeline via WSL
:: =============================================================================

:: ── Resolve project root ─────────────────────────────────────────────────────
set "SCRIPT_DIR=%~dp0"
set "ROOT=%SCRIPT_DIR%.."
pushd "%ROOT%"
set "ROOT=%CD%"
popd

:: ── Config ───────────────────────────────────────────────────────────────────
set "SERVICE_DIR=%ROOT%\revit_server\csharp_service"
set "REVIT_PORT=5000"
set "REVIT_KEY=my-revit-key-2023"
set "REVIT_EXE_2023=C:\Program Files\Autodesk\Revit 2023\Revit.exe"
set "REVIT_EXE_2022=C:\Program Files\Autodesk\Revit 2022\Revit.exe"
set "SERVICE_BIN=%SERVICE_DIR%\bin\Debug\net48\RevitService.exe"

:: ── Load .env if present ─────────────────────────────────────────────────────
if exist "%ROOT%\.env.bat" call "%ROOT%\.env.bat"

:: =============================================================================
if "%~1"=="" goto :show_help
if /i "%~1"=="--revit"         goto :start_revit
if /i "%~1"=="--revit-stop"    goto :stop_revit
if /i "%~1"=="--check"         goto :check
if /i "%~1"=="--pipeline"      goto :run_pipeline
if /i "%~1"=="--frontend"      goto :run_frontend
echo [ERROR] Unknown option: %~1
goto :show_help

:: =============================================================================
:show_help
echo.
echo  MCC-Amplify-v2 Pipeline Runner  (Windows)
echo  ==========================================
echo.
echo  Usage:
echo    script\run.bat --revit                  Build + start Revit C# service
echo    script\run.bat --revit-stop             Stop RevitService.exe
echo    script\run.bat --check                  Check Revit + service health
echo    script\run.bat --pipeline ^<pdf_path^>    Run pipeline (requires WSL)
echo    script\run.bat --frontend               Start web UI (requires WSL/Node)
echo.
echo  Config (edit this file to change):
echo    Revit service port : %REVIT_PORT%
echo    API key            : %REVIT_KEY%
echo    Service binary     : %SERVICE_BIN%
echo.
goto :eof

:: =============================================================================
:start_revit
echo.
echo [1/4] Checking Revit installation ...
if exist "%REVIT_EXE_2023%" (
    set "REVIT_EXE=%REVIT_EXE_2023%"
    echo  ^> Found Revit 2023
) else if exist "%REVIT_EXE_2022%" (
    set "REVIT_EXE=%REVIT_EXE_2022%"
    echo  ^> Found Revit 2022
) else (
    echo [ERROR] Revit 2022 or 2023 not found.
    echo  Please install Revit first.
    pause
    exit /b 1
)

echo.
echo [2/4] Building C# Revit service ...
cd /d "%SERVICE_DIR%"
call dotnet build
if %errorlevel% neq 0 (
    echo [ERROR] Build failed. Check C# syntax in RevitService/.
    pause
    exit /b 1
)
echo  ^> Build successful

echo.
echo [3/4] Registering DLL trust ...
set "DLL_PATH=%SERVICE_DIR%\bin\Debug\net48\RevitService.dll"
reg add "HKEY_CURRENT_USER\Software\Autodesk\Revit\Autodesk Revit 2023\CodeSigning" ^
    /v "%DLL_PATH%" /t REG_DWORD /d 1 /f >nul 2>&1
echo  ^> Registry updated

echo.
echo [4/4] Launching Revit ...
start "" "%REVIT_EXE%"
echo  ^> Revit starting — waiting for service to come up on port %REVIT_PORT% ...

:: Poll for port
set /a retries=0
:wait_loop
timeout /t 10 /nobreak >nul
set /a retries+=1
netstat -ano | findstr LISTENING | findstr ":%REVIT_PORT%" >nul
if %errorlevel% equ 0 goto :service_up
if %retries% lss 6 (
    echo  ^> Attempt %retries%/6 — port not yet active ...
    goto :wait_loop
)
echo [WARNING] Service did not start within 60 s.
echo  Open Revit manually and verify the Add-in loaded.
goto :eof

:service_up
echo.
echo  ================================================
echo   Revit service is LIVE on port %REVIT_PORT%
echo   Health: http://localhost:%REVIT_PORT%/health
echo  ================================================
echo.
:: Quick TCP test
powershell -Command "Test-NetConnection -ComputerName localhost -Port %REVIT_PORT% -InformationLevel Quiet" >nul 2>&1
if %errorlevel% equ 0 (echo  ^> TCP handshake OK) else (echo  ^> TCP handshake failed — check firewall)
goto :eof

:: =============================================================================
:stop_revit
echo.
echo Stopping RevitService.exe ...
taskkill /f /im RevitService.exe 2>nul
if %errorlevel% equ 0 (echo  ^> Stopped.) else (echo  ^> Not running.)
goto :eof

:: =============================================================================
:check
echo.
echo [CHECK] Revit installation
if exist "%REVIT_EXE_2023%" (echo  ^> Revit 2023: FOUND) else (echo  ^> Revit 2023: not found)
if exist "%REVIT_EXE_2022%" (echo  ^> Revit 2022: FOUND) else (echo  ^> Revit 2022: not found)

echo.
echo [CHECK] Service binary
if exist "%SERVICE_BIN%" (echo  ^> RevitService.exe: BUILT) else (echo  ^> RevitService.exe: NOT BUILT — run --revit first)

echo.
echo [CHECK] Port %REVIT_PORT%
netstat -ano | findstr LISTENING | findstr ":%REVIT_PORT%" >nul
if %errorlevel% equ 0 (
    echo  ^> Port %REVIT_PORT%: LISTENING
) else (
    echo  ^> Port %REVIT_PORT%: not listening — Revit service not running
)

echo.
echo [CHECK] HTTP health endpoint
powershell -Command ^
    "try { $r=(Invoke-WebRequest 'http://localhost:%REVIT_PORT%/health' -UseBasicParsing -TimeoutSec 3).StatusCode; Write-Host '  > Health check: HTTP '$r } catch { Write-Host '  > Health check: FAILED (service not reachable)' }"

echo.
echo [CHECK] WSL availability
wsl --status >nul 2>&1
if %errorlevel% equ 0 (echo  ^> WSL: available) else (echo  ^> WSL: not found — Python pipeline requires WSL or native Python)
goto :eof

:: =============================================================================
:run_pipeline
shift
set "PDF_PATH=%~1"
if "%PDF_PATH%"=="" (
    echo [ERROR] No PDF path provided.
    echo  Usage: script\run.bat --pipeline C:\path\to\floor_plan.pdf
    exit /b 1
)

:: Convert Windows path to WSL path
for /f "delims=" %%i in ('wsl wslpath -a "%PDF_PATH%" 2^>nul') do set "WSL_PDF=%%i"
if "%WSL_PDF%"=="" set "WSL_PDF=%PDF_PATH%"

:: Convert project root to WSL path
for /f "delims=" %%i in ('wsl wslpath -a "%ROOT%" 2^>nul') do set "WSL_ROOT=%%i"

echo.
echo Running pipeline via WSL ...
echo  PDF:  %PDF_PATH%  ^>^>  %WSL_PDF%
echo.

wsl bash -c "cd '%WSL_ROOT%' && ^
    WINDOWS_REVIT_SERVER=http://localhost:%REVIT_PORT% ^
    REVIT_SERVER_API_KEY=%REVIT_KEY% ^
    python3 controller.py '%WSL_PDF%' %2 %3 %4 %5"

if %errorlevel% equ 0 (
    echo.
    echo  ================================================
    echo   Pipeline complete. RVT saved to data/models/rvt/
    echo  ================================================
) else (
    echo.
    echo [ERROR] Pipeline exited with error code %errorlevel%
)
goto :eof

:: =============================================================================
:run_frontend
echo.
echo Starting frontend via WSL ...
for /f "delims=" %%i in ('wsl wslpath -a "%ROOT%\frontend" 2^>nul') do set "WSL_FRONTEND=%%i"
wsl bash -c "cd '%WSL_FRONTEND%' && npm install --silent && npm run dev"
goto :eof
