@echo off
setlocal
echo ============================================
echo 🛠️  REVIT BRIDGE INITIALIZATION
echo ============================================

:: 1. Navigation
cd /d "C:\MyDocuments\mcc-amplify-ai\revit_server\csharp_service"
echo 📂 Working Directory: %CD%

:: 2. Ubuntu Host Mapping Instruction (Reminder)
echo.
echo [1/5] NETWORK CONFIGURATION:
echo Run this on your UBUNTU machine (not here) to map the hostname:
echo echo "191.168.124.64 LT-HQ-277" ^| sudo tee -a /etc/hosts
echo --------------------------------------------

:: 3. Project Clean
echo.
echo [2/5] Cleaning Project Binaries...
dotnet clean
if %errorlevel% neq 0 echo ⚠️ Clean failed (files might be locked by Revit).

:: 4. Project Build
echo.
echo [3/5] Building Revit Service (net48)...
dotnet build
if %errorlevel% neq 0 (
    echo ❌ ERROR: Build failed! Check C# code for syntax errors.
    pause
    exit /b
)
echo ✅ Build Successful!

:: 5. Registry Bypass (Trust the DLL)
echo.
echo [4/5] Registering Trusted DLL...
set "DLL_PATH=%CD%\bin\Debug\net48\RevitService.dll"
reg add "HKEY_CURRENT_USER\Software\Autodesk\Revit\Autodesk Revit 2023\CodeSigning" /v "%DLL_PATH%" /t REG_DWORD /d 1 /f

:: 6. Launch Revit 2023
echo.
echo [5/5] Launching Revit 2023...
start "" "C:\Program Files\Autodesk\Revit 2023\Revit.exe"

echo.
echo ⏳ Waiting for Revit to initialize...
set /a retry_count=0

:CHECK_PORT
timeout /t 10 /nobreak > nul
set /a retry_count+=1

echo.
echo [Attempt %retry_count%] Checking Port 5000 Status...
netstat -ano | findstr LISTENING | findstr :5000 >nul

if %errorlevel% neq 0 (
    if %retry_count% lss 6 (
        echo ⏳ Port not active yet. Retrying in 10s...
        goto CHECK_PORT
    ) else (
        echo ❌ TIMEOUT: Revit took too long to open the port.
        goto END
    )
)

:: 7. TCP Connection Test
echo.
echo ✅ Port is LISTENING! Running final TCP Handshake...
powershell -Command "Test-NetConnection -ComputerName localhost -Port 5000"

:END
echo.
echo ============================================
echo 🚀 SETUP SEQUENCE COMPLETE
echo ============================================
pause