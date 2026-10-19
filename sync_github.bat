@echo off
REM Double-click to upload everything in this folder to GitHub (yxriam/memory_effect_in_traffic_flow).
REM Optional: sync_github.bat "your commit message"
setlocal
cd /d "%~dp0"
where git >nul 2>nul || (echo [ERROR] Git is not installed or not on PATH. & pause & exit /b 1)
if not exist ".git" (echo [ERROR] This folder is not a git repository: %CD% & pause & exit /b 1)

echo === Changes to upload ===
git -c core.quotepath=false status --short
echo.

set "MSG=%~1"
if "%MSG%"=="" set /p "MSG=Commit message (press Enter for 'Update'): "
if "%MSG%"=="" set "MSG=Update"

git add -A || goto :fail
powershell -NoProfile -ExecutionPolicy Bypass -Command "$big = git -c core.quotepath=false diff --cached --name-only | Where-Object { (Test-Path -LiteralPath $_) -and ((Get-Item -LiteralPath $_).Length -gt 50MB) }; if ($big) { Write-Host 'Files over 50 MB (GitHub refuses files over 100 MB):'; $big; exit 1 }"
if errorlevel 1 (git reset -q & echo Nothing was uploaded. Add these files to .gitignore and run again. & pause & exit /b 1)

git diff --cached --quiet
if errorlevel 1 (git commit -q -m "%MSG%" || goto :fail) else (echo No new changes to commit.)

echo === Getting the latest version from GitHub ===
git pull --rebase origin main || goto :fail
echo === Uploading ===
git push origin main || goto :fail
echo.
git log --oneline -3
echo.
echo Done: https://github.com/yxriam/memory_effect_in_traffic_flow
pause
exit /b 0

:fail
echo.
echo [ERROR] The step above failed. Your files are still safe in this folder.
echo Copy the message above and send it to Claude.
pause
exit /b 1
