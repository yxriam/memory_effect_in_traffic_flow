@echo off
REM Run from anywhere: the script switches to the prediction\ folder first.
cd /d "%~dp0.."
if exist "scripts\local_python.bat" call "scripts\local_python.bat"
if not defined PY set "PY=python"
REM Chapter 6: install the official Infomap once and run it on infomap_input\*.net (made by ch6_networks.py / export_surrogates.py).
REM Output: results\infomap\
if not exist results\infomap mkdir results\infomap
"%PY%" -m pip install infomap > results\infomap\infomap_install_log.txt 2>&1
"%PY%" run_infomap.py > results\infomap\infomap_log.txt 2>&1
echo finished > results\infomap\INFOMAP_DONE.txt
