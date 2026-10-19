@echo off
REM History-length ablation of the chapter 5 LSTM: three lanes in three windows, in parallel.
REM Keep the computer awake. Each lane writes results\neural\hist\<lane>_DONE.txt when finished.
REM Runs listed in results\neural\hist\SKIP.txt ("<dataset> <k> <seed>") are skipped.
cd /d "%~dp0"
start "ablation taxi" cmd /c lane_taxi.bat
start "ablation highd" cmd /c lane_highd.bat
start "ablation geolife" cmd /c lane_geolife.bat
