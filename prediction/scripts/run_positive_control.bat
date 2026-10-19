@echo off
REM P0-5 on the GPU: builds the synthetic sets if needed, then two lanes in two windows, in parallel:
REM   lane_positive_control.bat      positive control (5 sets x 3 seeds x {LSTM, Transformer} x {k=4, 128})
REM   lane_transformer_ablation.bat  Transformer history ablation on the real taxi data
REM Finished runs are skipped, so the script can be restarted after an interruption. Keep the computer awake.
REM When both lanes are done:  python prediction\positive_control.py summary   and   python prediction\summarise_neural.py
cd /d "%~dp0.."
if exist "scripts\local_python.bat" call "scripts\local_python.bat"
if not defined PY set "PY=python"
if not exist data\taxi_traj20.npz "%PY%" export_data.py taxi
if not exist data\pc_L16_I3.npz "%PY%" positive_control.py make
start "positive control" cmd /c scripts\lane_positive_control.bat
start "transformer ablation" cmd /c scripts\lane_transformer_ablation.bat
