@echo off
REM Run from anywhere: the script switches to the prediction\ folder first.
cd /d "%~dp0.."
if exist "scripts\local_python.bat" call "scripts\local_python.bat"
if not defined PY set "PY=python"
REM P0-5 positive control: k = 4 (neural_hist.py) vs 128-step context (neural_ch5.py), LSTM and Transformer, on the
REM synthetic sets with a planted long-range dependence (data\pc_*.npz, built by positive_control.py make).
REM Seed 0 of every set first, then seeds 1 and 2. Output: results\neural\positive_control\
set "O=results\neural\positive_control"
if not exist %O% mkdir %O%
echo running > %O%\RUNNING.txt
for %%S in (0 1 2) do (
  for %%D in (pc_L16_I2 pc_L8_I2 pc_L32_I2 pc_L16_I1 pc_L16_I3) do (
    if not exist %O%\hist_%%D_k4_s%%S.json "%PY%" neural_hist.py --data data\%%D.npz --hist 4 --seed %%S --out %O% > %O%\log_%%D_lstm_k4_s%%S.txt 2>&1
    if not exist %O%\neural_%%D_lstm_s%%S.json "%PY%" neural_ch5.py --data data\%%D.npz --model lstm --seed %%S --tag _s%%S --out %O% > %O%\log_%%D_lstm_ctx128_s%%S.txt 2>&1
    if not exist %O%\hist_transformer_%%D_k4_s%%S.json "%PY%" neural_hist.py --data data\%%D.npz --model transformer --hist 4 --seed %%S --out %O% > %O%\log_%%D_transformer_k4_s%%S.txt 2>&1
    if not exist %O%\neural_%%D_transformer_s%%S.json "%PY%" neural_ch5.py --data data\%%D.npz --model transformer --seed %%S --tag _s%%S --out %O% > %O%\log_%%D_transformer_ctx128_s%%S.txt 2>&1
  )
)
del %O%\RUNNING.txt
echo finished > %O%\DONE.txt
