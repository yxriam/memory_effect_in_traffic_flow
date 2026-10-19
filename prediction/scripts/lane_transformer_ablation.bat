@echo off
REM Run from anywhere: the script switches to the prediction\ folder first.
cd /d "%~dp0.."
if exist "scripts\local_python.bat" call "scripts\local_python.bat"
if not defined PY set "PY=python"
REM P0-5 (second part): history-length ablation of the Transformer on the real taxi data, as lane_taxi.bat does for
REM the LSTM (exactly the last k symbols), plus two more seeds of the 128-step Transformer.
REM Output: results\neural\hist\hist_transformer_taxi_traj20_k<k>_s<seed>.json, results\neural\neural_taxi_traj20_transformer_s1/_s2.json
set "O=results\neural\hist"
if not exist %O% mkdir %O%
echo running > %O%\transformer_RUNNING.txt
for %%K in (1 2 3 4 8 16 32) do (
  if not exist %O%\hist_transformer_taxi_traj20_k%%K_s0.json "%PY%" neural_hist.py --data data\taxi_traj20.npz --model transformer --hist %%K --seed 0 > %O%\log_transformer_taxi_traj20_k%%K_s0.txt 2>&1
)
for %%S in (1 2) do (
  if not exist %O%\hist_transformer_taxi_traj20_k4_s%%S.json "%PY%" neural_hist.py --data data\taxi_traj20.npz --model transformer --hist 4 --seed %%S > %O%\log_transformer_taxi_traj20_k4_s%%S.txt 2>&1
  if not exist results\neural\neural_taxi_traj20_transformer_s%%S.json "%PY%" neural_ch5.py --data data\taxi_traj20.npz --model transformer --seed %%S --tag _s%%S > %O%\log_taxi_traj20_transformer128_s%%S.txt 2>&1
)
del %O%\transformer_RUNNING.txt
echo finished > %O%\transformer_DONE.txt
