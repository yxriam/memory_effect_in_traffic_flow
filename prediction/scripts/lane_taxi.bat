@echo off
REM Run from anywhere: the script switches to the prediction\ folder first.
cd /d "%~dp0.."
if exist "scripts\local_python.bat" call "scripts\local_python.bat"
if not defined PY set "PY=python"
REM History-length ablation, taxi lane. Started by run_hist_all.bat or on its own. Output: results\neural\hist\
if not exist results\neural\hist mkdir results\neural\hist
echo running > results\neural\hist\taxi_RUNNING.txt
"%PY%" neural_hist.py --data data\taxi_traj20.npz --hist 1 --seed 0 > results\neural\hist\log_taxi_traj20_k1_s0.txt 2>&1
"%PY%" neural_hist.py --data data\taxi_traj20.npz --hist 2 --seed 0 > results\neural\hist\log_taxi_traj20_k2_s0.txt 2>&1
"%PY%" neural_hist.py --data data\taxi_traj20.npz --hist 3 --seed 0 > results\neural\hist\log_taxi_traj20_k3_s0.txt 2>&1
"%PY%" neural_hist.py --data data\taxi_traj20.npz --hist 4 --seed 0 > results\neural\hist\log_taxi_traj20_k4_s0.txt 2>&1
"%PY%" neural_hist.py --data data\taxi_traj20.npz --hist 8 --seed 0 > results\neural\hist\log_taxi_traj20_k8_s0.txt 2>&1
"%PY%" neural_hist.py --data data\taxi_traj20.npz --hist 16 --seed 0 > results\neural\hist\log_taxi_traj20_k16_s0.txt 2>&1
"%PY%" neural_hist.py --data data\taxi_traj20.npz --hist 32 --seed 0 > results\neural\hist\log_taxi_traj20_k32_s0.txt 2>&1
"%PY%" neural_hist.py --data data\taxi_traj20.npz --hist 2 --seed 1 > results\neural\hist\log_taxi_traj20_k2_s1.txt 2>&1
"%PY%" neural_hist.py --data data\taxi_traj20.npz --hist 32 --seed 1 > results\neural\hist\log_taxi_traj20_k32_s1.txt 2>&1
"%PY%" neural_ch5.py --data data\taxi_traj20.npz --model lstm --seed 1 --tag _s1 > results\neural\hist\log_taxi_traj20_lstm128_s1.txt 2>&1
"%PY%" neural_ch5.py --data data\taxi_traj20.npz --model lstm --seed 2 --tag _s2 > results\neural\hist\log_taxi_traj20_lstm128_s2.txt 2>&1
del results\neural\hist\taxi_RUNNING.txt
echo finished > results\neural\hist\taxi_DONE.txt
