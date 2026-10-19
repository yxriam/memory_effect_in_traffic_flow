@echo off
REM Run from anywhere: the script switches to the prediction\ folder first.
cd /d "%~dp0.."
if exist "scripts\local_python.bat" call "scripts\local_python.bat"
if not defined PY set "PY=python"
REM History-length ablation, highd lane. Started by run_hist_all.bat or on its own. Output: results\neural\hist\
if not exist results\neural\hist mkdir results\neural\hist
echo running > results\neural\hist\highd_RUNNING.txt
"%PY%" neural_hist.py --data data\highd_v20.npz --hist 1 --seed 0 > results\neural\hist\log_highd_v20_k1_s0.txt 2>&1
"%PY%" neural_hist.py --data data\highd_v20.npz --hist 2 --seed 0 > results\neural\hist\log_highd_v20_k2_s0.txt 2>&1
"%PY%" neural_hist.py --data data\highd_v20.npz --hist 4 --seed 0 > results\neural\hist\log_highd_v20_k4_s0.txt 2>&1
"%PY%" neural_hist.py --data data\highd_v20.npz --hist 8 --seed 0 > results\neural\hist\log_highd_v20_k8_s0.txt 2>&1
"%PY%" neural_hist.py --data data\highd_v20.npz --hist 32 --seed 0 > results\neural\hist\log_highd_v20_k32_s0.txt 2>&1
"%PY%" neural_hist.py --data data\highd_v100.npz --hist 1 --seed 0 > results\neural\hist\log_highd_v100_k1_s0.txt 2>&1
"%PY%" neural_hist.py --data data\highd_v100.npz --hist 2 --seed 0 > results\neural\hist\log_highd_v100_k2_s0.txt 2>&1
"%PY%" neural_hist.py --data data\highd_v100.npz --hist 4 --seed 0 > results\neural\hist\log_highd_v100_k4_s0.txt 2>&1
"%PY%" neural_hist.py --data data\highd_v100.npz --hist 8 --seed 0 > results\neural\hist\log_highd_v100_k8_s0.txt 2>&1
"%PY%" neural_hist.py --data data\highd_v100.npz --hist 32 --seed 0 > results\neural\hist\log_highd_v100_k32_s0.txt 2>&1
del results\neural\hist\highd_RUNNING.txt
echo finished > results\neural\hist\highd_DONE.txt
