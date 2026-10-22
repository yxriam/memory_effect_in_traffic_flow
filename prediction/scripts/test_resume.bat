@echo off
REM Checks on this computer that an interrupted training run continues and ends with the same result as an
REM uninterrupted one (resume.py): a short LSTM run (k = 4, 4 epochs of 20,000 targets, about 1-2 minutes) once straight
REM through (A), and once stopped after epoch 2 and started again (B). Result line in results\neural\resume_test\RESULT.txt:
REM PASS if the test cross-entropies of A and B differ by less than 0.005 bits (GPU kernels are not bit-exact).
cd /d "%~dp0.."
if exist "scripts\local_python.bat" call "scripts\local_python.bat"
if not defined PY set "PY=python"
set "T=results\neural\resume_test"
if not exist %T%\A mkdir %T%\A
if not exist %T%\B mkdir %T%\B
if exist %T%\A\hist_taxi_traj20_k4_s0.resume.pt del %T%\A\hist_taxi_traj20_k4_s0.resume.pt
if exist %T%\B\hist_taxi_traj20_k4_s0.resume.pt del %T%\B\hist_taxi_traj20_k4_s0.resume.pt
set "ARGS=--data data\taxi_traj20.npz --hist 4 --seed 0 --epochs 4 --patience 10 --per_epoch 20000 --max_val 20000"
echo [%date% %time%] resume test: run A (straight through)
"%PY%" neural_hist.py %ARGS% --out %T%\A > %T%\log_A.txt 2>&1
echo [%date% %time%] resume test: run B (stopped after epoch 2, then started again)
"%PY%" neural_hist.py %ARGS% --out %T%\B --stop_after 2 > %T%\log_B1.txt 2>&1
"%PY%" neural_hist.py %ARGS% --out %T%\B > %T%\log_B2.txt 2>&1
"%PY%" -c "import json,sys; a=json.load(open(sys.argv[1]))['test']['bits']; b=json.load(open(sys.argv[2]))['test']['bits']; print('A %%.6f  B %%.6f  diff %%.1e  %%s' %% (a, b, abs(a-b), 'PASS' if abs(a-b) < 0.005 else 'CHECK'))" %T%\A\hist_taxi_traj20_k4_s0.json %T%\B\hist_taxi_traj20_k4_s0.json > %T%\RESULT.txt 2>&1
type %T%\RESULT.txt
