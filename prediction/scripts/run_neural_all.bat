@echo off
REM Run from anywhere: the script switches to the prediction\ folder first.
cd /d "%~dp0.."
if exist "scripts\local_python.bat" call "scripts\local_python.bat"
if not defined PY set "PY=python"
REM Chapter 5 neural baselines (LSTM, Transformer, 128-step context) on all four token datasets.
REM Output: results\neural\
if not exist results\neural mkdir results\neural
"%PY%" -c "import torch,sys;print('python',sys.version);print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" > results\neural\env_check.txt 2>&1
for %%D in (taxi_traj20 geolife_bj500m highd_v20 highd_v100) do (
  "%PY%" neural_ch5.py --data data\%%D.npz --model lstm > results\neural\log_%%D_lstm.txt 2>&1
  "%PY%" neural_ch5.py --data data\%%D.npz --model transformer > results\neural\log_%%D_transformer.txt 2>&1
)
echo finished > results\neural\ALL_DONE.txt
