@echo off
REM P0-5 on the GPU, one job at a time in this window (two Transformers in parallel on the 2 GB MX250 were about
REM three times slower in total than running them one after the other):
REM   0. test_resume.bat (once, about 2 minutes): checks that an interrupted run continues correctly on this computer
REM   1. lane_positive_control.bat      positive control (5 sets x 3 seeds x {LSTM, Transformer} x {k=4, 128});
REM                                     seed 0 of every set first, then seeds 1 and 2
REM   2. lane_transformer_ablation.bat  Transformer history ablation on the real taxi data
REM   3. positive_control.py summary and summarise_neural.py (logs in results\neural\)
REM Interruptions: finished runs are skipped, and a run that was stopped part-way continues after its last finished
REM epoch (resume.py), so after a restart, power cut or closed window just start this script again.
REM The training scripts ask Windows not to sleep while they run (keep_awake.py). Closing the lid still follows the
REM power plan: set "When I close the lid" (plugged in) to "Do nothing" before closing it. Keep the charger in.
cd /d "%~dp0.."
if exist "scripts\local_python.bat" call "scripts\local_python.bat"
if not defined PY set "PY=python"
if not exist data\taxi_traj20.npz "%PY%" export_data.py taxi
if not exist data\pc_L16_I3.npz "%PY%" positive_control.py make
if not exist results\neural\resume_test\RESULT.txt call scripts\test_resume.bat
cd /d "%~dp0.."
echo [%date% %time%] 1/3 positive control (log files in results\neural\positive_control\)
call scripts\lane_positive_control.bat
cd /d "%~dp0.."
echo [%date% %time%] 2/3 Transformer history ablation (log files in results\neural\hist\)
call scripts\lane_transformer_ablation.bat
cd /d "%~dp0.."
echo [%date% %time%] 3/3 summaries
"%PY%" positive_control.py summary > results\neural\positive_control\summary_log.txt 2>&1
"%PY%" summarise_neural.py > results\neural\summarise_log.txt 2>&1
echo [%date% %time%] all done. Results: results\neural\positive_control\summary_log.txt
pause
