$ErrorActionPreference = "Stop"

# Use the Python on PATH by default; override with  $env:PYTHON = "C:\path\to\python.exe"
$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$Base = Split-Path -Parent $MyInvocation.MyCommand.Path

& $Python "$Base\00_check_environment.py"
& $Python "$Base\01_preprocess_no_leakage.py"
& $Python "$Base\02_memory_flow_metrics.py"
& $Python "$Base\03_memory_driver_correlation.py"
& $Python "$Base\04a_resample_sensitivity_select.py" --sensitivity-max-files-per-mode 100000 --gap-sample-files-per-mode 100000 --max-gaps-per-mode 10000000
& $Python "$Base\05_summarize_results.py"
