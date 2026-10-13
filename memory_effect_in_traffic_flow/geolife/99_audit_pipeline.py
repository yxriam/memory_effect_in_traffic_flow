from __future__ import annotations

import py_compile
import subprocess
import sys
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
AUDIT = BASE / "_audit"


def run_step(name: str, args: list[str], timeout: int = 600) -> None:
    cmd = [sys.executable] + args
    print(f"\n=== {name} ===")
    print(" ".join(str(x) for x in cmd))
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {result.returncode}")


def compile_scripts() -> None:
    print("\n=== compile scripts ===")
    for path in sorted(BASE.glob("*.py")):
        py_compile.compile(str(path), doraise=True)
        print(f"OK {path.name}")


def require_nonempty_csv(path: Path, min_rows: int = 1) -> None:
    if not path.exists():
        raise RuntimeError(f"missing output: {path}")
    df = pd.read_csv(path)
    if len(df) < min_rows:
        raise RuntimeError(f"output has too few rows: {path} rows={len(df)}")
    print(f"checked {path}: rows={len(df)}")


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    cache_dir = AUDIT / "cache"

    run_step("environment", [str(BASE / "00_check_environment.py")], timeout=120)
    compile_scripts()

    run_step(
        "preprocess smoke",
        [
            str(BASE / "01_preprocess_no_leakage.py"),
            "--cache-dir",
            str(cache_dir),
            "--max-users",
            "4",
            "--max-trips-per-user",
            "3",
        ],
        timeout=300,
    )
    require_nonempty_csv(cache_dir / "_summary.csv")
    summary = pd.read_csv(cache_dir / "_summary.csv")
    ok_users = int((summary["status"] == "ok").sum())
    if ok_users < 1:
        raise RuntimeError("preprocess smoke did not create any valid cached users")

    run_step(
        "memory-flow smoke",
        [
            str(BASE / "02_memory_flow_metrics.py"),
            "--cache-dir",
            str(cache_dir),
            "--out-dir",
            str(AUDIT / "02_memory_flow"),
            "--max-users",
            "4",
            "--min-trigrams",
            "5",
        ],
        timeout=300,
    )
    require_nonempty_csv(AUDIT / "02_memory_flow" / "memory_flow_metrics.csv")

    run_step(
        "correlation smoke",
        [
            str(BASE / "03_memory_driver_correlation.py"),
            "--cache-dir",
            str(cache_dir),
            "--memory-flow-dir",
            str(AUDIT / "02_memory_flow"),
            "--out-dir",
            str(AUDIT / "03_correlation_analysis"),
            "--fig-dir",
            str(AUDIT / "figures"),
            "--max-users",
            "4",
        ],
        timeout=300,
    )
    require_nonempty_csv(AUDIT / "03_correlation_analysis" / "geolife_driver_memory_merged.csv")

    run_step(
        "mode memory smoke",
        [
            str(BASE / "04_mode_hurst_memory.py"),
            "--out-dir",
            str(AUDIT / "04_mode_memory"),
            "--max-modes",
            "3",
            "--max-files-per-mode",
            "1",
            "--max-segments-per-file",
            "3",
            "--max-lag",
            "10",
            "--fit-max-lag",
            "10",
            "--min-points",
            "10",
        ],
        timeout=600,
    )
    require_nonempty_csv(AUDIT / "04_mode_memory" / "mode_memory_summary.csv")

    run_step(
        "resample sensitivity smoke",
        [
            str(BASE / "04a_resample_sensitivity_select.py"),
            "--candidate-seconds",
            "2,5",
            "--out-dir",
            str(AUDIT / "04_resample_sensitivity"),
            "--final-out-dir",
            str(AUDIT / "04_mode_memory_selected"),
            "--sensitivity-max-modes",
            "3",
            "--sensitivity-max-files-per-mode",
            "1",
            "--sensitivity-max-segments-per-file",
            "2",
            "--gap-sample-files-per-mode",
            "2",
            "--max-lag-seconds",
            "60",
            "--fit-min-seconds",
            "10",
            "--fit-max-seconds",
            "40",
            "--skip-final-rerun",
        ],
        timeout=300,
    )
    require_nonempty_csv(AUDIT / "04_resample_sensitivity" / "resample_selection_summary.csv")

    run_step(
        "summary smoke",
        [
            str(BASE / "05_summarize_results.py"),
            "--base-out-dir",
            str(AUDIT),
        ],
        timeout=120,
    )
    if not (AUDIT / "summary_report.txt").exists():
        raise RuntimeError("summary report was not created")

    print("\nAUDIT PASSED")
    print(f"audit outputs: {AUDIT}")


if __name__ == "__main__":
    main()
