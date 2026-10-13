from __future__ import annotations

import importlib
import sys
from pathlib import Path


REQUIRED = [
    "numpy",
    "pandas",
    "torch",
    "sklearn",
    "scipy",
    "matplotlib",
    "tqdm",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def pipeline_root() -> Path:
    return Path(__file__).resolve().parent


def first_existing_path(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def main() -> None:
    root = project_root()
    base = pipeline_root()
    print(f"python: {sys.executable}")
    print(f"version: {sys.version.split()[0]}")
    print(f"project_root: {root}")
    print(f"pipeline_root: {base}")

    missing = []
    for name in REQUIRED:
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", "unknown")
            print(f"{name}: {version}")
        except Exception as exc:
            missing.append((name, str(exc)))
            print(f"{name}: MISSING ({exc})")

    required_paths = [
        first_existing_path([base / "data_raw" / "Data", root / "Data"]),
        first_existing_path([base / "data_raw" / "trips_by_mode_data", root / "trips_by_mode" / "data"]),
    ]
    for path in required_paths:
        print(f"path {path}: {'OK' if path.exists() else 'MISSING'}")

    if missing:
        names = ", ".join(name for name, _ in missing)
        raise SystemExit(f"Missing required packages: {names}")


if __name__ == "__main__":
    main()
