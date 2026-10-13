from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


SEED = 42
SEQ_LEN_DEFAULT = 5


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TrajRNN(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.rnn = nn.RNN(2, hidden_size, 1, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


class TrajLSTM(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(2, hidden_size, 1, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class DeltaWindowDataset(Dataset):
    def __init__(
        self,
        trips_lat: list[np.ndarray],
        trips_lon: list[np.ndarray],
        seq_len: int,
        sigma_lon: float,
        sigma_lat: float,
    ) -> None:
        xs = []
        ys = []
        anchors = []
        targets = []
        for lat, lon in zip(trips_lat, trips_lon):
            lat = np.asarray(lat, dtype=np.float32)
            lon = np.asarray(lon, dtype=np.float32)
            if len(lat) < seq_len + 2:
                continue
            dlat = np.diff(lat)
            dlon = np.diff(lon)
            dlat_n = dlat / sigma_lat
            dlon_n = dlon / sigma_lon
            for i in range(len(dlat) - seq_len):
                xs.append(np.stack([dlon_n[i : i + seq_len], dlat_n[i : i + seq_len]], axis=1))
                ys.append(np.array([dlon_n[i + seq_len], dlat_n[i + seq_len]], dtype=np.float32))
                anchors.append(np.array([lon[i + seq_len], lat[i + seq_len]], dtype=np.float32))
                targets.append(np.array([lon[i + seq_len + 1], lat[i + seq_len + 1]], dtype=np.float32))

        if xs:
            self.x = np.asarray(xs, dtype=np.float32)
            self.y = np.asarray(ys, dtype=np.float32)
            self.anchor = np.asarray(anchors, dtype=np.float32)
            self.target = np.asarray(targets, dtype=np.float32)
        else:
            self.x = np.zeros((0, seq_len, 2), dtype=np.float32)
            self.y = np.zeros((0, 2), dtype=np.float32)
            self.anchor = np.zeros((0, 2), dtype=np.float32)
            self.target = np.zeros((0, 2), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.x[idx]),
            torch.from_numpy(self.y[idx]),
            torch.from_numpy(self.anchor[idx]),
            torch.from_numpy(self.target[idx]),
        )


def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    radius = 6371000.0
    lat1r = np.deg2rad(lat1)
    lat2r = np.deg2rad(lat2)
    dlat = lat2r - lat1r
    dlon = np.deg2rad(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def split_train_val(
    trips_lat: list[np.ndarray],
    trips_lon: list[np.ndarray],
    val_frac: float,
    seq_len: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    train_lat = []
    train_lon = []
    val_lat = []
    val_lon = []
    for lat, lon in zip(trips_lat, trips_lon):
        n = len(lat)
        if n < seq_len * 2 + 4:
            train_lat.append(lat)
            train_lon.append(lon)
            continue
        cut = int(n * (1.0 - val_frac))
        if cut < seq_len + 2 or n - cut < seq_len + 2:
            train_lat.append(lat)
            train_lon.append(lon)
            continue
        train_lat.append(lat[:cut])
        train_lon.append(lon[:cut])
        val_lat.append(lat[cut:])
        val_lon.append(lon[cut:])
    return train_lat, train_lon, val_lat, val_lon


def delta_stats(trips_lat: list[np.ndarray], trips_lon: list[np.ndarray]) -> tuple[float, float, float, float]:
    dlat_all = []
    dlon_all = []
    for lat, lon in zip(trips_lat, trips_lon):
        if len(lat) >= 2:
            dlat_all.append(np.diff(np.asarray(lat, dtype=np.float64)))
            dlon_all.append(np.diff(np.asarray(lon, dtype=np.float64)))
    if not dlat_all:
        return 0.0, 0.0, 1e-6, 1e-6
    dlat = np.concatenate(dlat_all)
    dlon = np.concatenate(dlon_all)
    mu_lat = float(np.mean(dlat))
    mu_lon = float(np.mean(dlon))
    sigma_lat = float(np.std(dlat) + 1e-9)
    sigma_lon = float(np.std(dlon) + 1e-9)
    return mu_lat, mu_lon, sigma_lat, sigma_lon


def adaptive_epochs(n_train_windows: int, override: int | None) -> int:
    if override is not None:
        return override
    if n_train_windows < 100:
        return 20
    if n_train_windows < 1000:
        return 30
    if n_train_windows < 10000:
        return 40
    return 50


def train_one(
    model_name: str,
    train_ds: DeltaWindowDataset,
    val_ds: DeltaWindowDataset,
    hidden_size: int,
    batch_size: int,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> tuple[nn.Module, list[dict[str, float]], float]:
    set_seed()
    model: nn.Module = TrajRNN(hidden_size) if model_name == "RNN" else TrajLSTM(hidden_size)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if len(val_ds) else None

    best_state = None
    best_val = float("inf")
    bad = 0
    history = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        batches = 0
        for x, y, _, _ in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
            batches += 1
        train_loss /= max(batches, 1)

        if val_loader is None:
            val_loss = train_loss
        else:
            model.eval()
            val_loss = 0.0
            batches = 0
            with torch.no_grad():
                for x, y, _, _ in val_loader:
                    x = x.to(device)
                    y = y.to(device)
                    val_loss += float(loss_fn(model(x), y).item())
                    batches += 1
            val_loss /= max(batches, 1)

        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val - 1e-9:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_val


@torch.no_grad()
def evaluate(
    model_or_baseline,
    ds: DeltaWindowDataset,
    batch_size: int,
    sigma_lon: float,
    sigma_lat: float,
    mu_lon: float,
    mu_lat: float,
    device: torch.device,
) -> dict[str, float | int]:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    pred_lat = []
    pred_lon = []
    true_lat = []
    true_lon = []

    is_model = isinstance(model_or_baseline, nn.Module)
    if is_model:
        model_or_baseline.eval()

    mean_delta_norm = np.array([mu_lon / sigma_lon, mu_lat / sigma_lat], dtype=np.float32)
    for x, _, anchor, target in loader:
        if is_model:
            out = model_or_baseline(x.to(device)).detach().cpu().numpy()
        elif model_or_baseline == "last_value":
            out = np.zeros((len(x), 2), dtype=np.float32)
        elif model_or_baseline == "mean_delta":
            out = np.tile(mean_delta_norm, (len(x), 1))
        else:
            raise ValueError(f"Unknown evaluator: {model_or_baseline}")

        anchor_np = anchor.numpy()
        target_np = target.numpy()
        lon_hat = anchor_np[:, 0] + out[:, 0] * sigma_lon
        lat_hat = anchor_np[:, 1] + out[:, 1] * sigma_lat
        pred_lon.append(lon_hat)
        pred_lat.append(lat_hat)
        true_lon.append(target_np[:, 0])
        true_lat.append(target_np[:, 1])

    plon = np.concatenate(pred_lon)
    plat = np.concatenate(pred_lat)
    tlon = np.concatenate(true_lon)
    tlat = np.concatenate(true_lat)
    dist = haversine_m(tlat, tlon, plat, plon)
    return {
        "samples": int(len(dist)),
        "haversine_mean_m": float(np.mean(dist)),
        "haversine_median_m": float(np.median(dist)),
        "haversine_p90_m": float(np.percentile(dist, 90)),
        "haversine_p99_m": float(np.percentile(dist, 99)),
    }


def load_user_cache(cache_file: Path) -> dict[str, list[np.ndarray]]:
    data = np.load(cache_file, allow_pickle=True)
    return {
        "train_lat": list(data["train_trips_lat"]),
        "train_lon": list(data["train_trips_lon"]),
        "test_lat": list(data["test_trips_lat"]),
        "test_lon": list(data["test_trips_lon"]),
    }


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Residual RNN/LSTM next-step prediction.")
    parser.add_argument("--cache-dir", type=Path, default=base / "cache_clean")
    parser.add_argument("--out-dir", type=Path, default=base / "outputs_clean" / "03_rnn_lstm")
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN_DEFAULT)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--models", nargs="+", default=["RNN", "LSTM"], choices=["RNN", "LSTM"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = args.out_dir / "weights"
    logs_dir = args.out_dir / "training_logs"
    weights_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"device: {device}")
    files = sorted(args.cache_dir.glob("*.npz"))
    if args.max_users is not None:
        files = files[: args.max_users]

    rows = []
    skipped = []
    start = time.time()
    for file in tqdm(files, desc="rnn/lstm users"):
        user = file.stem
        try:
            cached = load_user_cache(file)
            train_lat, train_lon, val_lat, val_lon = split_train_val(
                cached["train_lat"], cached["train_lon"], args.val_frac, args.seq_len
            )
            mu_lat, mu_lon, sigma_lat, sigma_lon = delta_stats(train_lat, train_lon)
            train_ds = DeltaWindowDataset(train_lat, train_lon, args.seq_len, sigma_lon, sigma_lat)
            val_ds = DeltaWindowDataset(val_lat, val_lon, args.seq_len, sigma_lon, sigma_lat)
            test_ds = DeltaWindowDataset(cached["test_lat"], cached["test_lon"], args.seq_len, sigma_lon, sigma_lat)
            if len(train_ds) < 10 or len(test_ds) < 1:
                skipped.append({"user": user, "reason": f"too few windows train={len(train_ds)} test={len(test_ds)}"})
                continue

            for baseline in ["last_value", "mean_delta"]:
                metrics = evaluate(
                    baseline, test_ds, args.batch_size, sigma_lon, sigma_lat, mu_lon, mu_lat, device
                )
                rows.append(
                    {
                        "user": int(user),
                        "model": f"baseline_{baseline}",
                        "seq_len": args.seq_len,
                        "epochs_run": 0,
                        "n_train_windows": len(train_ds),
                        "n_val_windows": len(val_ds),
                        "best_val_loss": np.nan,
                        "sigma_lat": sigma_lat,
                        "sigma_lon": sigma_lon,
                        **metrics,
                    }
                )

            max_epochs = adaptive_epochs(len(train_ds), args.epochs)
            for model_name in args.models:
                model, history, best_val = train_one(
                    model_name,
                    train_ds,
                    val_ds,
                    args.hidden_size,
                    args.batch_size,
                    max_epochs,
                    args.patience,
                    args.lr,
                    args.weight_decay,
                    device,
                )
                torch.save(model.state_dict(), weights_dir / f"{user}_{model_name}_seq{args.seq_len}.pth")
                pd.DataFrame(history).to_csv(logs_dir / f"{user}_{model_name}_seq{args.seq_len}.csv", index=False)
                metrics = evaluate(
                    model, test_ds, args.batch_size, sigma_lon, sigma_lat, mu_lon, mu_lat, device
                )
                rows.append(
                    {
                        "user": int(user),
                        "model": model_name,
                        "seq_len": args.seq_len,
                        "epochs_run": len(history),
                        "n_train_windows": len(train_ds),
                        "n_val_windows": len(val_ds),
                        "best_val_loss": best_val,
                        "sigma_lat": sigma_lat,
                        "sigma_lon": sigma_lon,
                        **metrics,
                    }
                )
        except Exception as exc:
            skipped.append({"user": user, "reason": repr(exc)})

    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "main_metrics_residual.csv", index=False)
    if skipped:
        pd.DataFrame(skipped).to_csv(args.out_dir / "main_skipped_residual.csv", index=False)

    print(f"rows: {len(df)} skipped: {len(skipped)} elapsed_min: {(time.time() - start) / 60:.2f}")
    if not df.empty:
        print(df.groupby("model")["haversine_median_m"].describe())


if __name__ == "__main__":
    main()
