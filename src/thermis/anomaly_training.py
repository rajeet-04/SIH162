"""CUDA denoising autoencoder for unlabeled thermal novelty, not fire-type truth."""

import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from thermis.retrain_models import run_epoch_with_backoff

FEATURES = [
    "log_frp",
    "brightness_temperature",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "latitude",
    "longitude",
]


def features(frame):
    stamp = pd.to_datetime(frame.timestamp_utc, utc=True)

    def numeric(name):
        return pd.to_numeric(
            frame.get(name, pd.Series(np.nan, index=frame.index)), errors="coerce"
        ).to_numpy(float)

    hour = (stamp.dt.hour + stamp.dt.minute / 60).to_numpy() * 2 * np.pi / 24
    month = (stamp.dt.month - 1).to_numpy() * 2 * np.pi / 12
    with np.errstate(invalid="ignore"):
        matrix = np.column_stack(
            [
                np.log1p(numeric("frp")),
                numeric("brightness_temperature"),
                np.sin(hour),
                np.cos(hour),
                np.sin(month),
                np.cos(month),
                numeric("latitude"),
                numeric("longitude"),
            ]
        )
    matrix[~np.isfinite(matrix)] = np.nan
    return matrix


def transform(matrix, center, scale):
    missing = np.isnan(matrix)
    x = np.clip((np.where(missing, center, matrix) - center) / scale, -20, 20)
    return np.concatenate([x, missing.astype(float)], axis=1).astype("float32")


def network():
    return nn.Sequential(
        nn.Linear(16, 32),
        nn.GELU(),
        nn.Linear(32, 4),
        nn.GELU(),
        nn.Linear(4, 32),
        nn.GELU(),
        nn.Linear(32, 16),
    )


def scores(model, x, device, batch=256):
    result = []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(x), batch):
            values = torch.from_numpy(x[offset : offset + batch]).to(device)
            result.append((model(values) - values).square().mean(1).cpu().numpy())
    return np.concatenate(result) if result else np.empty(0)


def train_anomaly(frame, output, device="cuda", epochs=20, batch_size=256, resume=False):
    output = Path(output)
    if output.name in ("tabular", "image", "models"):
        raise ValueError("Use an isolated research output")
    if epochs < 1 or batch_size < 1:
        raise ValueError("Positive epoch and batch counts required")
    frame = frame.copy()
    frame["timestamp_utc"] = pd.to_datetime(frame.timestamp_utc, utc=True, errors="coerce")
    for name in ("frp", "latitude", "longitude"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    valid = (
        frame.timestamp_utc.notna()
        & frame.frp.ge(0)
        & np.isfinite(frame.frp)
        & frame.latitude.between(4, 38)
        & frame.longitude.between(68, 98)
        & frame.event_id.notna()
    )
    invalid = int((~valid).sum())
    frame = frame[valid]
    count = len(frame)
    frame = (
        frame.drop_duplicates("event_id")
        .sort_values(["timestamp_utc", "event_id"])
        .reset_index(drop=True)
    )
    duplicates = count - len(frame)
    if len(frame) < 30:
        raise ValueError("At least 30 valid distinct observations required")
    stamps = frame.timestamp_utc
    val_start = stamps.iloc[int(len(frame) * 0.8)]
    test_start = stamps.iloc[int(len(frame) * 0.9)]
    split = np.where(
        stamps >= test_start, "test", np.where(stamps >= val_start, "validation", "train")
    )
    split_rows = {name: int((split == name).sum()) for name in ("train", "validation", "test")}
    if min(split_rows.values()) == 0:
        raise ValueError("Tied timestamps prevent three chronological partitions")
    # Sealed rows leave the pipeline before any numerical feature processing.
    development = frame[split != "test"].copy()
    train_mask = split[split != "test"] == "train"
    matrix = features(development)
    training = matrix[train_mask]
    center = np.array([np.nanmedian(c) if np.isfinite(c).any() else 0.0 for c in training.T])
    filled = np.where(np.isnan(training), center, training)
    scale = np.quantile(filled, 0.9, axis=0) - np.quantile(filled, 0.1, axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    x = transform(matrix, center, scale)
    train_x, val_x = x[train_mask], x[~train_mask]
    identity = hashlib.sha256(
        pd.util.hash_pandas_object(development[["event_id", "timestamp_utc"]], index=False)
        .to_numpy()
        .tobytes()
        + x.tobytes()
        + train_mask.tobytes()
    ).hexdigest()
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    torch.manual_seed(162)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(162)
    model = network().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    config = dict(batch_size=min(batch_size, 256), device=str(device), seed=162)
    batch = config["batch_size"]
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint.pt"
    if checkpoint.exists() and not resume:
        raise ValueError("Output already trained; use a new run or --resume")
    best, best_loss, start, history = None, float("inf"), 0, []

    def snapshot(epoch):
        return copy.deepcopy(
            dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scaler=scaler.state_dict(),
                rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                best=best,
                best_loss=best_loss,
                epoch=epoch,
                identity=identity,
                config=config,
                history=history,
                batch=batch,
            )
        )

    def restore(state):
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["rng"].cpu())
        if device.type == "cuda" and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda_rng"]])

    if resume:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if state["identity"] != identity or state["config"] != config:
            raise ValueError("Resume data/config identity mismatch")
        restore(state)
        best, best_loss, start, history, batch = (
            state[k] for k in ("best", "best_loss", "epoch", "history", "batch")
        )
    for epoch in range(start, epochs):
        before = snapshot(epoch)

        def run(active):
            model.train()
            order = torch.randperm(len(train_x)).numpy()
            for offset in range(0, len(order), active):
                clean = torch.from_numpy(train_x[order[offset : offset + active]]).to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    noisy = clean + 0.05 * torch.randn_like(clean)
                    loss = (model(noisy) - clean).square().mean()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        batch = run_epoch_with_backoff(run, lambda: restore(before), batch)
        value = float(scores(model, val_x, device, batch).mean())
        if not np.isfinite(value):
            raise ValueError("Nonfinite validation loss")
        if value < best_loss:
            best_loss = value
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(dict(epoch=epoch + 1, validation_reconstruction_mse=value, batch=batch))
        temporary = checkpoint.with_suffix(".tmp")
        torch.save(snapshot(epoch + 1), temporary)
        temporary.replace(checkpoint)
        print(json.dumps(history[-1]), flush=True)
    model.load_state_dict(best)
    validation_scores = scores(model, val_x, device, batch)
    threshold = float(np.quantile(validation_scores, 0.99))
    payload = dict(
        model=best,
        center=center.tolist(),
        scale=scale.tolist(),
        features=FEATURES,
        threshold=threshold,
        identity=identity,
        kind="thermal_novelty_autoencoder",
    )
    torch.save(payload, output / "anomaly.pt")
    ranked = development.loc[~train_mask, ["event_id", "timestamp_utc"]].copy()
    ranked["novelty_score"] = validation_scores
    ranked.sort_values("novelty_score", ascending=False).to_parquet(
        output / "validation_rankings.parquet", index=False
    )
    report = dict(
        status="complete",
        identity=identity,
        rows=len(frame),
        split_rows=split_rows,
        duplicates_excluded=duplicates,
        invalid_excluded=invalid,
        device=str(device),
        epochs_completed=max(epochs, start),
        active_batch_size=batch,
        features=FEATURES,
        validation_start=val_start.isoformat(),
        sealed_test_start=test_start.isoformat(),
        validation_reconstruction_mse=best_loss,
        review_threshold_p99=threshold,
        classification_accuracy=None,
        promotion_allowed=False,
        meaning="Thermal novelty score; not probability of fire, accident, or industrial origin",
        spatial_holdout="Not performed; chronological evaluation only",
        threshold_meaning="Validation novelty quantile, not a verified false-alert rate",
        torch_version=str(torch.__version__),
        history=history,
        selected_epoch=min(history, key=lambda item: item["validation_reconstruction_mse"])[
            "epoch"
        ],
        dependencies={
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "pandas", "scikit-learn")
        },
        artifact_sha256=hashlib.file_digest(
            (output / "anomaly.pt").open("rb"), "sha256"
        ).hexdigest(),
    )
    (output / "run_report.json").write_text(json.dumps(report, indent=2))
    (output / "model_card.md").write_text(
        "# Thermal novelty research model\n\n"
        "Denoising autoencoder trained without target labels. Higher scores indicate "
        "unusual input combinations; they do not establish fire type or accident probability.\n\n"
        "Chronological 80% fit / 10% validation / 10% sealed test, with timestamp ties "
        "kept together. All numeric scaling is fit on training rows. No test metrics "
        "are reported. Validation chooses the checkpoint and 99th-percentile review "
        "threshold. This threshold is not a measured false-alert rate.\n\n"
        "No independent reference-label or spatial holdout evaluation exists. Latitude "
        "and longitude are inputs, so geographic transfer is not established. All "
        "thermal source types participate in unsupervised training. Sensor calibration "
        "and cloud/overpass coverage can change scores. Deployment promotion is blocked.\n\n"
        "CUDA uses mixed precision and at most 256 rows per microbatch, with epoch "
        "rollback on training OOM. Checkpoint contains optimizer, scaler, RNG and "
        "data/config identity. CPU inference is supported. Load only trusted local "
        "PyTorch bundles. Feature order and scaling are embedded in anomaly.pt.\n"
    )
    return report


class AnomalyModel:
    def __init__(self, path, device="cpu"):
        self.device = torch.device(device)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.center = np.array(payload["center"])
        self.scale = np.array(payload["scale"])
        self.threshold = payload["threshold"]
        self.model = network().to(self.device)
        self.model.load_state_dict(payload["model"])

    def score(self, frame):
        return scores(self.model, transform(features(frame), self.center, self.scale), self.device)


def load_anomaly(path, device="cpu"):
    return AnomalyModel(path, device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    frame = pd.concat([pd.read_parquet(path) for path in args.observations], ignore_index=True)
    result = train_anomaly(
        frame, args.output, args.device, args.epochs, args.batch_size, args.resume
    )
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
