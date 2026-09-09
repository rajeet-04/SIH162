"""Isolated challengers. Sealed test data is discarded before any training work.

Artifacts/checkpoints are trusted local files, not an untrusted interchange format.
Validation is chronological, teacher-free context: only preceding feature observations
are used, never preceding targets. No function in this module promotes a model.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

SEED = 162
METADATA = [
    "timestamp_utc",
    "event_id",
    "site_id",
    "episode_id",
    "region_id",
    "split",
    "target",
    "label_tier",
]
INDUSTRIAL = {"industrial", "industrial_fire", "industrial_fire_candidate"}


def _known(frame):
    return (
        frame.label_tier.isin(["reference", "weak"])
        & frame.target.notna()
        & ~frame.target.astype(str).str.lower().isin(["unknown", "", "nan"])
    )


def _numeric(frame, columns):
    return (
        frame.reindex(columns=columns)
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy(dtype=np.float64)
    )


@dataclass
class Preprocessor:
    columns: list
    medians: list
    means: list
    scales: list

    @classmethod
    def fit(cls, frame, columns):
        values = _numeric(frame.loc[(frame.split == "train") & _known(frame)], columns)
        medians = [
            float(np.median(c[np.isfinite(c)])) if np.isfinite(c).any() else 0.0 for c in values.T
        ]
        filled = np.where(np.isnan(values), medians, values)
        means = filled.mean(axis=0) if len(filled) else np.zeros(len(columns))
        scales = filled.std(axis=0) if len(filled) else np.ones(len(columns))
        scales = np.where(scales > 1e-12, scales, 1.0)
        return cls(list(columns), medians, means.tolist(), scales.tolist())

    def transform(self, frame):
        values = _numeric(frame, self.columns)
        missing = np.isnan(values)
        normalized = (np.where(missing, self.medians, values) - self.means) / self.scales
        # Bound adversarial/out-of-range values before the float32 cast.
        return np.concatenate([np.clip(normalized, -1e6, 1e6), missing], axis=1).astype(np.float32)


class SequenceDataset(Dataset):
    """CPU arrays plus per-site indices; padded sequences are built only on access."""

    def __init__(self, frame, preprocessor, classes, split=None):
        self.frame = frame.reset_index(drop=True)
        self.values = preprocessor.transform(self.frame)
        self.classes = list(classes)
        self.times = (
            pd.to_datetime(self.frame.timestamp_utc, utc=True, errors="raise")
            .astype("int64")
            .to_numpy()
        )
        self.sites = self.frame.site_id.to_numpy()
        self.splits = self.frame.get("split", pd.Series("inference", index=self.frame.index))
        self.rows = (
            list(range(len(frame)))
            if split is None
            else self.frame.index[
                (self.splits == split)
                & _known(self.frame)
                & self.frame.target.astype(str).isin(classes)
            ].tolist()
        )
        self.groups = {}
        for site, indices in self.frame.groupby("site_id", dropna=True).indices.items():
            indices = np.asarray(indices)
            order = indices[np.argsort(self.times[indices], kind="stable")]
            for mode, allowed in (
                ("train", ["train"]),
                ("other", ["train", "validation", "inference"]),
            ):
                selected = order[self.splits.iloc[order].isin(allowed).to_numpy()]
                # Keep one stable observation per timestamp, so the entire
                # history (not just its boundary) is strictly increasing.
                selected = (
                    selected[np.r_[self.times[selected][1:] != self.times[selected][:-1], True]]
                    if len(selected)
                    else selected
                )
                self.groups[(site, mode)] = selected

    def __len__(self):
        return len(self.rows)

    def history_indices(self, row):
        mode = "train" if self.splits.iloc[row] == "train" else "other"
        indices = self.groups.get((self.sites[row], mode), np.empty(0, dtype=int))
        end = np.searchsorted(self.times[indices], self.times[row], side="left")
        return indices[max(0, end - 16) : end].tolist()

    def __getitem__(self, index):
        row = self.rows[index]
        history = self.history_indices(row)
        sequence = np.zeros((16, self.values.shape[1]), dtype=np.float32)
        sequence[: len(history)] = self.values[history]
        label = str(self.frame.iloc[row].get("target", ""))
        target = self.classes.index(label) if label in self.classes else -1
        return sequence, self.values[row], len(history), target


class _GRU(nn.Module):
    def __init__(self, width, classes):
        super().__init__()
        self.gru = nn.GRU(width, 32, batch_first=True)
        self.head = nn.Sequential(nn.Linear(32 + width, 32), nn.ReLU(), nn.Linear(32, classes))

    def forward(self, sequence, numeric, lengths):
        hidden, _ = self.gru(sequence)
        last = hidden[torch.arange(len(hidden), device=hidden.device), (lengths - 1).clamp(min=0)]
        last = last * (lengths > 0).unsqueeze(1)
        return self.head(torch.cat([last, numeric], dim=1))


def _device(device):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(device)
    if resolved.type not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu, cuda, or auto")
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return resolved


def _rng(cuda=False):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if cuda else None,
    }


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu(v) for v in value)
    return copy.deepcopy(value)


def _save(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _json(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def run_epoch_with_backoff(run, restore, batch_size):
    """Retry a whole epoch; release traceback references before allocator cleanup."""
    batch = batch_size
    while True:
        failed = False
        try:
            run(batch)
        except torch.OutOfMemoryError:
            restore()
            if batch == 1:
                raise
            failed = True
        if not failed:
            return batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        batch = max(1, batch // 2)


def _probabilities(model, dataset, device, batch_size=256):
    model.eval()
    chunks = []
    with torch.inference_mode():
        for sequence, numeric, lengths, _ in DataLoader(dataset, batch_size=batch_size):
            logits = model(sequence.to(device), numeric.to(device), lengths.to(device))
            chunks.append(logits.softmax(dim=1).cpu().numpy())
    return np.concatenate(chunks) if chunks else np.empty((0, model.head[-1].out_features))


def evaluate_metrics(frame, probabilities, classes):
    """PR AUC is average precision. Null denotes unsupported denominators/strata."""
    classes = list(classes)
    y = frame.target.astype(str).to_numpy()
    predicted = np.asarray(classes)[probabilities.argmax(axis=1)] if len(frame) else np.array([])
    per_class = {}
    for j, label in enumerate(classes):
        truth = y == label
        guess = predicted == label
        tp = int((truth & guess).sum())
        per_class[label] = {
            "precision": tp / int(guess.sum()) if guess.any() else None,
            "recall": tp / int(truth.sum()) if truth.any() else None,
            "support": int(truth.sum()),
            "pr_auc": float(average_precision_score(truth, probabilities[:, j]))
            if truth.any() and (~truth).any()
            else None,
        }
    ece = None
    if len(frame):
        confidence = probabilities.max(axis=1)
        correct = predicted == y
        bins = np.minimum((confidence * 10).astype(int), 9)
        ece = float(
            sum(
                np.mean(bins == b) * abs(correct[bins == b].mean() - confidence[bins == b].mean())
                for b in range(10)
                if (bins == b).any()
            )
        )
    industrial = [c for c in classes if c in INDUSTRIAL]
    persistent = {"events": None, "eligible_events": None, "rate": None}
    if industrial and len(frame):
        evaluated = frame.copy()
        evaluated["_alert"] = np.isin(predicted, industrial)
        count = eligible = 0
        for _, event in evaluated.groupby(["site_id", "episode_id", "event_id"], dropna=True):
            event = event.sort_values("timestamp_utc")
            # Two distinct observations with known negative truth are needed to
            # assess persistence. Mixed-positive events are not false events.
            if event.timestamp_utc.nunique() < 2 or event.target.isin(industrial).any():
                continue
            eligible += 1
            alerts = event.groupby("timestamp_utc")["_alert"].max().to_numpy()
            count += int((alerts[1:] & alerts[:-1]).any())
        persistent = {
            "events": count if eligible else None,
            "eligible_events": eligible,
            "rate": count / eligible if eligible else None,
        }
    return {
        "rows": len(frame),
        "macro_f1": float(f1_score(y, predicted, labels=classes, average="macro", zero_division=0))
        if len(frame)
        else None,
        "per_class": per_class,
        "ece": ece,
        "persistent_false_industrial_alerts": persistent,
    }


def _strata(frame, probabilities, classes, train):
    result = {"overall": evaluate_metrics(frame, probabilities, classes)}
    masks = {
        "reference": frame.label_tier == "reference",
        "weak": frame.label_tier == "weak",
        "unseen_site": ~frame.site_id.isin(train.site_id),
        "unseen_region": ~frame.region_id.isin(train.region_id),
    }
    for name, mask in masks.items():
        result[name] = evaluate_metrics(frame.loc[mask], probabilities[mask.to_numpy()], classes)
    return result


def _output_path(output):
    path = Path(output).resolve()
    # Protect deployed model trees, including paths reached through symlinks.
    if any(part.lower() in {"models", "deployed"} for part in path.parts):
        raise ValueError("output must be outside deployed models locations")
    return path


def train_challengers(
    frame, output, feature_columns, epochs=20, batch_size=256, device="auto", resume=False
):
    """Train two challengers using only train/validation rows; never evaluate test.

    Epochs is the total desired GRU epoch count (it may increase on resume).
    Other configuration and the ordered nonsealed dataset must match exactly.
    CatBoost is deterministically refit on resume; its artifact is not a GRU
    optimizer checkpoint. Actual GPU CatBoost reductions can be nondeterministic.
    """
    output = _output_path(output)
    columns = list(feature_columns)
    if not columns or len(set(columns)) != len(columns) or set(columns) & set(METADATA):
        raise ValueError("feature_columns must be unique numeric features, not metadata/labels")
    if (
        not isinstance(epochs, int)
        or epochs < 1
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("epochs and batch_size must be positive integers")
    if not set(METADATA).issubset(frame.columns):
        raise ValueError(f"missing metadata columns: {sorted(set(METADATA) - set(frame.columns))}")
    # Do not hash, parse, impute, or copy sealed features/labels.
    data = (
        frame.loc[
            frame.split.isin(["train", "validation"]), METADATA + [c for c in columns if c in frame]
        ]
        .copy()
        .reset_index(drop=True)
    )
    data["timestamp_utc"] = pd.to_datetime(data.timestamp_utc, utc=True, errors="raise")
    if data.timestamp_utc.isna().any() or data.site_id.isna().any():
        raise ValueError("nonsealed rows require valid timestamp_utc and site_id")
    train_mask = (data.split == "train") & _known(data)
    classes = sorted(data.loc[train_mask, "target"].astype(str).unique().tolist())
    promotion = {
        "allowed": False,
        "reasons": [
            "No automatic promotion.",
            "Adequate independent industrial labels have not been established.",
            "Weak-label agreement does not measure industrial accident accuracy.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    if len(classes) < 2:
        report = {
            "status": "blocked",
            "reason": "Training requires at least two represented classes",
            "training_rows": int(train_mask.sum()),
            "classes": classes,
            "promotion": promotion,
        }
        _json(report, output / "run_report.json")
        return report
    resolved = _device(device)
    config = {
        "version": 1,
        "features": columns,
        "batch_size": batch_size,
        "device_type": resolved.type,
        "seed": SEED,
        "max_history": 16,
        "hidden_size": 32,
        "learning_rate": 0.001,
        "effective_batch": max(256, batch_size),
    }
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    digest.update(
        data.reindex(columns=METADATA + columns)
        .to_json(orient="split", date_format="iso", date_unit="ns")
        .encode()
    )
    identity = digest.hexdigest()
    checkpoint_path = output / "checkpoint.pt"
    checkpoint = None
    if resume:
        if not checkpoint_path.exists():
            raise ValueError("resume checkpoint is missing")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["identity"] != identity or checkpoint["config"] != config:
            raise ValueError("resume dataset/config identity mismatch")
    elif checkpoint_path.exists():
        raise ValueError("output already contains a checkpoint; use resume or a new output")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if resolved.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    prep = Preprocessor.fit(data, columns)
    train_ds = SequenceDataset(data, prep, classes, "train")
    val_ds = SequenceDataset(data, prep, classes, "validation")
    train = data.iloc[train_ds.rows]
    validation = data.iloc[val_ds.rows]
    versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "catboost", "numpy", "pandas", "scikit-learn")
    }
    history = data.loc[data.split == "train"].copy()
    history["timestamp_utc"] = history.timestamp_utc.astype(str)
    # Labels are unnecessary at prediction time and are not packaged as history.
    history = history.drop(columns=["target", "label_tier"])
    metadata = {
        "preprocessing": asdict(prep),
        "classes": classes,
        "history": history.to_dict(orient="list"),
        "identity": identity,
        "config": config,
    }

    catboost = CatBoostClassifier(
        iterations=max(20, epochs * 5),
        depth=5,
        learning_rate=0.05,
        loss_function="MultiClass",
        random_seed=SEED,
        task_type="GPU" if resolved.type == "cuda" else "CPU",
        **(
            {"devices": str(resolved.index or 0), "gpu_ram_part": 0.65}
            if resolved.type == "cuda"
            else {"thread_count": 2}
        ),
        allow_writing_files=False,
        verbose=False,
    )
    # CatBoost GPU and PyTorch allocations never overlap during fitting.
    catboost.fit(
        prep.transform(train),
        [classes.index(str(y)) for y in train.target],
        eval_set=(prep.transform(validation), [classes.index(str(y)) for y in validation.target])
        if len(validation)
        else None,
        use_best_model=bool(len(validation)),
    )
    cat_path = output / "catboost.cbm"
    catboost.save_model(str(cat_path))
    _save(metadata, output / "catboost.metadata.pt")
    cat_probs = (
        catboost.predict_proba(prep.transform(validation))
        if len(validation)
        else np.empty((0, len(classes)))
    )

    model = _GRU(len(columns) * 2, len(classes)).to(resolved)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler("cuda", enabled=resolved.type == "cuda")
    start = 0
    best_score = -math.inf
    best_model = _cpu(model.state_dict())
    active_batch = min(batch_size, 256)
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        _restore_rng(checkpoint["rng"])
        start, active_batch = checkpoint["epoch"], checkpoint["active_batch"]
        best_score, best_model = checkpoint["best_score"], checkpoint["best_model"]

    def snapshot(epoch):
        return {
            "model": _cpu(model.state_dict()),
            "optimizer": _cpu(optimizer.state_dict()),
            "scaler": _cpu(scaler.state_dict()),
            "rng": _rng(resolved.type == "cuda"),
            "identity": identity,
            "config": config,
            "epoch": epoch,
            "active_batch": active_batch,
            "best_model": best_model,
            "best_score": best_score,
            "metadata": metadata,
        }

    backoffs = []
    for epoch in range(start, epochs):
        epoch_start = snapshot(epoch)
        _save(epoch_start, checkpoint_path)

        def restore():
            optimizer.zero_grad(set_to_none=True)
            model.load_state_dict(epoch_start["model"])
            # Optimizer loading can alias CPU tensors. A fresh copy protects
            # the epoch snapshot's moments/step counters across repeated OOMs.
            optimizer.load_state_dict(copy.deepcopy(epoch_start["optimizer"]))
            scaler.load_state_dict(epoch_start["scaler"])
            _restore_rng(epoch_start["rng"])

        def run(batch):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0)
            accumulation = math.ceil(config["effective_batch"] / batch)
            for step, (sequence, numeric, lengths, targets) in enumerate(loader):
                group_start = (step // accumulation) * accumulation * batch
                group_rows = min(accumulation * batch, len(train_ds) - group_start)
                with torch.autocast(device_type=resolved.type, enabled=resolved.type == "cuda"):
                    logits = model(
                        sequence.to(resolved), numeric.to(resolved), lengths.to(resolved)
                    )
                    loss = nn.functional.cross_entropy(
                        logits, targets.to(resolved), reduction="sum"
                    )
                    loss = loss / group_rows
                scaler.scale(loss).backward()
                if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            # Validation OOM also causes a full epoch restart from saved state.
            nonlocal epoch_probabilities
            epoch_probabilities = _probabilities(model, val_ds, resolved, batch)

        epoch_probabilities = None
        previous_batch = active_batch
        active_batch = run_epoch_with_backoff(run, restore, active_batch)
        if active_batch != previous_batch:
            backoffs.append({"epoch": epoch, "from": previous_batch, "to": active_batch})
        score = evaluate_metrics(validation, epoch_probabilities, classes)["macro_f1"]
        if score is None or score > best_score:
            best_model = _cpu(model.state_dict())
            best_score = score if score is not None else -math.inf
        _save(snapshot(epoch + 1), checkpoint_path)
    model.load_state_dict(best_model)
    gru_path = output / "gru.pt"
    _save({**metadata, "model": best_model}, gru_path)
    gru_probs = _probabilities(model, val_ds, resolved, active_batch)
    report = {
        "status": "complete",
        "identity": identity,
        "device": str(resolved),
        "classes": classes,
        "training_rows": len(train_ds),
        "validation_rows": len(val_ds),
        "validation_excluded_rows": int((data.split == "validation").sum()) - len(val_ds),
        "selection": "validation macro F1; latest epoch when validation is unavailable",
        "epochs_completed": max(start, epochs),
        "active_batch_size": active_batch,
        "effective_batch_size": math.ceil(config["effective_batch"] / active_batch) * active_batch,
        "oom_backoffs": backoffs,
        "promotion": promotion,
        "dependency_versions": versions,
        "feature_schema": {
            "columns": columns,
            "missingness_flags": columns,
            "absent_columns": [c for c in columns if c not in data],
            "preprocessing": asdict(prep),
        },
        "metrics": {
            "catboost": _strata(validation, cat_probs, classes, train),
            "gru": _strata(validation, gru_probs, classes, train),
        },
        "artifacts": {
            "catboost": str(cat_path),
            "gru": str(gru_path),
            "checkpoint": str(checkpoint_path),
        },
    }
    _json(report, output / "run_report.json")
    _json(report["feature_schema"], output / "feature_schema.json")
    _json(versions, output / "dependency_versions.json")
    (output / "model_card.md").write_text(
        "# Retraining challengers\n\nNo automatic promotion. Adequate independent industrial "
        "labels have not been established. Weak-label agreement is not accident accuracy.\n\n"
        "Test and purged rows are excluded before preprocessing, fitting, selection and metrics. "
        "Train histories contain only earlier train features; validation can use earlier train "
        "and validation features at the same site. Timestamp ties are excluded. Maximum history "
        "is 16 observations. Unknown labels never contribute supervised targets.\n\n"
        "Missing imagery/context is not synthesized: numeric NaNs and absent features receive "
        "train-fitted imputation and explicit missingness flags. No image model is trained.\n\n"
        "PR AUC uses average precision; ECE uses ten equal-width confidence bins. Persistence "
        "means two consecutive distinct timestamps predicted industrial within a known-negative "
        "site/episode/event; rate denominator is negative events with at least two timestamps. "
        "Reference and weak strata are reported separately; unsupported metrics are null.\n\n"
        "GRU selection uses validation macro F1; CatBoost uses validation MultiClass loss. "
        "Without validation, final models are unselected. CatBoost GPU reductions may remain "
        "nondeterministic despite fixed seeds. CPU tests do not certify CUDA memory usage. "
        "GRU uses a 32-unit hidden state, host-resident lazy sequences, AMP and <=256-row GPU "
        "microbatches with whole-epoch OOM rollback. CatBoost uses 65% GPU RAM budget.\n\n"
        "Resume preserves GRU model/optimizer/AMP/RNG plus dataset/config identity; CatBoost "
        "is refit. Load only trusted local artifacts.\n",
        encoding="utf-8",
    )
    return report


class _LoadedChallenger:
    def __init__(self, path, device):
        self.device = _device(device)
        self.kind = "catboost" if path.suffix == ".cbm" else "gru"
        payload = torch.load(
            path.with_name("catboost.metadata.pt") if self.kind == "catboost" else path,
            map_location="cpu",
            weights_only=False,
        )
        self.preprocessor = Preprocessor(**payload["preprocessing"])
        self.classes_ = np.asarray(payload["classes"])
        self.history = pd.DataFrame(payload["history"])
        if self.kind == "catboost":
            self.model = CatBoostClassifier()
            self.model.load_model(str(path))
        else:
            self.model = _GRU(len(self.preprocessor.columns) * 2, len(self.classes_))
            self.model.load_state_dict(payload["model"])
            self.model.to(self.device).eval()

    def predict_proba(self, frame):
        if not len(frame):
            return np.empty((0, len(self.classes_)))
        if self.kind == "catboost":
            return self.model.predict_proba(self.preprocessor.transform(frame), task_type="CPU")
        query = frame.copy().reset_index(drop=True)
        query["timestamp_utc"] = pd.to_datetime(query.timestamp_utc, utc=True, errors="raise")
        if query.timestamp_utc.isna().any() or query.site_id.isna().any():
            raise ValueError("predictions require valid timestamp_utc and site_id")
        if "split" not in query:
            query["split"] = "inference"
        history = self.history.copy()
        history["timestamp_utc"] = pd.to_datetime(history.timestamp_utc, utc=True)
        keys = ["site_id", "timestamp_utc"]
        if "event_id" in query:
            keys.append("event_id")
        matched = pd.MultiIndex.from_frame(history[keys]).isin(
            pd.MultiIndex.from_frame(query[keys])
        )
        history = history.loc[~matched]
        combined = pd.concat([history, query], ignore_index=True)
        dataset = SequenceDataset(combined, self.preprocessor, self.classes_.tolist())
        dataset.rows = list(range(len(history), len(combined)))
        return _probabilities(self.model, dataset, self.device)


def load_challenger(path, device="cpu"):
    """Load a trusted gru.pt or catboost.cbm with its adjacent metadata file."""
    return _LoadedChallenger(Path(path), device)
