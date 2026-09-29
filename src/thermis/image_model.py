import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    scene_group: str
    split: str
    label: str
    path: str


@dataclass(frozen=True)
class ImageQuality:
    readable: bool
    channels: int | None
    height: int | None
    width: int | None
    reason: str | None = None


def assign_image_splits(frame, ranking_fraction: float = 0.10):
    """Assign deterministic grouped holdout splits when image timestamps are absent."""
    if not 0 < ranking_fraction < 1:
        raise ValueError("ranking_fraction must be between zero and one")
    result = frame.copy()
    group_values = result["scene_group"].fillna(result["image_id"]).astype(str)
    official_test = (
        result["source_family"].astype(str).eq("fire_test")
        if "source_family" in result
        else result.index.to_series().map(lambda _: False)
    )
    groups = sorted(group_values[~official_test].unique())
    ranking_count = max(1, round(len(groups) * ranking_fraction))
    ranking_groups = set(groups[-ranking_count:])
    result["split"] = group_values.map(
        lambda group: "ranking" if group in ranking_groups else "development"
    )
    result.loc[official_test, "split"] = "ranking"
    return result


def assert_scene_isolation(records: list[ImageRecord]) -> None:
    groups: dict[str, str] = {}
    for record in records:
        previous = groups.setdefault(record.scene_group, record.split)
        if previous != record.split:
            raise ValueError(f"scene group crosses split boundary: {record.scene_group}")


def inspect_image(path: Path) -> ImageQuality:
    """Read image headers/array shape only; do not materialize pixel batches."""
    try:
        if path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                arrays = [archive[key] for key in archive.files]
                if not arrays:
                    return ImageQuality(False, None, None, None, "empty_npz")
                shape = arrays[0].shape
                if len(shape) < 2:
                    return ImageQuality(False, None, None, None, "not_an_image_array")
                channels = shape[2] if len(shape) == 3 else 1
                return ImageQuality(True, int(channels), int(shape[0]), int(shape[1]))
        with Image.open(path) as image:
            channels = len(image.getbands())
            return ImageQuality(True, channels, image.height, image.width)
    except (OSError, ValueError, KeyError) as error:
        return ImageQuality(False, None, None, None, str(error))


def _image_tensor(path: Path, augment: bool = False):
    import torch
    from torchvision import transforms as T

    with Image.open(path) as image:
        image = image.convert("RGB")
        if augment:
            image = T.Compose(
                [
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomRotation(degrees=15),
                    T.ColorJitter(brightness=0.2, contrast=0.2),
                ]
            )(image)
        image = image.resize((224, 224))
        values = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(values).permute(2, 0, 1)


class _ImageDataset:
    def __init__(
        self, records: list[ImageRecord], label_ids: dict[str, int], augment: bool = False
    ):
        self.records = records
        self.label_ids = label_ids
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        return _image_tensor(Path(record.path), augment=self.augment), self.label_ids[
            record.label
        ]


def _grouped_dev_val_split(
    development: list[ImageRecord], val_fraction: float
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """Split development rows into fit/dev-val by scene group (newest groups validate)."""
    groups: dict[str, list[ImageRecord]] = {}
    for record in development:
        groups.setdefault(record.scene_group, []).append(record)
    names = sorted(groups)
    val_count = max(1, round(len(names) * val_fraction))
    val_names = set(names[-val_count:])
    fit = [r for r in development if r.scene_group not in val_names]
    val = [r for r in development if r.scene_group in val_names]
    if not fit or not val:
        raise ValueError("grouped dev-val split left an empty side")
    return fit, val


def train_image_smoke(
    records: list[ImageRecord],
    output: Path,
    epochs: int = 1,
    max_records: int | None = 128,
    batch_size: int = 16,
    val_fraction: float = 0.0,
    weight_decay: float = 0.0,
    augment: bool = False,
) -> dict[str, Any]:
    """Train a bounded image verifier and export a CPU-loadable TorchScript model.

    When ``val_fraction`` is positive, the newest scene groups inside development
    form a dev-val set used only for best-epoch selection; ranking rows are never
    touched.
    """
    import torch
    from torch.utils.data import DataLoader
    from torchvision.models import resnet18

    candidates = [record for record in records if record.split != "ranking"]
    ranking_rows = sum(record.split == "ranking" for record in records)
    random.Random(26162).shuffle(candidates)
    if max_records is not None and max_records > 0:
        development = candidates[:max_records]
    else:
        development = candidates
    if not development:
        raise ValueError("image development set is empty")
    labels = sorted({record.label for record in development})
    if len(labels) < 2:
        raise ValueError("image development set needs at least two classes")
    label_ids = {label: index for index, label in enumerate(labels)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(26162)
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(labels))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    fit_rows = development
    val_rows: list[ImageRecord] = []
    if val_fraction > 0:
        fit_rows, val_rows = _grouped_dev_val_split(development, val_fraction)
    num_workers = 2 if len(fit_rows) >= 1024 else 0
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "generator": torch.Generator().manual_seed(26162),
    }
    if num_workers:
        loader_kwargs.update(prefetch_factor=2, persistent_workers=True)
    loader = DataLoader(_ImageDataset(fit_rows, label_ids, augment=augment), **loader_kwargs)
    val_loader = None
    if val_rows:
        val_loader = DataLoader(
            _ImageDataset(val_rows, label_ids),
            batch_size=min(512, batch_size * 2),
            shuffle=False,
            num_workers=0,
        )
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    losses: list[float] = []
    best_state = None
    best_val_nll: float | None = None
    best_epoch = epochs
    model.train()
    for epoch in range(epochs):
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                loss = criterion(model(inputs), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        if val_loader is not None:
            model.eval()
            total_nll, total_count = 0.0, 0
            with torch.inference_mode():
                for inputs, targets in val_loader:
                    logits = model(inputs.to(device))
                    total_nll += float(
                        torch.nn.functional.cross_entropy(
                            logits, targets.to(device), reduction="sum"
                        ).cpu()
                    )
                    total_count += len(targets)
            val_nll = total_nll / max(1, total_count)
            if best_val_nll is None or val_nll < best_val_nll:
                best_val_nll = val_nll
                best_epoch = epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.train()
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.cpu().eval()
    output.mkdir(parents=True, exist_ok=True)
    torch.jit.trace(model, torch.zeros(1, 3, 224, 224)).save(str(output / "image_verifier.ts"))
    metadata = {
        "classes": labels,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "training_rows": len(fit_rows),
        "dev_val_rows": len(val_rows),
        "val_fraction": val_fraction,
        "weight_decay": weight_decay,
        "augment": augment,
        "batch_size": batch_size,
        "amp": amp_enabled,
        "num_workers": num_workers,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "val_nll_best": best_val_nll,
        "loss_final": losses[-1] if losses else None,
        "ranking_rows": ranking_rows,
        "ranking_rows_used": 0,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
