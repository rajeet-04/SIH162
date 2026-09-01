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


def _image_tensor(path: Path):
    import torch

    with Image.open(path) as image:
        image = image.convert("RGB").resize((224, 224))
        values = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(values).permute(2, 0, 1)


class _ImageDataset:
    def __init__(self, records: list[ImageRecord], label_ids: dict[str, int]):
        self.records = records
        self.label_ids = label_ids

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        return _image_tensor(Path(record.path)), self.label_ids[record.label]


def train_image_smoke(
    records: list[ImageRecord],
    output: Path,
    epochs: int = 1,
    max_records: int | None = 128,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Train a bounded image verifier and export a CPU-loadable TorchScript model."""
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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    num_workers = 2 if len(development) >= 1024 else 0
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "generator": torch.Generator().manual_seed(26162),
    }
    if num_workers:
        loader_kwargs.update(prefetch_factor=2, persistent_workers=True)
    loader = DataLoader(_ImageDataset(development, label_ids), **loader_kwargs)
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    losses: list[float] = []
    model.train()
    for _ in range(epochs):
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
    model = model.cpu().eval()
    output.mkdir(parents=True, exist_ok=True)
    torch.jit.trace(model, torch.zeros(1, 3, 224, 224)).save(str(output / "image_verifier.ts"))
    metadata = {
        "classes": labels,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "training_rows": len(development),
        "batch_size": batch_size,
        "amp": amp_enabled,
        "num_workers": num_workers,
        "epochs": epochs,
        "loss_final": losses[-1] if losses else None,
        "ranking_rows": ranking_rows,
        "ranking_rows_used": 0,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
