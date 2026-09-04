"""Segmentation baseline smoke train on FireSat-style 12-band NPZ patches.

Split is scene-grouped (no neighboring-patch leakage):
  fit:     D:/data/scene1 + D:/data/scene2
  dev-val: D:/data/scene3  (best-Dice epoch selection)
  holdout: D:/data/scene4  (evaluation only, never trained)

Model: tiny 2-level U-Net, 12ch 128px input -> 1ch fire mask.
Loss: BCE + (1 - Dice). Exports CPU TorchScript to models/seg-baseline/.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "models" / "seg-baseline"
REPORT = ROOT / "reports" / "seg_baseline.json"
SCENES = {
    "fit": [r"D:/data/scene1", r"D:/data/scene2"],
    "val": [r"D:/data/scene3"],
    "holdout": [r"D:/data/scene4"],
}
SIZE = 128
EPOCHS = 8
BATCH = 128
SEED = 26162
FIRE_OVERSAMPLE = 8
POS_WEIGHT = 60.0


class NpzDataset(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with np.load(self.paths[index], allow_pickle=False) as archive:
            image = archive["image"].astype(np.float32)
            mask = archive["label"].astype(np.float32)
        image = np.clip(image / 10000.0, 0.0, 1.0)
        image_t = torch.from_numpy(image).unsqueeze(0)
        image_t = torch.nn.functional.interpolate(
            image_t, size=(SIZE, SIZE), mode="bilinear", align_corners=False
        )[0]
        mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
        mask_t = torch.nn.functional.interpolate(mask_t, size=(SIZE, SIZE), mode="nearest")[0]
        return image_t, (mask_t > 0.5).float()


class TinyUNet(nn.Module):
    def __init__(self, in_channels: int = 12):
        super().__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
        )
        self.pool = nn.MaxPool2d(2)
        self.mid = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.up = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.down1(x)
        x = self.mid(self.pool(skip))
        x = self.up(x)
        return self.head(torch.cat([x, skip[:, :, : x.shape[2], : x.shape[3]]], dim=1))


def dice_score(prob: torch.Tensor, truth: torch.Tensor, eps: float = 1e-6) -> float:
    pred = (prob > 0.5).float()
    inter = float((pred * truth).sum().cpu())
    return (2 * inter + eps) / (float(pred.sum().cpu()) + float(truth.sum().cpu()) + eps)


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    splits = {
        name: sorted(p for root in roots for p in Path(root).glob("*.npz"))
        for name, roots in SCENES.items()
    }
    print({name: len(paths) for name, paths in splits.items()}, flush=True)
    fire_paths = [
        p
        for p in splits["fit"]
        if bool((np.load(p, allow_pickle=False)["label"] > 0).any())
    ]
    print(f"fit fire patches={len(fire_paths)}/{len(splits['fit'])}", flush=True)
    fit_paths = splits["fit"] + fire_paths * FIRE_OVERSAMPLE
    rng = np.random.RandomState(SEED)
    rng.shuffle(fit_paths)
    print(f"fit rows after oversample={len(fit_paths)}", flush=True)

    def loader(name: str, shuffle: bool) -> DataLoader:
        paths = fit_paths if name == "fit" else splits[name]
        return DataLoader(
            NpzDataset(paths),
            batch_size=BATCH,
            shuffle=shuffle,
            num_workers=4 if len(paths) >= 512 else 0,
            pin_memory=device.type == "cuda",
            persistent_workers=len(paths) >= 512,
            generator=torch.Generator().manual_seed(SEED) if shuffle else None,
        )

    train_loader, val_loader, hold_loader = (
        loader("fit", True),
        loader("val", False),
        loader("holdout", False),
    )
    model = TinyUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT], device=device))
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best_dice, best_state, best_epoch = -1.0, None, EPOCHS
    for epoch in range(EPOCHS):
        model.train()
        for images, masks in train_loader:
            images, masks = images.to(device, non_blocking=True), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(images)
                probs = torch.sigmoid(logits)
                inter = (probs * masks).sum()
                loss = bce(logits, masks) + 1 - (2 * inter + 1e-6) / (
                    probs.sum() + masks.sum() + 1e-6
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        model.eval()
        dices: list[float] = []
        with torch.inference_mode():
            for images, masks in val_loader:
                probs = torch.sigmoid(model(images.to(device))).cpu()
                dices.append(dice_score(probs, masks))
        val_dice = sum(dices) / len(dices)
        print(f"epoch={epoch + 1}/{EPOCHS} val_dice={val_dice:.4f}", flush=True)
        if val_dice > best_dice:
            best_dice, best_epoch = val_dice, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)

    model.eval()
    hold_dices, ious, correct, total = [], [], 0, 0
    with torch.inference_mode():
        for images, masks in hold_loader:
            probs = torch.sigmoid(model(images.to(device))).cpu()
            pred = (probs > 0.5).float()
            hold_dices.append(dice_score(probs, masks))
            inter = float(((pred == 1) & (masks == 1)).sum())
            union = float(((pred == 1) | (masks == 1)).sum())
            ious.append(inter / union if union else 1.0)
            correct += int((pred == masks).sum())
            total += masks.numel()
    payload = {
        "fit_patches": len(splits["fit"]),
        "val_patches": len(splits["val"]),
        "holdout_patches": len(splits["holdout"]),
        "best_epoch": best_epoch,
        "val_dice_best": best_dice,
        "holdout_dice": sum(hold_dices) / len(hold_dices),
        "holdout_iou": sum(ious) / len(ious),
        "holdout_pixel_accuracy": correct / total,
        "input": "12ch_128px_s2_reflectance",
        "device": str(device),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    model.cpu().eval()
    torch.jit.trace(model, torch.zeros(1, 12, SIZE, SIZE)).save(str(OUT / "seg_unet.ts"))
    (OUT / "metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
