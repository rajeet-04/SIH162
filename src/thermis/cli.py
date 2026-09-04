import json
from hashlib import sha256
from pathlib import Path

import pandas as pd
import typer

from thermis.adapters.environmental import build_environment_manifest
from thermis.adapters.fire_atlas import build_fire_atlas_manifest
from thermis.adapters.structured import read_frp_events
from thermis.api import ModelRuntime, create_app
from thermis.config import load_settings
from thermis.evaluation import evaluate_ranking_set
from thermis.features import build_event_features
from thermis.image_model import ImageRecord, assign_image_splits, train_image_smoke
from thermis.images import build_image_manifest
from thermis.inventory import inventory_sources
from thermis.labels import assign_labels
from thermis.splits import (
    assert_no_group_overlap,
    assert_ranking_is_newest,
    make_grouped_time_split,
)
from thermis.training import train_tabular

app = typer.Typer(help="SIH26162 data and model pipeline")


@app.command()
def version() -> None:
    """Print the pipeline version."""
    typer.echo("thermis 0.1.0")


@app.command("check-config")
def check_config(path: str = "config/sources.yaml") -> None:
    """Validate the source configuration file."""
    settings = load_settings(__import__("pathlib").Path(path))
    typer.echo(f"validated {len(settings.sources)} source roots")


@app.command("inventory")
def inventory(
    config: str = "config/sources.yaml",
    output: str = "data/manifests/source_manifest.parquet",
) -> None:
    """Create a source manifest while leaving configured roots untouched."""
    config_path = Path(config)
    settings = load_settings(config_path)
    destination = Path(output)
    artifacts_root = settings.artifacts_root
    if not artifacts_root.is_absolute():
        artifacts_root = config_path.parent.parent / artifacts_root
    if not destination.is_absolute():
        destination = config_path.parent.parent / destination
    if artifacts_root.resolve() not in destination.resolve().parents:
        raise typer.BadParameter("output must be inside artifacts_root")
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = inventory_sources(settings)
    manifest.to_parquet(destination, index=False)
    typer.echo(f"wrote {len(manifest)} rows to {destination}")


@app.command("prepare-manifests")
def prepare_manifests(
    config: str = "config/sources.yaml",
    verify_checksums: bool = typer.Option(False, help="Hash every Fire Atlas archive."),
) -> None:
    """Build lightweight provenance manifests from configured source roots."""
    config_path = Path(config)
    settings = load_settings(config_path)
    root = config_path.parent.parent / settings.artifacts_root
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    env = build_environment_manifest(settings.sources)
    env.to_parquet(manifests / "environment_manifest.parquet", index=False)
    atlas_root = settings.sources.get("global_fire_atlas")
    atlas = (
        build_fire_atlas_manifest(atlas_root, verify_checksums=verify_checksums)
        if atlas_root and atlas_root.exists()
        else None
    )
    if atlas is not None:
        atlas.to_parquet(manifests / "fire_atlas_manifest.parquet", index=False)
    images = build_image_manifest(settings.sources)
    images.to_parquet(manifests / "image_manifest.parquet", index=False)
    typer.echo(f"environment={len(env)} images={len(images)}")
    if atlas is not None:
        typer.echo(f"fire_atlas_archives={len(atlas)} checksums={int(atlas['checksum_ok'].sum())}")


@app.command("label-events")
def label_events(
    events: str = "data/cleaned/events.parquet",
    output: str = "data/labels/events_labeled.parquet",
) -> None:
    """Apply conservative labels and write a label-distribution report."""
    frame = pd.read_parquet(events)
    labeled = assign_labels(frame)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_parquet(destination, index=False)
    report = labeled["label_stage_2"].value_counts(dropna=False).to_dict()
    report_path = Path("reports/data/label_distribution.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_payload = {str(key): int(value) for key, value in report.items()}
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    typer.echo(f"wrote {len(labeled)} labeled events to {destination}")


@app.command("prepare-events")
def prepare_events(
    config: str = "config/sources.yaml",
    output: str = "data/cleaned/events.parquet",
    rejects: str = "data/cleaned/event_rejects.parquet",
) -> None:
    """Normalize structured FRP CSV files and preserve rejected rows."""
    settings = load_settings(Path(config))
    structured_root = settings.sources["structured"]
    frames = [read_frp_events(path) for path in sorted(structured_root.glob("*.csv"))]
    if not frames:
        raise typer.BadParameter(f"no CSV files found under {structured_root}")
    combined = pd.concat(frames, ignore_index=True)
    destination = Path(output)
    reject_path = Path(rejects)
    destination.parent.mkdir(parents=True, exist_ok=True)
    reject_path.parent.mkdir(parents=True, exist_ok=True)
    combined.loc[combined["reject_reason"].isna()].to_parquet(destination, index=False)
    combined.loc[combined["reject_reason"].notna()].to_parquet(reject_path, index=False)
    typer.echo(
        f"accepted={combined['reject_reason'].isna().sum()} "
        f"rejected={combined['reject_reason'].notna().sum()}"
    )


@app.command("build-features")
def build_features(
    events: str = "data/cleaned/events.parquet",
    output: str = "data/features/event_features.parquet",
) -> None:
    """Build point-in-time persistence features for normalized events."""
    frame = pd.read_parquet(events)
    settings = load_settings(Path("config/sources.yaml"))
    flare_path = settings.sources["flaresat"] / "sources" / "flare" / "gas_flaring_points.csv"
    flare_points = pd.read_csv(flare_path) if flare_path.exists() else None
    features = build_event_features(frame, {"flare_points": flare_points})
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(destination, index=False)
    typer.echo(f"wrote {len(features)} feature rows to {destination}")


@app.command("make-splits")
def make_splits(
    events: str = "data/labels/events_labeled.parquet",
    images: str = "data/manifests/image_manifest.parquet",
    output: str = "data/splits",
) -> None:
    """Freeze grouped chronological event and image split manifests."""
    del images
    frame = pd.read_parquet(events)
    group_cols = ["scene_group"] if "scene_group" in frame.columns else ["event_id"]
    split = make_grouped_time_split(frame, "timestamp_utc", group_cols, 0.10)
    assert_no_group_overlap(split, group_cols)
    assert_ranking_is_newest(split, "timestamp_utc")
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "event_splits.parquet"
    split.to_parquet(target, index=False)
    digest = sha256(target.read_bytes()).hexdigest()
    Path(f"{target}.sha256").write_text(f"{digest}  {target.name}\n", encoding="ascii")
    typer.echo(f"wrote {len(split)} split rows to {target}")


@app.command("train-tabular")
def train_tabular_command(
    features: str = "data/features/event_features.parquet",
    splits: str = "data/splits/event_splits.parquet",
    output: str = "models/tabular",
) -> None:
    """Train Stage 1 and Stage 2 CatBoost models on development rows only."""
    feature_frame = pd.read_parquet(features)
    split_frame = pd.read_parquet(splits)[["event_id", "split"]]
    frame = feature_frame.merge(split_frame, on="event_id", validate="one_to_one")
    if "label_stage_1" not in frame.columns:
        labels = pd.read_parquet("data/labels/events_labeled.parquet")
        label_columns = ["event_id", "label_stage_1", "label_stage_2", "supervised_eligible"]
        frame = frame.merge(labels[label_columns], on="event_id", validate="one_to_one")
    bundles = train_tabular(frame, Path(output))
    typer.echo(
        f"stage1_rows={bundles['stage1']['training_rows']} "
        f"stage2_rows={bundles['stage2']['training_rows']} ranking_rows_used=0"
    )


@app.command("serve")
def serve(
    demo: bool = typer.Option(False, help="Run the offline demo mode."),
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Serve the loaded tabular model and API contracts."""
    import uvicorn

    demo_path = Path("data/demo/events.json") if demo else None
    runtime = ModelRuntime.from_paths(Path("models/tabular"), demo_path=demo_path)
    if demo:
        runtime.model_version = "tabular-local-0.1-offline"
    uvicorn.run(create_app(runtime), host=host, port=port)


@app.command("evaluate-ranking")
def evaluate_ranking(
    features: str = "data/features/event_features.parquet",
    splits: str = "data/splits/event_splits.parquet",
    models: str = "models/tabular",
    output: str = "reports/ranking",
) -> None:
    """Evaluate the untouched ranking rows without model mutation."""
    report = evaluate_ranking_set(Path(features), Path(splits), Path(models), Path(output))
    typer.echo(
        f"ranking_rows={report['ranking_rows']} promoted={report['promoted']} "
        f"reasons={','.join(report['reasons'])}"
    )


@app.command("train-image")
def train_image(
    manifest: str = "data/manifests/image_manifest.parquet",
    output: str = "models/image-smoke",
    epochs: int = 1,
    max_records: int = 0,
    batch_size: int = 16,
    val_fraction: float = 0.0,
    weight_decay: float = 0.0,
    augment: bool = False,
) -> None:
    """Run the bounded CUDA image-verifier smoke train and export TorchScript."""
    frame = pd.read_parquet(manifest)
    frame = assign_image_splits(frame)
    split_path = Path("data/splits/image_splits.parquet")
    split_path.parent.mkdir(parents=True, exist_ok=True)
    frame[["image_id", "scene_group", "split"]].to_parquet(split_path, index=False)
    records = [
        ImageRecord(
            image_id=str(row.image_id),
            scene_group=str(row.scene_group),
            split=str(row.split),
            label=str(row.label_source),
            path=str(row.path),
        )
        for row in frame.itertuples()
        if bool(row.readable) and str(row.label_source).lower() in {"fire", "no_fire"}
    ]
    metadata = train_image_smoke(
        records,
        Path(output),
        epochs=epochs,
        max_records=None if max_records <= 0 else max_records,
        batch_size=batch_size,
        val_fraction=val_fraction,
        weight_decay=weight_decay,
        augment=augment,
    )
    typer.echo(
        f"image_rows={metadata['training_rows']} device={metadata['device']} "
        f"ranking={metadata['ranking_rows']} used={metadata['ranking_rows_used']} "
        f"best_epoch={metadata.get('best_epoch')} val_nll={metadata.get('val_nll_best')}"
    )
