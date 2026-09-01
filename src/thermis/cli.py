from pathlib import Path

import typer

from thermis.adapters.environmental import build_environment_manifest
from thermis.adapters.fire_atlas import build_fire_atlas_manifest
from thermis.config import load_settings
from thermis.images import build_image_manifest
from thermis.inventory import inventory_sources

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
def prepare_manifests(config: str = "config/sources.yaml") -> None:
    """Build lightweight provenance manifests from configured source roots."""
    config_path = Path(config)
    settings = load_settings(config_path)
    root = config_path.parent.parent / settings.artifacts_root
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    env = build_environment_manifest(settings.sources)
    env.to_parquet(manifests / "environment_manifest.parquet", index=False)
    atlas_root = settings.sources.get("global_fire_atlas")
    atlas = build_fire_atlas_manifest(atlas_root) if atlas_root and atlas_root.exists() else None
    if atlas is not None:
        atlas.to_parquet(manifests / "fire_atlas_manifest.parquet", index=False)
    images = build_image_manifest(settings.sources)
    images.to_parquet(manifests / "image_manifest.parquet", index=False)
    typer.echo(f"environment={len(env)} images={len(images)}")
    if atlas is not None:
        typer.echo(f"fire_atlas_archives={len(atlas)} checksums={int(atlas['checksum_ok'].sum())}")
