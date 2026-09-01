from pathlib import Path

import typer

from thermis.config import load_settings
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
