import typer

from thermis.config import load_settings

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
