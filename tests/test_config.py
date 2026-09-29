from pathlib import Path

from thermis.config import load_settings


def test_load_settings_resolves_source_roots(tmp_path: Path) -> None:
    cfg = tmp_path / "sources.yaml"
    cfg.write_text(
        "sources:\n"
        "  structured: D:/data/csv_output\n"
        "  micasa_daily: R:/MiCASA_FLUX_DAILY\n"
        "artifacts_root: data\n",
        encoding="utf-8",
    )
    settings = load_settings(cfg)
    assert settings.sources["structured"] == Path("D:/data/csv_output")
    assert settings.artifacts_root == Path("data")
