from pathlib import Path

from thermis.adapters.environmental import parse_micasa_date, parse_modis_identity


def test_parse_micasa_daily_date() -> None:
    path = Path("MiCASA_v1_flux_x3600_y1800_daily_20240229.nc4")
    assert parse_micasa_date(path).isoformat() == "2024-02-29"


def test_modis_identity_groups_processing_versions() -> None:
    path = Path("MCD64A1.A2024153.h10v06.061.2024225140057.hdf")
    identity = parse_modis_identity(path)
    assert identity == (2024, 153, "h10v06", "061")
