from pathlib import Path

from thermis.inventory import inventory_root


def test_inventory_marks_partial_files_excluded(tmp_path: Path) -> None:
    (tmp_path / "event.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (tmp_path / "event.nc4.part").write_bytes(b"partial")
    manifest = inventory_root(tmp_path, "structured")
    status = dict(zip(manifest["extension"], manifest["status"], strict=True))
    assert status[".csv"] == "candidate"
    assert status[".part"] == "excluded"


def test_inventory_assigns_same_fast_hash_to_equal_content(tmp_path: Path) -> None:
    payload = b"same-content"
    (tmp_path / "a.csv").write_bytes(payload)
    (tmp_path / "b.csv").write_bytes(payload)
    manifest = inventory_root(tmp_path, "structured")
    assert manifest["fast_hash"].nunique() == 1
