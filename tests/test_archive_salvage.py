import zipfile


def test_salvage_handles_streaming_deflate_data_descriptors(tmp_path):
    import io

    from thermis.archive_salvage import salvage

    class Streaming(io.BytesIO):
        def seek(self, *args):
            raise io.UnsupportedOperation()

    stream = Streaming()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("a.tiff", b"pixels" * 100)
        z.writestr("b.tiff", b"other pixels" * 100)
    data = stream.getvalue()
    path = tmp_path / "stream.zip"
    path.write_bytes(data[: data.index(b"PK\x01\x02")])
    report = salvage(path, tmp_path / "out")
    assert report["verified_members"] == 2
    assert (tmp_path / "out" / "a.tiff").read_bytes() == b"pixels" * 100


def test_salvage_verifies_complete_members_without_central_directory(tmp_path):
    from thermis.archive_salvage import salvage

    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("patches/a.tiff", b"complete image bytes")
        z.writestr("patches/b.tiff", b"second image bytes")
    data = archive.read_bytes()
    truncated = data[: data.index(b"PK\x01\x02")]
    archive.write_bytes(truncated)
    report = salvage(archive, tmp_path / "recovered")
    assert report["verified_members"] == 2
    assert (tmp_path / "recovered" / "a.tiff").read_bytes() == b"complete image bytes"
    assert archive.read_bytes() == truncated


def test_salvage_never_admits_corrupt_or_incomplete_members(tmp_path):
    from thermis.archive_salvage import salvage

    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr("first.tiff", b"good")
        z.writestr("last.tiff", b"incomplete content")
    data = archive.read_bytes()
    archive.write_bytes(data[: data.index(b"incomplete content") + 3])
    report = salvage(archive, tmp_path / "recovered")
    assert report["verified_members"] == 1
    assert report["stop_reason"] == "incomplete_member"
    assert not (tmp_path / "recovered" / "last.tiff").exists()
