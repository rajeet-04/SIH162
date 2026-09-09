"""Recover CRC-verified local ZIP entries; never claim the broken archive is valid."""

import argparse
import hashlib
import json
import struct
import zlib
from pathlib import Path, PurePosixPath


def salvage(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    members = []
    reason = "end_of_file"
    with source.open("rb") as stream:
        while True:
            offset = stream.tell()
            header = stream.read(30)
            if not header:
                break
            if len(header) < 30 or header[:4] != b"PK\x03\x04":
                reason = "end_of_local_entries"
                break
            _, version, flags, method, _, _, crc, compressed, size, namesize, extrasize = (
                struct.unpack("<4s5H3I2H", header)
            )
            if flags & 1 or method not in (0, 8) or max(compressed, size) > 64_000_000:
                reason = "unsupported_or_oversize_member"
                break
            name = stream.read(namesize).decode("utf-8" if flags & 2048 else "cp437")
            extra = stream.read(extrasize)
            if len(extra) != extrasize:
                reason = "incomplete_member"
                break
            try:
                if flags & 8:
                    if method != 8:
                        reason = "unsupported_streaming_method"
                        break
                    decoder = zlib.decompressobj(-15)
                    decoded = bytearray()
                    consumed = 0
                    while not decoder.eof:
                        block = stream.read(65536)
                        if not block:
                            break
                        consumed += len(block)
                        decoded.extend(decoder.decompress(block, 64_000_001 - len(decoded)))
                        if consumed > 64_000_000 or len(decoded) > 64_000_000:
                            raise ValueError("Member exceeds bounded recovery size")
                    if not decoder.eof:
                        reason = "incomplete_member"
                        break
                    consumed -= len(decoder.unused_data)
                    stream.seek(-len(decoder.unused_data), 1)
                    descriptor = stream.read(4)
                    descriptor = (
                        stream.read(12)
                        if descriptor == b"PK\x07\x08"
                        else descriptor + stream.read(8)
                    )
                    if len(descriptor) != 12:
                        reason = "incomplete_descriptor"
                        break
                    crc, compressed, size = struct.unpack("<3I", descriptor)
                    if compressed != consumed:
                        reason = "compressed_size_failure"
                        break
                elif method == 8:
                    data = stream.read(compressed)
                    if len(data) != compressed:
                        reason = "incomplete_member"
                        break
                    decoder = zlib.decompressobj(-15)
                    decoded = decoder.decompress(data, size + 1)
                    if not decoder.eof or decoder.unused_data:
                        reason = "invalid_compressed_member"
                        break
                else:
                    decoded = stream.read(compressed)
                    if len(decoded) != compressed:
                        reason = "incomplete_member"
                        break
            except zlib.error:
                reason = "invalid_compressed_member"
                break
            if len(decoded) != size or zlib.crc32(decoded) & 0xFFFFFFFF != crc:
                reason = "crc_or_size_failure"
                break
            if name.endswith("/"):
                continue
            basename = PurePosixPath(name).name
            if (
                PurePosixPath(name).is_absolute()
                or ".." in PurePosixPath(name).parts
                or "\\" in name
            ):
                reason = "unsafe_member_name"
                break
            destination = output / basename
            with destination.open("xb") as handle:
                handle.write(decoded)
            members.append(
                dict(
                    name=name,
                    file=basename,
                    bytes=size,
                    crc32=f"{crc:08x}",
                    sha256=hashlib.sha256(decoded).hexdigest(),
                    offset=offset,
                )
            )
    report = dict(
        source=str(source),
        source_sha256=hashlib.file_digest(source.open("rb"), "sha256").hexdigest(),
        verified_members=len(members),
        stop_reason=reason,
        members=members,
        admission="CRC integrity only; imagery semantics and label alignment still required",
    )
    (output / "salvage.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            {k: v for k, v in salvage(args.source, args.output).items() if k != "members"}, indent=2
        )
    )
