from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def split_file(source_tar: Path, out_dir: Path, chunk_size_bytes: int, prefix: str) -> list[Path]:
    parts: list[Path] = []
    with source_tar.open("rb") as src:
        i = 1
        while True:
            data = src.read(chunk_size_bytes)
            if not data:
                break
            part = out_dir / f"{prefix}.part-{i:05d}"
            part.write_bytes(data)
            parts.append(part)
            i += 1
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description="Split bge-m3 runtime bundle into 45MiB chunks")
    parser.add_argument("--source-dir", required=True, help="Prepared bge-m3 runtime source directory")
    parser.add_argument("--output-dir", default="models/chunks/bge-m3", help="Output chunk directory")
    parser.add_argument("--prefix", default="bge-m3-runtime", help="Chunk prefix")
    parser.add_argument("--chunk-size-mib", type=int, default=45, help="Chunk size in MiB")
    args = parser.parse_args()

    source_dir = Path(args.source_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle_tar = out_dir / f"{args.prefix}.tar"
    with tarfile.open(bundle_tar, "w") as tar:
        tar.add(source_dir, arcname="bge-m3")

    chunk_size_bytes = args.chunk_size_mib * 1024 * 1024
    parts = split_file(bundle_tar, out_dir, chunk_size_bytes, args.prefix)

    bundle_sha = sha256_file(bundle_tar)
    sha_lines = []
    part_entries = []
    for p in parts:
        part_sha = sha256_file(p)
        sha_lines.append(f"{part_sha}  {p.name}")
        part_entries.append({"file": p.name, "size": p.stat().st_size, "sha256": part_sha})

    (out_dir / "SHA256SUMS").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")

    manifest = {
        "name": "bge-m3-runtime",
        "format": "tar+parts",
        "chunk_size_mib": args.chunk_size_mib,
        "bundle": {
            "file": bundle_tar.name,
            "size": bundle_tar.stat().st_size,
            "sha256": bundle_sha,
        },
        "parts": part_entries,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Created {len(parts)} parts in {out_dir}")


if __name__ == "__main__":
    main()
