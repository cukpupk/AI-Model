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


def main() -> None:
    parser = argparse.ArgumentParser(description="Reassemble bge-m3 runtime from chunk files")
    parser.add_argument("--chunks-dir", default="models/chunks/bge-m3", help="Chunk directory")
    parser.add_argument("--output-dir", default="models/local", help="Reassembly output parent")
    args = parser.parse_args()

    chunks_dir = Path(args.chunks_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((chunks_dir / "manifest.json").read_text(encoding="utf-8"))
    bundle = chunks_dir / manifest["bundle"]["file"]

    with bundle.open("wb") as out:
        for part in manifest["parts"]:
            part_file = chunks_dir / part["file"]
            actual = sha256_file(part_file)
            if actual != part["sha256"]:
                raise ValueError(f"Checksum mismatch for {part_file.name}")
            with part_file.open("rb") as pf:
                for block in iter(lambda: pf.read(1024 * 1024), b""):
                    out.write(block)

    if sha256_file(bundle) != manifest["bundle"]["sha256"]:
        raise ValueError("Reassembled tar checksum mismatch")

    with tarfile.open(bundle, "r") as tar:
        tar.extractall(path=output_dir, filter="data")

    print(f"Reassembled and extracted to {output_dir / 'bge-m3'}")


if __name__ == "__main__":
    main()
