"""Verify recorded original FITS without changing or downloading data."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    work = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=work/"SOL2026-07-07/data_manifest.json")
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    missing, mismatched = [], []
    for record in manifest["files"]:
        path = (root/record["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Manifest path escapes its event directory")
        if not path.is_file():
            missing.append(record["path"])
            continue
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream,"sha256").hexdigest()
        if actual != record["sha256"] or path.stat().st_size != record["bytes"]:
            mismatched.append(record["path"])
    print(f"Checked {len(manifest['files'])} recorded FITS: {len(missing)} missing, {len(mismatched)} mismatched")
    for name in missing: print("MISSING:",name)
    for name in mismatched: print("MISMATCH:",name)
    return 1 if missing or mismatched else 0


if __name__ == "__main__":
    raise SystemExit(main())
