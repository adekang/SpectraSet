"""Verify the bundled FCG, LEC, and GAS split files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED_COUNTS = {
    "fcg": {"train": 6342, "valid": 1387, "test": 933},
    "lec": {"train": 17502, "valid": 3501, "test": 2334},
    "gas": {"train": 4683, "valid": 585, "test": 586},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_split(split_dir: Path) -> dict[str, Any]:
    spectra = {path.stem: path for path in split_dir.glob("*.npy")}
    labels = {path.stem: path for path in split_dir.glob("*.txt")}
    unpaired_spectra = sorted(spectra.keys() - labels.keys())
    unpaired_labels = sorted(labels.keys() - spectra.keys())
    if unpaired_spectra or unpaired_labels:
        raise ValueError(
            f"unpaired files in {split_dir}: "
            f"spectra={unpaired_spectra[:5]}, labels={unpaired_labels[:5]}"
        )

    names = sorted(spectra)
    names_digest = hashlib.sha256()
    content_digest = hashlib.sha256()
    total_bytes = 0
    for stem in names:
        signal_path = spectra[stem]
        label_path = labels[stem]
        signal_hash = sha256_file(signal_path)
        label_hash = sha256_file(label_path)
        for value in (signal_path.name, signal_hash, label_hash):
            content_digest.update(value.encode("utf-8"))
            content_digest.update(b"\0")
        names_digest.update(signal_path.name.encode("utf-8"))
        names_digest.update(b"\0")
        total_bytes += signal_path.stat().st_size + label_path.stat().st_size
    return {
        "count": len(names),
        "names_sha256": names_digest.hexdigest(),
        "content_sha256": content_digest.hexdigest(),
        "total_bytes": total_bytes,
    }


def inspect_datasets(root: Path) -> dict[str, Any]:
    result = {"schema": "spectraset_dataset_files_v1", "datasets": {}}
    for dataset, split_counts in EXPECTED_COUNTS.items():
        dataset_root = root / dataset
        if not dataset_root.is_dir():
            raise FileNotFoundError(f"data set directory not found: {dataset_root}")
        split_records = {}
        for split, expected_count in split_counts.items():
            record = inspect_split(dataset_root / split)
            if record["count"] != expected_count:
                raise ValueError(
                    f"{dataset}/{split} count mismatch: "
                    f"expected {expected_count}, got {record['count']}"
                )
            split_records[split] = record
        result["datasets"][dataset] = {
            "split_manifest_sha256": sha256_file(dataset_root / "split_manifest.json"),
            "splits": split_records,
        }
    return result


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=default_root / "dataset_files.json",
    )
    parser.add_argument("--write-manifest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    observed = inspect_datasets(args.root)
    if args.write_manifest:
        args.manifest.write_text(
            json.dumps(observed, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.manifest}")
        return
    if not args.manifest.is_file():
        raise FileNotFoundError(f"verification manifest not found: {args.manifest}")
    expected = json.loads(args.manifest.read_text(encoding="utf-8"))
    if observed != expected:
        raise RuntimeError("data set content differs from dataset_files.json")
    counts = {
        dataset: sum(split["count"] for split in record["splits"].values())
        for dataset, record in observed["datasets"].items()
    }
    print(json.dumps({"status": "PASS", "counts": counts}))


if __name__ == "__main__":
    main()
