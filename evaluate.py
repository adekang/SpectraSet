"""Evaluate SpectraSet checkpoints on released FTIR directories.

FCG and LEC spectra are independently min-max-normalized and cubically resized.
GAS1647 spectra are already normalized on their native 1,647-point grid and are
kept unchanged when the GAS configuration is selected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from spectraset import SpectraSet, SpectraSetConfig, load_spectraset_checkpoint

CLASS_NAMES = (
    "alkane",
    "methyl",
    "alkene",
    "alkyne",
    "alcohols",
    "amines",
    "nitriles",
    "aromatics",
    "alkyl halides",
    "esters",
    "ketones",
    "aldehydes",
    "carboxylic acids",
    "ether",
    "acyl halides",
    "amides",
    "nitro",
)
SUMMARY_METRICS = (
    "label_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "micro_f1",
    "macro_auprc",
    "exact_match_accuracy",
)


def list_samples(split_dir: str | Path) -> list[str]:
    """Return sorted spectra that have a paired label file."""

    split_dir = Path(split_dir)
    if not split_dir.is_dir():
        raise FileNotFoundError(f"split directory not found: {split_dir}")
    names = sorted(
        path.name
        for path in split_dir.glob("*.npy")
        if path.with_suffix(".txt").is_file()
    )
    if not names:
        raise ValueError(f"no paired .npy/.txt samples found in {split_dir}")
    return names


def names_sha256(names: list[str]) -> str:
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def preprocess_spectrum(
    spectrum: np.ndarray,
    signal_size: int,
    *,
    normalize: bool = True,
) -> Tensor:
    """Convert a stored spectrum to a model signal with explicit preprocessing."""

    spectrum = np.asarray(spectrum)
    if spectrum.ndim != 1:
        raise ValueError(f"expected a rank-1 spectrum, got {spectrum.shape}")
    if signal_size < 2:
        raise ValueError("signal_size must be at least two")
    if spectrum.size < 2 or not np.isfinite(spectrum).all():
        raise ValueError("spectrum must contain at least two finite values")

    signal = spectrum.astype(np.float32, copy=False)
    if normalize:
        maximum = np.max(signal)
        minimum = np.min(signal)
        if maximum <= minimum:
            raise ValueError("constant spectra cannot be min-max normalized")
        signal = (signal - minimum) / (maximum - minimum)
    if signal.size != signal_size:
        signal = cv2.resize(
            np.expand_dims(signal, axis=0),
            (signal_size, 1),
            interpolation=cv2.INTER_CUBIC,
        )
    else:
        signal = np.expand_dims(signal, axis=0)
    return torch.from_numpy(signal).to(torch.float32)


class ReleasedFTIRDataset(Dataset):
    """Paired NumPy-spectrum and text-label benchmark split."""

    def __init__(
        self,
        split_dir: str | Path,
        signal_size: int = 1024,
        *,
        normalize: bool = True,
    ) -> None:
        self.split_dir = Path(split_dir)
        self.signal_size = signal_size
        self.normalize = normalize
        self.names = list_samples(self.split_dir)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        name = self.names[index]
        signal_path = self.split_dir / name
        label_path = signal_path.with_suffix(".txt")
        spectrum = np.load(signal_path, allow_pickle=False)
        signal = preprocess_spectrum(
            spectrum,
            self.signal_size,
            normalize=self.normalize,
        )
        labels = np.fromstring(label_path.read_text(encoding="utf-8").strip(), sep=" ")
        if labels.shape != (len(CLASS_NAMES),) or not np.isin(labels, (0, 1)).all():
            raise ValueError(f"invalid 17-label vector: {label_path}")
        return {
            "signal": signal,
            "label": torch.from_numpy(labels.astype(np.float32)),
            "name": name,
        }


def uses_native_gas_preprocessing(
    signal_size: int,
    minimum_wavenumber: float,
    maximum_wavenumber: float,
) -> bool:
    """Identify the documented native GAS1647 input representation."""

    return (
        signal_size == 1647
        and minimum_wavenumber == 552.0
        and maximum_wavenumber == 3844.0
    )


def finite_float(value: Any, name: str) -> float:
    """Convert a metric scalar while preserving a useful evaluator failure."""

    try:
        converted = float(value)
    except (RuntimeError, TypeError, ValueError) as error:
        raise RuntimeError(f"could not convert {name} to a scalar") from error
    if not math.isfinite(converted):
        raise RuntimeError(f"{name} is not finite: {converted}")
    return converted


def nonnegative_int(value: Any, name: str) -> int:
    """Convert a count-like metric value and reject invalid negative supports."""

    try:
        converted = int(value)
    except (RuntimeError, TypeError, ValueError) as error:
        raise RuntimeError(f"could not convert {name} to an integer") from error
    if converted < 0:
        raise RuntimeError(f"{name} must be nonnegative: {converted}")
    return converted


def metric_components(result: Any, name: str) -> tuple[Any, Any, Any, Any]:
    """Validate the four-value metric tuple returned by scikit-learn."""

    if not isinstance(result, tuple) or len(result) != 4:
        raise RuntimeError(f"unexpected {name} metric result")
    return result


def peak_gpu_memory_bytes(device: torch.device) -> int:
    """Read CUDA allocation statistics with an actionable failure on reset."""

    if device.type != "cuda":
        return 0
    try:
        return nonnegative_int(torch.cuda.max_memory_allocated(device), "GPU memory")
    except RuntimeError as error:
        raise RuntimeError("could not read CUDA peak-memory statistics") from error


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute the fixed-threshold metrics used by the released benchmarks."""

    labels = np.asarray(labels, dtype=np.uint8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.shape != probabilities.shape or labels.ndim != 2:
        raise ValueError("labels and probabilities must have the same rank-2 shape")
    if labels.shape[1] != len(CLASS_NAMES):
        raise ValueError("expected 17 functional-group columns")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be binary")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities contain non-finite values")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must lie in (0, 1)")

    predictions = (probabilities >= threshold).astype(np.uint8)
    micro = precision_recall_fscore_support(
        labels.reshape(-1),
        predictions.reshape(-1),
        average="binary",
        zero_division=cast(Any, 0),
    )
    macro = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=cast(Any, 0),
    )
    per_class = precision_recall_fscore_support(
        labels,
        predictions,
        average=None,
        zero_division=cast(Any, 0),
    )
    micro_precision, micro_recall, micro_f1, _ = metric_components(micro, "micro")
    macro_precision, macro_recall, macro_f1, _ = metric_components(macro, "macro")
    per_precision, per_recall, per_f1, per_support = metric_components(
        per_class,
        "per-class",
    )
    return {
        "label_accuracy": finite_float(
            (labels == predictions).mean(),
            "label accuracy",
        ),
        "exact_match_accuracy": finite_float(
            np.all(labels == predictions, axis=1).mean(),
            "exact-match accuracy",
        ),
        "micro_precision": finite_float(micro_precision, "micro precision"),
        "micro_recall": finite_float(micro_recall, "micro recall"),
        "micro_f1": finite_float(micro_f1, "micro F1"),
        "macro_precision": finite_float(macro_precision, "macro precision"),
        "macro_recall": finite_float(macro_recall, "macro recall"),
        "macro_f1": finite_float(macro_f1, "macro F1"),
        "macro_auprc": finite_float(
            average_precision_score(labels, probabilities, average="macro"),
            "macro AUPRC",
        ),
        "per_class": {
            name: {
                "precision": finite_float(per_precision[index], f"{name} precision"),
                "recall": finite_float(per_recall[index], f"{name} recall"),
                "f1": finite_float(per_f1[index], f"{name} F1"),
                "support": nonnegative_int(per_support[index], f"{name} support"),
            }
            for index, name in enumerate(CLASS_NAMES)
        },
    }


def _checkpoint_seed(metadata: dict[str, Any], path: Path, index: int) -> str:
    for key in ("model_seed", "seed", "run_seed"):
        if key in metadata:
            return str(metadata[key])
    return f"checkpoint_{index}_{path.stem}"


@torch.inference_mode()
def predict_split(
    model: SpectraSet,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names: list[str] = []
    labels = []
    probabilities = []
    model.eval()
    for batch in loader:
        signal = batch["signal"].to(device, non_blocking=True)
        logits = model(signal)["logits"]
        if logits is None:
            raise RuntimeError("model did not return logits")
        names.extend(batch["name"])
        labels.append(batch["label"].numpy().astype(np.uint8))
        probabilities.append(logits.float().sigmoid().cpu().numpy())
    return (
        np.asarray(names),
        np.concatenate(labels),
        np.concatenate(probabilities),
    )


def aggregate_metrics(per_checkpoint: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate checkpoint metrics as mean and sample standard deviation."""

    aggregate: dict[str, Any] = {"count": len(per_checkpoint), "mean": {}, "std": {}}
    for metric in SUMMARY_METRICS:
        values = np.asarray(
            [record["metrics"][metric] for record in per_checkpoint],
            dtype=np.float64,
        )
        aggregate["mean"][metric] = finite_float(values.mean(), f"mean {metric}")
        aggregate["std"][metric] = (
            finite_float(values.std(ddof=1), f"standard deviation {metric}")
            if len(values) > 1
            else 0.0
        )
        aggregate.setdefault("values", {})[metric] = values.tolist()
    return aggregate


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    native_gas = uses_native_gas_preprocessing(
        args.signal_size,
        args.minimum_wavenumber,
        args.maximum_wavenumber,
    )
    dataset = ReleasedFTIRDataset(
        args.split_dir,
        signal_size=args.signal_size,
        normalize=not native_gas,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    checkpoint_records = []
    canonical_names = None
    canonical_labels = None
    for index, checkpoint_path in enumerate(args.checkpoint):
        checkpoint_path = Path(checkpoint_path)
        config = SpectraSetConfig(
            signal_size=args.signal_size,
            minimum_wavenumber=args.minimum_wavenumber,
            maximum_wavenumber=args.maximum_wavenumber,
        )
        model = SpectraSet(config)
        receipt = load_spectraset_checkpoint(model, checkpoint_path, strict=True)
        seed = _checkpoint_seed(receipt["metadata"], checkpoint_path, index)
        model = model.to(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        started = time.monotonic()
        names, labels, probabilities = predict_split(model, loader, device)
        elapsed = time.monotonic() - started
        if canonical_names is None:
            canonical_names = names
            canonical_labels = labels
        else:
            if canonical_labels is None:
                raise RuntimeError("canonical labels were not initialized")
            if not np.array_equal(canonical_names, names) or not np.array_equal(
                canonical_labels,
                labels,
            ):
                raise RuntimeError("sample order or labels changed between checkpoints")

        metrics = classification_metrics(labels, probabilities, args.threshold)
        prediction_path = output_dir / f"predictions_seed{seed}.npz"
        np.savez_compressed(
            prediction_path,
            test_names=names,
            test_labels=labels.astype(np.uint8),
            test_probabilities=probabilities.astype(np.float64),
        )
        record = {
            "seed": seed,
            "checkpoint": str(checkpoint_path.resolve()),
            "prediction_file": prediction_path.name,
            "metrics": metrics,
            "inference_seconds": elapsed,
            "max_gpu_memory_bytes": peak_gpu_memory_bytes(device),
        }
        checkpoint_records.append(record)
        _write_json(output_dir / f"metrics_seed{seed}.json", record)
        del model

    if canonical_names is None or canonical_labels is None:
        raise RuntimeError("no checkpoints were evaluated")
    summary = {
        "status": "complete",
        "split_dir": str(Path(args.split_dir).resolve()),
        "samples": len(dataset),
        "names_sha256": names_sha256(dataset.names),
        "signal_size": args.signal_size,
        "preprocessing": "native_gas" if native_gas else "normalize_and_resize",
        "classes": list(CLASS_NAMES),
        "threshold": args.threshold,
        "device": str(device),
        "parameters": 9_490_053 if args.signal_size == 1024 else None,
        "checkpoints": checkpoint_records,
        "aggregate": aggregate_metrics(checkpoint_records),
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dataset", choices=("fcg", "lec", "gas"))
    parser.add_argument("--signal-size", type=int)
    parser.add_argument("--minimum-wavenumber", type=float)
    parser.add_argument("--maximum-wavenumber", type=float)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    profile = {
        "fcg": (1024, 400.0, 4000.0),
        "lec": (1024, 400.0, 4000.0),
        "gas": (1647, 552.0, 3844.0),
    }[args.dataset or "fcg"]
    args.signal_size = args.signal_size if args.signal_size is not None else profile[0]
    args.minimum_wavenumber = (
        args.minimum_wavenumber if args.minimum_wavenumber is not None else profile[1]
    )
    args.maximum_wavenumber = (
        args.maximum_wavenumber if args.maximum_wavenumber is not None else profile[2]
    )
    if args.batch_size < 1 or args.workers < 0:
        parser.error("batch-size must be positive and workers nonnegative")
    if not 0.0 < args.threshold < 1.0:
        parser.error("threshold must lie in (0, 1)")
    return args


def main() -> None:
    summary = evaluate(parse_args())
    print(json.dumps({"status": summary["status"], "samples": summary["samples"]}))


if __name__ == "__main__":
    main()
