"""Train fixed-epoch SpectraSet checkpoints on bundled FTIR splits.

The command deliberately saves the final epoch rather than selecting a checkpoint
from validation metrics. Test splits are not read here; use ``spectraset-evaluate``
after training for held-out metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from evaluate import ReleasedFTIRDataset, classification_metrics
from spectraset import SpectraSet, SpectraSetConfig, config_dict, spectraset_loss


@dataclass(frozen=True)
class DatasetSpec:
    signal_size: int
    minimum_wavenumber: float
    maximum_wavenumber: float
    normalize: bool


DATASET_SPECS = {
    "fcg": DatasetSpec(1024, 400.0, 4000.0, True),
    "lec": DatasetSpec(1024, 400.0, 4000.0, True),
    "gas": DatasetSpec(1647, 552.0, 3844.0, False),
}
AUGMENTATION_PROBABILITY = 0.3
SNR_RANGE_DB = (2.0, 20.0)
ZERO_MASK_FRACTION_RANGE = (0.01, 0.15)
VERTICAL_SHIFT_FRACTION_RANGE = (0.01, 0.10)
PROJECT_ROOT = Path(__file__).resolve().parent


def project_output_path(path: Path) -> Path:
    """Resolve an artifact path and keep training outputs inside this project."""

    try:
        resolved = path.resolve()
        resolved.relative_to(PROJECT_ROOT)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"output path must stay under {PROJECT_ROOT}: {path}"
        ) from error
    return resolved


def finite_float(value: Any, name: str) -> float:
    """Convert a scalar-like value while preserving a useful training failure."""

    try:
        converted = float(value)
    except (RuntimeError, TypeError, ValueError) as error:
        raise RuntimeError(f"could not convert {name} to a scalar") from error
    if not math.isfinite(converted):
        raise RuntimeError(f"{name} is not finite: {converted}")
    return converted


def peak_gpu_memory_bytes(device: torch.device) -> int:
    """Return CUDA peak allocation with an actionable failure if CUDA resets."""

    if device.type != "cuda":
        return 0
    try:
        return int(torch.cuda.max_memory_allocated(device))
    except RuntimeError as error:
        raise RuntimeError("could not read CUDA peak-memory statistics") from error


def seed_everything(seed: int) -> None:
    """Seed model initialization, dropout, augmentation, and data loading."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(_: int) -> None:
    """Give each spawned data worker an independent deterministic seed."""

    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_model_config(dataset: str) -> SpectraSetConfig:
    """Return the fixed architecture configuration for one bundled data set."""

    if dataset == "gas":
        return SpectraSetConfig.gas1647()
    spec = DATASET_SPECS[dataset]
    return SpectraSetConfig(
        signal_size=spec.signal_size,
        minimum_wavenumber=spec.minimum_wavenumber,
        maximum_wavenumber=spec.maximum_wavenumber,
    )


def build_loader(
    split_dir: Path,
    spec: DatasetSpec,
    batch_size: int,
    workers: int,
    *,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    """Build a deterministic loader using the evaluator's preprocessing contract."""

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        ReleasedFTIRDataset(
            split_dir,
            signal_size=spec.signal_size,
            normalize=spec.normalize,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def _sample_uniform(
    shape: tuple[int, int, int],
    lower: float,
    upper: float,
    device: torch.device,
) -> Tensor:
    return torch.empty(shape, device=device).uniform_(lower, upper)


def augment_training_signals(signal: Tensor) -> Tensor:
    """Apply the documented training-only perturbations independently per sample.

    The archive specifies ranges and per-transform probability but not a historical
    implementation. This uses standard per-sample RMS-SNR noise, pointwise zero
    masking, and a symmetric additive vertical offset. Values are intentionally not
    clamped because the documented perturbations are distribution shifts.
    """

    if signal.ndim != 3 or signal.shape[1] != 1:
        raise ValueError(
            f"expected [batch, 1, points] signals, got {tuple(signal.shape)}"
        )

    batch_size = signal.shape[0]
    sample_shape = (batch_size, 1, 1)
    device = signal.device
    epsilon = torch.finfo(signal.dtype).eps
    augmented = signal.clone()
    maximum_intensity = signal.amax(dim=-1, keepdim=True).clamp_min(epsilon)

    apply_noise = torch.rand(sample_shape, device=device) < AUGMENTATION_PROBABILITY
    snr_db = _sample_uniform(sample_shape, *SNR_RANGE_DB, device)
    rms = signal.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(epsilon)
    noise_std = rms / torch.pow(10.0, snr_db / 20.0)
    augmented = augmented + torch.randn_like(augmented) * noise_std * apply_noise

    apply_mask = torch.rand(sample_shape, device=device) < AUGMENTATION_PROBABILITY
    mask_fraction = _sample_uniform(sample_shape, *ZERO_MASK_FRACTION_RANGE, device)
    point_mask = torch.rand_like(augmented) < mask_fraction
    augmented = augmented.masked_fill(apply_mask & point_mask, 0.0)

    apply_shift = torch.rand(sample_shape, device=device) < AUGMENTATION_PROBABILITY
    shift_fraction = _sample_uniform(
        sample_shape,
        *VERTICAL_SHIFT_FRACTION_RANGE,
        device,
    )
    shift_direction = torch.where(
        torch.rand(sample_shape, device=device) < 0.5,
        -torch.ones(sample_shape, device=device),
        torch.ones(sample_shape, device=device),
    )
    augmented = (
        augmented + apply_shift * shift_direction * shift_fraction * maximum_intensity
    )
    return augmented


def forward_logits(model: SpectraSet, signal: Tensor, device: torch.device) -> Tensor:
    """Run the model with CUDA bfloat16 AMP when it is available."""

    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        logits = model(signal)["logits"]
    if logits is None:
        raise RuntimeError("SpectraSet did not return logits")
    return logits


def train_epoch(
    model: SpectraSet,
    loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
) -> dict[str, float]:
    """Train one epoch and return sample-weighted loss components."""

    model.train()
    totals = {"total": 0.0, "bce": 0.0, "soft_macro_f1": 0.0, "set_margin": 0.0}
    samples = 0

    for batch in loader:
        signal = batch["signal"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        signal = augment_training_signals(signal)
        optimizer.zero_grad(set_to_none=True)
        logits = forward_logits(model, signal, device)
        losses = spectraset_loss(logits, labels)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        batch_size = labels.shape[0]
        samples += batch_size
        for name, loss in losses.items():
            totals[name] += finite_float(loss.detach(), f"{name} loss") * batch_size

    if samples == 0:
        raise RuntimeError("training loader produced no samples")
    return {name: value / samples for name, value in totals.items()}


@torch.inference_mode()
def evaluate_validation(
    model: SpectraSet,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate the final epoch on validation data without stochastic augmentation."""

    model.eval()
    all_labels: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    for batch in loader:
        signal = batch["signal"].to(device, non_blocking=True)
        logits = forward_logits(model, signal, device)
        all_labels.append(batch["label"].numpy().astype(np.uint8))
        all_probabilities.append(logits.float().sigmoid().cpu().numpy())

    if not all_labels:
        raise RuntimeError("validation loader produced no samples")
    return classification_metrics(
        np.concatenate(all_labels),
        np.concatenate(all_probabilities),
    )


def write_json(path: Path, value: Any) -> None:
    path = project_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def save_checkpoint(
    path: Path,
    model: SpectraSet,
    *,
    model_seed: int,
    run_seed: int,
    epoch: int,
    training_config: dict[str, Any],
) -> None:
    """Atomically save the evaluator-compatible final model checkpoint."""

    path = project_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": {
            name: parameter.detach().cpu()
            for name, parameter in model.state_dict().items()
        },
        "model_seed": model_seed,
        "run_seed": run_seed,
        "epoch": epoch,
        "model_config": config_dict(model.config),
        "training_config": training_config,
    }
    temporary_path = path.with_suffix(".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def training_config(args: argparse.Namespace, spec: DatasetSpec) -> dict[str, Any]:
    """Record every fixed recipe setting alongside a checkpoint."""

    return {
        "dataset": args.dataset,
        "train_dir": str(args.train_dir.resolve()),
        "valid_dir": str(args.valid_dir.resolve()),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "scheduler": {"name": "CosineAnnealingLR", "T_max": args.epochs},
        "gradient_clip_norm": 5.0,
        "amp": "bfloat16" if args.device.type == "cuda" else "disabled",
        "loss": "0.5*bce + 0.5*soft_macro_f1 + 0.1*set_margin",
        "set_margin_temperature": 0.25,
        "signal_size": spec.signal_size,
        "minimum_wavenumber": spec.minimum_wavenumber,
        "maximum_wavenumber": spec.maximum_wavenumber,
        "preprocessing": ("normalize_and_resize" if spec.normalize else "native_gas"),
        "augmentation": {
            "probability_each": AUGMENTATION_PROBABILITY,
            "noise_snr_db": list(SNR_RANGE_DB),
            "zero_mask_fraction": list(ZERO_MASK_FRACTION_RANGE),
            "vertical_shift_fraction": list(VERTICAL_SHIFT_FRACTION_RANGE),
            "implementation": "per-sample RMS noise, pointwise masks, signed additive offsets",
            "clamp_after_augmentation": False,
        },
    }


def train_one_seed(
    args: argparse.Namespace,
    spec: DatasetSpec,
    model_seed: int,
) -> dict[str, Any]:
    """Train one fixed seed and save its final epoch checkpoint."""

    run_seed = model_seed * 100
    run_dir = args.output_dir / f"seed{model_seed}"
    checkpoint_path = run_dir / "last.pt"
    if checkpoint_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"checkpoint already exists: {checkpoint_path}; use --overwrite to replace it"
        )

    seed_everything(run_seed)
    model = SpectraSet(build_model_config(args.dataset)).to(args.device)
    # Reset post-construction so data order, augmentation, and dropout do not
    # depend on the number of random draws used during parameter initialization.
    seed_everything(run_seed)
    train_loader = build_loader(
        args.train_dir,
        spec,
        args.batch_size,
        args.workers,
        shuffle=True,
        seed=run_seed,
        device=args.device,
    )
    valid_loader = build_loader(
        args.valid_dir,
        spec,
        args.batch_size,
        args.workers,
        shuffle=False,
        seed=run_seed + 1,
        device=args.device,
    )
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    if args.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(args.device)

    for epoch in range(1, args.epochs + 1):
        learning_rate = finite_float(optimizer.param_groups[0]["lr"], "learning rate")
        losses = train_epoch(model, train_loader, optimizer, args.device)
        scheduler.step()
        record = {"epoch": epoch, "learning_rate": learning_rate, **losses}
        history.append(record)
        if epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                json.dumps(
                    {
                        "status": "training",
                        "seed": model_seed,
                        "epoch": epoch,
                        "epochs": args.epochs,
                        "loss": round(losses["total"], 6),
                        "learning_rate": learning_rate,
                    }
                ),
                flush=True,
            )

    validation_metrics = evaluate_validation(model, valid_loader, args.device)
    elapsed_seconds = time.perf_counter() - started
    recipe = training_config(args, spec)
    save_checkpoint(
        checkpoint_path,
        model,
        model_seed=model_seed,
        run_seed=run_seed,
        epoch=args.epochs,
        training_config=recipe,
    )
    summary = {
        "status": "complete",
        "dataset": args.dataset,
        "model_seed": model_seed,
        "run_seed": run_seed,
        "epochs": args.epochs,
        "checkpoint": str(checkpoint_path.resolve()),
        "elapsed_seconds": elapsed_seconds,
        "max_gpu_memory_bytes": peak_gpu_memory_bytes(args.device),
        "final_train_losses": history[-1],
        "validation_metrics": validation_metrics,
        "training_config": recipe,
    }
    write_json(run_dir / "history.json", history)
    write_json(run_dir / "run_summary.json", summary)
    print(
        json.dumps(
            {
                "status": "complete",
                "seed": model_seed,
                "checkpoint": str(checkpoint_path),
                "validation_macro_f1": validation_metrics["macro_f1"],
                "validation_exact_match_accuracy": validation_metrics[
                    "exact_match_accuracy"
                ],
            }
        ),
        flush=True,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASET_SPECS), default="gas")
    parser.add_argument("--train-dir", type=Path)
    parser.add_argument("--valid-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[70, 71, 72])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_root = Path("datasets") / args.dataset
    args.train_dir = args.train_dir or dataset_root / "train"
    args.valid_dir = args.valid_dir or dataset_root / "valid"
    args.output_dir = args.output_dir or Path("checkpoints") / args.dataset
    args.device = torch.device(args.device)
    if args.device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.log_every < 1:
        parser.error(
            "epochs, batch-size, and log-every must be positive; workers cannot be negative"
        )
    if not args.seeds or any(seed < 0 for seed in args.seeds):
        parser.error("at least one nonnegative model seed is required")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    for split_dir in (args.train_dir, args.valid_dir):
        if not split_dir.is_dir():
            parser.error(f"split directory not found: {split_dir}")
    return args


def main() -> None:
    args = parse_args()
    spec = DATASET_SPECS[args.dataset]
    summaries = [train_one_seed(args, spec, seed) for seed in args.seeds]
    write_json(
        args.output_dir / "training_summary.json",
        {
            "status": "complete",
            "dataset": args.dataset,
            "seeds": args.seeds,
            "runs": summaries,
        },
    )


if __name__ == "__main__":
    main()
