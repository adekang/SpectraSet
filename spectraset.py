"""Self-contained PyTorch implementation of SpectraSet.

SpectraSet is a single-forward-graph, single-classifier network for multilabel
functional-group identification from one-dimensional FTIR spectra.  The model
keeps an unpooled, wavenumber-aligned token state until a gated, order-sensitive
decision interface while retaining a conventional pooled context path.

This module intentionally contains every model component and training loss.  It
depends only on PyTorch and does not import code from the research workspace.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

REGION_NAMES = ("fingerprint", "double_bond", "triple_bond", "x_h", "global")
REGION_WAVENUMBERS = (
    (400.0, 1500.0),
    (1500.0, 2000.0),
    (2000.0, 2500.0),
    (2500.0, 4000.0),
)
ORDERED_MODES = ("ordered", "fixed_permutation", "mean_collapse")


@dataclass(frozen=True)
class SpectraSetConfig:
    """Architecture configuration for SpectraSet.

    The defaults reproduce the selected 1024-point FCG/LEC architecture.
    ``ordered_token_count`` is inferred from the global path when omitted.
    """

    signal_size: int = 1024
    num_classes: int = 17
    channels: int = 256
    attention_heads: int = 4
    head_dim: int = 128
    head_hidden: int = 256
    dropout: float = 0.1
    region_tokens: int = 32
    minimum_wavenumber: float = 400.0
    maximum_wavenumber: float = 4000.0
    full_resolution_kernels: tuple[int, ...] = (63, 31)
    sharpening_channels: int = 2
    sharpening_kernel_size: int = 13
    maximum_sharpening_gate: float = 1.0
    sharpening_epsilon: float = 1e-6
    ordered_mode: str = "ordered"
    ordered_token_count: int | None = None
    ordered_position_rank: int = 128
    ordered_channel_rank: int = 256

    def __post_init__(self) -> None:
        if self.signal_size < 64:
            raise ValueError("signal_size must be at least 64")
        if self.num_classes < 1:
            raise ValueError("num_classes must be positive")
        if self.channels < 1 or self.channels % self.attention_heads:
            raise ValueError(
                "channels must be positive and divisible by attention_heads"
            )
        if self.head_dim < 1 or self.head_hidden < 1:
            raise ValueError("head dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if self.region_tokens < 2:
            raise ValueError("region_tokens must be at least two")
        if self.maximum_wavenumber <= self.minimum_wavenumber:
            raise ValueError("maximum_wavenumber must exceed minimum_wavenumber")
        if not self.full_resolution_kernels:
            raise ValueError("full_resolution_kernels cannot be empty")
        if any(
            kernel < 1 or kernel % 2 == 0 for kernel in self.full_resolution_kernels
        ):
            raise ValueError("full-resolution kernels must be positive odd integers")
        if self.sharpening_channels < 1:
            raise ValueError("sharpening_channels must be positive")
        if self.sharpening_kernel_size < 3 or self.sharpening_kernel_size % 2 == 0:
            raise ValueError("sharpening_kernel_size must be an odd integer >= 3")
        if not 0.0 < self.maximum_sharpening_gate <= 1.0:
            raise ValueError("maximum_sharpening_gate must lie in (0, 1]")
        if self.sharpening_epsilon <= 0.0:
            raise ValueError("sharpening_epsilon must be positive")
        if self.ordered_mode not in ORDERED_MODES:
            raise ValueError(f"unsupported ordered_mode: {self.ordered_mode}")

        expected_tokens = self.global_token_count
        if (
            self.ordered_token_count is not None
            and self.ordered_token_count != expected_tokens
        ):
            raise ValueError(
                "ordered_token_count must match the global-path output: "
                f"expected {expected_tokens}, got {self.ordered_token_count}"
            )
        if not 1 <= self.ordered_position_rank <= expected_tokens:
            raise ValueError(
                "ordered_position_rank must lie in [1, global_token_count]"
            )
        if not 1 <= self.ordered_channel_rank <= self.channels:
            raise ValueError("ordered_channel_rank must lie in [1, channels]")

    @property
    def global_token_count(self) -> int:
        """Token count produced by the stride-8, same-padded global path."""

        return (self.signal_size + 7) // 8

    @property
    def resolved_ordered_token_count(self) -> int:
        return self.global_token_count

    @classmethod
    def gas1647(cls, **overrides: Any) -> SpectraSetConfig:
        """Return the full-resolution GAS1647 configuration."""

        values: dict[str, Any] = {
            "signal_size": 1647,
            "minimum_wavenumber": 552.0,
            "maximum_wavenumber": 3844.0,
            "ordered_position_rank": 128,
            "ordered_channel_rank": 256,
        }
        values.update(overrides)
        return cls(**values)


def _normalization_groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class DepthwiseResidualBlock(nn.Module):
    """Full-resolution depthwise filtering followed by pointwise mixing."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        padding = dilation * (kernel_size - 1) // 2
        self.norm = nn.GroupNorm(_normalization_groups(channels), channels)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            padding_mode="replicate",
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(channels, channels, 1, bias=False)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: Tensor) -> Tensor:
        update = self.norm(features)
        update = self.depthwise(update)
        update = self.activation(update)
        update = self.pointwise(update)
        return features + self.dropout(update)


class SpectralDownsamplePath(nn.Module):
    """Depthwise-separable path that converts features into spectral tokens."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        stride: int,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        padding = dilation * (kernel_size - 1) // 2
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            padding_mode="replicate",
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(channels, channels, 1, bias=False)
        self.norm = nn.GroupNorm(_normalization_groups(channels), channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: Tensor) -> Tensor:
        features = self.depthwise(features)
        features = self.pointwise(features)
        features = self.norm(features)
        return self.dropout(self.activation(features))


class RegionalEvidenceEncoder(nn.Module):
    """Encode fixed spectral regions and route them to individual classes."""

    def __init__(
        self,
        channels: int,
        num_classes: int,
        signal_size: int,
        dropout: float,
        region_tokens: int,
        minimum_wavenumber: float,
        maximum_wavenumber: float,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.num_classes = int(num_classes)
        self.signal_size = int(signal_size)
        self.region_tokens = int(region_tokens)
        self.minimum_wavenumber = float(minimum_wavenumber)
        self.maximum_wavenumber = float(maximum_wavenumber)

        self.shared_block = DepthwiseResidualBlock(channels, 15, 1, dropout)
        self.global_block = DepthwiseResidualBlock(channels, 31, 2, dropout)
        self.scorer = nn.Conv1d(channels, 1, 1)
        self.global_scorer = nn.Conv1d(channels, 1, 1)
        nn.init.zeros_(self.scorer.weight)
        nn.init.zeros_(self.scorer.bias)
        nn.init.zeros_(self.global_scorer.weight)
        nn.init.zeros_(self.global_scorer.bias)
        self.output_norm = nn.LayerNorm(channels)
        self.region_embeddings = nn.Parameter(torch.zeros(len(REGION_NAMES), channels))
        self.router_logits = nn.Parameter(torch.zeros(num_classes, len(REGION_NAMES)))

    @property
    def region_bounds(self) -> tuple[tuple[int, int], ...]:
        span = self.maximum_wavenumber - self.minimum_wavenumber
        bounds = []
        for lower, upper in REGION_WAVENUMBERS:
            start = round((lower - self.minimum_wavenumber) / span * self.signal_size)
            stop = round((upper - self.minimum_wavenumber) / span * self.signal_size)
            start = max(0, min(self.signal_size, start))
            stop = max(0, min(self.signal_size, stop))
            if stop <= start:
                raise ValueError(
                    "a configured spectral region collapsed to zero samples"
                )
            bounds.append((start, stop))
        return tuple(bounds)

    @property
    def router_weights(self) -> Tensor:
        return self.router_logits.softmax(dim=-1)

    def _pool_bounded_region(
        self,
        features: Tensor,
        embedding: Tensor,
    ) -> tuple[Tensor, Tensor]:
        pooled = F.adaptive_avg_pool1d(features, self.region_tokens)
        pooled = pooled + embedding.view(1, -1, 1).to(dtype=pooled.dtype)
        pooled = self.shared_block(pooled)
        scores = self.scorer(pooled).squeeze(1)
        weights = scores.float().softmax(dim=-1).to(dtype=pooled.dtype)
        vector = torch.einsum("bct,bt->bc", pooled, weights)
        return self.output_norm(vector), weights

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        vectors = []
        attentions = []
        for index, (start, stop) in enumerate(self.region_bounds):
            vector, weights = self._pool_bounded_region(
                features[..., start:stop],
                self.region_embeddings[index],
            )
            vectors.append(vector)
            attentions.append(weights)

        global_features = F.adaptive_avg_pool1d(features, self.region_tokens)
        global_features = global_features + self.region_embeddings[-1].view(1, -1, 1)
        global_features = self.global_block(global_features)
        global_scores = self.global_scorer(global_features).squeeze(1)
        global_weights = global_scores.float().softmax(dim=-1).to(global_features.dtype)
        global_vector = torch.einsum("bct,bt->bc", global_features, global_weights)
        vectors.append(self.output_norm(global_vector))
        attentions.append(global_weights)

        region_vectors = torch.stack(vectors, dim=1)
        class_evidence = torch.einsum(
            "kr,brc->bkc",
            self.router_weights,
            region_vectors,
        )
        return class_evidence, torch.stack(attentions, dim=1), region_vectors


class OrderedSpectralStateInterface(nn.Module):
    """Rank-controlled, order-sensitive map from token state to context."""

    def __init__(
        self,
        token_count: int,
        channels: int,
        output_dim: int,
        position_rank: int,
        channel_rank: int,
    ) -> None:
        super().__init__()
        if min(token_count, channels, output_dim, position_rank, channel_rank) < 1:
            raise ValueError("interface dimensions and ranks must be positive")
        if position_rank > token_count:
            raise ValueError("position_rank cannot exceed token_count")
        if channel_rank > channels:
            raise ValueError("channel_rank cannot exceed channels")
        self.token_count = int(token_count)
        self.channels = int(channels)
        self.output_dim = int(output_dim)
        self.position_rank = int(position_rank)
        self.channel_rank = int(channel_rank)

        self.position_factor = nn.Parameter(torch.empty(token_count, position_rank))
        self.channel_factor = nn.Parameter(torch.empty(channels, channel_rank))
        self.core = nn.Parameter(torch.empty(position_rank, channel_rank, output_dim))
        self.gate = nn.Parameter(torch.zeros(()))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.position_factor)
        nn.init.orthogonal_(self.channel_factor)
        nn.init.xavier_uniform_(self.core)
        nn.init.zeros_(self.gate)

    def delta(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 3:
            raise ValueError("ordered interface expects [batch, tokens, channels]")
        if tokens.shape[1:] != (self.token_count, self.channels):
            raise ValueError(
                "ordered interface received an incompatible token shape: "
                f"{tuple(tokens.shape)}"
            )
        reduced = torch.einsum(
            "btc,ta,cr->bar",
            tokens,
            self.position_factor,
            self.channel_factor,
        )
        return torch.einsum("bar,ard->bd", reduced, self.core)

    def forward(self, tokens: Tensor) -> Tensor:
        return torch.tanh(self.gate) * self.delta(tokens)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class SpectraSet(nn.Module):
    """Single-model FTIR functional-group classifier with delayed collapse."""

    def __init__(self, config: SpectraSetConfig | None = None) -> None:
        super().__init__()
        self.config = config or SpectraSetConfig()
        cfg = self.config

        self.signal_size = cfg.signal_size
        self.num_classes = cfg.num_classes
        self.channels = cfg.channels
        self.attention_heads = cfg.attention_heads
        self.head_dim = cfg.head_dim
        self.head_hidden = cfg.head_hidden
        self.minimum_wavenumber = cfg.minimum_wavenumber
        self.maximum_wavenumber = cfg.maximum_wavenumber

        coordinate = torch.linspace(-1.0, 1.0, cfg.signal_size).view(1, 1, -1)
        wavenumbers = torch.linspace(
            cfg.minimum_wavenumber,
            cfg.maximum_wavenumber,
            cfg.signal_size,
        ).view(1, 1, -1)
        self.register_buffer("physical_coordinate", coordinate, persistent=True)
        self.register_buffer("wavenumbers_cm1", wavenumbers, persistent=True)

        groups = _normalization_groups(cfg.channels)
        self.stem = nn.Sequential(
            nn.Conv1d(2, cfg.channels, 1, bias=False),
            nn.GroupNorm(groups, cfg.channels),
            nn.GELU(),
            nn.Conv1d(cfg.channels, cfg.channels, 1, bias=False),
            nn.GroupNorm(groups, cfg.channels),
            nn.GELU(),
        )
        self.full_resolution_blocks = nn.ModuleList(
            DepthwiseResidualBlock(cfg.channels, kernel, 1, cfg.dropout)
            for kernel in cfg.full_resolution_kernels
        )
        self.regional_encoder = RegionalEvidenceEncoder(
            channels=cfg.channels,
            num_classes=cfg.num_classes,
            signal_size=cfg.signal_size,
            dropout=cfg.dropout,
            region_tokens=cfg.region_tokens,
            minimum_wavenumber=cfg.minimum_wavenumber,
            maximum_wavenumber=cfg.maximum_wavenumber,
        )
        self.local_path = SpectralDownsamplePath(cfg.channels, 11, 4, 1, cfg.dropout)
        self.mid_path = SpectralDownsamplePath(cfg.channels, 21, 8, 1, cfg.dropout)
        self.global_path = SpectralDownsamplePath(cfg.channels, 31, 8, 2, cfg.dropout)
        self.cross_attention = nn.MultiheadAttention(
            cfg.channels,
            cfg.attention_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(cfg.channels)
        self.context_projection = nn.Sequential(
            nn.Linear(cfg.channels, cfg.channels),
            nn.GELU(),
        )
        self.class_embeddings = nn.Parameter(torch.zeros(cfg.num_classes, cfg.channels))
        nn.init.normal_(self.class_embeddings, std=0.02)
        self.head_norm = nn.LayerNorm(cfg.channels)
        self.head_projection = nn.Sequential(
            nn.Linear(cfg.channels, cfg.head_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, cfg.head_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.classifier_weight = nn.Parameter(
            torch.empty(cfg.num_classes, cfg.head_dim)
        )
        self.classifier_bias = nn.Parameter(torch.zeros(cfg.num_classes))
        nn.init.kaiming_uniform_(self.classifier_weight, a=math.sqrt(5))

        self.sharpening_conv = nn.Conv1d(
            1,
            cfg.sharpening_channels,
            cfg.sharpening_kernel_size,
            padding=cfg.sharpening_kernel_size // 2,
            padding_mode="replicate",
            bias=False,
        )
        self.sharpening_gate = nn.Parameter(torch.zeros(cfg.sharpening_channels))

        old_input_projection = self.stem[0]
        if not isinstance(old_input_projection, nn.Conv1d):
            raise TypeError("unexpected input projection")
        widened_input_projection = nn.Conv1d(
            2 + cfg.sharpening_channels,
            cfg.channels,
            1,
            bias=old_input_projection.bias is not None,
        )
        with torch.no_grad():
            widened_input_projection.weight[:, :2, :].copy_(old_input_projection.weight)
            if old_input_projection.bias is not None:
                widened_input_projection.bias.copy_(old_input_projection.bias)
        self.stem[0] = widened_input_projection

        self.ordered_mode = cfg.ordered_mode
        self.ordered_interface = OrderedSpectralStateInterface(
            token_count=cfg.resolved_ordered_token_count,
            channels=cfg.channels,
            output_dim=cfg.channels,
            position_rank=cfg.ordered_position_rank,
            channel_rank=cfg.ordered_channel_rank,
        )

    @property
    def region_bounds(self) -> tuple[tuple[int, int], ...]:
        return self.regional_encoder.region_bounds

    @property
    def router_weights(self) -> Tensor:
        return self.regional_encoder.router_weights

    def _validate_signal(self, signal: Tensor) -> None:
        expected = (1, self.signal_size)
        if signal.ndim != 3 or signal.shape[1:] != expected:
            raise ValueError(
                f"SpectraSet expects [batch, 1, {self.signal_size}], "
                f"got {tuple(signal.shape)}"
            )

    def _gated_sharpening(self, signal: Tensor) -> Tensor:
        padding = self.config.sharpening_kernel_size // 2
        padded = F.pad(signal, (padding, padding), mode="replicate")
        sharpened = self.sharpening_conv(padded)[..., padding:-padding]
        rms = (
            sharpened.square()
            .mean(dim=-1, keepdim=True)
            .add(self.config.sharpening_epsilon)
            .sqrt()
        )
        sharpened = sharpened / rms
        gate = self.config.maximum_sharpening_gate * torch.tanh(self.sharpening_gate)
        return sharpened * gate.to(dtype=signal.dtype).view(1, -1, 1)

    def _prepare_input(self, signal: Tensor) -> Tensor:
        self._validate_signal(signal)
        coordinate = self.physical_coordinate.expand(signal.shape[0], -1, -1)
        sharpening = self._gated_sharpening(signal)
        return torch.cat(
            (signal, coordinate.to(dtype=signal.dtype), sharpening),
            dim=1,
        )

    def _select_ordered_tokens(
        self,
        global_tokens: Tensor,
        token_permutation: Tensor | None,
    ) -> Tensor:
        if self.ordered_mode == "ordered":
            if token_permutation is not None:
                raise ValueError("ordered mode does not accept token_permutation")
            return global_tokens
        if self.ordered_mode == "mean_collapse":
            if token_permutation is not None:
                raise ValueError("mean_collapse mode does not accept token_permutation")
            return global_tokens.mean(dim=1, keepdim=True).expand_as(global_tokens)
        if token_permutation is None:
            raise ValueError("fixed_permutation mode requires token_permutation")
        if token_permutation.shape != global_tokens.shape[:2]:
            raise ValueError(
                "token_permutation must have shape [batch, tokens], got "
                f"{tuple(token_permutation.shape)}"
            )

        permutation = token_permutation.to(
            device=global_tokens.device, dtype=torch.long
        )
        expected = torch.arange(
            global_tokens.shape[1],
            device=global_tokens.device,
            dtype=torch.long,
        )
        if not torch.equal(
            permutation.sort(dim=1).values, expected.expand_as(permutation)
        ):
            raise ValueError(
                "each token_permutation row must be a complete permutation"
            )
        gather_index = permutation.unsqueeze(-1).expand_as(global_tokens)
        return torch.gather(global_tokens, dim=1, index=gather_index)

    def forward(
        self,
        signal: Tensor,
        token_permutation: Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> dict[str, Tensor | None]:
        features = self.stem(self._prepare_input(signal))
        for block in self.full_resolution_blocks:
            features = block(features)

        region_evidence, region_attention, region_vectors = self.regional_encoder(
            features
        )

        local_tokens = self.local_path(features).transpose(1, 2)
        mid_tokens = self.mid_path(features).transpose(1, 2)
        global_tokens = self.global_path(features).transpose(1, 2)
        evidence_tokens = torch.cat((local_tokens, mid_tokens), dim=1)
        attended, cross_weights = self.cross_attention(
            global_tokens,
            evidence_tokens,
            evidence_tokens,
            need_weights=return_diagnostics,
            average_attn_weights=False,
        )
        global_tokens = self.cross_norm(global_tokens + attended)

        pooled_context = self.context_projection(
            global_tokens.mean(dim=1)
            + local_tokens.mean(dim=1)
            + mid_tokens.mean(dim=1)
        )
        interface_tokens = self._select_ordered_tokens(global_tokens, token_permutation)
        context_delta = self.ordered_interface.delta(interface_tokens)
        ordered_gate = torch.tanh(self.ordered_interface.gate)
        context = pooled_context + ordered_gate * context_delta

        class_features = context[:, None, :]
        class_features = class_features + region_evidence
        class_features = class_features + self.class_embeddings[None, :, :]
        projected = self.head_projection(self.head_norm(class_features))
        logits = torch.einsum("bkh,kh->bk", projected, self.classifier_weight)
        logits = logits + self.classifier_bias

        output: dict[str, Tensor | None] = {"logits": logits}
        if return_diagnostics:
            output.update(
                {
                    "full_resolution_features": features,
                    "region_evidence": region_evidence,
                    "region_vectors": region_vectors,
                    "region_attention": region_attention,
                    "router_weights": self.router_weights,
                    "local_tokens": local_tokens,
                    "mid_tokens": mid_tokens,
                    "global_tokens": global_tokens,
                    "cross_attention_weights": cross_weights,
                    "pooled_context": pooled_context,
                    "ordered_interface_tokens": interface_tokens,
                    "ordered_context_delta": context_delta,
                    "ordered_gate": ordered_gate,
                    "context": context,
                    "class_features": class_features,
                }
            )
        return output

    @torch.inference_mode()
    def predict_proba(self, signal: Tensor) -> Tensor:
        logits = self(signal)["logits"]
        if logits is None:
            raise RuntimeError("forward pass did not return logits")
        return logits.float().sigmoid()

    @torch.inference_mode()
    def predict(self, signal: Tensor, threshold: float = 0.5) -> Tensor:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must lie in [0, 1]")
        return self.predict_proba(signal) >= threshold

    def model_report(self) -> dict[str, Any]:
        parameters = sum(parameter.numel() for parameter in self.parameters())
        return {
            "model_family": "spectraset",
            "variant": "ordered_spectral_state",
            "signal_size": self.signal_size,
            "num_classes": self.num_classes,
            "channels": self.channels,
            "attention_heads": self.attention_heads,
            "head_dim": self.head_dim,
            "head_hidden": self.head_hidden,
            "full_resolution_kernels": list(self.config.full_resolution_kernels),
            "sharpening_channels": self.config.sharpening_channels,
            "ordered_mode": self.ordered_mode,
            "ordered_token_count": self.ordered_interface.token_count,
            "ordered_position_rank": self.ordered_interface.position_rank,
            "ordered_channel_rank": self.ordered_interface.channel_rank,
            "ordered_interface_parameters": self.ordered_interface.parameter_count(),
            "parameters": parameters,
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            ),
            "deployed_parameters": parameters,
            "one_shared_encoder": True,
            "one_classifier": True,
            "independent_prediction_branches": 0,
            "probability_fusion": False,
            "teacher_checkpoint": False,
            "dormant_parameters": 0,
        }


def soft_macro_f1_loss(
    logits: Tensor,
    targets: Tensor,
    epsilon: float = 1e-7,
) -> Tensor:
    """Differentiable macro-F1 loss used by the released training recipe."""

    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have identical shapes")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    probabilities = logits.float().sigmoid()
    targets = targets.float()
    true_positive = (probabilities * targets).sum(dim=0)
    false_positive = (probabilities * (1.0 - targets)).sum(dim=0)
    false_negative = ((1.0 - probabilities) * targets).sum(dim=0)
    soft_f1 = (2.0 * true_positive + epsilon) / (
        2.0 * true_positive + false_positive + false_negative + epsilon
    )
    return 1.0 - soft_f1.mean()


def smooth_set_margin_loss(
    logits: Tensor,
    targets: Tensor,
    temperature: float = 0.25,
) -> Tensor:
    """Penalize a smooth approximation to each sample's worst signed logit."""

    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have identical shapes")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    signed_violations = (1.0 - 2.0 * targets.float()) * logits.float()
    smooth_worst = temperature * torch.logsumexp(
        signed_violations / temperature,
        dim=1,
    )
    return F.softplus(smooth_worst).mean()


def spectraset_loss(
    logits: Tensor,
    targets: Tensor,
    bce_weight: float = 0.5,
    soft_f1_weight: float = 0.5,
    set_margin_weight: float = 0.1,
    soft_f1_epsilon: float = 1e-7,
    set_margin_temperature: float = 0.25,
) -> dict[str, Tensor]:
    """Return the total training loss and its three named components."""

    weights = (bce_weight, soft_f1_weight, set_margin_weight)
    if any(weight < 0.0 for weight in weights):
        raise ValueError("loss weights must be nonnegative")
    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have identical shapes")

    targets = targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    soft_f1 = soft_macro_f1_loss(logits, targets, soft_f1_epsilon)
    set_margin = smooth_set_margin_loss(logits, targets, set_margin_temperature)
    total = bce_weight * bce + soft_f1_weight * soft_f1 + set_margin_weight * set_margin
    return {
        "total": total,
        "bce": bce,
        "soft_macro_f1": soft_f1,
        "set_margin": set_margin,
    }


_EXACT_CHECKPOINT_KEY_RENAMES = {
    "deriv_gate": "sharpening_gate",
}
_CHECKPOINT_PREFIX_RENAMES = (
    ("peak_blocks.", "full_resolution_blocks."),
    ("regional.", "regional_encoder."),
    ("sharpen_conv.", "sharpening_conv."),
    ("ordered_adapter.", "ordered_interface."),
)
_IGNORED_CHECKPOINT_KEYS = {"sg_kernel"}


def _rename_checkpoint_key(key: str) -> str | None:
    key = key.removeprefix("module.")
    if key in _IGNORED_CHECKPOINT_KEYS:
        return None
    if key in _EXACT_CHECKPOINT_KEY_RENAMES:
        return _EXACT_CHECKPOINT_KEY_RENAMES[key]
    for old_prefix, new_prefix in _CHECKPOINT_PREFIX_RENAMES:
        if key.startswith(old_prefix):
            return new_prefix + key[len(old_prefix) :]
    return key


def _extract_state_dict(
    payload: Mapping[str, Any],
) -> tuple[Mapping[str, Tensor], dict[str, Any]]:
    for state_key in ("model", "state_dict"):
        candidate = payload.get(state_key)
        if isinstance(candidate, Mapping):
            metadata = {
                key: value for key, value in payload.items() if key != state_key
            }
            return candidate, metadata
    if payload and all(isinstance(value, Tensor) for value in payload.values()):
        return payload, {}
    raise ValueError("checkpoint does not contain a model state dictionary")


def load_spectraset_checkpoint(
    model: SpectraSet,
    checkpoint: str | Path | Mapping[str, Any],
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load either a public SpectraSet state dict or an experiment checkpoint.

    The loader translates historical internal module keys to the public names
    used in this file and ignores one unused fixed-filter buffer that was stored
    by the experiment harness.
    """

    if isinstance(checkpoint, (str, Path)):
        try:
            payload = torch.load(
                checkpoint,
                map_location=map_location,
                weights_only=False,
            )
        except TypeError:
            payload = torch.load(checkpoint, map_location=map_location)
    else:
        payload = checkpoint
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint must resolve to a mapping")

    state_dict, metadata = _extract_state_dict(payload)
    renamed_state = {}
    for key, value in state_dict.items():
        renamed_key = _rename_checkpoint_key(str(key))
        if renamed_key is not None:
            renamed_state[renamed_key] = value

    incompatible = model.load_state_dict(renamed_state, strict=strict)
    return {
        "metadata": metadata,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def build_spectraset(
    config: SpectraSetConfig | None = None,
    **config_overrides: Any,
) -> SpectraSet:
    """Build SpectraSet from a config or from keyword configuration values."""

    if config is not None and config_overrides:
        raise ValueError("pass either config or keyword overrides, not both")
    return SpectraSet(config or SpectraSetConfig(**config_overrides))


def config_dict(config: SpectraSetConfig) -> dict[str, Any]:
    """Return a JSON-serializable configuration dictionary."""

    values = asdict(config)
    values["full_resolution_kernels"] = list(config.full_resolution_kernels)
    values["ordered_token_count"] = config.resolved_ordered_token_count
    return values


__all__ = [
    "ORDERED_MODES",
    "REGION_NAMES",
    "REGION_WAVENUMBERS",
    "DepthwiseResidualBlock",
    "OrderedSpectralStateInterface",
    "RegionalEvidenceEncoder",
    "SpectraSet",
    "SpectraSetConfig",
    "SpectralDownsamplePath",
    "build_spectraset",
    "config_dict",
    "load_spectraset_checkpoint",
    "smooth_set_margin_loss",
    "soft_macro_f1_loss",
    "spectraset_loss",
]
