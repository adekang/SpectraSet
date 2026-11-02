import torch
from spectraset import (
    SpectraSet,
    SpectraSetConfig,
    config_dict,
    load_spectraset_checkpoint,
    spectraset_loss,
)


def small_config(**overrides):
    values = {
        "signal_size": 64,
        "num_classes": 5,
        "channels": 32,
        "attention_heads": 4,
        "head_dim": 16,
        "head_hidden": 32,
        "dropout": 0.0,
        "region_tokens": 8,
        "full_resolution_kernels": (15, 7),
        "sharpening_kernel_size": 5,
        "ordered_position_rank": 4,
        "ordered_channel_rank": 8,
    }
    values.update(overrides)
    return SpectraSetConfig(**values)


def test_default_model_report_matches_released_architecture():
    model = SpectraSet()
    report = model.model_report()

    assert report["model_family"] == "spectraset"
    assert report["parameters"] == 9_490_053
    assert report["ordered_interface_parameters"] == 8_470_529
    assert report["ordered_token_count"] == 128
    assert report["ordered_position_rank"] == 128
    assert report["ordered_channel_rank"] == 256
    assert report["one_classifier"] is True
    assert report["independent_prediction_branches"] == 0
    assert report["probability_fusion"] is False


def test_gas1647_configuration_matches_released_architecture():
    model = SpectraSet(SpectraSetConfig.gas1647())
    report = model.model_report()

    assert report["parameters"] == 9_500_037
    assert report["ordered_interface_parameters"] == 8_480_513
    assert report["ordered_token_count"] == 206
    assert config_dict(model.config)["ordered_token_count"] == 206


def test_forward_loss_and_backward_on_small_model():
    torch.manual_seed(7)
    model = SpectraSet(small_config())
    spectrum = torch.rand(3, 1, 64)
    targets = torch.randint(0, 2, (3, 5), dtype=torch.float32)

    output = model(spectrum, return_diagnostics=True)
    logits = output["logits"]
    assert logits.shape == (3, 5)
    assert output["global_tokens"].shape == (3, 8, 32)
    assert output["region_evidence"].shape == (3, 5, 32)

    losses = spectraset_loss(logits, targets)
    losses["total"].backward()

    assert torch.isfinite(losses["total"])
    assert model.ordered_interface.gate.grad is not None
    assert model.sharpening_gate.grad is not None


def test_mean_collapse_control_accepts_no_permutation():
    model = SpectraSet(small_config(ordered_mode="mean_collapse"))
    output = model(torch.rand(2, 1, 64), return_diagnostics=True)
    tokens = output["ordered_interface_tokens"]

    assert torch.equal(tokens[:, :1, :].expand_as(tokens), tokens)


def test_checkpoint_loader_translates_experiment_module_names():
    torch.manual_seed(11)
    source = SpectraSet(small_config())
    experiment_state = {}
    for key, value in source.state_dict().items():
        translated = key
        translated = translated.replace("full_resolution_blocks.", "peak_blocks.")
        translated = translated.replace("regional_encoder.", "regional.")
        translated = translated.replace("sharpening_conv.", "sharpen_conv.")
        translated = translated.replace("ordered_interface.", "ordered_adapter.")
        if translated == "sharpening_gate":
            translated = "deriv_gate"
        experiment_state[translated] = value.clone()
    experiment_state["sg_kernel"] = torch.zeros(2, 1, 5)

    target = SpectraSet(small_config())
    receipt = load_spectraset_checkpoint(target, {"model": experiment_state})

    assert receipt["missing_keys"] == []
    assert receipt["unexpected_keys"] == []
    for key, value in source.state_dict().items():
        assert torch.equal(value, target.state_dict()[key])


def test_invalid_signal_shape_fails_closed():
    model = SpectraSet(small_config())
    bad_signal = torch.rand(2, 64)

    try:
        model(bad_signal)
    except ValueError as error:
        assert "expects [batch, 1, 64]" in str(error)
    else:
        raise AssertionError("invalid signal shape did not raise ValueError")
