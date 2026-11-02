"""Minimal SpectraSet inference example."""

import torch
from spectraset import SpectraSet


def main() -> None:
    torch.manual_seed(0)
    model = SpectraSet().eval()
    spectrum = torch.rand(1, 1, 1024)

    with torch.inference_mode():
        output = model(spectrum)
        probabilities = output["logits"].sigmoid()
        predictions = probabilities >= 0.5

    print(model.model_report())
    print("logits shape:", tuple(output["logits"].shape))
    print("positive labels:", int(predictions.sum()))


if __name__ == "__main__":
    main()
