# SpectraSet

PyTorch implementation of **SpectraSet: A Wavenumber-Aware Multiscale Network
for Multilabel Functional-Group Identification from FTIR Spectra**.

## Installation

Requires Python 3.10+, PyTorch 2.2+, and [Git LFS](https://git-lfs.com/).

```bash
git lfs install
git clone https://github.com/adekang/SpectraSet.git
cd SpectraSet
git lfs pull
python -m pip install -e ".[eval]"
```

## Datasets

- **GAS1647**: included in `datasets/gas/`, with 4,683 training, 585 validation,
  and 586 test spectra from the NIST Chemistry WebBook.
- **FCG**: [dataset release](https://github.com/HoangTran223/Fcg-Former_FTIR_GasClassification).
- **LEC**: [Zenodo release](https://doi.org/10.5281/zenodo.17067390).

See [datasets/README.md](datasets/README.md) for the data format and preparation.

## Training

```bash
python train.py --dataset gas --output-dir checkpoints/gas --seeds 70 71 72
```

Checkpoints are saved to `checkpoints/gas/seed{70,71,72}/last.pt`.

## Evaluation

Evaluate the checkpoints produced by training:

```bash
python evaluate.py --dataset gas \
  --split-dir datasets/gas/test \
  --checkpoint checkpoints/gas/seed70/last.pt \
  --checkpoint checkpoints/gas/seed71/last.pt \
  --checkpoint checkpoints/gas/seed72/last.pt \
  --output-dir outputs/gas
```

Training and evaluation use CUDA by default. Add `--device cpu` to use the CPU.

## License

The software is released under the [MIT License](LICENSE). GAS1647 is curated
from the [NIST Chemistry WebBook](https://webbook.nist.gov/chemistry/); its data
are subject to the source's terms of use.
