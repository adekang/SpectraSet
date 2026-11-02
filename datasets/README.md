# FTIR Data Sets

This directory contains the GAS1647 data set used by the SpectraSet
experiments, with fixed `train`, `valid`, and `test` directories plus the
source `split_manifest.json`.

## Overview

| Data set | Train | Valid | Test | Stored points | Source |
|---|---:|---:|---:|---:|---|
| GAS | 4,683 | 585 | 586 | 1,647 | NIST Chemistry WebBook gas-phase IR records |

The FCG and LEC benchmark data sets are available from their original releases:

- **FCG**: [Fcg-Former](https://github.com/HoangTran223/Fcg-Former_FTIR_GasClassification) release
- **LEC**: LEC-Former Zenodo archive, DOI [10.5281/zenodo.17067390](https://doi.org/10.5281/zenodo.17067390) (CC BY 4.0)

To use FCG or LEC with SpectraSet, download the splits and place them under
`datasets/fcg/` or `datasets/lec/` following the same `{train,valid,test}/`
structure as `datasets/gas/`.

## File Format

Each sample is stored as a pair whose basename is the compound identifier:

```text
<sample>.npy   one-dimensional FTIR absorbance array
<sample>.txt   one line containing 17 space-separated binary labels
```

The output order is:

```text
alkane, methyl, alkene, alkyne, alcohols, amines, nitriles, aromatics,
alkyl halides, esters, ketones, aldehydes, carboxylic acids, ether,
acyl halides, amides, nitro
```

FCG and LEC spectra are independently min-max normalized and resized from
3,602 to 1,024 points with cubic interpolation by `../evaluate.py`. GAS spectra
are already min-max-normalized 1,647-point arrays spanning 552--3844 cm-1 at
2 cm-1 intervals and should remain at their native resolution.

## Important Cohort Boundaries

- FCG and LEC are official released benchmark splits, not independent external
  replications. LEC contains 8,152 CAS/spectrum/label identities also present
  in FCG.
- FCG retains known exact cross-split duplicates and label ambiguities from the
  released benchmark.
- GAS contains one documented train-test exact signal-and-label duplicate. Its
  fixed split is retained for benchmark compatibility.
- GAS is the Fcg-17 version used by SpectraSet. The separate 21-label DH-CNN
  data set is not included.

## Integrity Verification

`dataset_files.json` records deterministic aggregate hashes for every split.
Verify all paired files after copying, downloading, or checking out with Git
LFS:

```bash
python datasets/verify_datasets.py
```

The verifier checks expected split counts, unmatched files, sorted-name hashes,
content hashes, and total bytes.

## Git LFS

The repository-level `.gitattributes` routes all `datasets/**/*.npy` files
through Git LFS. Install and initialize Git LFS before staging the data:

```bash
git lfs install
git add .gitattributes datasets/
```

Do not stage the `.npy` files with ordinary Git. The bundled splits contain
more than 75,000 files and approximately 1.2 GB of working-tree data. For a
public release, a DOI-backed data archive with download scripts may be more
practical than storing every sample directly in the code repository.

## Evaluation Examples

FCG:

```bash
spectraset-evaluate \
  --split-dir datasets/fcg/test \
  --checkpoint /path/to/fcg_seed70.pt \
  --output-dir outputs/fcg
```

LEC uses the same command with `datasets/lec/test`. GAS uses
`datasets/gas/test` together with `--signal-size 1647`,
`--minimum-wavenumber 552`, and `--maximum-wavenumber 3844`.

## Redistribution

Bundling data locally does not establish redistribution rights. Before making
this directory public, retain the LEC attribution and license notice and verify
the FCG and NIST/GAS source terms. The software license for SpectraSet does not
automatically apply to third-party data.
