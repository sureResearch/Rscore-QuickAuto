# Rscore QuickAuto

Rscore QuickAuto is a research pipeline that converts a conventional short-axis
(SAX) cine cardiac MRI DICOM series into a locked radiomics score (`Rscore_z`).
It performs DICOM cine reconstruction, CorSeg-CineSAX segmentation, end-diastolic
phase selection, myocardial radiomics extraction, and calculation of the locked
23-feature Rscore.

> **Research use only.** This software is not a medical device and must not be
> used for clinical diagnosis, treatment planning, or unsupervised clinical
> decision-making.

## What is included

- One executable script: `rscore_quickauto.py`
- Locked model: `Rscore-CMRRI-v1`
- Primary program output: `Rscore_z`
- Model-development target: `CMRRI_z`
- Locked radiomics pipeline: `CineSAX-Rad-v1`
- Single-case, batch, resume, and environment-check commands
- Formula and pipeline self-checks
- No patient images, patient-level results, or pretrained CorSeg weights

The final Rscore formula is embedded in the Python script. The coefficient CSV
in `models/` is an audit copy and is checked against the code by automated tests.

## Important upstream dependency

This repository does **not** redistribute the CorSeg pretrained weight file.
Download the MedNeXt-L checkpoint from the official CorSeg project:

- Official repository: <https://github.com/RunhaoXu2003/CorSeg>
- Official download link currently listed by CorSeg:
  <https://pan.baidu.com/s/1BM9viKgzGoECovzjxbMgtg?pwd=4396>

Place the required checkpoint at:

```text
Rscore-QuickAuto/
└── weights/
    └── ModelWeight-CorSeg-CineSAX_MedNextL.pth
```

If the downloaded checkpoint has a different filename, either rename the
appropriate MedNeXt-L checkpoint or pass its location with `--weight`.

The CorSeg README states that the upstream project is MIT-licensed, but no
standalone upstream `LICENSE` file was visible when this release was prepared.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before redistribution.

## Installation

Python 3.9 or later is required. Python 3.10 is recommended.

### Windows example

```powershell
py -3.10 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

Install PyTorch using the command appropriate for the computer's CPU or CUDA
configuration from <https://pytorch.org/get-started/locally/>. Then install the
remaining dependencies:

```powershell
pip install -r requirements.txt
python rscore_quickauto.py check
```

Do not silently replace PyRadiomics 3.1.0 with a different version. Radiomics
implementations and feature definitions can change across releases.

## Input data organization

No example patient data are included. Use only appropriately de-identified
research data.

### One case

Each case folder must contain exactly one complete conventional, single-frame,
short-axis cine DICOM series. DICOM files may be nested inside subfolders, but
the case folder must not contain scout, long-axis, LGE, mapping, duplicated, or
other DICOM series.

```text
input/
└── HCM0001/                 # de-identified research ID
    ├── image0001.dcm
    ├── image0002.dcm
    └── ...
```

The program intentionally stops rather than guessing when it detects:

- more than one DICOM series;
- inconsistent matrix size, pixel spacing, or orientation;
- different phase counts across slices;
- irregular slice spacing;
- ambiguous temporal ordering;
- enhanced multi-frame DICOM.

### Batch input

For batch processing, every immediate child folder of `input/` is treated as
one research case:

```text
input/
├── HCM0001/
│   └── *.dcm
├── HCM0002/
│   └── *.dcm
└── HCM0003/
    └── *.dcm
```

Folder names are written into output tables and logs. They must be
de-identified research IDs, not names, medical-record numbers, or dates of birth.

## Usage

### Check the environment

```powershell
python rscore_quickauto.py check
```

### Run one case

```powershell
python rscore_quickauto.py run --input "input/HCM0001"
```

Specify a de-identified identifier and output folder explicitly if needed:

```powershell
python rscore_quickauto.py run `
  --input "input/HCM0001" `
  --patient-id "HCM0001" `
  --output "output/HCM0001" `
  --device cpu
```

The default device is `auto`. Available values are `auto`, `cpu`, and `cuda`.

The program first attempts the safer `torch.load(..., weights_only=True)` mode.
If the official checkpoint cannot be loaded in this mode, verify that it came
from the official CorSeg source before using:

```powershell
python rscore_quickauto.py run `
  --input "input/HCM0001" `
  --allow-unsafe-checkpoint
```

Never use that option with an untrusted `.pth` file.

### Run a batch

```powershell
python rscore_quickauto.py batch `
  --input-root "input" `
  --output-root "output" `
  --device cpu
```

Completed cases are skipped by default. Use `--rerun-success` only for a
documented reason.

### Recalculate only Rscore_z

If radiomics extraction completed but the scoring step did not, run:

```powershell
python rscore_quickauto.py resume --output "output/HCM0001"
```

This reads the existing `radiomics_features.csv` and does not rerun CorSeg.

## Main outputs

```text
output/HCM0001/
├── phase_cavity_summary.csv
├── ED_native_image.nii.gz
├── ED_native_LVmyocardium_mask.nii.gz
├── ED_radiomics_image_1x1x8_0to255.nii.gz
├── ED_radiomics_LVmyocardium_mask_1x1x8.nii.gz
├── radiomics_features.csv
├── radiomics_features.xlsx
├── radiomics_qc.json
├── Rscore_result.csv
├── Rscore_result.xlsx
├── Rscore_feature_parameters.csv
├── corseg_postprocess_stats.json
└── run_summary.json
```

`Rscore_result.xlsx` contains the primary `Rscore_z` output and the unstandardized
`raw_Rscore`. `run_summary.json` records model, pipeline, program, weight, and
package versions and SHA-256 identifiers for reproducibility.

## Locked model definition

The elastic-net model was developed in 610 A-center participants.

- Training target: `CMRRI_z`
- Primary exported outcome: `Rscore_z`
- Nonzero radiomics features: 23
- Selected alpha: 0.6
- Selected lambda: 0.0300311770330127
- Selection rule: global minimum inner-CV MSE
- A-center OOF raw Rscore mean: 0.000397984875998731
- A-center OOF raw Rscore SD: 0.629442418702308

For feature value `x_j`, training mean `mu_j`, training SD `sigma_j`, and locked
coefficient `beta_j`:

```text
raw_Rscore = intercept + sum(beta_j * (x_j - mu_j) / sigma_j)
Rscore_z   = (raw_Rscore - OOF_mean) / OOF_SD
```

The model-development target and exported score should not be conflated:
`CMRRI_z` is the training target, whereas `Rscore_z` is the standardized
radiomics prediction used as the primary program output.

## Locked radiomics method

`CineSAX-Rad-v1` uses:

- ED phase defined by maximum summed LV blood-pool area across SAX slices;
- DICOM geometry reconstruction from Image Position/Orientation and Pixel Spacing;
- 1 × 1 × 8 mm³ resampling;
- B-spline image interpolation and label-preserving mask interpolation;
- myocardial ROI P1–P99 clipping and linear rescaling to 0–255;
- fixed bin width 5;
- `force2D=True`, `force2Ddimension=0`;
- Original and LoG sigma 1, 2, and 3 mm image types;
- no wavelet features;
- PyRadiomics 3.1.0.

When absolute `TriggerTime` values differ slightly across slices but every
slice has the same phase count and a unique within-slice temporal ordering, the
program preserves the locked development behavior and aligns frames by their
within-slice temporal rank. This is recorded as a warning and in
`run_summary.json`; no frame is fabricated, discarded, or reassigned.

Any change to these settings requires a new pipeline ID and model validation.

## Validation and limitations

- Automated segmentation must be visually reviewed before scientific use.
- This release supports conventional single-frame-per-file SAX cine DICOM only.
- It does not automatically select a SAX series from a mixed study folder.
- ED is inferred from LV cavity area and may differ from scanner-reported timing.
- Generalizability depends on scanner, sequence, population, segmentation quality,
  and adherence to the locked preprocessing method.
- No clinical safety, diagnostic-performance, or regulatory claim is made.

Run formula tests without patient data:

```powershell
python -m unittest discover -s tests -v
```

## Citation

Use `CITATION.cff` to cite this software. Also cite the upstream CorSeg work:

Xu R, Jiang S, Zhai Y, Chen Y. *CorSeg-CineSAX: An Open-Source Deep Learning
Framework for Fully Automatic Segmentation of Short-Axis Cine Cardiac MRI
Across Multiple Cardiac Diseases*. 2026. DOI:
<https://doi.org/10.64898/2026.04.01.26349955>.

## License

Original contributions by Shuo Shi are released under the MIT License. Adapted
CorSeg portions and model weights remain subject to the upstream project's terms.
See `LICENSE`, `THIRD_PARTY_NOTICES.md`, and `licenses/CorSeg-NOTICE.txt`.
