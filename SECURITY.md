# Security and sensitive-data guidance

## Model checkpoints

PyTorch checkpoint files can contain pickle-based data. Download the CorSeg
checkpoint only from the official upstream project. Rscore QuickAuto first uses
`weights_only=True`. Use `--allow-unsafe-checkpoint` only after verifying the
checkpoint source.

## Patient information

Do not commit DICOM, NIfTI, patient-level spreadsheets, logs, or output folders.
Use de-identified research IDs for input folder names and `--patient-id`.

The included `.gitignore` blocks common medical-image and result formats, but it
does not replace institutional de-identification review.

## Clinical use

This is research software and is not intended for clinical diagnosis, treatment,
or autonomous decision-making.

