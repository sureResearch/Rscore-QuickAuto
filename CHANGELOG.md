# Changelog

## 1.1.0 — 2026-10-17

- Updated to the 21-feature model fitted in 895 development patients.
- Replaced coefficients, intercept, feature scaling and OOF score scaling.
- Renamed the model target to CMRPI_z and model ID to Rscore-CMRPI-v2.
- Updated the coefficient audit CSV, model checks and documentation.
- Kept radiomics preprocessing and segmentation unchanged.


## 1.0.1 — 2026-08-16

- Corrected an overly strict cross-slice absolute `TriggerTime` equality check
  introduced during the single-file consolidation.
- Retained the locked development behavior: frames are ordered by their unique
  temporal rank within each slice when phase counts are consistent.
- Added an auditable warning and run-summary fields when absolute temporal
  values differ across slices.
- No CorSeg inference, radiomics, or Rscore formula parameter changed.

## 1.0.0 — 2026-08-16

- Combined the complete executable workflow into one Python script.
- Embedded the locked 23-feature `Rscore-CMRRI-v1` formula.
- Defined `Rscore_z` as the primary exported outcome.
- Locked the `CineSAX-Rad-v1` radiomics configuration and SHA-256 signature.
- Added conventional DICOM geometry and temporal-alignment QC.
- Added safe-first checkpoint loading and reproducibility hashes.
- Added single-case, batch, resume, and environment-check subcommands.
- Added formula audit tests, privacy exclusions, citation metadata, and
  third-party attribution.
