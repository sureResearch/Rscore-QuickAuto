# Locked model audit files

`Final_model_coefficients.csv` is a human-readable audit copy of the locked
23-feature `Rscore-CMRRI-v1` model.

The executable model parameters are embedded in `rscore_quickauto.py`. Automated
tests require the CSV and the embedded constants to agree exactly, preventing
silent divergence between the published formula and the running program.

No patient-level development data or identifiers are included.

