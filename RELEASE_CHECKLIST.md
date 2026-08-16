# Pre-publication checklist

Complete these checks before creating a public GitHub release:

- [ ] Confirm that no `.dcm`, `.nii`, `.nii.gz`, `.pth`, patient-level
      spreadsheet, log, or output file is staged by Git.
- [ ] Confirm that all folder names and screenshots use synthetic or
      de-identified research IDs.
- [ ] Download the CorSeg checkpoint only from the official upstream source.
- [ ] Run `python rscore_quickauto.py check` in the intended release environment.
- [ ] Run `python -m unittest discover -s tests -v`.
- [ ] Run at least one local end-to-end case and visually review the segmentation.
- [ ] Compare Python `Rscore_z` with the locked development implementation on
      prespecified local validation cases.
- [ ] Record the tested Python, PyTorch, MONAI, PyRadiomics, SimpleITK, and
      pydicom versions in the release notes.
- [ ] Recheck the upstream CorSeg repository for an authoritative `LICENSE`
      file or author response; update `licenses/CorSeg-NOTICE.txt` if available.
- [ ] Add the final article citation/DOI when the Rscore manuscript is published.

