# Third-party notices

## CorSeg-CineSAX

This project contains adapted model-loading, two-dimensional inference, and
anatomical post-processing logic from CorSeg-CineSAX.

- Upstream repository: <https://github.com/RunhaoXu2003/CorSeg>
- Upstream work: Runhao Xu, Siyuan Jiang, Yujia Zhai, and Yucheng Chen,
  *CorSeg-CineSAX: An Open-Source Deep Learning Framework for Fully Automatic
  Segmentation of Short-Axis Cine Cardiac MRI Across Multiple Cardiac Diseases*
- DOI: <https://doi.org/10.64898/2026.04.01.26349955>

### Changes made in this project

- Removed the upstream PyQt graphical interface.
- Refactored model loading and inference into a research command-line workflow.
- Integrated inference with strict conventional DICOM cine reconstruction.
- Applied the three anatomical post-processing steps in a locked sequence.
- Added ED phase detection, geometry-aware 3D reconstruction, radiomics,
  pixel-change QC, batch execution, hashing, and Rscore calculation.

### Licensing status

As checked on 2026-08-16, the upstream README states: "This project is licensed
under the MIT License." However, the upstream repository file list did not
contain a standalone `LICENSE` file, and the authors had not separately
confirmed the licensing request made by this project's author.

Accordingly:

- this repository preserves clear attribution and links to the upstream source;
- it does not claim ownership of CorSeg-derived code;
- it does not redistribute the CorSeg pretrained `.pth` checkpoint;
- `LICENSE` applies to Shuo Shi's original contributions, while third-party
  material remains subject to the upstream terms.

If the upstream authors later publish an authoritative license file or copyright
notice, preserve it verbatim in this repository and update this notice.

## Other dependencies

PyTorch, MONAI, PyRadiomics, SimpleITK, pydicom, SciPy, NumPy, pandas, and
openpyxl are external packages. They are not redistributed in this repository
and retain their own licenses.

