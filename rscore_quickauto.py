#!/usr/bin/env python3
"""Rscore QuickAuto: conventional SAX cine DICOM to locked Rscore_z.

Copyright (c) 2026 Shuo Shi, Beijing Anzhen Hospital.

The original Rscore integration, DICOM reconstruction, radiomics pipeline,
quality-control logic, and command-line workflow are released under the MIT
License in this repository.

The CorSeg model-loading, inference, and anatomical post-processing portions
are adapted from CorSeg-CineSAX by Runhao Xu et al. The upstream README states
that CorSeg is MIT-licensed, but no standalone upstream LICENSE file was present
when this release was prepared. See THIRD_PARTY_NOTICES.md and
licenses/CorSeg-NOTICE.txt before redistribution.

Research software only. Not for clinical diagnosis or treatment decisions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


PROGRAM_NAME = "Rscore QuickAuto"
PROGRAM_VERSION = "1.0.1"
MODEL_ID = "Rscore-CMRRI-v1"
MODEL_TARGET = "CMRRI_z"
PRIMARY_OUTCOME = "Rscore_z"
PIPELINE_ID = "CineSAX-Rad-v1"

PROGRAM_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHT = (
    PROGRAM_DIR
    / "weights"
    / "ModelWeight-CorSeg-CineSAX_MedNextL.pth"
)

# Locked model fitted in the A-center development cohort (N=610).
# Each entry is: feature_name: (coefficient, training_mean, training_SD).
INTERCEPT = -6.12521947480383e-17
OOF_RSCORE_MEAN = 0.000397984875998731
OOF_RSCORE_SD = 0.629442418702308
SELECTED_ALPHA = 0.6
SELECTED_LAMBDA = 0.0300311770330127
TUNING_RULE = "global_minimum_MSE"

FEATURE_SPECS: Dict[str, Tuple[float, float, float]] = {
    "original_shape_Compactness2": (
        0.196518898648995,
        0.0371269637728555,
        0.0149293261944881,
    ),
    "original_shape_Maximum2DDiameterSlice": (
        0.231545534510617,
        80.1961024578862,
        8.38394566419164,
    ),
    "original_glcm_Autocorrelation": (
        -0.00173108337031287,
        588.051670909677,
        172.493532350894,
    ),
    "original_glcm_ClusterShade": (
        0.0165335248649646,
        2299.42143471462,
        3031.07728443994,
    ),
    "original_glcm_Correlation": (
        0.115897353697732,
        0.8853173021163,
        0.0457880370122781,
    ),
    "original_glcm_InverseVariance": (
        -0.126986163165913,
        0.359628625082663,
        0.0354709444892982,
    ),
    "original_glszm_GrayLevelNonUniformity": (
        0.00199862384032442,
        220.122820116924,
        96.4713924944949,
    ),
    "original_glszm_LargeAreaLowGrayLevelEmphasis": (
        -0.00185502219811432,
        0.778035895826026,
        1.11521610951685,
    ),
    "log-sigma-1-0-mm-3D_firstorder_90Percentile": (
        -0.0776442368715164,
        18.8932198069526,
        4.86659223629072,
    ),
    "log-sigma-1-0-mm-3D_firstorder_Kurtosis": (
        0.0373628347790785,
        6.53309639175009,
        1.7199921759456,
    ),
    "log-sigma-1-0-mm-3D_glcm_InverseVariance": (
        0.0161070665136766,
        0.413699794550222,
        0.0245435334204463,
    ),
    "log-sigma-1-0-mm-3D_gldm_DependenceEntropy": (
        -0.0363688854893148,
        5.79292121347163,
        0.145484231431054,
    ),
    "log-sigma-2-0-mm-3D_firstorder_Skewness": (
        0.0475034811846754,
        0.249834364092006,
        0.33480661329703,
    ),
    "log-sigma-2-0-mm-3D_glszm_GrayLevelNonUniformityNormalized": (
        -0.12810179877559,
        0.0630799943965918,
        0.0059800039912888,
    ),
    "log-sigma-2-0-mm-3D_gldm_DependenceEntropy": (
        0.0806471919720448,
        6.52604609573996,
        0.107803448935442,
    ),
    "log-sigma-2-0-mm-3D_gldm_DependenceVariance": (
        0.0555215091093061,
        3.37529031402775,
        0.890760288802934,
    ),
    "log-sigma-3-0-mm-3D_firstorder_InterquartileRange": (
        -0.0349252210258058,
        30.28571265029,
        5.22623758506106,
    ),
    "log-sigma-3-0-mm-3D_firstorder_Skewness": (
        0.0340632267551152,
        0.0233941227783107,
        0.298259491287254,
    ),
    "log-sigma-3-0-mm-3D_firstorder_Kurtosis": (
        0.0498230336354101,
        3.67214369973822,
        0.65736045649235,
    ),
    "log-sigma-3-0-mm-3D_glszm_LargeAreaHighGrayLevelEmphasis": (
        -0.00968484373085998,
        35304.3545997395,
        33475.7589365849,
    ),
    "log-sigma-3-0-mm-3D_glszm_ZoneEntropy": (
        0.0806283519633917,
        6.79707555592151,
        0.238868837306891,
    ),
    "log-sigma-3-0-mm-3D_gldm_DependenceEntropy": (
        -0.139343439126078,
        6.91293094079228,
        0.128894364168613,
    ),
    "log-sigma-3-0-mm-3D_ngtdm_Contrast": (
        -0.0880349607233836,
        0.0202699167832803,
        0.00923356357248006,
    ),
}

TARGET_SPACING = [1.0, 1.0, 8.0]
LOWER_PERCENTILE = 1.0
UPPER_PERCENTILE = 99.0
INTENSITY_MIN = 0.0
INTENSITY_MAX = 255.0
BIN_WIDTH = 5.0
LOG_SIGMAS = [1.0, 2.0, 3.0]
PAD_DISTANCE = 5
LABEL = 1
EXPECTED_FEATURE_COUNT = 393
PYRADIOMICS_VERSION = "3.1.0"

PIPELINE_CONFIG = {
    "pipeline_id": PIPELINE_ID,
    "target_spacing_mm": TARGET_SPACING,
    "intensity_percentiles_roi": [LOWER_PERCENTILE, UPPER_PERCENTILE],
    "intensity_output_range": [INTENSITY_MIN, INTENSITY_MAX],
    "bin_width": BIN_WIDTH,
    "force2D": True,
    "force2Ddimension": 0,
    "image_types": {"Original": {}, "LoG": {"sigma_mm": LOG_SIGMAS}},
    "feature_classes": [
        "shape",
        "firstorder",
        "glcm",
        "glrlm",
        "glszm",
        "gldm",
        "ngtdm",
    ],
    "wavelet": False,
    "label": LABEL,
    "pad_distance": PAD_DISTANCE,
    "pyradiomics_version": PYRADIOMICS_VERSION,
}

# Fixed signature of the radiomics configuration used to fit the model.
LOCKED_PIPELINE_SHA256 = (
    "6ffc276087ef241226cf7236cd29d693f45c8b92e60305c5d27e89a514c0d37d"
)

_RUNTIME_LOADED = False


def _load_runtime() -> None:
    """Load heavy scientific dependencies only when image processing is used."""
    global _RUNTIME_LOADED
    if _RUNTIME_LOADED:
        return
    try:
        import numpy as _np
        import pandas as _pd
        import pydicom as _pydicom
        import radiomics as _radiomics
        import SimpleITK as _sitk
        import torch as _torch
        import torch.nn.functional as _F
        from monai.networks.nets.mednext import create_mednext as _create_mednext
        from radiomics import featureextractor as _featureextractor
        from radiomics import imageoperations as _imageoperations
        from scipy import ndimage as _sp_ndimage
    except Exception as exc:
        raise RuntimeError(
            "Scientific dependencies are unavailable. Run "
            "`python rscore_quickauto.py check` and install requirements. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    globals().update({
        "np": _np,
        "pd": _pd,
        "pydicom": _pydicom,
        "radiomics": _radiomics,
        "sitk": _sitk,
        "torch": _torch,
        "F": _F,
        "create_mednext": _create_mednext,
        "featureextractor": _featureextractor,
        "imageoperations": _imageoperations,
        "sp_ndimage": _sp_ndimage,
    })
    _RUNTIME_LOADED = True


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def pipeline_sha256() -> str:
    return _canonical_sha256(PIPELINE_CONFIG)


def model_sha256() -> str:
    return _canonical_sha256({
        "model_id": MODEL_ID,
        "model_target": MODEL_TARGET,
        "primary_outcome": PRIMARY_OUTCOME,
        "intercept": INTERCEPT,
        "oof_mean": OOF_RSCORE_MEAN,
        "oof_sd": OOF_RSCORE_SD,
        "alpha": SELECTED_ALPHA,
        "lambda": SELECTED_LAMBDA,
        "tuning_rule": TUNING_RULE,
        "features": FEATURE_SPECS,
    })


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def validate_locked_model() -> None:
    if len(FEATURE_SPECS) != 23:
        raise RuntimeError(
            f"Locked model must contain 23 features; found {len(FEATURE_SPECS)}."
        )
    for name, (_, mean, sd) in FEATURE_SPECS.items():
        if not name or not math.isfinite(mean) or not math.isfinite(sd) or sd <= 0:
            raise RuntimeError(f"Invalid locked model parameters for {name!r}.")
    if not math.isfinite(OOF_RSCORE_SD) or OOF_RSCORE_SD <= 0:
        raise RuntimeError("OOF Rscore SD must be positive.")
    actual = pipeline_sha256()
    if actual != LOCKED_PIPELINE_SHA256:
        raise RuntimeError(
            "Radiomics configuration differs from the configuration used to "
            f"fit {MODEL_ID}. Expected {LOCKED_PIPELINE_SHA256}; got {actual}. "
            "Do not calculate Rscore until the mismatch is resolved."
        )


def calculate_rscore(feature_values: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate the locked raw Rscore and its OOF-referenced Rscore_z."""
    validate_locked_model()
    missing = [name for name in FEATURE_SPECS if name not in feature_values]
    if missing:
        raise RuntimeError(
            "Missing locked-model radiomics features:\n"
            + "\n".join(f"  - {name}" for name in missing)
        )

    raw = float(INTERCEPT)
    details = []
    for name, (coefficient, mean, sd) in FEATURE_SPECS.items():
        try:
            value = float(feature_values[name])
        except Exception as exc:
            raise RuntimeError(f"Feature is not numeric: {name}") from exc
        if not math.isfinite(value):
            raise RuntimeError(f"Feature is non-finite: {name}")
        z_value = (value - mean) / sd
        contribution = coefficient * z_value
        raw += contribution
        details.append({
            "Feature": name,
            "Coefficient": coefficient,
            "Training_mean": mean,
            "Training_SD": sd,
            "Current_value": value,
            "Current_Z": z_value,
            "Contribution_to_raw_Rscore": contribution,
        })

    rscore_z = (raw - OOF_RSCORE_MEAN) / OOF_RSCORE_SD
    return {
        "raw_Rscore": float(raw),
        "Rscore_z": float(rscore_z),
        "details": details,
    }


@dataclass
class FrameMeta:
    path: Path
    series_uid: str
    series_number: str
    series_description: str
    rows: int
    cols: int
    pixel_spacing_row: float
    pixel_spacing_col: float
    image_position: Any
    row_direction: Any
    col_direction: Any
    normal_direction: Any
    slice_coordinate: float
    trigger_time: Optional[float]
    temporal_position: Optional[int]
    instance_number: Optional[int]
    cardiac_number_of_images: Optional[int]


@dataclass
class CineGrid:
    slices: List[List[FrameMeta]]
    slice_coordinates: List[float]
    phase_count: int
    row_direction: Any
    col_direction: Any
    normal_direction: Any
    pixel_spacing_row: float
    pixel_spacing_col: float
    rows: int
    cols: int
    series_uid: str
    series_number: str
    series_description: str
    temporal_ordering_basis: List[str]
    absolute_phase_values_aligned: Optional[bool]

    @property
    def slice_count(self) -> int:
        return len(self.slices)


def _is_dicom(path: Path) -> bool:
    try:
        return path.suffix.lower() == ".dcm" or pydicom.misc.is_dicom(str(path))
    except Exception:
        return False


def _collect_dicoms(root: Path) -> List[Path]:
    if root.is_file():
        return [root]
    files = []
    for base, _, names in os.walk(root):
        for name in names:
            path = Path(base) / name
            if _is_dicom(path):
                files.append(path)
    return sorted(files)


def _to_float(value: Any, default: Any = None) -> Any:
    try:
        return float(value)
    except Exception:
        return default


def _to_int(value: Any, default: Any = None) -> Any:
    try:
        return int(value)
    except Exception:
        return default


def _read_header(path: Path) -> FrameMeta:
    ds = pydicom.dcmread(str(path), stop_before_pixels=True)
    if not hasattr(ds, "ImageOrientationPatient"):
        raise ValueError(f"Missing ImageOrientationPatient: {path}")
    if not hasattr(ds, "ImagePositionPatient"):
        raise ValueError(f"Missing ImagePositionPatient: {path}")
    iop = np.asarray([float(x) for x in ds.ImageOrientationPatient], dtype=float)
    if iop.size != 6:
        raise ValueError(f"Invalid ImageOrientationPatient: {path}")
    row_dir = iop[:3]
    col_dir = iop[3:]
    row_dir = row_dir / np.linalg.norm(row_dir)
    col_dir = col_dir / np.linalg.norm(col_dir)
    normal = np.cross(row_dir, col_dir)
    normal = normal / np.linalg.norm(normal)
    ipp = np.asarray([float(x) for x in ds.ImagePositionPatient], dtype=float)
    if hasattr(ds, "PixelSpacing"):
        spacing = [float(x) for x in ds.PixelSpacing]
    elif hasattr(ds, "ImagerPixelSpacing"):
        spacing = [float(x) for x in ds.ImagerPixelSpacing]
    else:
        raise ValueError(f"Missing PixelSpacing: {path}")
    return FrameMeta(
        path=path,
        series_uid=str(getattr(ds, "SeriesInstanceUID", "")),
        series_number=str(getattr(ds, "SeriesNumber", "")),
        series_description=str(getattr(ds, "SeriesDescription", "")),
        rows=int(getattr(ds, "Rows")),
        cols=int(getattr(ds, "Columns")),
        pixel_spacing_row=float(spacing[0]),
        pixel_spacing_col=float(spacing[1]),
        image_position=ipp,
        row_direction=row_dir,
        col_direction=col_dir,
        normal_direction=normal,
        slice_coordinate=float(np.dot(ipp, normal)),
        trigger_time=_to_float(getattr(ds, "TriggerTime", None)),
        temporal_position=_to_int(
            getattr(ds, "TemporalPositionIdentifier", None)
        ),
        instance_number=_to_int(getattr(ds, "InstanceNumber", None)),
        cardiac_number_of_images=_to_int(
            getattr(ds, "CardiacNumberOfImages", None)
        ),
    )


def _order_slice_phases(
    items: Sequence[FrameMeta],
) -> Tuple[List[FrameMeta], str, Optional[List[float]]]:
    triggers = [x.trigger_time for x in items]
    temporal = [x.temporal_position for x in items]
    instances = [x.instance_number for x in items]
    if all(x is not None for x in triggers) and len(set(triggers)) == len(items):
        ordered = sorted(items, key=lambda x: (x.trigger_time, x.instance_number or 10**9))
        return ordered, "TriggerTime", [float(x.trigger_time) for x in ordered]
    if all(x is not None for x in temporal) and len(set(temporal)) == len(items):
        ordered = sorted(items, key=lambda x: (x.temporal_position, x.instance_number or 10**9))
        return ordered, "TemporalPositionIdentifier", [float(x.temporal_position) for x in ordered]
    if all(x is not None for x in instances) and len(set(instances)) == len(items):
        ordered = sorted(items, key=lambda x: x.instance_number)
        return ordered, "InstanceNumber", None
    raise RuntimeError(
        "Unable to establish a unique temporal ordering within at least one slice."
    )


def build_cine_grid(input_path: Path) -> CineGrid:
    paths = _collect_dicoms(Path(input_path))
    if not paths:
        raise RuntimeError("No DICOM files were found.")
    records = [_read_header(path) for path in paths]
    series_uids = sorted(set(x.series_uid for x in records))
    if len(series_uids) != 1:
        details = []
        for uid in series_uids:
            subset = [x for x in records if x.series_uid == uid]
            details.append(
                f"UID={uid}; SeriesNumber={subset[0].series_number}; "
                f"Description={subset[0].series_description}; Files={len(subset)}"
            )
        raise RuntimeError(
            f"Expected exactly one SAX cine series, found {len(series_uids)}. "
            "No automatic series selection was performed. Candidates: "
            + " | ".join(details)
        )

    ref = records[0]
    for item in records:
        if item.rows != ref.rows or item.cols != ref.cols:
            raise RuntimeError(
                "DICOM matrix size is inconsistent: "
                f"reference={ref.rows}x{ref.cols}; "
                f"offending={item.rows}x{item.cols}; file={item.path}"
            )
        if not np.isclose(item.pixel_spacing_row, ref.pixel_spacing_row, atol=1e-4):
            raise RuntimeError("Row PixelSpacing is inconsistent across the series.")
        if not np.isclose(item.pixel_spacing_col, ref.pixel_spacing_col, atol=1e-4):
            raise RuntimeError("Column PixelSpacing is inconsistent across the series.")
        row_dot = float(np.dot(item.row_direction, ref.row_direction))
        col_dot = float(np.dot(item.col_direction, ref.col_direction))
        if row_dot < 0.999 or col_dot < 0.999:
            raise RuntimeError(
                "ImageOrientationPatient is inconsistent across the series: "
                f"row_dot={row_dot:.6f}; col_dot={col_dot:.6f}; file={item.path}"
            )

    for item in records:
        item.normal_direction = ref.normal_direction
        item.slice_coordinate = float(
            np.dot(item.image_position, ref.normal_direction)
        )

    centers: List[float] = []
    for coordinate in sorted(x.slice_coordinate for x in records):
        if not centers or abs(coordinate - centers[-1]) > 0.2:
            centers.append(coordinate)
    grouped: List[List[FrameMeta]] = [[] for _ in centers]
    for item in records:
        index = int(
            np.argmin(np.abs(np.asarray(centers) - item.slice_coordinate))
        )
        if abs(item.slice_coordinate - centers[index]) > 0.2:
            raise RuntimeError("Failed to assign a DICOM frame to a slice.")
        grouped[index].append(item)

    frame_counts = [len(x) for x in grouped]
    if len(set(frame_counts)) != 1:
        raise RuntimeError(f"Frame counts differ across slices: {frame_counts}")
    phase_count = frame_counts[0]
    if phase_count < 2:
        raise RuntimeError("The series does not appear to be a cine acquisition.")

    cardiac_counts = sorted(set(
        x.cardiac_number_of_images
        for x in records
        if x.cardiac_number_of_images is not None
    ))
    if cardiac_counts and any(x != phase_count for x in cardiac_counts):
        raise RuntimeError(
            f"Resolved phase count {phase_count} does not match "
            f"CardiacNumberOfImages {cardiac_counts}."
        )

    ordered_slices: List[List[FrameMeta]] = []
    ordering_basis: List[str] = []
    phase_keys: List[Optional[List[float]]] = []
    for items in grouped:
        ordered, basis, keys = _order_slice_phases(items)
        ordered_slices.append(ordered)
        ordering_basis.append(basis)
        phase_keys.append(keys)
    absolute_phase_values_aligned: Optional[bool] = None
    if (
        len(set(ordering_basis)) == 1
        and ordering_basis[0] in {
            "TriggerTime",
            "TemporalPositionIdentifier",
        }
    ):
        reference_keys = phase_keys[0]
        assert reference_keys is not None
        tolerance = 2.0 if ordering_basis[0] == "TriggerTime" else 0.0
        absolute_phase_values_aligned = True
        for keys in phase_keys[1:]:
            assert keys is not None
            if not np.allclose(keys, reference_keys, atol=tolerance, rtol=0.0):
                absolute_phase_values_aligned = False
                break
        if not absolute_phase_values_aligned:
            print(
                "[Warning] Absolute temporal values differ across slices. "
                "Frames remain ordered by within-slice temporal rank, matching "
                "the locked development workflow; no frames were discarded "
                "or reassigned."
            )
    elif len(set(ordering_basis)) != 1:
        print(
            "[Warning] DICOM temporal ordering tags differ across slices. "
            "Each slice was ordered using its available unique temporal tag, "
            "matching the locked development workflow."
        )

    return CineGrid(
        slices=ordered_slices,
        slice_coordinates=centers,
        phase_count=phase_count,
        row_direction=ref.row_direction,
        col_direction=ref.col_direction,
        normal_direction=ref.normal_direction,
        pixel_spacing_row=ref.pixel_spacing_row,
        pixel_spacing_col=ref.pixel_spacing_col,
        rows=ref.rows,
        cols=ref.cols,
        series_uid=ref.series_uid,
        series_number=ref.series_number,
        series_description=ref.series_description,
        temporal_ordering_basis=ordering_basis,
        absolute_phase_values_aligned=absolute_phase_values_aligned,
    )


def load_frame_pixels(frame: FrameMeta) -> Any:
    reader = sitk.ImageFileReader()
    reader.SetImageIO("GDCMImageIO")
    reader.SetFileName(str(frame.path))
    try:
        image = reader.Execute()
    except Exception as exc:
        ds = pydicom.dcmread(str(frame.path), stop_before_pixels=True)
        transfer_syntax = getattr(
            getattr(ds, "file_meta", None),
            "TransferSyntaxUID",
            None,
        )
        raise RuntimeError(
            "SimpleITK GDCMImageIO failed to decode DICOM pixel data. "
            f"File={frame.path}; TransferSyntaxUID={transfer_syntax}; "
            f"error={type(exc).__name__}: {exc}"
        ) from exc
    data = sitk.GetArrayFromImage(image)
    if data.ndim == 3 and data.shape[0] == 1:
        data = data[0]
    elif data.ndim != 2:
        raise RuntimeError(
            "Only conventional single-frame 2D DICOM is supported: "
            f"{frame.path}; decoded shape={data.shape}"
        )
    data = np.asarray(data, dtype=np.float32)
    if data.shape != (frame.rows, frame.cols):
        raise RuntimeError(
            "Decoded matrix does not match header metadata: "
            f"decoded={data.shape}; header={(frame.rows, frame.cols)}; "
            f"file={frame.path}"
        )
    if not np.all(np.isfinite(data)):
        raise RuntimeError(f"DICOM contains non-finite pixels: {frame.path}")
    return data


def get_slice_spacing(grid: CineGrid) -> float:
    if grid.slice_count > 1:
        diffs = np.diff(np.asarray(grid.slice_coordinates, dtype=float))
        spacing = float(np.median(np.abs(diffs)))
        if spacing <= 0:
            raise RuntimeError("Invalid slice spacing.")
        if not np.allclose(np.abs(diffs), spacing, atol=0.2, rtol=0.02):
            raise RuntimeError(
                f"Slice spacing is not sufficiently regular: {diffs.tolist()}"
            )
        return spacing
    frame = grid.slices[0][0]
    ds = pydicom.dcmread(str(frame.path), stop_before_pixels=True)
    for name in ("SpacingBetweenSlices", "SliceThickness"):
        value = _to_float(getattr(ds, name, None))
        if value is not None and value > 0:
            return float(value)
    raise RuntimeError("Unable to determine slice spacing.")


def build_ed_sitk_volume(
    grid: CineGrid,
    phase_index: int,
    masks: Sequence[Any],
) -> Tuple[Any, Any]:
    if len(masks) != grid.slice_count:
        raise ValueError("Mask count does not match slice count.")
    image_slices = []
    mask_slices = []
    for slice_index in range(grid.slice_count):
        frame = grid.slices[slice_index][phase_index]
        image = load_frame_pixels(frame)
        mask = np.asarray(masks[slice_index], dtype=np.uint8)
        if image.shape != (grid.rows, grid.cols):
            raise RuntimeError("Unexpected DICOM pixel matrix shape.")
        if mask.shape != image.shape:
            raise RuntimeError("CorSeg mask shape does not match DICOM image.")
        image_slices.append(image)
        mask_slices.append((mask == 1).astype(np.uint8))

    image_array = np.stack(image_slices, axis=0).astype(np.float32)
    mask_array = np.stack(mask_slices, axis=0).astype(np.uint8)
    image_sitk = sitk.GetImageFromArray(image_array)
    mask_sitk = sitk.GetImageFromArray(mask_array)
    spacing = (
        float(grid.pixel_spacing_col),
        float(grid.pixel_spacing_row),
        get_slice_spacing(grid),
    )
    first_frame = grid.slices[0][phase_index]
    origin = tuple(float(x) for x in first_frame.image_position)
    direction_matrix = np.column_stack([
        grid.row_direction,
        grid.col_direction,
        grid.normal_direction,
    ])
    direction = tuple(float(x) for x in direction_matrix.reshape(-1))
    image_sitk.SetSpacing(spacing)
    image_sitk.SetOrigin(origin)
    image_sitk.SetDirection(direction)
    mask_sitk.CopyInformation(image_sitk)
    return image_sitk, mask_sitk


def load_corseg_model(
    weight_path: Path,
    device: Any,
    allow_unsafe_checkpoint: bool = False,
) -> Tuple[Any, Dict[str, Any], Tuple[int, int]]:
    weight_path = Path(weight_path)
    if not weight_path.exists():
        raise FileNotFoundError(
            "CorSeg weight not found: "
            f"{weight_path}\nDownload it from the official CorSeg repository "
            "and place it in weights/, or pass --weight."
        )
    try:
        checkpoint = torch.load(
            str(weight_path),
            map_location=device,
            weights_only=True,
        )
    except Exception as safe_exc:
        if not allow_unsafe_checkpoint:
            raise RuntimeError(
                "The checkpoint could not be loaded with weights_only=True. "
                "Only if this file was downloaded from the official CorSeg "
                "source, rerun with --allow-unsafe-checkpoint. Loading a "
                "pickle-based checkpoint can execute untrusted code. "
                f"Original error: {type(safe_exc).__name__}: {safe_exc}"
            ) from safe_exc
        checkpoint = torch.load(
            str(weight_path),
            map_location=device,
            weights_only=False,
        )

    if not isinstance(checkpoint, dict):
        raise RuntimeError("CorSeg checkpoint must be a dictionary-like object.")
    config = checkpoint.get("config", {}) or {
        "spatial_dims": 2,
        "in_channels": 1,
        "num_classes": 4,
        "mednext_variant": "L",
        "mednext_kernel": 5,
        "img_size": (224, 224),
    }
    model = create_mednext(
        variant=config.get("mednext_variant", "L"),
        spatial_dims=config.get("spatial_dims", 2),
        in_channels=config.get("in_channels", 1),
        out_channels=config.get("num_classes", 4),
        kernel_size=config.get("mednext_kernel", 5),
        deep_supervision=False,
    )
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.to(device).eval()
    img_size = tuple(int(x) for x in config.get("img_size", (224, 224)))
    if len(img_size) != 2:
        raise RuntimeError(f"Invalid CorSeg image size: {img_size}")
    return model, config, img_size


def detect_violations(mask: Any) -> Dict[str, bool]:
    stats = {
        "has_fragment": False,
        "has_containment_violation": False,
        "has_gap": False,
    }
    structure = sp_ndimage.generate_binary_structure(2, 1)
    for label_value in (1, 2, 3):
        binary = mask == label_value
        if binary.any():
            _, count = sp_ndimage.label(binary)
            if count > 1:
                stats["has_fragment"] = True
                break
    lv_cavity = mask == 2
    if lv_cavity.any():
        non_lv = (mask == 0) | (mask == 3)
        if (
            lv_cavity
            & sp_ndimage.binary_dilation(non_lv, structure=structure)
        ).any():
            stats["has_containment_violation"] = True
    cardiac = mask > 0
    if cardiac.any():
        filled = sp_ndimage.binary_fill_holes(cardiac)
        if (filled & ~cardiac).any():
            stats["has_gap"] = True
        else:
            lvm = mask == 1
            rv = mask == 3
            if lvm.any() and rv.any():
                lvm_dilated = sp_ndimage.binary_dilation(
                    lvm, structure=structure
                )
                rv_dilated = sp_ndimage.binary_dilation(
                    rv, structure=structure
                )
                if (lvm_dilated & rv_dilated & (mask == 0)).any():
                    stats["has_gap"] = True
    return stats


def pp_step1_largest_component(mask: Any) -> Any:
    result = np.zeros_like(mask)
    for label_value in (1, 2, 3):
        binary = mask == label_value
        if not binary.any():
            continue
        labeled, count = sp_ndimage.label(binary)
        if count <= 1:
            result[binary] = label_value
            continue
        sizes = sp_ndimage.sum(binary, labeled, range(1, count + 1))
        largest = int(np.argmax(sizes)) + 1
        result[labeled == largest] = label_value
    return result


def pp_step2_containment(mask: Any) -> Any:
    result = mask.copy()
    structure = sp_ndimage.generate_binary_structure(2, 1)
    original_cavity = int(np.sum(result == 2))
    if original_cavity == 0:
        return result
    for _ in range(50):
        cavity = result == 2
        non_lv = (result == 0) | (result == 3)
        exposed = cavity & sp_ndimage.binary_dilation(
            non_lv, structure=structure
        )
        if not exposed.any():
            break
        result[exposed] = 1
        if int(np.sum(result == 2)) < original_cavity * 0.5:
            break
    return result


def pp_step3_fill_gaps(mask: Any) -> Any:
    result = mask.copy()
    structure = sp_ndimage.generate_binary_structure(2, 1)
    cardiac = result > 0
    if cardiac.any():
        holes = sp_ndimage.binary_fill_holes(cardiac) & ~cardiac
        if holes.any():
            labeled, count = sp_ndimage.label(holes)
            for hole_id in range(1, count + 1):
                hole = labeled == hole_id
                border = (
                    sp_ndimage.binary_dilation(
                        hole,
                        structure=structure,
                        iterations=2,
                    )
                    & ~hole
                    & (result > 0)
                )
                if border.any():
                    counts = np.bincount(result[border], minlength=4)
                    fill_value = (
                        int(np.argmax(counts[1:])) + 1
                        if counts[1:].sum() > 0
                        else 1
                    )
                else:
                    fill_value = 1
                result[hole] = fill_value
    background = result == 0
    lvm = result == 1
    rv = result == 3
    if background.any() and lvm.any() and rv.any():
        lvm_adjacent = (
            sp_ndimage.binary_dilation(lvm, structure=structure) & background
        )
        rv_adjacent = (
            sp_ndimage.binary_dilation(rv, structure=structure) & background
        )
        result[lvm_adjacent & rv_adjacent] = 1
    return result


def apply_postprocessing(mask: Any) -> Tuple[Any, Dict[str, Any]]:
    pre_stats = detect_violations(mask)
    result = mask.copy()
    pixels_changed: Dict[str, int] = defaultdict(int)
    before = result.copy()
    result = pp_step1_largest_component(result)
    pixels_changed["step1"] = int(np.sum(before != result))
    before = result.copy()
    result = pp_step2_containment(result)
    pixels_changed["step2"] = int(np.sum(before != result))
    before = result.copy()
    result = pp_step3_fill_gaps(result)
    pixels_changed["step3"] = int(np.sum(before != result))
    return result, {
        "pre": pre_stats,
        "post": detect_violations(result),
        "pixels_changed": dict(pixels_changed),
    }


def infer_array(
    model: Any,
    image: Any,
    img_size: Tuple[int, int],
    device: Any,
) -> Tuple[Any, Dict[str, Any]]:
    image = np.asarray(image, dtype=np.float32)
    original_shape = image.shape
    if image.ndim != 2:
        raise ValueError(f"CorSeg expects a 2D image; got {image.shape}")
    tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(
        tensor,
        size=img_size,
        mode="bilinear",
        align_corners=False,
    )
    nonzero = tensor != 0
    if nonzero.any():
        mean = tensor[nonzero].mean()
        sd = tensor[nonzero].std()
        if sd > 1e-8:
            tensor = (tensor - mean) / sd
        tensor[~nonzero] = 0.0
    tensor = tensor.to(device)
    with torch.no_grad():
        if device.type == "cuda":
            with torch.amp.autocast(device_type="cuda"):
                logits = model(tensor)
        else:
            logits = model(tensor)
    prediction = logits.argmax(dim=1).squeeze(0).cpu()
    if tuple(prediction.shape) != original_shape:
        prediction = F.interpolate(
            prediction.float().unsqueeze(0).unsqueeze(0),
            size=original_shape,
            mode="nearest",
        ).squeeze().to(torch.uint8)
    prediction_np = prediction.numpy().astype(np.uint8)
    return apply_postprocessing(prediction_np)


def _jsonable(value: Any) -> Any:
    if _RUNTIME_LOADED and isinstance(value, np.generic):
        return value.item()
    if _RUNTIME_LOADED and isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(x) for x in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def add_deprecated_features(
    feature_values: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    required_shape = [
        "original_shape_MeshVolume",
        "original_shape_SurfaceArea",
        "original_shape_Sphericity",
    ]
    missing = [x for x in required_shape if x not in feature_values]
    if missing:
        raise RuntimeError(
            "Cannot reconstruct deprecated shape features; missing: "
            + ", ".join(missing)
        )
    volume = float(feature_values["original_shape_MeshVolume"])
    area = float(feature_values["original_shape_SurfaceArea"])
    sphericity = float(feature_values["original_shape_Sphericity"])
    if (
        not np.isfinite(volume)
        or not np.isfinite(area)
        or not np.isfinite(sphericity)
        or area <= 0
        or sphericity <= 0
    ):
        raise RuntimeError("Invalid inputs for deprecated shape reconstruction.")
    feature_values["original_shape_Compactness1"] = float(
        volume / (np.sqrt(np.pi) * np.power(area, 1.5))
    )
    feature_values["original_shape_Compactness2"] = float(
        36.0 * np.pi * np.power(volume, 2.0) / np.power(area, 3.0)
    )
    feature_values["original_shape_SphericalDisproportion"] = float(
        1.0 / sphericity
    )
    variance_to_sd = {
        "original_firstorder_Variance":
            "original_firstorder_StandardDeviation",
        "log-sigma-1-0-mm-3D_firstorder_Variance":
            "log-sigma-1-0-mm-3D_firstorder_StandardDeviation",
        "log-sigma-2-0-mm-3D_firstorder_Variance":
            "log-sigma-2-0-mm-3D_firstorder_StandardDeviation",
        "log-sigma-3-0-mm-3D_firstorder_Variance":
            "log-sigma-3-0-mm-3D_firstorder_StandardDeviation",
    }
    for variance_name, sd_name in variance_to_sd.items():
        if variance_name not in feature_values:
            raise RuntimeError(
                "Cannot reconstruct deprecated standard deviation; missing "
                f"{variance_name}."
            )
        variance = float(feature_values[variance_name])
        if not np.isfinite(variance) or variance < -1e-12:
            raise RuntimeError(
                f"Invalid variance for {variance_name}: {variance}"
            )
        feature_values[sd_name] = float(np.sqrt(max(variance, 0.0)))
    reconstructed = [
        "original_shape_Compactness1",
        "original_shape_Compactness2",
        "original_shape_SphericalDisproportion",
        "original_firstorder_StandardDeviation",
        "log-sigma-1-0-mm-3D_firstorder_StandardDeviation",
        "log-sigma-2-0-mm-3D_firstorder_StandardDeviation",
        "log-sigma-3-0-mm-3D_firstorder_StandardDeviation",
    ]
    return feature_values, reconstructed


def resample_image_and_mask(image: Any, mask: Any) -> Tuple[Any, Any]:
    resampled_image, resampled_mask = imageoperations.resampleImage(
        image,
        mask,
        resampledPixelSpacing=TARGET_SPACING,
        interpolator=sitk.sitkBSpline,
        padDistance=PAD_DISTANCE,
        label=LABEL,
    )
    if resampled_image is None or resampled_mask is None:
        raise RuntimeError("PyRadiomics resampling failed.")
    return resampled_image, resampled_mask


def percentile_clip_and_rescale(
    image: Any,
    mask: Any,
) -> Tuple[Any, Dict[str, float]]:
    image_array = sitk.GetArrayFromImage(image).astype(np.float32)
    mask_array = sitk.GetArrayFromImage(mask)
    roi = mask_array == LABEL
    if not np.any(roi):
        raise ValueError("Myocardial label is absent after resampling.")
    roi_values = image_array[roi]
    roi_values = roi_values[np.isfinite(roi_values)]
    if roi_values.size == 0:
        raise ValueError("No finite myocardial intensity values were found.")
    low, high = np.percentile(
        roi_values,
        [LOWER_PERCENTILE, UPPER_PERCENTILE],
    )
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError(f"Invalid percentile range: P1={low}; P99={high}")
    clipped = np.clip(image_array, low, high)
    scaled = (clipped - low) / (high - low)
    scaled = (
        scaled * (INTENSITY_MAX - INTENSITY_MIN) + INTENSITY_MIN
    ).astype(np.float32)
    scaled = np.clip(scaled, INTENSITY_MIN, INTENSITY_MAX).astype(np.float32)
    scaled_image = sitk.GetImageFromArray(scaled)
    scaled_image.CopyInformation(image)
    scaled_roi = scaled[roi]
    return scaled_image, {
        "p01_before_scaling": float(low),
        "p99_before_scaling": float(high),
        "roi_min_after_scaling": float(np.min(scaled_roi)),
        "roi_max_after_scaling": float(np.max(scaled_roi)),
    }


def preprocess_radiomics(image: Any, raw_mask: Any) -> Tuple[Any, Any, Dict[str, Any]]:
    if image.GetDimension() != 3:
        raise ValueError(f"A 3D image is required; got {image.GetDimension()}D.")
    mask = imageoperations.getMask(raw_mask, label=LABEL, label_channel=0)
    source_spacing = tuple(float(x) for x in image.GetSpacing())
    source_size = tuple(int(x) for x in image.GetSize())
    resampled_image, resampled_mask = resample_image_and_mask(image, mask)
    scaled_image, intensity_qc = percentile_clip_and_rescale(
        resampled_image,
        resampled_mask,
    )
    qc = {
        "pipeline_id": PIPELINE_ID,
        "pipeline_sha256": pipeline_sha256(),
        "pyradiomics_version": radiomics.__version__,
        "simpleitk_version": sitk.Version_VersionString(),
        "source_spacing_x": source_spacing[0],
        "source_spacing_y": source_spacing[1],
        "source_spacing_z": source_spacing[2],
        "source_size_x": source_size[0],
        "source_size_y": source_size[1],
        "source_size_z": source_size[2],
        "resampled_spacing_x": float(resampled_image.GetSpacing()[0]),
        "resampled_spacing_y": float(resampled_image.GetSpacing()[1]),
        "resampled_spacing_z": float(resampled_image.GetSpacing()[2]),
        "resampled_size_x": int(resampled_image.GetSize()[0]),
        "resampled_size_y": int(resampled_image.GetSize()[1]),
        "resampled_size_z": int(resampled_image.GetSize()[2]),
    }
    qc.update(intensity_qc)
    return scaled_image, resampled_mask, qc


def build_radiomics_extractor() -> Any:
    extractor = featureextractor.RadiomicsFeatureExtractor(
        binWidth=BIN_WIDTH,
        normalize=False,
        resampledPixelSpacing=None,
        force2D=True,
        force2Ddimension=0,
        label=LABEL,
        voxelArrayShift=0,
        additionalInfo=True,
    )
    extractor.disableAllImageTypes()
    extractor.enableImageTypeByName("Original")
    extractor.enableImageTypeByName("LoG", customArgs={"sigma": LOG_SIGMAS})
    extractor.disableAllFeatures()
    for feature_class in PIPELINE_CONFIG["feature_classes"]:
        extractor.enableFeatureClassByName(feature_class)
    return extractor


def _try_write_image(
    image: Any,
    path: Path,
    label: str,
) -> Tuple[Optional[Path], Optional[str]]:
    try:
        sitk.WriteImage(image, str(path), True)
        return path, None
    except Exception as exc:
        warning = (
            f"{label} could not be saved; in-memory processing continued. "
            f"{type(exc).__name__}: {exc}"
        )
        print(f"[Warning] {warning}")
        return None, warning


def extract_radiomics(
    image: Any,
    mask: Any,
    output_dir: Path,
    patient_id: str,
) -> Dict[str, Any]:
    validate_locked_model()
    normalized_version = str(radiomics.__version__).lstrip("vV")
    if normalized_version != PYRADIOMICS_VERSION:
        raise RuntimeError(
            f"PyRadiomics {PYRADIOMICS_VERSION} is required; found "
            f"{radiomics.__version__}."
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_image, processed_mask, qc = preprocess_radiomics(image, mask)
    image_path, image_warning = _try_write_image(
        processed_image,
        output_dir / "ED_radiomics_image_1x1x8_0to255.nii.gz",
        "Processed radiomics image",
    )
    mask_path, mask_warning = _try_write_image(
        processed_mask,
        output_dir / "ED_radiomics_LVmyocardium_mask_1x1x8.nii.gz",
        "Processed radiomics mask",
    )
    extractor = build_radiomics_extractor()
    result = extractor.execute(processed_image, processed_mask)
    feature_values: Dict[str, Any] = {}
    diagnostics: Dict[str, Any] = {}
    for key, value in result.items():
        if str(key).startswith("diagnostics_"):
            diagnostics[str(key)] = _jsonable(value)
        else:
            try:
                feature_values[str(key)] = float(value)
            except Exception:
                feature_values[str(key)] = _jsonable(value)
    feature_values, reconstructed = add_deprecated_features(feature_values)
    if len(feature_values) != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            "Unexpected radiomics feature count: "
            f"{len(feature_values)}; expected {EXPECTED_FEATURE_COUNT}."
        )
    if any(name.startswith("wavelet-") for name in feature_values):
        raise RuntimeError("Wavelet features were unexpectedly extracted.")
    if qc["roi_min_after_scaling"] < -1e-5:
        raise RuntimeError("ROI intensity below 0 after scaling.")
    if qc["roi_max_after_scaling"] > 255.0 + 1e-5:
        raise RuntimeError("ROI intensity above 255 after scaling.")
    row = {"Patient": patient_id, **qc, **feature_values}
    feature_csv = output_dir / "radiomics_features.csv"
    feature_xlsx = output_dir / "radiomics_features.xlsx"
    pd.DataFrame([row]).to_csv(feature_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame([row]).to_excel(
        feature_xlsx,
        sheet_name="features",
        index=False,
    )
    qc_path = output_dir / "radiomics_qc.json"
    qc_payload = {
        "patient": patient_id,
        "qc": qc,
        "pipeline_config": PIPELINE_CONFIG,
        "extractor_settings": _jsonable(extractor.settings),
        "enabled_image_types": _jsonable(extractor.enabledImagetypes),
        "enabled_feature_classes": list(extractor.enabledFeatures.keys()),
        "program_version": PROGRAM_VERSION,
        "feature_count": len(feature_values),
        "expected_feature_count": EXPECTED_FEATURE_COUNT,
        "deprecated_features_reconstructed": reconstructed,
        "intermediate_write_warnings": [
            item for item in [image_warning, mask_warning] if item
        ],
        "diagnostics": diagnostics,
    }
    qc_path.write_text(
        json.dumps(qc_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "features": feature_values,
        "qc": qc,
        "csv_path": feature_csv,
        "xlsx_path": feature_xlsx,
        "processed_image_path": image_path,
        "processed_mask_path": mask_path,
        "qc_path": qc_path,
    }


def score_radiomics_csv(
    radiomics_csv: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    _load_runtime()
    validate_locked_model()
    radiomics_csv = Path(radiomics_csv)
    output_dir = Path(output_dir)
    if not radiomics_csv.exists():
        raise FileNotFoundError(f"Radiomics CSV not found: {radiomics_csv}")
    data = pd.read_csv(radiomics_csv)
    if len(data) != 1:
        raise RuntimeError(
            f"Expected exactly one radiomics row; found {len(data)}."
        )
    feature_values = {
        name: data.loc[0, name]
        for name in FEATURE_SPECS
        if name in data.columns
    }
    score = calculate_rscore(feature_values)
    patient = (
        str(data.loc[0, "Patient"])
        if "Patient" in data.columns
        else "Patient"
    )
    result = {
        "Patient": patient,
        "Rscore_z": score["Rscore_z"],
        "raw_Rscore": score["raw_Rscore"],
        "Primary_outcome": PRIMARY_OUTCOME,
        "Model_target": MODEL_TARGET,
        "Model_ID": MODEL_ID,
        "Pipeline_ID": PIPELINE_ID,
        "Model_SHA256": model_sha256(),
        "Pipeline_SHA256": pipeline_sha256(),
        "Nonzero_feature_count": len(FEATURE_SPECS),
        "A_center_OOF_reference_mean": OOF_RSCORE_MEAN,
        "A_center_OOF_reference_SD": OOF_RSCORE_SD,
        "Selected_alpha": SELECTED_ALPHA,
        "Selected_lambda": SELECTED_LAMBDA,
        "Tuning_rule": TUNING_RULE,
    }
    result_df = pd.DataFrame([result])
    details_df = pd.DataFrame(score["details"])
    result_csv = output_dir / "Rscore_result.csv"
    result_xlsx = output_dir / "Rscore_result.xlsx"
    parameters_csv = output_dir / "Rscore_feature_parameters.csv"
    result_df.to_csv(result_csv, index=False, encoding="utf-8-sig")
    details_df.to_csv(parameters_csv, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(result_xlsx, engine="openpyxl") as writer:
        result_df.to_excel(writer, sheet_name="Rscore_result", index=False)
        details_df.to_excel(writer, sheet_name="Feature_parameters", index=False)
    return {
        **result,
        "result_csv": str(result_csv),
        "result_xlsx": str(result_xlsx),
        "feature_parameters_csv": str(parameters_csv),
    }


def choose_device(requested: str) -> Any:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def default_output(input_path: Path) -> Path:
    path = Path(input_path)
    base = path if path.is_dir() else path.parent
    return base.parent / f"{base.name}_Rscore_output"


def _prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise RuntimeError(
            f"Output folder is not empty: {output_dir}\n"
            "Choose a new folder or explicitly pass --overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def _package_versions() -> Dict[str, str]:
    versions = {}
    for label, module in [
        ("python", None),
        ("torch", torch),
        ("monai", importlib.import_module("monai")),
        ("numpy", np),
        ("scipy", importlib.import_module("scipy")),
        ("pydicom", pydicom),
        ("SimpleITK", sitk),
        ("pyradiomics", radiomics),
        ("pandas", pd),
    ]:
        if module is None:
            versions[label] = sys.version.split()[0]
        elif label == "SimpleITK":
            versions[label] = sitk.Version_VersionString()
        else:
            versions[label] = str(getattr(module, "__version__", "unknown"))
    return versions


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    _load_runtime()
    validate_locked_model()
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    output_dir = (
        Path(args.output).resolve()
        if args.output
        else default_output(input_path).resolve()
    )
    _prepare_output(output_dir, args.overwrite)
    patient_id = args.patient_id or (
        input_path.name if input_path.is_dir() else input_path.parent.name
    )
    if not patient_id.strip():
        raise RuntimeError("Patient/research identifier is empty.")
    weight_path = Path(args.weight).resolve()
    device = choose_device(args.device)

    print(f"{PROGRAM_NAME} v{PROGRAM_VERSION}")
    print(f"Research ID: {patient_id}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"Model: {MODEL_ID} | Output: {PRIMARY_OUTCOME}")
    print(f"Radiomics pipeline: {PIPELINE_ID}")

    print("1/6 Building cine grid...")
    grid = build_cine_grid(input_path)
    print(
        f"  Series: {grid.series_number} | {grid.series_description}\n"
        f"  Slices: {grid.slice_count} | Phases: {grid.phase_count}"
    )
    print("2/6 Loading CorSeg model...")
    model, model_config, img_size = load_corseg_model(
        weight_path,
        device,
        allow_unsafe_checkpoint=args.allow_unsafe_checkpoint,
    )

    phase_rows = []
    masks_by_phase = []
    postprocess_stats = []
    total = grid.slice_count * grid.phase_count
    completed = 0
    pixel_area = grid.pixel_spacing_row * grid.pixel_spacing_col
    slice_spacing = get_slice_spacing(grid)
    print("3/6 Segmenting cine and detecting ED...")
    for phase in range(grid.phase_count):
        phase_masks = []
        cavity_pixels = 0
        myocardium_pixels = 0
        rv_pixels = 0
        for slice_index in range(grid.slice_count):
            frame = grid.slices[slice_index][phase]
            image = load_frame_pixels(frame)
            mask, stats = infer_array(model, image, img_size, device)
            phase_masks.append(mask)
            cavity_pixels += int(np.sum(mask == 2))
            myocardium_pixels += int(np.sum(mask == 1))
            rv_pixels += int(np.sum(mask == 3))
            postprocess_stats.append({
                "phase": phase,
                "slice": slice_index,
                **stats,
            })
            completed += 1
            print(
                f"\r  CorSeg inference {completed}/{total}",
                end="",
                flush=True,
            )
        masks_by_phase.append(phase_masks)
        area_sum = cavity_pixels * pixel_area
        phase_rows.append({
            "PhaseIndex": phase,
            "LV_cavity_pixels_total": cavity_pixels,
            "LV_cavity_area_mm2_total": area_sum,
            "LV_cavity_volume_proxy_mm3": area_sum * slice_spacing,
            "LV_myocardium_pixels_total": myocardium_pixels,
            "RV_pixels_total": rv_pixels,
        })
    print("")
    ed_phase = int(max(
        phase_rows,
        key=lambda row: row["LV_cavity_area_mm2_total"],
    )["PhaseIndex"])
    es_phase = int(min(
        phase_rows,
        key=lambda row: row["LV_cavity_area_mm2_total"],
    )["PhaseIndex"])
    pd.DataFrame(phase_rows).to_csv(
        output_dir / "phase_cavity_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"  Detected ED phase: {ed_phase}")
    print(f"  Detected ES phase: {es_phase}")

    print("4/6 Building geometry-correct ED 3D volume...")
    ed_image, ed_mask = build_ed_sitk_volume(
        grid,
        ed_phase,
        masks_by_phase[ed_phase],
    )
    native_image_path, native_image_warning = _try_write_image(
        ed_image,
        output_dir / "ED_native_image.nii.gz",
        "Native ED image",
    )
    native_mask_path, native_mask_warning = _try_write_image(
        ed_mask,
        output_dir / "ED_native_LVmyocardium_mask.nii.gz",
        "Native LV myocardium mask",
    )

    print("5/6 Running locked PyRadiomics pipeline...")
    radiomics_result = extract_radiomics(
        ed_image,
        ed_mask,
        output_dir,
        patient_id,
    )
    print("6/6 Calculating locked Rscore_z...")
    rscore_result = score_radiomics_csv(
        radiomics_result["csv_path"],
        output_dir,
    )
    summary = {
        "ResearchID": patient_id,
        "Program": PROGRAM_NAME,
        "ProgramVersion": PROGRAM_VERSION,
        "ModelID": MODEL_ID,
        "ModelTarget": MODEL_TARGET,
        "PrimaryOutcome": PRIMARY_OUTCOME,
        "ModelSHA256": model_sha256(),
        "PipelineID": PIPELINE_ID,
        "PipelineSHA256": pipeline_sha256(),
        "ProgramFileSHA256": file_sha256(Path(__file__)),
        "CorSegWeightFile": weight_path.name,
        "CorSegWeightSHA256": file_sha256(weight_path),
        "CorSegModelConfig": _jsonable(model_config),
        "PackageVersions": _package_versions(),
        "SeriesNumber": grid.series_number,
        "SeriesDescription": grid.series_description,
        "TemporalOrderingBasisBySlice": grid.temporal_ordering_basis,
        "AbsolutePhaseValuesAligned": grid.absolute_phase_values_aligned,
        "SliceCount": grid.slice_count,
        "PhaseCount": grid.phase_count,
        "ED_phase": ed_phase,
        "ES_phase": es_phase,
        "SourceSpacing": list(ed_image.GetSpacing()),
        "SourceSize": list(ed_image.GetSize()),
        "RadiomicsFeatureCount": len(radiomics_result["features"]),
        "ExpectedRadiomicsFeatureCount": EXPECTED_FEATURE_COUNT,
        "Device": str(device),
        "NativeImage": str(native_image_path) if native_image_path else None,
        "NativeMask": str(native_mask_path) if native_mask_path else None,
        "NativeImageWriteWarning": native_image_warning,
        "NativeMaskWriteWarning": native_mask_warning,
        "RscoreCalculated": True,
        "raw_Rscore": rscore_result["raw_Rscore"],
        "Rscore_z": rscore_result["Rscore_z"],
        "CompletedAt": datetime.now().isoformat(timespec="seconds"),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (output_dir / "corseg_postprocess_stats.json").write_text(
        json.dumps(
            postprocess_stats,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print("")
    print("PIPELINE SUCCESS")
    print(f"Rscore_z: {float(rscore_result['Rscore_z']):.6f}")
    print(f"raw Rscore: {float(rscore_result['raw_Rscore']):.6f}")
    print(f"Result: {rscore_result['result_xlsx']}")
    return summary


def _same_or_child(path: Path, parent: Path) -> bool:
    try:
        path = path.resolve()
        parent = parent.resolve()
        return os.path.commonpath([str(path), str(parent)]) == str(parent)
    except Exception:
        return False


def _read_success_summary(output_dir: Path) -> Optional[Dict[str, Any]]:
    summary_path = output_dir / "run_summary.json"
    result_path = output_dir / "Rscore_result.xlsx"
    if not summary_path.exists() or not result_path.exists():
        return None
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if data.get("RscoreCalculated") is True and data.get("Rscore_z") is not None:
        return data
    return None


def _write_batch_summary(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "Patient",
        "Status",
        "Elapsed_seconds",
        "Return_code",
        "Input_folder",
        "Output_folder",
        "SliceCount",
        "PhaseCount",
        "ED_phase",
        "ES_phase",
        "raw_Rscore",
        "Rscore_z",
        "Error_tail",
    ]
    def _write(target: Path) -> None:
        temporary = target.with_suffix(target.suffix + ".tmp")
        try:
            with temporary.open(
                "w",
                newline="",
                encoding="utf-8-sig",
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for row in rows:
                    writer.writerow({
                        key: row.get(key, "")
                        for key in fields
                    })
            temporary.replace(target)
        finally:
            if temporary.exists():
                try:
                    temporary.unlink()
                except OSError:
                    pass

    try:
        _write(path)
    except PermissionError as exc:
        recovery = path.with_name("batch_summary_recovery.csv")
        print(
            "[Warning] batch_summary.csv appears to be open in Excel/WPS; "
            f"writing recovery summary instead. Original error: {exc}"
        )
        _write(recovery)


def run_batch(args: argparse.Namespace) -> None:
    root = Path(args.input_root).resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Input root not found: {root}")
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else root.parent / f"{root.name}_Rscore_batch_output"
    )
    if _same_or_child(output_root, root):
        raise RuntimeError("Output root must not be inside the DICOM input root.")
    output_root.mkdir(parents=True, exist_ok=True)
    patient_dirs = sorted(
        [path for path in root.iterdir() if path.is_dir()],
        key=lambda path: path.name.lower(),
    )
    if not patient_dirs:
        raise RuntimeError(f"No patient folders found directly under {root}")
    rows: List[Dict[str, Any]] = []
    summary_path = output_root / "batch_summary.csv"
    weight = Path(args.weight).resolve()
    for index, patient_dir in enumerate(patient_dirs, start=1):
        patient = patient_dir.name
        patient_output = output_root / patient
        existing = _read_success_summary(patient_output)
        if existing and not args.rerun_success:
            print(f"[{index}/{len(patient_dirs)}] SKIP: {patient}")
            rows.append({
                "Patient": patient,
                "Status": "SKIPPED_ALREADY_SUCCESS",
                "Return_code": 0,
                "Input_folder": str(patient_dir),
                "Output_folder": str(patient_output),
                "SliceCount": existing.get("SliceCount", ""),
                "PhaseCount": existing.get("PhaseCount", ""),
                "ED_phase": existing.get("ED_phase", ""),
                "ES_phase": existing.get("ES_phase", ""),
                "raw_Rscore": existing.get("raw_Rscore", ""),
                "Rscore_z": existing.get("Rscore_z", ""),
            })
            _write_batch_summary(summary_path, rows)
            continue
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "run",
            "--input",
            str(patient_dir),
            "--output",
            str(patient_output),
            "--patient-id",
            patient,
            "--weight",
            str(weight),
            "--device",
            args.device,
        ]
        if patient_output.exists() and any(patient_output.iterdir()):
            command.append("--overwrite")
        if args.allow_unsafe_checkpoint:
            command.append("--allow-unsafe-checkpoint")
        print(f"[{index}/{len(patient_dirs)}] RUN: {patient}")
        started = time.time()
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        elapsed = round(time.time() - started, 1)
        log_path = patient_output / "batch_run.log"
        patient_output.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            completed.stdout + "\n" + completed.stderr,
            encoding="utf-8",
        )
        success = _read_success_summary(patient_output)
        if completed.returncode == 0 and success:
            status = "SUCCESS"
            error_tail = ""
            print(f"[{index}/{len(patient_dirs)}] SUCCESS: {patient}")
        else:
            status = "FAILED"
            error_tail = (completed.stdout + completed.stderr)[-3000:]
            print(f"[{index}/{len(patient_dirs)}] FAILED: {patient}")
        success = success or {}
        rows.append({
            "Patient": patient,
            "Status": status,
            "Elapsed_seconds": elapsed,
            "Return_code": completed.returncode,
            "Input_folder": str(patient_dir),
            "Output_folder": str(patient_output),
            "SliceCount": success.get("SliceCount", ""),
            "PhaseCount": success.get("PhaseCount", ""),
            "ED_phase": success.get("ED_phase", ""),
            "ES_phase": success.get("ES_phase", ""),
            "raw_Rscore": success.get("raw_Rscore", ""),
            "Rscore_z": success.get("Rscore_z", ""),
            "Error_tail": error_tail,
        })
        _write_batch_summary(summary_path, rows)
    print(f"Batch summary: {summary_path}")


def resume_rscore(args: argparse.Namespace) -> None:
    output_dir = Path(args.output).resolve()
    result = score_radiomics_csv(
        output_dir / "radiomics_features.csv",
        output_dir,
    )
    summary_path = output_dir / "run_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update({
            "ModelID": MODEL_ID,
            "ModelTarget": MODEL_TARGET,
            "PrimaryOutcome": PRIMARY_OUTCOME,
            "ModelSHA256": model_sha256(),
            "PipelineID": PIPELINE_ID,
            "PipelineSHA256": pipeline_sha256(),
            "RscoreCalculated": True,
            "raw_Rscore": result["raw_Rscore"],
            "Rscore_z": result["Rscore_z"],
            "RscoreRecalculatedAt": datetime.now().isoformat(timespec="seconds"),
        })
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print("RSCORE RESUME SUCCESS")
    print(f"Rscore_z: {float(result['Rscore_z']):.6f}")
    print(f"raw Rscore: {float(result['raw_Rscore']):.6f}")


def check_environment(args: argparse.Namespace) -> int:
    print(f"{PROGRAM_NAME} v{PROGRAM_VERSION}")
    print(f"Python: {sys.version}")
    failures = []
    if sys.version_info < (3, 9):
        failures.append("Python >=3.9 is required.")
    modules = [
        ("torch", "torch"),
        ("monai", "monai"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("pydicom", "pydicom"),
        ("SimpleITK", "SimpleITK"),
        ("pyradiomics", "radiomics"),
        ("pandas", "pandas"),
        ("openpyxl", "openpyxl"),
    ]
    imported = {}
    for label, module_name in modules:
        try:
            module = importlib.import_module(module_name)
            imported[label] = module
            version = getattr(module, "__version__", "unknown")
            print(f"{label}: {version}")
        except Exception as exc:
            failures.append(f"{label}: {type(exc).__name__}: {exc}")
            print(f"{label}: FAILED")
    if "pyradiomics" in imported:
        found = str(imported["pyradiomics"].__version__).lstrip("vV")
        if found != PYRADIOMICS_VERSION:
            failures.append(
                f"PyRadiomics {PYRADIOMICS_VERSION} required; found {found}."
            )
    if "SimpleITK" in imported:
        try:
            reader = imported["SimpleITK"].ImageFileReader()
            if "GDCMImageIO" not in list(reader.GetRegisteredImageIOs()):
                failures.append("SimpleITK GDCMImageIO is unavailable.")
        except Exception as exc:
            failures.append(f"SimpleITK GDCM check failed: {exc}")
    try:
        validate_locked_model()
        means = {name: spec[1] for name, spec in FEATURE_SPECS.items()}
        score = calculate_rscore(means)
        if not math.isclose(score["raw_Rscore"], INTERCEPT, abs_tol=1e-12):
            raise RuntimeError("Formula mean-vector self-test failed.")
        print(f"Locked model self-test: OK ({len(FEATURE_SPECS)} features)")
        print(f"Model SHA256: {model_sha256()}")
        print(f"Pipeline SHA256: {pipeline_sha256()}")
    except Exception as exc:
        failures.append(f"Locked model: {exc}")
    weight = Path(args.weight).resolve()
    if weight.exists():
        print(f"CorSeg weight: FOUND ({weight.name})")
    else:
        print(f"CorSeg weight: NOT FOUND ({weight})")
        print("  Download it before running image inference.")
    if failures:
        print("\nENVIRONMENT CHECK FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("\nENVIRONMENT CHECK PASSED")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Conventional SAX cine DICOM -> CorSeg -> ED myocardium -> "
            "locked radiomics -> Rscore_z"
        )
    )
    parser.add_argument("--version", action="version", version=PROGRAM_VERSION)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Process one research case.")
    run_parser.add_argument("--input", required=True)
    run_parser.add_argument("--output", default=None)
    run_parser.add_argument(
        "--patient-id",
        default=None,
        help="De-identified research ID; defaults to the input folder name.",
    )
    run_parser.add_argument("--weight", default=str(DEFAULT_WEIGHT))
    run_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    run_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output folder.",
    )
    run_parser.add_argument(
        "--allow-unsafe-checkpoint",
        action="store_true",
        help=(
            "Allow pickle-based torch.load only for a checkpoint obtained "
            "from the official CorSeg source."
        ),
    )
    run_parser.set_defaults(function=run_pipeline)

    batch_parser = subparsers.add_parser(
        "batch",
        help="Process immediate child folders as separate research cases.",
    )
    batch_parser.add_argument("--input-root", required=True)
    batch_parser.add_argument("--output-root", default=None)
    batch_parser.add_argument("--weight", default=str(DEFAULT_WEIGHT))
    batch_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="cpu",
    )
    batch_parser.add_argument("--rerun-success", action="store_true")
    batch_parser.add_argument(
        "--allow-unsafe-checkpoint",
        action="store_true",
    )
    batch_parser.set_defaults(function=run_batch)

    resume_parser = subparsers.add_parser(
        "resume",
        help="Recalculate Rscore_z from an existing radiomics_features.csv.",
    )
    resume_parser.add_argument("--output", required=True)
    resume_parser.set_defaults(function=resume_rscore)

    check_parser = subparsers.add_parser(
        "check",
        help="Check dependencies, locked formula, and weight location.",
    )
    check_parser.add_argument("--weight", default=str(DEFAULT_WEIGHT))
    check_parser.set_defaults(function=check_environment)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    result = args.function(args)
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
