"""
Dataset Adapter for DICOM Series
=================================
Converts DICOM patient studies into per-modality NIfTI files.

Production features:
  - Enhanced modality detection for GE, Siemens, Philips scanners
  - DICOM series validation (slice consistency, orientation)
  - Structured logging
"""
import os
from typing import Dict, List, Optional, Tuple

import pydicom
from pydicom.misc import is_dicom
import SimpleITK as sitk

from utils.pipeline_logger import get_logger

logger = get_logger("DICOM_Adapter")

REQUIRED_MODALITIES = ("t1", "t1ce", "t2", "flair")
OPTIONAL_MODALITIES = ("seg",)


def _is_hidden(path: str) -> bool:
    return os.path.basename(path).startswith(".")


def _list_patient_dirs(data_dir: str) -> List[str]:
    patients = []
    for name in sorted(os.listdir(data_dir)):
        full = os.path.join(data_dir, name)
        if os.path.isdir(full) and not _is_hidden(full):
            patients.append(full)
    return patients


def _dicom_files_in_dir(folder: str) -> List[str]:
    files = []
    for name in sorted(os.listdir(folder)):
        full = os.path.join(folder, name)
        if not os.path.isfile(full):
            continue
        if is_dicom(full):
            files.append(full)
            continue
        # Some valid DICOM files do not include the "DICM" preamble.
        # Accept files that pydicom can parse in force mode.
        try:
            pydicom.dcmread(full, stop_before_pixels=True, force=True)
            files.append(full)
        except Exception:
            pass
    return files


def _scan_series_dirs(patient_dir: str) -> List[Tuple[str, List[str]]]:
    """Return series folders that contain at least one DICOM file."""
    series_dirs: List[Tuple[str, List[str]]] = []
    for root, _, _ in os.walk(patient_dir):
        if _is_hidden(root):
            continue
        dicom_files = _dicom_files_in_dir(root)
        if dicom_files:
            series_dirs.append((root, dicom_files))
    return series_dirs


def _safe_attr(ds: pydicom.Dataset, key: str) -> str:
    val = getattr(ds, key, "")
    return str(val).strip() if val is not None else ""


def _modality_hint(series_dir: str, sample_file: str) -> str:
    """
    Infer pipeline modality from folder name and DICOM metadata.
    Returns one of: t1, t1ce, t2, flair, seg, unknown.
    """
    folder_name = os.path.basename(series_dir).lower()
    hint_text = [folder_name]

    try:
        ds = pydicom.dcmread(sample_file, stop_before_pixels=True, force=True)
        hint_text.extend([
            _safe_attr(ds, "SeriesDescription").lower(),
            _safe_attr(ds, "ProtocolName").lower(),
            _safe_attr(ds, "SequenceName").lower(),
            _safe_attr(ds, "Modality").lower(),
        ])
    except Exception:
        pass

    text = " ".join([x for x in hint_text if x])

    # Contrast-enhanced T1 checks first so "t1ce" doesn't get classified as "t1".
    if any(token in text for token in (
        "t1ce", "t1c", "t1 gd", "t1-gd", "post contrast", "post-contrast",
        "contrast enhanced", "contrast-enhanced", "gad", "mprage c", "mpragec",
        "t1w+c", "t1_post", "ax t1 c+", "sag t1 c+", "cor t1 c+",
        "bravo+c", "fspgr+c", "3d t1 gd", "t1 with gad",
    )):
        return "t1ce"
    if "flair" in text:
        return "flair"
    if "t2" in text:
        return "t2"
    if "seg" in text or "label" in text or "rtstruct" in text:
        return "seg"
    if "t1" in text or "mprage" in text or "bravo" in text or "fspgr" in text:
        return "t1"
    return "unknown"


def _read_series_image(series_dir: str, sample_file: str) -> sitk.Image:
    """Read a DICOM series directory into a SimpleITK image."""
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(series_dir)
    if series_ids:
        # Use the first series ID found in this folder.
        file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(series_dir, series_ids[0])
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(file_names)
        return reader.Execute()

    # Fallback for single-file DICOM objects.
    return sitk.ReadImage(sample_file)


def discover_dicom_series(data_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Discover DICOM modalities by patient.

    Returns:
      {
        "patient_id": {
          "t1": "/path/to/series_dir",
          "t1ce": "/path/to/series_dir",
          ...
        }
      }

    Selection rule when duplicates exist for one modality:
      keep the series folder with the most DICOM files.
    """
    if not os.path.isdir(data_dir):
        return {}

    discovered: Dict[str, Dict[str, str]] = {}

    for patient_dir in _list_patient_dirs(data_dir):
        patient_id = os.path.basename(patient_dir)
        modality_to_series: Dict[str, str] = {}
        modality_counts: Dict[str, int] = {}
        largest_unknown_series: Optional[str] = None
        largest_unknown_count = -1

        for series_dir, dicom_files in _scan_series_dirs(patient_dir):
            modality = _modality_hint(series_dir, dicom_files[0])
            if modality == "unknown":
                if len(dicom_files) > largest_unknown_count:
                    largest_unknown_series = series_dir
                    largest_unknown_count = len(dicom_files)
                continue

            current_count = modality_counts.get(modality, -1)
            if len(dicom_files) > current_count:
                modality_to_series[modality] = series_dir
                modality_counts[modality] = len(dicom_files)

        if not modality_to_series and largest_unknown_series:
            # Fallback for sparse metadata datasets: keep one representative
            # MR series so downstream conversion can still proceed.
            modality_to_series["t2"] = largest_unknown_series

        if modality_to_series:
            discovered[patient_id] = modality_to_series

    logger.info(f"Discovered {len(discovered)} DICOM patient(s)")
    return discovered


def validate_dicom_series(series_dir: str) -> List[str]:
    """Validate a DICOM series for consistency issues."""
    issues = []
    dicom_files = _dicom_files_in_dir(series_dir)
    if len(dicom_files) < 2:
        return issues

    # Check slice thickness consistency
    thicknesses = set()
    orientations = set()
    for dcm_path in dicom_files[:20]:  # Sample first 20
        try:
            ds = pydicom.dcmread(dcm_path, stop_before_pixels=True, force=True)
            st = getattr(ds, "SliceThickness", None)
            if st is not None:
                thicknesses.add(round(float(st), 2))
            iop = getattr(ds, "ImageOrientationPatient", None)
            if iop is not None:
                orientations.add(tuple(round(float(x), 3) for x in iop))
        except Exception:
            pass

    if len(thicknesses) > 1:
        issues.append(f"Inconsistent slice thickness in {os.path.basename(series_dir)}: {thicknesses}")
    if len(orientations) > 1:
        issues.append(f"Inconsistent orientation in {os.path.basename(series_dir)}")

    return issues


def convert_dicom_patient(modality_series: Dict[str, str]) -> Dict[str, sitk.Image]:
    """Read all discovered DICOM modalities for a patient as SimpleITK images."""
    images: Dict[str, sitk.Image] = {}

    for modality, series_dir in modality_series.items():
        dicom_files = _dicom_files_in_dir(series_dir)
        if not dicom_files:
            continue
        images[modality] = _read_series_image(series_dir, dicom_files[0])

    missing = [m for m in REQUIRED_MODALITIES if m not in images]
    allow_fallback = os.environ.get("ALLOW_DICOM_MODALITY_FALLBACK", "1").strip().lower() in ("1", "true", "yes")
    if missing and allow_fallback and images:
        fallback_modality = next(iter(images.keys()))
        fallback_image = images[fallback_modality]
        for modality in missing:
            images[modality] = fallback_image
        print(
            "[WARNING] Missing DICOM modalities: "
            f"{', '.join(missing)}. "
            f"Falling back to '{fallback_modality}' image for missing channels."
        )

    missing = [m for m in REQUIRED_MODALITIES if m not in images]
    if missing:
        raise ValueError(f"Missing required DICOM modalities: {', '.join(missing)}")

    return images


def save_dicom_as_nifti(images: Dict[str, sitk.Image], patient_id: str,
                        output_dir: str) -> Tuple[str, Dict[str, str]]:
    """Save modality images to the reconstructed patient directory as NIfTI."""
    patient_dir = os.path.join(output_dir, "reconstructed", patient_id)
    os.makedirs(patient_dir, exist_ok=True)

    saved: Dict[str, str] = {}
    for modality in REQUIRED_MODALITIES + OPTIONAL_MODALITIES:
        image = images.get(modality)
        if image is None:
            continue
        path = os.path.join(patient_dir, f"{patient_id}_{modality}.nii.gz")
        sitk.WriteImage(image, path)
        saved[modality] = path

    return patient_dir, saved


def normalize_dicom_patient_id(raw_patient_id: str) -> str:
    """Normalize incoming patient IDs to the folder naming used in discovery."""
    return raw_patient_id.strip()
