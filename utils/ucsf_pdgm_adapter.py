"""
UCSF-PDGM Dataset Adapter
============================
Dedicated adapter for the UCSF-PDGM (Public Dataset of Glioma MRI)
NIfTI dataset format.

Handles:
  - _nifti suffixes in filenames
  - Multi-sequence layouts (T1, T1c/T1CE, T2, FLAIR)
  - Auto-detection of modalities from filenames
  - Modality alignment validation (spacing, affine, dimensions)
  - Conversion to standard pipeline naming convention
"""
import os
import re
import shutil
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np

from utils.pipeline_logger import get_logger

logger = get_logger("UCSF_PDGM")


def _get_ucsf_config():
    try:
        from config.config_loader import get_config
        return get_config().ucsf_pdgm
    except Exception:
        from dataclasses import dataclass as dc, field
        @dc
        class _D:
            modality_patterns: Dict[str, List[str]] = field(default_factory=lambda: {
                "t1": ["_T1_", "_t1_", "_T1."],
                "t1ce": ["_T1c_", "_t1c_", "_T1CE_", "_t1ce_", "_T1GD_", "_T1post_"],
                "t2": ["_T2_", "_t2_", "_T2."],
                "flair": ["_FLAIR_", "_flair_", "_Flair_"],
            })
            nifti_suffix: str = "_nifti"
            expected_extensions: List[str] = field(default_factory=lambda: [".nii.gz", ".nii"])
        return _D()


@dataclass
class ValidationResult:
    """Result of modality alignment validation."""
    valid: bool
    issues: List[str]
    spacing_consistent: bool = True
    dimensions_consistent: bool = True
    affine_consistent: bool = True


def detect_modality(filename: str) -> Optional[str]:
    """
    Pattern-match a UCSF-PDGM filename to determine its MRI modality.

    Examples:
        UCSF-PDGM-0001_T1_nifti.nii.gz → "t1"
        UCSF-PDGM-0001_T1c_nifti.nii.gz → "t1ce"
        UCSF-PDGM-0001_FLAIR_nifti.nii.gz → "flair"
    """
    cfg = _get_ucsf_config()

    # Strip path and extension
    basename = os.path.basename(filename)
    stem = basename
    for ext in cfg.expected_extensions:
        if stem.lower().endswith(ext):
            stem = stem[:-len(ext)]
            break

    # Remove _nifti suffix if present
    if stem.lower().endswith(cfg.nifti_suffix):
        stem = stem[:-len(cfg.nifti_suffix)]

    # Check modality patterns (order matters: t1ce before t1)
    check_order = ["t1ce", "flair", "t2", "t1"]
    for mod in check_order:
        patterns = cfg.modality_patterns.get(mod, [])
        for pattern in patterns:
            if pattern in stem or pattern in basename:
                return mod

    # Fallback: check if filename ends with modality name
    stem_lower = stem.lower()
    for mod in check_order:
        if stem_lower.endswith(f"_{mod}") or stem_lower.endswith(f"-{mod}"):
            return mod

    # Check for segmentation mask
    if any(seg in stem.lower() for seg in ["_seg", "_label", "_mask", "_tumor"]):
        return "seg"

    return None


def discover_ucsf_patients(data_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Scan a UCSF-PDGM dataset directory for patients and their modalities.

    Expected structure:
        data_dir/
            UCSF-PDGM-0001/
                UCSF-PDGM-0001_T1_nifti.nii.gz
                UCSF-PDGM-0001_T1c_nifti.nii.gz
                UCSF-PDGM-0001_T2_nifti.nii.gz
                UCSF-PDGM-0001_FLAIR_nifti.nii.gz
            UCSF-PDGM-0002/
                ...

    Returns:
        {patient_id: {modality: filepath}}
    """
    if not os.path.isdir(data_dir):
        logger.error(f"UCSF-PDGM data directory not found: {data_dir}")
        return {}

    discovered = {}

    for entry in sorted(os.listdir(data_dir)):
        patient_dir = os.path.join(data_dir, entry)
        if not os.path.isdir(patient_dir) or entry.startswith("."):
            continue

        modalities = {}

        # Scan for NIfTI files in patient directory (may be nested)
        for root, _, files in os.walk(patient_dir):
            for fname in files:
                fpath = os.path.join(root, fname)
                fname_lower = fname.lower()

                if not any(fname_lower.endswith(ext) for ext in [".nii.gz", ".nii"]):
                    continue

                mod = detect_modality(fname)
                if mod and mod not in modalities:
                    modalities[mod] = fpath

        if modalities:
            discovered[entry] = modalities
            found_mods = list(modalities.keys())
            logger.info(f"Discovered {entry}: {found_mods}")

    logger.info(f"Total UCSF-PDGM patients discovered: {len(discovered)}")
    return discovered


def validate_patient_modalities(patient_dir: str) -> ValidationResult:
    """
    Validate modality alignment for a UCSF-PDGM patient.
    Checks spacing consistency, dimension matching, and affine alignment.
    """
    issues = []

    try:
        import nibabel as nib
    except ImportError:
        return ValidationResult(valid=True, issues=["nibabel not available for validation"])

    nifti_files = []
    for fname in os.listdir(patient_dir):
        fpath = os.path.join(patient_dir, fname)
        if fname.lower().endswith((".nii.gz", ".nii")) and os.path.isfile(fpath):
            nifti_files.append(fpath)

    if len(nifti_files) < 2:
        return ValidationResult(valid=True, issues=["Too few files for cross-validation"])

    # Load headers
    headers = {}
    for fpath in nifti_files:
        try:
            img = nib.load(fpath)
            headers[os.path.basename(fpath)] = {
                "shape": img.shape[:3],
                "spacing": tuple(round(float(s), 3) for s in img.header.get_zooms()[:3]),
                "affine": img.affine,
            }
        except Exception as e:
            issues.append(f"Could not read {os.path.basename(fpath)}: {e}")

    if len(headers) < 2:
        return ValidationResult(valid=len(issues) == 0, issues=issues)

    # Check consistency
    ref_name = list(headers.keys())[0]
    ref = headers[ref_name]

    spacing_ok = True
    dims_ok = True
    affine_ok = True

    for name, h in headers.items():
        if name == ref_name:
            continue

        if h["spacing"] != ref["spacing"]:
            spacing_ok = False
            issues.append(
                f"Spacing mismatch: {name} {h['spacing']} vs {ref_name} {ref['spacing']}"
            )

        if h["shape"] != ref["shape"]:
            dims_ok = False
            issues.append(
                f"Dimension mismatch: {name} {h['shape']} vs {ref_name} {ref['shape']}"
            )

        if not np.allclose(h["affine"], ref["affine"], atol=1e-3):
            affine_ok = False
            issues.append(f"Affine mismatch between {name} and {ref_name}")

    valid = len(issues) == 0

    return ValidationResult(
        valid=valid,
        issues=issues,
        spacing_consistent=spacing_ok,
        dimensions_consistent=dims_ok,
        affine_consistent=affine_ok,
    )


def convert_ucsf_patient(patient_modalities: Dict[str, str],
                          patient_id: str,
                          output_dir: str) -> str:
    """
    Convert UCSF-PDGM files into the standard pipeline naming convention.
    Creates symlinks (or copies on Windows) in the reconstructed directory.

    Returns:
        Path to the reconstructed patient directory.
    """
    recon_dir = os.path.join(output_dir, "reconstructed", patient_id)
    os.makedirs(recon_dir, exist_ok=True)

    for mod, src_path in patient_modalities.items():
        ext = ".nii.gz" if src_path.lower().endswith(".nii.gz") else ".nii"
        dst_path = os.path.join(recon_dir, f"{patient_id}_{mod}{ext}")

        if os.path.exists(dst_path):
            continue

        try:
            os.symlink(os.path.abspath(src_path), dst_path)
        except (OSError, NotImplementedError):
            # Symlinks may not be available on Windows without admin
            shutil.copy2(src_path, dst_path)

        logger.info(f"  {mod}: {os.path.basename(src_path)} → {os.path.basename(dst_path)}")

    return recon_dir
