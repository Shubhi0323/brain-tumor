"""
Pipeline Preflight Checks
============================
Validates pipeline inputs before execution begins.
Catches common problems early (missing files, corrupt inputs, missing deps).

Usage:
    from utils.preflight import run_preflight_checks
    ok, issues = run_preflight_checks(state)
    if not ok:
        for issue in issues:
            print(f"  PREFLIGHT FAIL: {issue}")
"""
import os
import shutil
from typing import Dict, List, Tuple


def _check_directory_exists(path: str, label: str) -> List[str]:
    """Verify a directory exists and is readable."""
    issues = []
    if not path:
        issues.append(f"{label}: path is empty or None")
    elif not os.path.exists(path):
        issues.append(f"{label}: directory does not exist: {path}")
    elif not os.path.isdir(path):
        issues.append(f"{label}: path is not a directory: {path}")
    elif not os.access(path, os.R_OK):
        issues.append(f"{label}: directory is not readable: {path}")
    return issues


def _check_modality_files(base_dir: str) -> Tuple[List[str], List[str]]:
    """Check which modality files are present in a patient directory."""
    found = []
    missing = []
    if not os.path.isdir(base_dir):
        return [], ["t1", "t1ce", "t2", "flair"]

    files_lower = [f.lower() for f in os.listdir(base_dir)]
    modality_patterns = {
        "t1": ["_t1.", "_t1_", "-t1.", "-t1_", "t1.nii"],
        "t1ce": ["_t1ce", "_t1c.", "_t1c_", "-t1ce", "-t1c.", "_t1gd", "_t1post"],
        "t2": ["_t2.", "_t2_", "-t2.", "-t2_", "t2.nii"],
        "flair": ["_flair", "-flair", "flair.nii"],
    }

    for mod, patterns in modality_patterns.items():
        mod_found = False
        for f in files_lower:
            if any(p in f for p in patterns):
                mod_found = True
                break
        if mod_found:
            found.append(mod)
        else:
            missing.append(mod)

    return found, missing


def _check_file_integrity(base_dir: str) -> List[str]:
    """Check for obviously corrupt files (zero-byte, truncated)."""
    issues = []
    if not os.path.isdir(base_dir):
        return issues

    for fname in os.listdir(base_dir):
        fpath = os.path.join(base_dir, fname)
        if not os.path.isfile(fpath):
            continue
        if fname.lower().endswith((".nii.gz", ".nii", ".dcm")):
            size = os.path.getsize(fpath)
            if size == 0:
                issues.append(f"Empty file detected: {fname} (0 bytes)")
            elif fname.endswith(".nii.gz") and size < 1000:
                issues.append(f"Suspiciously small NIfTI file: {fname} ({size} bytes)")
    return issues


def _check_disk_space(output_dir: str, min_gb: float = 1.0) -> List[str]:
    """Verify sufficient disk space for outputs."""
    issues = []
    try:
        target = output_dir if os.path.exists(output_dir) else os.path.dirname(output_dir)
        if not target:
            target = "."
        usage = shutil.disk_usage(target)
        free_gb = usage.free / (1024 ** 3)
        if free_gb < min_gb:
            issues.append(
                f"Low disk space: {free_gb:.1f} GB free "
                f"(minimum {min_gb:.1f} GB recommended)"
            )
    except Exception as e:
        issues.append(f"Could not check disk space: {e}")
    return issues


def _check_dependencies() -> List[str]:
    """Verify required Python packages are importable."""
    issues = []
    required = {
        "numpy": "numpy",
        "SimpleITK": "SimpleITK",
        "torch": "torch",
        "monai": "monai",
    }
    optional = {
        "nibabel": "nibabel",
        "scipy": "scipy",
        "pydicom": "pydicom",
    }

    for display_name, import_name in required.items():
        try:
            __import__(import_name)
        except ImportError:
            issues.append(f"Required dependency missing: {display_name}")

    for display_name, import_name in optional.items():
        try:
            __import__(import_name)
        except ImportError:
            # Optional deps are warnings, not hard failures
            pass

    return issues


def run_preflight_checks(state: Dict) -> Tuple[bool, List[str]]:
    """
    Run all preflight validation checks before pipeline execution.

    Args:
        state: Pipeline state dict with at least 'patient_id', 'base_dir', 'output_dir'

    Returns:
        (passed: bool, issues: list[str]) — True if all checks pass
    """
    issues = []

    patient_id = state.get("patient_id", "")
    base_dir = state.get("base_dir", "")
    output_dir = state.get("output_dir", "")

    # 1. Patient ID
    if not patient_id:
        issues.append("patient_id is empty")

    # 2. Input directory
    issues.extend(_check_directory_exists(base_dir, "base_dir"))

    # 3. Output directory writability
    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            issues.append(f"Cannot create output directory: {e}")

    # 4. Modality files
    if os.path.isdir(base_dir or ""):
        found, missing = _check_modality_files(base_dir)
        if len(found) == 0:
            issues.append(
                f"No MRI modality files found in {base_dir}. "
                f"Expected at least one of: t1, t1ce, t2, flair"
            )
        elif missing:
            # Partial availability is a warning, not a blocker
            pass  # Pipeline handles missing modalities with fallbacks

    # 5. File integrity
    if os.path.isdir(base_dir or ""):
        issues.extend(_check_file_integrity(base_dir))

    # 6. Disk space
    if output_dir:
        issues.extend(_check_disk_space(output_dir))

    # 7. Dependencies
    issues.extend(_check_dependencies())

    passed = len(issues) == 0
    return passed, issues
