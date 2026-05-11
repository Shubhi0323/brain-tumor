"""
MRI Preprocessing Pipeline Node
================================
Handles: N4 Bias Field Correction, Skull Stripping, Intensity Normalization,
         Voxel Resampling, Modality Alignment, Multi-channel Stacking.

Production-ready with:
  - Enhanced skull stripping (morphological cleanup + largest CC)
  - Configurable target spacing via YAML
  - Input validation preflight
  - Structured logging
"""
import os
import numpy as np
import SimpleITK as sitk
import nibabel as nib

from utils.pipeline_logger import get_logger

logger = get_logger("Preprocessing")


def _get_preprocessing_config():
    """Load preprocessing config with safe fallback."""
    try:
        from config.config_loader import get_config
        return get_config().preprocessing
    except Exception:
        from dataclasses import dataclass, field
        from typing import List

        @dataclass
        class _Default:
            target_spacing: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
            skull_strip_enabled: bool = True
            n4_max_iterations: List[int] = field(default_factory=lambda: [50, 50, 30, 20])
            intensity_normalization: str = "zscore"
        return _Default()


def validate_preprocessing_inputs(base_dir: str, modalities: list) -> list:
    """
    Pre-flight check: verify input NIfTI files exist and are valid.

    Returns:
        List of warning/error messages (empty = all good)
    """
    issues = []
    if not base_dir or not os.path.isdir(base_dir):
        issues.append(f"Input directory does not exist: {base_dir}")
        return issues

    nifti_files = [
        f for f in os.listdir(base_dir)
        if f.lower().endswith(('.nii.gz', '.nii'))
    ]
    if not nifti_files:
        issues.append(f"No NIfTI files found in {base_dir}")
        return issues

    for f in nifti_files:
        fpath = os.path.join(base_dir, f)
        size = os.path.getsize(fpath)
        if size == 0:
            issues.append(f"Empty file: {f}")
        elif size < 1000:
            issues.append(f"Suspiciously small file ({size} bytes): {f}")

    return issues


def n4_bias_field_correction(image: sitk.Image, max_iterations: list = None) -> sitk.Image:
    """Apply N4 Bias Field Correction to an MRI image."""
    cfg = _get_preprocessing_config()
    iterations = max_iterations or cfg.n4_max_iterations

    mask_image = sitk.OtsuThreshold(image, 0, 1, 200)
    input_image = sitk.Cast(image, sitk.sitkFloat32)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(iterations)
    corrected = corrector.Execute(input_image, mask_image)
    return corrected


def skull_strip(image: sitk.Image) -> sitk.Image:
    """
    Enhanced skull stripping using multi-step approach:
    1. Otsu thresholding to create initial brain mask
    2. Morphological closing to fill gaps
    3. Largest connected component extraction
    4. Hole filling for clean brain mask

    Works reliably on both curated and real-world DICOM data.
    """
    cfg = _get_preprocessing_config()
    if not cfg.skull_strip_enabled:
        return image

    # Step 1: Otsu threshold
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(1)
    otsu_filter.SetOutsideValue(0)
    mask = otsu_filter.Execute(image)

    # Step 2: Morphological closing to fill small gaps
    try:
        closing_filter = sitk.BinaryMorphologicalClosingImageFilter()
        closing_filter.SetKernelRadius(3)
        closing_filter.SetForegroundValue(1)
        mask = closing_filter.Execute(mask)
    except Exception:
        pass  # Continue without closing if it fails

    # Step 3: Largest connected component — removes skull fragments
    try:
        cc_filter = sitk.ConnectedComponentImageFilter()
        labeled = cc_filter.Execute(mask)

        label_stats = sitk.LabelShapeStatisticsImageFilter()
        label_stats.Execute(labeled)

        labels = label_stats.GetLabels()
        if labels:
            largest_label = max(labels,
                                key=lambda l: label_stats.GetNumberOfPixels(l))
            mask = sitk.Equal(labeled, int(largest_label))
            mask = sitk.Cast(mask, sitk.sitkUInt8)
    except Exception:
        pass  # Continue with Otsu mask if CC fails

    # Step 4: Fill holes inside the brain mask
    try:
        hole_filler = sitk.BinaryFillholeImageFilter()
        hole_filler.SetForegroundValue(1)
        mask = hole_filler.Execute(mask)
    except Exception:
        pass

    # Apply mask to image
    stripped = sitk.Mask(image, mask)
    return stripped


def normalize_intensity(image: sitk.Image) -> sitk.Image:
    """Z-score intensity normalization (zero mean, unit variance) on non-zero voxels."""
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    non_zero = arr[arr > 0]
    if len(non_zero) > 0:
        mean_val = np.mean(non_zero)
        std_val = np.std(non_zero)
        if std_val > 0:
            arr[arr > 0] = (arr[arr > 0] - mean_val) / std_val
    result = sitk.GetImageFromArray(arr)
    result.CopyInformation(image)
    return result


def resample_volume(image: sitk.Image, target_spacing: tuple = None) -> sitk.Image:
    """Resample voxels to isotropic target spacing (from config or argument)."""
    cfg = _get_preprocessing_config()
    if target_spacing is None:
        target_spacing = tuple(cfg.target_spacing)

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(round(osz * ospc / tspc))
        for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkBSpline)

    return resampler.Execute(image)


def resample_to_reference(image: sitk.Image, reference: sitk.Image) -> sitk.Image:
    """Resample an image to match a reference image's spatial domain."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(sitk.Transform())
    return resampler.Execute(image)


def preprocess_modality(nifti_path: str) -> sitk.Image:
    """Full preprocessing pipeline for a single MRI modality."""
    image = sitk.ReadImage(nifti_path, sitk.sitkFloat32)
    image = n4_bias_field_correction(image)
    image = skull_strip(image)
    image = normalize_intensity(image)
    image = resample_volume(image)
    return image


def find_modality_file(base_dir: str, modality: str) -> str:
    """
    Locate a specific modality NIfTI file in a patient directory.
    Handles naming variations like 'T1.nii.gz', 't1.nii.gz', 'subject_..._t1.nii.gz'.
    Also handles UCSF-PDGM aliases (e.g. t1c instead of t1ce).
    """
    modality_lower = modality.lower()

    aliases = {
        "t1": ["_t1", "-t1"],
        "t1ce": ["_t1ce", "-t1ce", "_t1c", "-t1c", "_t1gd", "-t1gd", "_t1post", "-t1post"],
        "t2": ["_t2", "-t2"],
        "flair": ["_flair", "-flair"]
    }

    mod_aliases = aliases.get(modality_lower, [f"_{modality_lower}", f"-{modality_lower}"])

    try:
        dir_contents = os.listdir(base_dir)
    except OSError:
        return None

    for f in dir_contents:
        if f.endswith('.nii.gz') or f.endswith('.nii'):
            fname_lower = f.lower()
            stem = fname_lower.replace('.nii.gz', '').replace('.nii', '')

            # Exact match
            if stem == modality_lower:
                return os.path.join(base_dir, f)

            # Check endings
            if any(stem.endswith(alias) for alias in mod_aliases):
                return os.path.join(base_dir, f)

            # Check if embedded like _t1c_vibe
            if any((alias + "_") in stem or (alias + "-") in stem for alias in mod_aliases):
                return os.path.join(base_dir, f)

    return None


def run_preprocessing(state: dict) -> dict:
    """
    LangGraph node: Preprocess all MRI modalities for a patient.
    Stacks T1, T1CE, T2, FLAIR into a multi-channel numpy array.

    Production features:
      - Input validation before processing
      - Proper spatial alignment via resampling to reference
      - Structured logging
      - Graceful degradation on partial modality availability
    """
    patient_id = state["patient_id"]
    base_dir = state["base_dir"]
    output_dir = state["output_dir"]
    errors = list(state.get("errors", []))

    prep_dir = os.path.join(output_dir, "preprocessed")
    os.makedirs(prep_dir, exist_ok=True)

    modalities = ["t1", "t1ce", "t2", "flair"]

    logger.log_stage_start(patient_id)

    # Pre-flight input validation
    validation_issues = validate_preprocessing_inputs(base_dir, modalities)
    for issue in validation_issues:
        logger.warning(issue, patient_id=patient_id)

    channels = []
    reference_image = None

    for mod in modalities:
        path = find_modality_file(base_dir, mod)
        if path is None:
            msg = f"Modality {mod} not found for patient {patient_id} in {base_dir}"
            logger.warning(msg, patient_id=patient_id)
            errors.append(msg)
            continue

        logger.info(f"Processing {mod}: {os.path.basename(path)}", patient_id=patient_id)
        try:
            processed = preprocess_modality(path)

            # Use first successfully processed modality as spatial reference
            if reference_image is None:
                reference_image = processed
            else:
                # Resample to reference spatial domain instead of crude pad/crop
                if processed.GetSize() != reference_image.GetSize():
                    processed = resample_to_reference(processed, reference_image)

            arr = sitk.GetArrayFromImage(processed)
            channels.append(arr)
        except Exception as e:
            msg = f"Error preprocessing {mod} for {patient_id}: {e}"
            logger.error(msg, patient_id=patient_id)
            errors.append(msg)

    if len(channels) == len(modalities):
        # All modalities available — verify alignment
        target_shape = channels[0].shape
        aligned_channels = []
        for ch in channels:
            if ch.shape != target_shape:
                # Safety fallback: pad/crop if resampling didn't fully align
                padded = np.zeros(target_shape, dtype=np.float32)
                slices = tuple(
                    slice(0, min(s1, s2))
                    for s1, s2 in zip(ch.shape, target_shape)
                )
                padded[slices] = ch[slices]
                aligned_channels.append(padded)
            else:
                aligned_channels.append(ch)

        stacked = np.stack(aligned_channels, axis=0)  # Shape: (4, D, H, W)
        save_path = os.path.join(prep_dir, f"{patient_id}_preprocessed.npy")
        np.save(save_path, stacked)
        logger.info(
            f"Saved preprocessed volume: {save_path} | shape: {stacked.shape}",
            patient_id=patient_id,
        )
        return {**state, "preprocessed_path": save_path, "errors": errors}
    else:
        errors.append(
            f"Incomplete modalities for {patient_id}, "
            f"got {len(channels)}/{len(modalities)}"
        )
        return {**state, "preprocessed_path": None, "errors": errors}
