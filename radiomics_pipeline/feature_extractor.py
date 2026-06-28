"""
Radiomics Feature Extraction Node
===================================
Uses PyRadiomics to extract shape, texture (GLCM), and intensity features
from the tumor segmentation mask overlaid on the original MRI.

NOTE: This module lives in radiomics_pipeline/ (not radiomics/) to avoid
shadowing the installed pyradiomics library, which also imports as
`import radiomics`. With the folder renamed, pyradiomics loads cleanly
with zero import hacks.
"""
import os
import json
import numpy as np
import SimpleITK as sitk

try:
    from radiomics import featureextractor
    PYRADIOMICS_AVAILABLE = True
except (ImportError, Exception):
    PYRADIOMICS_AVAILABLE = False


def get_radiomics_params() -> dict:
    """Define PyRadiomics extraction parameters."""
    params = {
        "setting": {
            "binWidth": 25,
            "resampledPixelSpacing": None,  # Already resampled
            "interpolator": "sitkBSpline",
            "minimumROISize": 2,
        },
        "featureClass": {
            "shape": [],       # All shape features
            "firstorder": [],  # Intensity statistics
            "glcm": [],        # Texture features
        }
    }
    return params


def extract_features_manual(image_arr: np.ndarray, mask_arr: np.ndarray) -> dict:
    """
    Manual feature extraction fallback if PyRadiomics is unavailable.
    Extracts basic shape, intensity, and simple texture proxies.
    """
    features = {}
    tumor_voxels = image_arr[mask_arr > 0]

    if len(tumor_voxels) == 0:
        return {"error": "No tumor voxels found"}

    # Shape features
    features["shape_volume_voxels"] = int(np.sum(mask_arr > 0))
    features["shape_volume_mm3"] = float(np.sum(mask_arr > 0))  # Approximate (1mm³ spacing)

    # Surface area approximation (count boundary voxels)
    from scipy import ndimage
    eroded = ndimage.binary_erosion(mask_arr > 0)
    surface = (mask_arr > 0).astype(int) - eroded.astype(int)
    features["shape_surface_area"] = float(np.sum(surface > 0))

    # Maximum diameter (approximate via bounding box diagonal)
    coords = np.argwhere(mask_arr > 0)
    if len(coords) > 1:
        bbox_min = coords.min(axis=0)
        bbox_max = coords.max(axis=0)
        features["shape_max_diameter"] = float(np.linalg.norm(bbox_max - bbox_min))
    else:
        features["shape_max_diameter"] = 0.0

    # Sphericity approximation
    volume = features["shape_volume_voxels"]
    sa = features["shape_surface_area"]
    if sa > 0:
        features["shape_sphericity"] = float(
            (np.pi ** (1/3)) * ((6 * volume) ** (2/3)) / sa
        )
    else:
        features["shape_sphericity"] = 0.0

    # Intensity / First-order features
    features["intensity_mean"] = float(np.mean(tumor_voxels))
    features["intensity_variance"] = float(np.var(tumor_voxels))
    features["intensity_std"] = float(np.std(tumor_voxels))
    features["intensity_skewness"] = float(
        np.mean(((tumor_voxels - np.mean(tumor_voxels)) / (np.std(tumor_voxels) + 1e-8)) ** 3)
    )
    features["intensity_kurtosis"] = float(
        np.mean(((tumor_voxels - np.mean(tumor_voxels)) / (np.std(tumor_voxels) + 1e-8)) ** 4) - 3
    )
    features["intensity_min"] = float(np.min(tumor_voxels))
    features["intensity_max"] = float(np.max(tumor_voxels))
    features["intensity_median"] = float(np.median(tumor_voxels))
    features["intensity_energy"] = float(np.sum(tumor_voxels ** 2))

    return features


def extract_features_pyradiomics(image_sitk: sitk.Image, mask_sitk: sitk.Image) -> dict:
    """Extract features using PyRadiomics library."""
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableFeatureClassByName("shape")
    extractor.enableFeatureClassByName("firstorder")
    extractor.enableFeatureClassByName("glcm")

    result = extractor.execute(image_sitk, mask_sitk)

    features = {}
    for key, val in result.items():
        if not key.startswith("diagnostics_"):
            features[key] = float(val) if hasattr(val, '__float__') else str(val)

    return features


def extract_radiomics(state: dict) -> dict:
    """
    LangGraph node: Extract radiomics features from the segmentation mask.
    Uses PyRadiomics if available, otherwise falls back to manual extraction.
    """
    patient_id = state["patient_id"]
    base_dir = state["base_dir"]
    output_dir = state["output_dir"]
    segmentation_path = state.get("segmentation_path")
    preprocessed_path = state.get("preprocessed_path")
    errors = list(state.get("errors", []))

    rad_dir = os.path.join(output_dir, "radiomics")
    os.makedirs(rad_dir, exist_ok=True)

    print(f"[Radiomics] Processing patient: {patient_id}")

    if segmentation_path is None:
        msg = f"No segmentation mask for {patient_id}, skipping radiomics."
        print(f"  [WARNING] {msg}")
        errors.append(msg)
        return {**state, "radiomics_features": {}, "errors": errors}

    try:
        # Load segmentation mask
        mask_sitk = sitk.ReadImage(segmentation_path, sitk.sitkInt32)
        mask_arr = sitk.GetArrayFromImage(mask_sitk)

        # Load image data (use T1CE channel for feature extraction)
        if preprocessed_path and os.path.exists(preprocessed_path):
            data = np.load(preprocessed_path)
            t1ce_arr = data[1]  # T1CE channel
        else:
            from preprocessing.mri_prep import find_modality_file
            t1ce_path = find_modality_file(base_dir, "t1ce")
            if t1ce_path:
                t1ce_sitk = sitk.ReadImage(t1ce_path, sitk.sitkFloat32)
                t1ce_arr = sitk.GetArrayFromImage(t1ce_sitk)
            else:
                msg = f"No T1CE data found for {patient_id}"
                errors.append(msg)
                return {**state, "radiomics_features": {}, "errors": errors}

        # Ensure shapes match
        if t1ce_arr.shape != mask_arr.shape:
            min_shape = tuple(min(s1, s2) for s1, s2 in zip(t1ce_arr.shape, mask_arr.shape))
            t1ce_arr = t1ce_arr[:min_shape[0], :min_shape[1], :min_shape[2]]
            mask_arr = mask_arr[:min_shape[0], :min_shape[1], :min_shape[2]]

        if PYRADIOMICS_AVAILABLE:
            print("  Using PyRadiomics for feature extraction.")
            image_sitk = sitk.GetImageFromArray(t1ce_arr.astype(np.float32))
            mask_sitk_aligned = sitk.GetImageFromArray(mask_arr.astype(np.int32))
            image_sitk.CopyInformation(mask_sitk_aligned)
            features = extract_features_pyradiomics(image_sitk, mask_sitk_aligned)
        else:
            print("  PyRadiomics not available. Using manual feature extraction.")
            features = extract_features_manual(t1ce_arr, mask_arr)

        save_path = os.path.join(rad_dir, f"{patient_id}_radiomics.json")
        with open(save_path, "w") as f:
            json.dump(features, f, indent=2)

        print(f"  Extracted {len(features)} features. Saved to: {save_path}")
        return {**state, "radiomics_features": features, "errors": errors}

    except Exception as e:
        msg = f"Error extracting radiomics for {patient_id}: {e}"
        print(f"  [ERROR] {msg}")
        errors.append(msg)
        return {**state, "radiomics_features": {}, "errors": errors}
