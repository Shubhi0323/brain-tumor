"""
Tumor Segmentation Node — SegResNet (MONAI)
=============================================
Uses MONAI's SegResNet with pretrained weights from the MONAI Model Zoo.

Production features:
  - Morphological cleanup (opening + closing)
  - Connected component filtering
  - Segmentation quality metrics and poor-quality detection
  - Structured logging
"""
import os
import subprocess
import numpy as np
import SimpleITK as sitk
import torch

from monai.networks.nets import SegResNet
from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    Spacingd, NormalizeIntensityd, ConcatItemsd, EnsureTyped,
    Activationsd, AsDiscreted,
)

from utils.pipeline_logger import get_logger

logger = get_logger("Segmentation")

# --- SegResNet architecture config ---
IN_CHANNELS = 4
OUT_CHANNELS = 3
BLOCKS_DOWN = [1, 2, 2, 4]
BLOCKS_UP = [1, 1, 1]
INIT_FILTERS = 16
DROPOUT_PROB = 0.2
ROI_SIZE = (240, 240, 160)
SW_BATCH_SIZE = 1
OVERLAP = 0.5

WEIGHTS_DIR = os.environ.get("DYNUNET_WEIGHTS_DIR", "weights")
WEIGHTS_FILE = os.environ.get("DYNUNET_WEIGHTS_FILE", "model_mri_segmentation.pt")
PRETRAINED_WEIGHTS_URL = os.environ.get("DYNUNET_WEIGHTS_URL", "")


def _get_seg_config():
    try:
        from config.config_loader import get_config
        return get_config().segmentation
    except Exception:
        from dataclasses import dataclass
        @dataclass
        class _D:
            cleanup_enabled: bool = True
            cleanup_kernel_radius: int = 1
            cc_filter_enabled: bool = True
            cc_min_voxels: int = 500
            cc_keep_top_n: int = 3
            quality_min_tumor_voxels: int = 100
            quality_max_tumor_fraction: float = 0.30
            quality_min_tumor_fraction: float = 0.0001
        return _D()


# ─── Post-processing ─────────────────────────────────────────────────

def morphological_cleanup(mask: np.ndarray, kernel_radius: int = 1) -> np.ndarray:
    """Apply binary opening (remove noise) then closing (fill holes)."""
    try:
        from scipy.ndimage import binary_opening, binary_closing, generate_binary_structure
        struct = generate_binary_structure(3, 1)
        cleaned = binary_opening(mask > 0, structure=struct, iterations=kernel_radius)
        cleaned = binary_closing(cleaned, structure=struct, iterations=kernel_radius)
        return cleaned.astype(np.int32)
    except ImportError:
        return mask


def connected_component_filter(mask: np.ndarray, min_voxels: int = 500,
                                keep_top_n: int = 3) -> np.ndarray:
    """Keep only the N largest connected components above min_voxels."""
    try:
        from scipy.ndimage import label
        labeled, num_features = label(mask > 0)
        if num_features == 0:
            return mask

        sizes = []
        for i in range(1, num_features + 1):
            sizes.append((i, int(np.sum(labeled == i))))

        sizes.sort(key=lambda x: x[1], reverse=True)
        keep_labels = [lbl for lbl, sz in sizes[:keep_top_n] if sz >= min_voxels]

        if not keep_labels:
            return mask

        filtered = np.isin(labeled, keep_labels).astype(np.int32)
        return filtered
    except ImportError:
        return mask


def compute_segmentation_quality(mask: np.ndarray) -> dict:
    """
    Compute segmentation quality metrics.
    Returns dict with quality indicators and poor-quality flag.
    """
    cfg = _get_seg_config()
    total_voxels = int(mask.size)
    tumor_voxels = int(np.sum(mask > 0))
    tumor_fraction = tumor_voxels / max(total_voxels, 1)

    # Count connected components
    num_components = 0
    largest_component_fraction = 0.0
    try:
        from scipy.ndimage import label
        labeled, num_components = label(mask > 0)
        if num_components > 0 and tumor_voxels > 0:
            sizes = [int(np.sum(labeled == i)) for i in range(1, num_components + 1)]
            largest_component_fraction = max(sizes) / tumor_voxels
    except ImportError:
        pass

    is_poor_quality = (
        tumor_voxels < cfg.quality_min_tumor_voxels
        or tumor_fraction > cfg.quality_max_tumor_fraction
        or tumor_fraction < cfg.quality_min_tumor_fraction
        or tumor_voxels == 0
    )

    return {
        "tumor_voxels": tumor_voxels,
        "total_voxels": total_voxels,
        "tumor_fraction": round(tumor_fraction, 6),
        "num_components": num_components,
        "largest_component_fraction": round(largest_component_fraction, 4),
        "is_poor_quality": is_poor_quality,
    }


# ─── Model loading ───────────────────────────────────────────────────

def build_segresnet_model() -> SegResNet:
    return SegResNet(
        blocks_down=BLOCKS_DOWN, blocks_up=BLOCKS_UP,
        init_filters=INIT_FILTERS, in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS, dropout_prob=DROPOUT_PROB,
    )


def find_weights_path() -> str | None:
    candidates = [
        os.path.join(WEIGHTS_DIR, WEIGHTS_FILE),
        os.path.join(os.path.dirname(__file__), "..", "weights", WEIGHTS_FILE),
        os.path.join(os.path.expanduser("~"), ".monai", WEIGHTS_FILE),
    ]
    for path in candidates:
        path = os.path.abspath(path)
        if os.path.isfile(path):
            return path
    return None


def download_weights() -> str | None:
    if not PRETRAINED_WEIGHTS_URL:
        logger.info("DYNUNET_WEIGHTS_URL is not set; skipping weight download.")
        return None

    dest_dir = os.path.abspath(WEIGHTS_DIR)
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, WEIGHTS_FILE)

    if os.path.isfile(dest_path):
        file_size = os.path.getsize(dest_path)
        if file_size > 1_000_000:
            logger.info(f"Weights already downloaded: {dest_path} ({file_size / 1e6:.1f} MB)")
            return dest_path
        else:
            os.remove(dest_path)

    logger.info(f"Downloading SegResNet weights from {PRETRAINED_WEIGHTS_URL}")
    try:
        result = subprocess.run(
            ["wget", "-q", "--show-progress", "-O", dest_path, PRETRAINED_WEIGHTS_URL],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode == 0 and os.path.isfile(dest_path):
            logger.info(f"Download complete: {os.path.getsize(dest_path) / 1e6:.1f} MB")
            return dest_path
    except Exception as e:
        logger.warning(f"wget download failed: {e}")

    try:
        import urllib.request
        urllib.request.urlretrieve(PRETRAINED_WEIGHTS_URL, dest_path)
        if os.path.isfile(dest_path) and os.path.getsize(dest_path) > 1_000_000:
            return dest_path
    except Exception as e:
        logger.warning(f"urllib fallback failed: {e}")

    if os.path.isfile(dest_path):
        os.remove(dest_path)
    return None


def load_model_with_weights(device: torch.device) -> SegResNet | None:
    weights_path = find_weights_path()
    if weights_path is None:
        weights_path = download_weights()
    if weights_path is None:
        logger.info("No pretrained SegResNet weights available.")
        return None

    logger.info(f"Loading SegResNet weights from: {weights_path}")
    model = build_segresnet_model()
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "net" in checkpoint:
        state_dict = checkpoint["net"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    logger.info("SegResNet model loaded successfully.")
    return model


# ─── Inference ────────────────────────────────────────────────────────

def build_preprocessing_transforms(keys: list[str]):
    return Compose([
        LoadImaged(keys=keys, image_only=True),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        NormalizeIntensityd(keys=keys, nonzero=True, channel_wise=True),
        ConcatItemsd(keys=keys, name="image"),
        EnsureTyped(keys=["image"], dtype=torch.float32),
    ])


def find_modality_files(base_dir: str) -> dict[str, str] | None:
    modality_map: dict[str, str | None] = {"t1": None, "t1ce": None, "t2": None, "flair": None}
    aliases = {
        "t1": ["_t1", "-t1"],
        "t1ce": ["_t1ce", "-t1ce", "_t1c", "-t1c", "_t1gd", "-t1gd", "_t1post", "-t1post"],
        "t2": ["_t2", "-t2"],
        "flair": ["_flair", "-flair"]
    }

    try:
        dir_contents = os.listdir(base_dir)
    except OSError:
        return None

    for f in dir_contents:
        f_lower = f.lower()
        if not (f_lower.endswith(".nii.gz") or f_lower.endswith(".nii")):
            continue
        stem = f_lower.replace(".nii.gz", "").replace(".nii", "")
        if stem.endswith("_seg") or stem == "seg":
            continue

        for mod in modality_map:
            if modality_map[mod] is not None:
                continue
            mod_aliases = aliases[mod]
            if stem == mod:
                modality_map[mod] = os.path.join(base_dir, f)
                break
            if any(stem.endswith(alias) for alias in mod_aliases):
                modality_map[mod] = os.path.join(base_dir, f)
                break
            if any((alias + "_") in stem or (alias + "-") in stem for alias in mod_aliases):
                modality_map[mod] = os.path.join(base_dir, f)
                break

    if all(v is not None for v in modality_map.values()):
        return modality_map
    missing = [k for k, v in modality_map.items() if v is None]
    logger.info(f"Missing modalities: {missing}")
    return None


def run_segresnet_inference(model: SegResNet, base_dir: str,
                            device: torch.device) -> np.ndarray | None:
    modality_files = find_modality_files(base_dir)
    if modality_files is None:
        return None

    keys = ["t1", "t1ce", "t2", "flair"]
    transforms = build_preprocessing_transforms(keys)
    data = {mod: modality_files[mod] for mod in keys}

    try:
        transformed = transforms(data)
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return None

    image_tensor = transformed["image"].unsqueeze(0).to(device)
    inferer = SlidingWindowInferer(
        roi_size=ROI_SIZE, sw_batch_size=SW_BATCH_SIZE,
        overlap=OVERLAP, mode="gaussian",
    )

    with torch.no_grad():
        output = inferer(image_tensor, model)

    post = Compose([
        Activationsd(keys="pred", sigmoid=True),
        AsDiscreted(keys="pred", threshold=0.5),
    ])
    post_data = post({"pred": output.squeeze(0)})
    pred = post_data["pred"].cpu().numpy()
    binary_mask = (pred.sum(axis=0) > 0).astype(np.int32)
    return binary_mask


def find_seg_file(base_dir: str) -> str | None:
    try:
        for f in os.listdir(base_dir):
            fname_lower = f.lower()
            if fname_lower.endswith(".nii.gz") or fname_lower.endswith(".nii"):
                stem = fname_lower.replace(".nii.gz", "").replace(".nii", "")
                if stem.endswith("_seg") or stem == "seg":
                    return os.path.join(base_dir, f)
    except OSError:
        pass
    return None


def load_and_binarize_mask(seg_path: str) -> tuple:
    seg_image = sitk.ReadImage(seg_path, sitk.sitkInt32)
    seg_arr = sitk.GetArrayFromImage(seg_image)
    binary_mask = (seg_arr > 0).astype(np.int32)
    binary_sitk = sitk.GetImageFromArray(binary_mask)
    binary_sitk.CopyInformation(seg_image)
    return binary_mask, binary_sitk


def heuristic_segmentation_from_preprocessed(
        preprocessed_path: str,
        ref_image: sitk.Image | None = None) -> tuple[np.ndarray, sitk.Image] | None:
    if not preprocessed_path or not os.path.exists(preprocessed_path):
        return None
    try:
        data = np.load(preprocessed_path)
    except Exception:
        return None
    if data.ndim != 4 or data.shape[0] == 0:
        return None

    channel_idx = 1 if data.shape[0] > 1 else 0
    vol = data[channel_idx].astype(np.float32)
    brain = vol != 0
    if int(np.sum(brain)) < 100:
        return None

    vals = vol[brain]
    thr = np.percentile(vals, 99.2)
    mask = (vol >= thr).astype(np.int32)

    try:
        from scipy import ndimage
        labeled, n = ndimage.label(mask)
        if n > 0:
            sizes = ndimage.sum(mask, labeled, range(1, n + 1))
            keep = [i + 1 for i, s in enumerate(sizes) if s >= 250]
            if keep:
                mask = np.isin(labeled, keep).astype(np.int32)
    except Exception:
        pass

    if int(np.sum(mask > 0)) < 100:
        return None

    mask_sitk = sitk.GetImageFromArray(mask)
    if ref_image is not None:
        mask_sitk.SetOrigin(ref_image.GetOrigin())
        mask_sitk.SetSpacing(ref_image.GetSpacing())
        mask_sitk.SetDirection(ref_image.GetDirection())

    return mask, mask_sitk


def _apply_postprocessing(binary_mask: np.ndarray) -> np.ndarray:
    """Apply morphological cleanup and CC filtering based on config."""
    cfg = _get_seg_config()

    if cfg.cleanup_enabled:
        binary_mask = morphological_cleanup(binary_mask, cfg.cleanup_kernel_radius)
        logger.info("Applied morphological cleanup")

    if cfg.cc_filter_enabled:
        binary_mask = connected_component_filter(
            binary_mask, cfg.cc_min_voxels, cfg.cc_keep_top_n
        )
        logger.info("Applied connected component filtering")

    return binary_mask


def run_segmentation(state: dict) -> dict:
    """
    LangGraph node: Extract tumor segmentation mask.
    Priority: 1. SegResNet  2. Ground-truth  3. Heuristic fallback
    Post-processing: morphological cleanup + CC filtering + quality check
    """
    patient_id = state["patient_id"]
    base_dir = state["base_dir"]
    output_dir = state["output_dir"]
    errors = list(state.get("errors", []))
    preprocessed_path = state.get("preprocessed_path")

    seg_dir = os.path.join(output_dir, "segmentation")
    os.makedirs(seg_dir, exist_ok=True)

    logger.log_stage_start(patient_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}", patient_id=patient_id)

    binary_mask = None
    source = None

    # --- Strategy 1: SegResNet ---
    try:
        model = load_model_with_weights(device)
        if model is not None:
            logger.info("Attempting SegResNet inference...", patient_id=patient_id)
            binary_mask = run_segresnet_inference(model, base_dir, device)
            if binary_mask is not None:
                source = "segresnet"
    except Exception as e:
        logger.warning(f"SegResNet failed: {e}", patient_id=patient_id)
        errors.append(f"SegResNet inference failed for {patient_id}: {e}")

    # --- Strategy 2: Ground-truth ---
    if binary_mask is None:
        seg_path = find_seg_file(base_dir)
        if seg_path is not None:
            try:
                logger.info(f"Using ground-truth: {os.path.basename(seg_path)}", patient_id=patient_id)
                binary_mask, _ = load_and_binarize_mask(seg_path)
                source = "ground_truth"
            except Exception as e:
                msg = f"Error loading GT seg for {patient_id}: {e}"
                logger.error(msg, patient_id=patient_id)
                errors.append(msg)

    # --- Strategy 3: Heuristic ---
    if binary_mask is None:
        use_heuristic = os.environ.get(
            "ALLOW_HEURISTIC_SEGMENTATION_FALLBACK", "1"
        ).strip().lower() in ("1", "true", "yes")
        if use_heuristic:
            ref_image = None
            modality_files = find_modality_files(base_dir)
            if modality_files:
                try:
                    ref_image = sitk.ReadImage(next(iter(modality_files.values())))
                except Exception:
                    pass
            heur = heuristic_segmentation_from_preprocessed(preprocessed_path, ref_image)
            if heur is not None:
                binary_mask, _ = heur
                source = "heuristic"
                errors.append(f"Used heuristic segmentation fallback for {patient_id}")

    # --- No segmentation available ---
    if binary_mask is None:
        msg = f"No segmentation available for {patient_id}"
        errors.append(msg)
        logger.error(msg, patient_id=patient_id)
        return {**state, "segmentation_path": None, "errors": errors}

    # --- Post-processing ---
    binary_mask = _apply_postprocessing(binary_mask)

    # --- Quality check ---
    quality = compute_segmentation_quality(binary_mask)
    tumor_voxels = quality["tumor_voxels"]
    total_voxels = quality["total_voxels"]
    logger.info(
        f"{source} segmentation: {tumor_voxels}/{total_voxels} tumor voxels "
        f"({100 * quality['tumor_fraction']:.2f}%)",
        patient_id=patient_id,
    )

    if quality["is_poor_quality"]:
        msg = (
            f"Poor segmentation quality for {patient_id}: "
            f"fraction={quality['tumor_fraction']:.4f}, "
            f"components={quality['num_components']}"
        )
        logger.warning(msg, patient_id=patient_id)
        errors.append(msg)

    # --- Save ---
    ref_image = None
    modality_files = find_modality_files(base_dir)
    if modality_files:
        try:
            ref_image = sitk.ReadImage(next(iter(modality_files.values())))
        except Exception:
            pass

    binary_sitk = sitk.GetImageFromArray(binary_mask)
    if ref_image is not None:
        binary_sitk.SetOrigin(ref_image.GetOrigin())
        binary_sitk.SetSpacing(ref_image.GetSpacing())
        binary_sitk.SetDirection(ref_image.GetDirection())

    save_path = os.path.join(seg_dir, f"{patient_id}_seg.nii.gz")
    sitk.WriteImage(binary_sitk, save_path)
    logger.info(f"Saved segmentation: {save_path}", patient_id=patient_id)

    return {**state, "segmentation_path": save_path, "errors": errors}
