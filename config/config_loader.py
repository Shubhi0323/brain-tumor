"""
Pipeline Configuration Loader
================================
Loads pipeline configuration from YAML with typed dataclass access.
Provides a singleton ``get_config()`` accessor used across all modules.

Environment variable overrides are supported:
  PIPELINE_PREPROCESSING__TARGET_SPACING="1.0,1.0,1.0"
  PIPELINE_SEGMENTATION__CC_MIN_VOXELS="1000"
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CONFIG_PATH = os.path.join(_CONFIG_DIR, "pipeline_config.yaml")


# ─── Typed Config Sections ───────────────────────────────────────────

@dataclass
class PreprocessingConfig:
    target_spacing: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    skull_strip_enabled: bool = True
    n4_max_iterations: List[int] = field(default_factory=lambda: [50, 50, 30, 20])
    intensity_normalization: str = "zscore"


@dataclass
class SegmentationConfig:
    cleanup_enabled: bool = True
    cleanup_kernel_radius: int = 1
    cc_filter_enabled: bool = True
    cc_min_voxels: int = 500
    cc_keep_top_n: int = 3
    quality_min_tumor_voxels: int = 100
    quality_max_tumor_fraction: float = 0.30
    quality_min_tumor_fraction: float = 0.0001


@dataclass
class ProgressionConfig:
    progression_threshold_mm3_per_day: float = 50.0
    regression_threshold_mm3_per_day: float = -50.0
    baseline_default_interval_days: int = 90
    doubling_times: Dict[str, int] = field(default_factory=lambda: {
        "glioblastoma": 50,
        "astrocytoma": 300,
        "oligodendroglioma": 400,
        "meningioma": 600,
        "glioma_nos": 200,
    })


@dataclass
class WHOClassificationConfig:
    differential_top_n: int = 5
    low_confidence_gap_threshold: float = 1.0
    calibration_method: str = "sigmoid"
    sigmoid_k: float = 1.2
    sigmoid_midpoint: float = 4.0
    default_molecular_markers: Dict[str, str] = field(default_factory=lambda: {
        "IDH": "unknown",
        "MGMT": "unknown",
        "1p19q": "unknown",
        "ATRX": "unknown",
        "TP53": "unknown",
    })


@dataclass
class SimilarityConfig:
    top_k: int = 5
    min_similarity_threshold: float = 0.5
    normalize_embeddings: bool = True
    embedding_dim: int = 768


@dataclass
class ErrorHandlingConfig:
    max_retries: int = 2
    log_level: str = "INFO"
    structured_log_file: str = "pipeline.log"
    graceful_degradation: bool = True


@dataclass
class UCSFPDGMConfig:
    modality_patterns: Dict[str, List[str]] = field(default_factory=lambda: {
        "t1": ["_T1_", "_t1_", "_T1."],
        "t1ce": ["_T1c_", "_t1c_", "_T1CE_", "_t1ce_", "_T1GD_", "_T1post_"],
        "t2": ["_T2_", "_t2_", "_T2."],
        "flair": ["_FLAIR_", "_flair_", "_Flair_"],
    })
    nifti_suffix: str = "_nifti"
    expected_extensions: List[str] = field(default_factory=lambda: [".nii.gz", ".nii"])


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration container."""
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    progression: ProgressionConfig = field(default_factory=ProgressionConfig)
    who_classification: WHOClassificationConfig = field(default_factory=WHOClassificationConfig)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    error_handling: ErrorHandlingConfig = field(default_factory=ErrorHandlingConfig)
    ucsf_pdgm: UCSFPDGMConfig = field(default_factory=UCSFPDGMConfig)


# ─── Loader Utilities ────────────────────────────────────────────────

def _apply_dict_to_dataclass(dc, data: dict):
    """Recursively apply a dict to a dataclass, updating only known fields."""
    if not isinstance(data, dict):
        return
    for key, value in data.items():
        if hasattr(dc, key):
            current = getattr(dc, key)
            # If the field is itself a dataclass, recurse
            if hasattr(current, "__dataclass_fields__"):
                _apply_dict_to_dataclass(current, value)
            else:
                setattr(dc, key, value)


def load_config(config_path: str = None) -> PipelineConfig:
    """
    Load pipeline configuration from YAML file.
    Falls back to defaults if YAML is unavailable or file is missing.
    """
    config = PipelineConfig()

    path = config_path or os.environ.get("PIPELINE_CONFIG_PATH", _DEFAULT_CONFIG_PATH)

    if YAML_AVAILABLE and os.path.isfile(path):
        try:
            with open(path, "r") as f:
                raw = yaml.safe_load(f)
            if isinstance(raw, dict):
                _apply_dict_to_dataclass(config, raw)
        except Exception as e:
            print(f"[CONFIG] Warning: Could not load {path}: {e}. Using defaults.")
    else:
        if not YAML_AVAILABLE:
            print("[CONFIG] PyYAML not installed. Using default configuration.")

    return config


# ─── Singleton Accessor ──────────────────────────────────────────────

_config_instance: Optional[PipelineConfig] = None


def get_config(config_path: str = None) -> PipelineConfig:
    """
    Get the singleton pipeline configuration.
    Loads from YAML on first call, returns cached instance thereafter.
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = load_config(config_path)
    return _config_instance


def reload_config(config_path: str = None) -> PipelineConfig:
    """Force reload configuration from YAML."""
    global _config_instance
    _config_instance = load_config(config_path)
    return _config_instance
