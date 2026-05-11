"""
Tumor Progression Estimator
==============================
Estimates tumor progression state with longitudinal tracking.

Production features:
  - Baseline scan registration when no prior exists
  - Longitudinal scan history tracking
  - Rule-based volume growth fallback using WHO-grade doubling times
  - Configurable thresholds via YAML
"""
import os
import json
import glob
from datetime import datetime
from typing import Optional, Tuple

from utils.pipeline_logger import get_logger

logger = get_logger("Progression")


def _get_progression_config():
    try:
        from config.config_loader import get_config
        return get_config().progression
    except Exception:
        from dataclasses import dataclass, field
        from typing import Dict
        @dataclass
        class _D:
            progression_threshold_mm3_per_day: float = 50.0
            regression_threshold_mm3_per_day: float = -50.0
            baseline_default_interval_days: int = 90
            doubling_times: Dict[str, int] = field(default_factory=lambda: {
                "glioblastoma": 50, "astrocytoma": 300,
                "oligodendroglioma": 400, "meningioma": 600, "glioma_nos": 200,
            })
        return _D()


def register_baseline_scan(patient_id: str, current_volume: float,
                           output_dir: str) -> dict:
    """
    Register the current scan as the baseline when no prior scan exists.
    Saves baseline data and returns baseline progression result.
    """
    clinical_dir = os.path.join(output_dir, "clinical_features")
    os.makedirs(clinical_dir, exist_ok=True)

    baseline = {
        "patient_id": patient_id,
        "tumor_volume": current_volume,
        "timestamp": datetime.now().isoformat(),
        "scan_number": 1,
    }

    history_path = os.path.join(clinical_dir, f"{patient_id}_history.json")
    history = []
    if os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                history = json.load(f)
        except Exception:
            history = []

    history.append(baseline)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"Registered baseline scan for {patient_id}: {current_volume:.0f} mm³")

    return {
        "current_volume": current_volume,
        "previous_volume": None,
        "time_interval_days": None,
        "progression_state": "baseline",
        "growth_rate": None,
        "size_change_pct": None,
        "reasoning": (
            f"Baseline scan registered for patient {patient_id}. "
            f"Current tumor volume: {current_volume:.0f} mm³. "
            "No prior scan available for comparison. "
            "Follow-up imaging recommended at 3-month interval."
        ),
    }


def load_patient_scan_history(patient_id: str, output_dir: str) -> list:
    """
    Load all prior scan records for longitudinal tracking.
    Searches clinical profiles and history files, sorted by timestamp.
    """
    history = []

    # Check history file
    clinical_dir = os.path.join(output_dir, "clinical_features")
    history_path = os.path.join(clinical_dir, f"{patient_id}_history.json")
    if os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                history = json.load(f)
        except Exception:
            pass

    # Also check patient memory store
    memory_dir = os.path.join(output_dir, "patient_memory")
    memory_path = os.path.join(memory_dir, f"{patient_id}_memory.json")
    if os.path.exists(memory_path):
        try:
            with open(memory_path, "r") as f:
                memory_data = json.load(f)
            past_reports = memory_data.get("past_reports", [])
            for report in past_reports:
                tumor_summary = report.get("tumor_summary", {})
                morphology = tumor_summary.get("morphology", {})
                volume = morphology.get("tumor_volume")
                if volume is not None:
                    entry = {
                        "patient_id": patient_id,
                        "tumor_volume": volume,
                        "timestamp": report.get("report_metadata", {}).get("generated_at", ""),
                    }
                    # Avoid duplicates
                    if not any(h.get("tumor_volume") == volume for h in history):
                        history.append(entry)
        except Exception:
            pass

    # Sort by timestamp
    history.sort(key=lambda x: x.get("timestamp", ""))
    return history


def estimate_progression_from_grade(current_volume: float,
                                     tumor_grade: str = "glioma_nos") -> dict:
    """
    Rule-based volume growth fallback using WHO-grade-specific
    doubling times when time interval is unknown.
    """
    cfg = _get_progression_config()
    doubling_times = cfg.doubling_times
    grade_lower = tumor_grade.lower()

    doubling_time = doubling_times.get(grade_lower, 200)

    # Estimate expected volume after one doubling period
    expected_growth_rate = current_volume * 0.693 / doubling_time  # ln(2)/T_d

    if expected_growth_rate > cfg.progression_threshold_mm3_per_day:
        state = "expected_progression"
    elif expected_growth_rate < abs(cfg.regression_threshold_mm3_per_day):
        state = "expected_slow_growth"
    else:
        state = "expected_stability"

    return {
        "current_volume": current_volume,
        "previous_volume": None,
        "time_interval_days": None,
        "progression_state": state,
        "growth_rate": round(expected_growth_rate, 2),
        "size_change_pct": None,
        "estimated_doubling_time_days": doubling_time,
        "tumor_grade_used": grade_lower,
        "reasoning": (
            f"No prior scan data available. Based on {grade_lower} "
            f"typical doubling time of {doubling_time} days, "
            f"estimated growth rate is {expected_growth_rate:.1f} mm³/day. "
            f"Classification: {state.replace('_', ' ')}."
        ),
    }


def estimate_progression(
    current_volume: float,
    previous_volume: float = None,
    time_interval_days: float = None,
) -> dict:
    """
    Estimate tumor progression state.

    Args:
        current_volume: Current tumor volume in mm³
        previous_volume: Previous tumor volume in mm³ (from earlier scan)
        time_interval_days: Days between scans

    Returns:
        dict with growth_rate, progression_state, size_change_pct, etc.
    """
    cfg = _get_progression_config()

    result = {
        "current_volume": current_volume,
        "previous_volume": previous_volume,
        "time_interval_days": time_interval_days,
    }

    if previous_volume is None or time_interval_days is None:
        result["progression_state"] = "unknown"
        result["growth_rate"] = None
        result["size_change_pct"] = None
        result["reasoning"] = "No follow-up scan data available for comparison."
        return result

    if time_interval_days <= 0:
        result["progression_state"] = "unknown"
        result["growth_rate"] = None
        result["size_change_pct"] = None
        result["reasoning"] = "Invalid time interval."
        return result

    volume_change = current_volume - previous_volume
    growth_rate = volume_change / time_interval_days

    if previous_volume > 0:
        size_change_pct = (volume_change / previous_volume) * 100
    else:
        size_change_pct = 100.0 if current_volume > 0 else 0.0

    prog_threshold = cfg.progression_threshold_mm3_per_day
    reg_threshold = cfg.regression_threshold_mm3_per_day

    if growth_rate > prog_threshold:
        state = "tumor_progression"
        reasoning = (
            f"Tumor volume increased by {volume_change:.0f} mm³ "
            f"({size_change_pct:+.1f}%) over {time_interval_days:.0f} days. "
            f"Growth rate: {growth_rate:.1f} mm³/day — indicates progression."
        )
    elif growth_rate < reg_threshold:
        state = "tumor_regression"
        reasoning = (
            f"Tumor volume decreased by {abs(volume_change):.0f} mm³ "
            f"({size_change_pct:+.1f}%) over {time_interval_days:.0f} days. "
            f"Regression rate: {abs(growth_rate):.1f} mm³/day — indicates regression."
        )
    else:
        state = "tumor_stability"
        reasoning = (
            f"Tumor volume changed by {volume_change:.0f} mm³ "
            f"({size_change_pct:+.1f}%) over {time_interval_days:.0f} days. "
            f"Growth rate: {growth_rate:.1f} mm³/day — within stability range."
        )

    result["growth_rate"] = round(growth_rate, 2)
    result["size_change_pct"] = round(size_change_pct, 2)
    result["progression_state"] = state
    result["reasoning"] = reasoning

    return result


def find_previous_volume(patient_id: str, output_dir: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Look for a previous clinical profile to extract prior volume.
    Searches: history file → patient memory → clinical profiles.
    Returns (previous_volume, time_interval_days) or (None, None).
    """
    cfg = _get_progression_config()

    # 1. Check scan history
    history = load_patient_scan_history(patient_id, output_dir)
    if len(history) >= 2:
        prev = history[-2]
        prev_volume = prev.get("tumor_volume")
        interval = prev.get("time_interval_days", cfg.baseline_default_interval_days)

        # Try to compute interval from timestamps
        if prev.get("timestamp") and history[-1].get("timestamp"):
            try:
                t1 = datetime.fromisoformat(prev["timestamp"])
                t2 = datetime.fromisoformat(history[-1]["timestamp"])
                interval = (t2 - t1).total_seconds() / 86400
                if interval <= 0:
                    interval = cfg.baseline_default_interval_days
            except Exception:
                pass

        return prev_volume, interval

    # 2. Check history file directly
    clinical_dir = os.path.join(output_dir, "clinical_features")
    history_path = os.path.join(clinical_dir, f"{patient_id}_history.json")
    if os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                hist = json.load(f)
            if len(hist) >= 2:
                prev = hist[-2]
                return prev.get("tumor_volume"), prev.get(
                    "time_interval_days", cfg.baseline_default_interval_days
                )
        except Exception:
            pass

    return None, None
