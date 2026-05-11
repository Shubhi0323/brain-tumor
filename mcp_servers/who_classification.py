"""
WHO CNS Tumor Classification MCP Server
==========================================
Implements WHO CNS5 (2021) classification with:
  - Molecular markers (IDH, MGMT, 1p/19q, ATRX, TP53)
  - Low-confidence detection via score gap analysis
  - Top-5 differential diagnosis
  - Sigmoid-calibrated confidence scoring
  - Uncertainty estimation

Can run as standalone MCP server or direct function call.
"""
import math
from utils.pipeline_logger import get_logger

logger = get_logger("WHO_Classification")


def _get_who_config():
    try:
        from config.config_loader import get_config
        return get_config().who_classification
    except Exception:
        from dataclasses import dataclass, field
        from typing import Dict
        @dataclass
        class _D:
            differential_top_n: int = 5
            low_confidence_gap_threshold: float = 1.0
            calibration_method: str = "sigmoid"
            sigmoid_k: float = 1.2
            sigmoid_midpoint: float = 4.0
            default_molecular_markers: Dict[str, str] = field(default_factory=lambda: {
                "IDH": "unknown", "MGMT": "unknown", "1p19q": "unknown",
                "ATRX": "unknown", "TP53": "unknown",
            })
        return _D()


# ─── Molecular Marker Profiles ───────────────────────────────────────

MOLECULAR_PROFILES = {
    "glioblastoma": {
        "IDH": "wildtype", "MGMT": "variable", "1p19q": "intact",
        "ATRX": "retained", "TP53": "variable",
    },
    "astrocytoma": {
        "IDH": "mutant", "MGMT": "variable", "1p19q": "intact",
        "ATRX": "lost", "TP53": "mutant",
    },
    "oligodendroglioma": {
        "IDH": "mutant", "MGMT": "methylated", "1p19q": "codeleted",
        "ATRX": "retained", "TP53": "wildtype",
    },
    "meningioma": {
        "IDH": "wildtype", "MGMT": "unmethylated", "1p19q": "intact",
        "ATRX": "retained", "TP53": "wildtype",
    },
    "glioma_nos": {
        "IDH": "unknown", "MGMT": "unknown", "1p19q": "unknown",
        "ATRX": "unknown", "TP53": "unknown",
    },
}


# ─── WHO CNS5 Classification Knowledge Base ──────────────────────────

WHO_CLASSIFICATIONS = {
    "glioblastoma": {
        "who_grade": "IV",
        "full_name": "Glioblastoma, IDH-wildtype",
        "description": (
            "Highly malignant diffuse astrocytic glioma. "
            "Characterized by microvascular proliferation and/or necrosis. "
            "Most common primary malignant brain tumor in adults."
        ),
        "typical_features": {
            "volume": "large", "enhancement": "ring-enhancing with central necrosis",
            "growth_rate": "rapid", "sphericity": "irregular (low sphericity)",
            "typical_location": ["frontal", "temporal", "parietal"],
        },
        "morphology_indicators": {
            "min_volume": 15000, "max_sphericity": 0.6,
            "intensity_heterogeneity": "high",
        },
        "molecular_profile": MOLECULAR_PROFILES["glioblastoma"],
        "prognosis": "Poor. Median survival 14-16 months with standard treatment.",
        "standard_treatment": "Maximal safe resection, radiotherapy, temozolomide.",
    },
    "astrocytoma": {
        "who_grade": "II-III",
        "full_name": "Astrocytoma, IDH-mutant",
        "description": (
            "Diffuse astrocytic glioma with IDH mutation. "
            "Grade II (low-grade) or Grade III (anaplastic). "
            "Better prognosis than glioblastoma."
        ),
        "typical_features": {
            "volume": "small to medium",
            "enhancement": "minimal or no enhancement (grade II), variable (grade III)",
            "growth_rate": "slow to moderate", "sphericity": "moderate",
            "typical_location": ["frontal", "temporal"],
        },
        "morphology_indicators": {
            "min_volume": 3000, "max_volume": 40000, "min_sphericity": 0.4,
        },
        "molecular_profile": MOLECULAR_PROFILES["astrocytoma"],
        "prognosis": "Variable. Grade II median survival 7-10 years. Grade III: 3-5 years.",
        "standard_treatment": "Surgery when feasible, radiation, chemotherapy for higher grades.",
    },
    "oligodendroglioma": {
        "who_grade": "II-III",
        "full_name": "Oligodendroglioma, IDH-mutant, 1p/19q-codeleted",
        "description": (
            "Diffuse glioma with IDH mutation and 1p/19q codeletion. "
            "Characteristically demonstrates calcifications on imaging. "
            "Generally better prognosis among diffuse gliomas."
        ),
        "typical_features": {
            "volume": "small to medium",
            "enhancement": "variable, often cortical involvement",
            "growth_rate": "slow", "sphericity": "moderate to high",
            "typical_location": ["frontal"],
        },
        "morphology_indicators": {
            "min_volume": 2000, "max_volume": 35000, "min_sphericity": 0.5,
        },
        "molecular_profile": MOLECULAR_PROFILES["oligodendroglioma"],
        "prognosis": "Favorable. Grade II median survival >10 years.",
        "standard_treatment": "Surgery, PCV chemotherapy, radiation.",
    },
    "meningioma": {
        "who_grade": "I-III",
        "full_name": "Meningioma",
        "description": (
            "Extra-axial tumor arising from meningothelial cells. "
            "Most common primary intracranial tumor. "
            "Majority are WHO Grade I (benign)."
        ),
        "typical_features": {
            "volume": "variable",
            "enhancement": "homogeneous, intense enhancement",
            "growth_rate": "slow",
            "sphericity": "high (well-circumscribed)",
            "typical_location": ["parietal", "frontal"],
        },
        "morphology_indicators": {"min_sphericity": 0.65},
        "molecular_profile": MOLECULAR_PROFILES["meningioma"],
        "prognosis": "Excellent for grade I. Grade II/III have higher recurrence.",
        "standard_treatment": "Surgical resection. Radiation for incompletely resected or higher grade.",
    },
    "glioma_nos": {
        "who_grade": "II-IV",
        "full_name": "Glioma, not otherwise specified",
        "description": (
            "Diffuse glioma that cannot be further classified due to "
            "insufficient molecular data. Grading based on histology."
        ),
        "typical_features": {
            "volume": "variable", "enhancement": "variable",
            "growth_rate": "variable", "sphericity": "variable",
            "typical_location": ["frontal", "temporal", "parietal", "occipital"],
        },
        "morphology_indicators": {},
        "molecular_profile": MOLECULAR_PROFILES["glioma_nos"],
        "prognosis": "Depends on histological grade.",
        "standard_treatment": "Surgery, radiation and/or chemotherapy based on grade.",
    },
}


def _sigmoid_confidence(score: float, k: float = 1.2,
                        midpoint: float = 4.0) -> float:
    """Sigmoid-calibrated confidence: prevents overconfident low scores."""
    try:
        return 1.0 / (1.0 + math.exp(-k * (score - midpoint)))
    except OverflowError:
        return 0.0 if score < midpoint else 1.0


def _compute_molecular_score(tumor_type: str,
                              molecular_markers: dict) -> tuple:
    """Score molecular marker concordance. Returns (score, reasons)."""
    expected = MOLECULAR_PROFILES.get(tumor_type, {})
    score = 0.0
    reasons = []

    for marker, patient_status in molecular_markers.items():
        if patient_status == "unknown":
            continue
        expected_status = expected.get(marker, "unknown")
        if expected_status == "unknown" or expected_status == "variable":
            continue

        if patient_status.lower() == expected_status.lower():
            bonus = 2.0 if marker == "IDH" else 1.5
            score += bonus
            reasons.append(
                f"{marker} {patient_status} matches {tumor_type} profile (+{bonus})"
            )
        else:
            penalty = -1.5 if marker == "IDH" else -0.5
            score += penalty
            reasons.append(
                f"{marker} {patient_status} contradicts {tumor_type} "
                f"(expected {expected_status}, {penalty})"
            )

    return score, reasons


def classify_tumor(morphology: dict, radiomics_patterns: dict = None,
                   growth_characteristics: dict = None,
                   molecular_markers: dict = None) -> dict:
    """
    Classify a tumor using WHO CNS5 guidelines.

    Args:
        morphology: dict with tumor_volume, max_diameter, sphericity, surface_area
        radiomics_patterns: dict with intensity/texture features (optional)
        growth_characteristics: dict with growth_rate, enhancement info (optional)
        molecular_markers: dict with IDH, MGMT, 1p19q, ATRX, TP53 status (optional)

    Returns:
        dict with classification, confidence, reasoning, differential, molecular info
    """
    cfg = _get_who_config()
    radiomics_patterns = radiomics_patterns or {}
    growth_characteristics = growth_characteristics or {}
    molecular_markers = molecular_markers or cfg.default_molecular_markers

    volume = morphology.get("tumor_volume", 0)
    sphericity = morphology.get("sphericity", 0.5)

    scores = {}
    reasoning = {}

    for tumor_type, info in WHO_CLASSIFICATIONS.items():
        score = 0.0
        reasons = []
        indicators = info["morphology_indicators"]

        # Volume scoring
        min_vol = indicators.get("min_volume", 0)
        max_vol = indicators.get("max_volume", float("inf"))
        if min_vol <= volume <= max_vol:
            score += 2.0
            reasons.append(f"Volume {volume:.0f} mm³ matches {tumor_type} range")
        elif volume > 0:
            if volume > min_vol * 0.5:
                score += 0.5

        # Sphericity scoring
        min_sph = indicators.get("min_sphericity", 0)
        max_sph = indicators.get("max_sphericity", 1.0)
        if min_sph <= sphericity <= max_sph:
            score += 2.0
            reasons.append(f"Sphericity {sphericity:.3f} consistent with {tumor_type}")
        elif sphericity > 0:
            score += 0.5

        # Size-specific scoring
        if tumor_type == "glioblastoma" and volume > 30000:
            score += 2.0
            reasons.append("Large volume strongly suggests high-grade lesion")
        elif tumor_type == "meningioma" and sphericity > 0.7:
            score += 2.0
            reasons.append("High sphericity suggests well-circumscribed extra-axial mass")

        # Growth rate
        growth_rate = growth_characteristics.get("growth_rate", None)
        if growth_rate is not None:
            typical_growth = info["typical_features"]["growth_rate"]
            if "rapid" in typical_growth and growth_rate > 0.5:
                score += 1.5
                reasons.append("Rapid growth rate matches profile")
            elif "slow" in typical_growth and growth_rate < 0.1:
                score += 1.5
                reasons.append("Slow growth rate matches profile")

        # Intensity heterogeneity
        intensity_std = radiomics_patterns.get("intensity_std", 0)
        if tumor_type == "glioblastoma" and intensity_std > 0.5:
            score += 1.0
            reasons.append("High intensity heterogeneity consistent with GBM")
        elif tumor_type in ["astrocytoma", "oligodendroglioma"] and intensity_std < 0.5:
            score += 1.0
            reasons.append("Low intensity heterogeneity consistent with lower-grade glioma")

        # Molecular markers scoring
        mol_score, mol_reasons = _compute_molecular_score(tumor_type, molecular_markers)
        score += mol_score
        reasons.extend(mol_reasons)

        scores[tumor_type] = score
        reasoning[tumor_type] = reasons

    # Sort by score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_type = ranked[0][0]
    top_score = ranked[0][1]
    second_score = ranked[1][1] if len(ranked) > 1 else 0

    # Confidence calibration
    if cfg.calibration_method == "sigmoid":
        confidence = _sigmoid_confidence(top_score, cfg.sigmoid_k, cfg.sigmoid_midpoint)
    else:
        max_possible = 12.0  # Updated for molecular markers
        confidence = min(top_score / max_possible, 1.0)

    # Low-confidence detection
    confidence_gap = top_score - second_score
    low_confidence = confidence_gap < cfg.low_confidence_gap_threshold

    # Uncertainty estimation
    all_scores = [s for _, s in ranked if s > 0]
    if len(all_scores) > 1:
        import statistics
        score_spread = statistics.stdev(all_scores)
    else:
        score_spread = 0.0

    # Build result
    classification = WHO_CLASSIFICATIONS[top_type].copy()
    classification["classified_as"] = top_type
    classification["confidence"] = round(confidence, 3)
    classification["confidence_gap"] = round(confidence_gap, 2)
    classification["low_confidence"] = low_confidence
    classification["reasoning"] = reasoning[top_type]

    if low_confidence:
        classification["reasoning"].append(
            f"⚠ Low confidence: score gap ({confidence_gap:.1f}) below "
            f"threshold ({cfg.low_confidence_gap_threshold}). "
            "Molecular testing strongly recommended."
        )

    # Top-N differential diagnosis
    diff_n = cfg.differential_top_n
    classification["differential"] = [
        {"type": t, "score": round(s, 2), "reasoning": reasoning[t]}
        for t, s in ranked[1:diff_n + 1]
        if s > 0
    ]

    # Molecular markers used
    classification["molecular_markers_input"] = molecular_markers
    classification["molecular_profile_expected"] = MOLECULAR_PROFILES.get(top_type, {})

    # Uncertainty estimation
    classification["uncertainty"] = {
        "score_spread": round(score_spread, 2),
        "confidence_interval": [
            round(max(0, confidence - 0.1 * score_spread), 3),
            round(min(1, confidence + 0.1 * score_spread), 3),
        ],
    }

    return classification


# ─── MCP Server (optional standalone mode) ────────────────────────────

def run_mcp_server():
    """Run as a standalone MCP server via stdio transport."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        print("ERROR: 'mcp' package not installed. pip install mcp[cli]")
        return

    mcp = FastMCP("WHO Tumor Classification")

    @mcp.tool()
    def who_classify_tumor(
        tumor_volume: float = 0, sphericity: float = 0.5,
        max_diameter: float = 0, surface_area: float = 0,
        intensity_std: float = 0, intensity_skewness: float = 0,
        growth_rate: float = None,
        idh_status: str = "unknown", mgmt_status: str = "unknown",
        codeletion_1p19q: str = "unknown",
        atrx_status: str = "unknown", tp53_status: str = "unknown",
    ) -> dict:
        """Classify a brain tumor using WHO CNS5 guidelines with molecular markers."""
        morphology = {
            "tumor_volume": tumor_volume, "sphericity": sphericity,
            "max_diameter": max_diameter, "surface_area": surface_area,
        }
        radiomics = {"intensity_std": intensity_std, "intensity_skewness": intensity_skewness}
        growth = {"growth_rate": growth_rate} if growth_rate is not None else {}
        molecular = {
            "IDH": idh_status, "MGMT": mgmt_status, "1p19q": codeletion_1p19q,
            "ATRX": atrx_status, "TP53": tp53_status,
        }
        return classify_tumor(morphology, radiomics, growth, molecular)

    mcp.run(transport="stdio")


if __name__ == "__main__":
    run_mcp_server()
