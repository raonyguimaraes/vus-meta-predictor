"""Data models for variant representation and prediction results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class VariantType(str, Enum):
    """Type of genetic variant."""
    MISSENSE = "missense"
    NONSENSE = "nonsense"
    SYNONYMOUS = "synonymous"
    SPLICE = "splice"
    FRAMESHIFT = "frameshift"
    INFRAME_INDEL = "inframe_indel"
    INTRONIC = "intronic"
    UTR = "utr"
    INTERGENIC = "intergenic"
    UNKNOWN = "unknown"


class ACMGClassification(str, Enum):
    """ACMG 5-tier classification."""
    PATHOGENIC = "Pathogenic"
    LIKELY_PATHOGENIC = "Likely Pathogenic"
    VUS = "Uncertain Significance"
    LIKELY_BENIGN = "Likely Benign"
    BENIGN = "Benign"


@dataclass
class Variant:
    """Normalized variant representation.

    Stores both genomic coordinates (chr, pos, ref, alt) and
    protein-level annotation when available.
    """
    chrom: str
    pos: int  # 1-based
    ref: str
    alt: str
    # Optional annotations
    gene: Optional[str] = None
    transcript: Optional[str] = None
    hgvs_c: Optional[str] = None
    hgvs_p: Optional[str] = None
    variant_type: VariantType = VariantType.UNKNOWN
    # Protein-level info for missense
    aa_ref: Optional[str] = None
    aa_alt: Optional[str] = None
    aa_pos: Optional[int] = None
    uniprot_id: Optional[str] = None

    @property
    def variant_id(self) -> str:
        return f"{self.chrom}-{self.pos}-{self.ref}-{self.alt}"

    def __repr__(self) -> str:
        label = self.hgvs_p or self.hgvs_c or self.variant_id
        return f"Variant({label}, gene={self.gene})"


@dataclass
class PredictorScore:
    """Score from a single predictor."""
    name: str
    score: Optional[float]  # None if predictor couldn't score this variant
    rank_score: Optional[float] = None  # Percentile rank [0, 1]
    raw_output: Optional[dict] = None  # Predictor-specific details


@dataclass
class VUSPrediction:
    """Full prediction result for a variant."""
    variant: Variant
    # Meta-predictor output
    meta_score: float  # [0, 1] calibrated probability of pathogenicity
    classification: ACMGClassification
    confidence: float  # [0, 1] prediction confidence
    # Individual predictor scores
    predictor_scores: dict[str, PredictorScore] = field(default_factory=dict)
    # Feature importance (SHAP values)
    feature_importances: dict[str, float] = field(default_factory=dict)
    # Feature vector used for meta-prediction
    feature_vector: Optional[np.ndarray] = None
    feature_names: Optional[list[str]] = None

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "variant_id": self.variant.variant_id,
            "gene": self.variant.gene,
            "hgvs_c": self.variant.hgvs_c,
            "hgvs_p": self.variant.hgvs_p,
            "variant_type": self.variant.variant_type.value,
            "meta_score": round(self.meta_score, 4),
            "classification": self.classification.value,
            "confidence": round(self.confidence, 3),
            "predictor_scores": {
                name: {"score": ps.score, "rank": ps.rank_score}
                for name, ps in self.predictor_scores.items()
            },
            "top_features": dict(
                sorted(
                    self.feature_importances.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )[:10]
            ),
        }

    def __repr__(self) -> str:
        return (
            f"VUSPrediction(variant={self.variant}, "
            f"score={self.meta_score:.3f}, "
            f"class='{self.classification.value}', "
            f"confidence={self.confidence:.3f})"
        )


def score_to_classification(score: float) -> ACMGClassification:
    """Convert calibrated pathogenicity score to ACMG classification.

    Thresholds based on ClinGen calibration recommendations:
    - >= 0.99 → Pathogenic (PP ≥ 0.99 corresponds to Strong evidence)
    - >= 0.90 → Likely Pathogenic
    - 0.10 - 0.90 → VUS
    - 0.01 - 0.10 → Likely Benign
    - < 0.01 → Benign
    """
    if score >= 0.99:
        return ACMGClassification.PATHOGENIC
    elif score >= 0.90:
        return ACMGClassification.LIKELY_PATHOGENIC
    elif score > 0.10:
        return ACMGClassification.VUS
    elif score > 0.01:
        return ACMGClassification.LIKELY_BENIGN
    else:
        return ACMGClassification.BENIGN
