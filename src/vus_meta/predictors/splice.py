"""Splice variant predictor.

Integrates splice prediction tools for variants near splice sites
and in non-coding regions. This covers the ~15% of pathogenic variants
that are splice-disrupting.

Features:
- SpliceAI delta scores (acceptor gain/loss, donor gain/loss)
- Distance to nearest splice site
- Canonical splice site flags (±1/2 positions)
- Consensus splice sequence disruption

For variants without SpliceAI pre-computed scores, uses
positional heuristics based on known splice site biology.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
from loguru import logger

from vus_meta.predictors.base import BasePredictor
from vus_meta.schemas import PredictorScore, Variant, VariantType


# Canonical splice site positions relative to exon boundary
# Donor: exon[-3:-1] + intron[+1:+6] → ...AGgt(a/g)agt...
# Acceptor: intron[-20:-1] + exon[+1:+1] → ...yyyyyyyyncagG...
CANONICAL_DONOR_POSITIONS = {1, 2}  # GT of donor
CANONICAL_ACCEPTOR_POSITIONS = {-1, -2}  # AG of acceptor
EXTENDED_DONOR = set(range(-3, 7))  # Extended donor region
EXTENDED_ACCEPTOR = set(range(-20, 4))  # Extended acceptor region


class SplicePredictor(BasePredictor):
    """Splice variant effect predictor.
    
    Combines:
    1. Pre-computed SpliceAI scores (if available from dbNSFP)
    2. Positional analysis relative to splice junctions
    3. Canonical splice site disruption flags
    """
    
    name = "splice"
    version = "1.0"
    supported_variant_types = [
        VariantType.SPLICE.value,
        VariantType.INTRONIC.value,
        VariantType.MISSENSE.value,  # Exonic variants can affect splicing too
        VariantType.SYNONYMOUS.value,
    ]
    
    def __init__(self):
        self.spliceai_precomputed = {}  # Optional lookup
    
    def setup(self) -> None:
        logger.info("SplicePredictor initialized")
    
    def predict(
        self,
        variant: Variant,
        spliceai_scores: Optional[dict] = None,
        distance_to_splice: Optional[int] = None,
    ) -> PredictorScore:
        """Score a variant for splice impact.
        
        Args:
            variant: Variant object
            spliceai_scores: Pre-computed SpliceAI delta scores
                {DS_AG, DS_AL, DS_DG, DS_DL, DP_AG, DP_AL, DP_DG, DP_DL}
            distance_to_splice: Distance to nearest exon boundary (negative=intronic)
        """
        features = {}
        
        # SpliceAI scores if available
        if spliceai_scores:
            ds_ag = spliceai_scores.get('DS_AG', 0)  # Acceptor Gain
            ds_al = spliceai_scores.get('DS_AL', 0)  # Acceptor Loss
            ds_dg = spliceai_scores.get('DS_DG', 0)  # Donor Gain
            ds_dl = spliceai_scores.get('DS_DL', 0)  # Donor Loss
            
            max_delta = max(ds_ag, ds_al, ds_dg, ds_dl)
            
            features['spliceai_max'] = max_delta
            features['spliceai_ds_ag'] = ds_ag
            features['spliceai_ds_al'] = ds_al
            features['spliceai_ds_dg'] = ds_dg
            features['spliceai_ds_dl'] = ds_dl
            features['spliceai_loss_max'] = max(ds_al, ds_dl)
            features['spliceai_gain_max'] = max(ds_ag, ds_dg)
            
            # Classification thresholds (from SpliceAI paper)
            if max_delta >= 0.8:
                splice_class = "high_impact"
            elif max_delta >= 0.5:
                splice_class = "moderate_impact"
            elif max_delta >= 0.2:
                splice_class = "low_impact"
            else:
                splice_class = "no_impact"
            
            features['splice_class'] = splice_class
            pathogenicity_score = min(max_delta / 0.8, 1.0)
        else:
            # Heuristic based on position
            if distance_to_splice is not None:
                abs_dist = abs(distance_to_splice)
                
                features['distance_to_splice'] = distance_to_splice
                features['abs_distance_to_splice'] = abs_dist
                
                # Canonical splice sites (±1, ±2)
                if abs_dist <= 2:
                    features['is_canonical_splice'] = 1.0
                    pathogenicity_score = 0.95
                elif abs_dist <= 6:
                    features['is_near_splice'] = 1.0
                    pathogenicity_score = 0.5
                elif abs_dist <= 20:
                    features['is_extended_splice'] = 1.0
                    pathogenicity_score = 0.2
                else:
                    pathogenicity_score = 0.05
            else:
                pathogenicity_score = None
        
        return PredictorScore(
            name=self.name,
            score=pathogenicity_score,
            raw_output=features,
        )
    
    @staticmethod
    def compute_splice_features(
        distance_to_donor: Optional[int],
        distance_to_acceptor: Optional[int],
        spliceai_max: Optional[float] = None,
    ) -> dict:
        """Compute a feature dict for splice-relevant features.
        
        Returns features suitable for the meta-learner.
        """
        features = {}
        
        if distance_to_donor is not None:
            features['dist_to_donor'] = float(distance_to_donor)
            features['abs_dist_to_donor'] = float(abs(distance_to_donor))
            features['in_donor_canonical'] = float(abs(distance_to_donor) <= 2)
            features['in_donor_extended'] = float(abs(distance_to_donor) <= 6)
        
        if distance_to_acceptor is not None:
            features['dist_to_acceptor'] = float(distance_to_acceptor)
            features['abs_dist_to_acceptor'] = float(abs(distance_to_acceptor))
            features['in_acceptor_canonical'] = float(abs(distance_to_acceptor) <= 2)
            features['in_acceptor_extended'] = float(abs(distance_to_acceptor) <= 20)
        
        if spliceai_max is not None:
            features['spliceai_max'] = spliceai_max
            features['spliceai_high'] = float(spliceai_max >= 0.8)
            features['spliceai_moderate'] = float(0.5 <= spliceai_max < 0.8)
        
        return features
