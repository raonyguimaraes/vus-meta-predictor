"""Core VUS Meta-Predictor: the main entry point.

Orchestrates:
1. Variant parsing and normalization
2. Feature extraction (all predictors)
3. Meta-learner inference
4. Calibration + explainability
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from loguru import logger

from vus_meta.schemas import (
    ACMGClassification,
    Variant,
    VariantType,
    VUSPrediction,
    PredictorScore,
    score_to_classification,
)
from vus_meta.models.meta_learner import MetaLearner
from vus_meta.data.features import FeatureExtractor


class VUSPredictor:
    """Main interface for VUS variant interpretation.

    Combines multiple predictors (ESM-2, DNABERT-2, dbNSFP, gnomAD, etc.)
    with a calibrated meta-learner to produce interpretable pathogenicity scores.

    Usage:
        # From pretrained model
        predictor = VUSPredictor.from_pretrained("models/meta_v1")
        result = predictor.predict_variant(variant)

        # From scratch
        predictor = VUSPredictor(config)
        predictor.setup()
        predictor.train(X, y)
        predictor.save("models/meta_v1")
    """

    def __init__(
        self,
        meta_learner: Optional[MetaLearner] = None,
        feature_extractor: Optional[FeatureExtractor] = None,
        config: Optional[dict] = None,
    ):
        self.meta_learner = meta_learner or MetaLearner()
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.config = config or {}

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> VUSPredictor:
        """Load a pretrained VUS Meta-Predictor.

        Args:
            model_dir: Directory containing saved model files

        Returns:
            Initialized VUSPredictor ready for inference
        """
        model_dir = Path(model_dir)
        meta_learner = MetaLearner.load(model_dir)

        # Feature extractor with feature names from the trained model
        feature_extractor = FeatureExtractor(
            feature_names=meta_learner.feature_names,
        )

        predictor = cls(
            meta_learner=meta_learner,
            feature_extractor=feature_extractor,
        )
        logger.info(f"Loaded pretrained VUS predictor from {model_dir}")
        return predictor

    def setup(self) -> None:
        """Initialize all predictors and models."""
        self.feature_extractor.setup()
        logger.info("VUS Predictor setup complete")

    def predict_variant(
        self,
        variant: Variant,
        protein_sequence: Optional[str] = None,
        genome_fasta: Optional[str] = None,
    ) -> VUSPrediction:
        """Predict pathogenicity for a single variant.

        Args:
            variant: Variant object
            protein_sequence: Protein sequence for ESM-2 (optional)
            genome_fasta: Path to reference genome FASTA (optional)

        Returns:
            VUSPrediction with score, classification, confidence, and explanations
        """
        # Extract features
        features, feature_names = self.feature_extractor.extract(
            variant,
            sequence=protein_sequence,
            genome_fasta=genome_fasta,
        )

        # Run meta-learner
        score, explanations = self.meta_learner.predict_single(features)

        # Get confidence
        _, confidences = self.meta_learner.predict_with_confidence(
            features.reshape(1, -1)
        )
        confidence = confidences[0]

        # Classify
        classification = score_to_classification(score)

        # Collect individual predictor scores
        predictor_scores = {}
        for pred in self.feature_extractor.predictors:
            if pred.can_score(variant):
                try:
                    ps = pred.predict(variant)
                    predictor_scores[pred.name] = ps
                except Exception:
                    pass

        return VUSPrediction(
            variant=variant,
            meta_score=score,
            classification=classification,
            confidence=confidence,
            predictor_scores=predictor_scores,
            feature_importances=explanations,
            feature_vector=features,
            feature_names=feature_names,
        )

    def predict_from_features(
        self,
        X: np.ndarray,
        variants: Optional[list[Variant]] = None,
    ) -> list[VUSPrediction]:
        """Predict from a precomputed feature matrix.

        This is the fast path when features are already extracted
        (e.g., from dbNSFP bulk annotation).

        Args:
            X: Feature matrix (n_variants, n_features)
            variants: Optional list of Variant objects for annotation

        Returns:
            List of VUSPrediction objects
        """
        scores, confidences = self.meta_learner.predict_with_confidence(X)

        predictions = []
        for i in range(len(scores)):
            variant = variants[i] if variants else Variant(
                chrom="?", pos=0, ref="?", alt="?"
            )

            # Get SHAP explanations for each prediction
            _, explanations = self.meta_learner.predict_single(X[i])

            pred = VUSPrediction(
                variant=variant,
                meta_score=float(scores[i]),
                classification=score_to_classification(float(scores[i])),
                confidence=float(confidences[i]),
                feature_importances=explanations,
                feature_vector=X[i],
                feature_names=self.meta_learner.feature_names,
            )
            predictions.append(pred)

        return predictions

    def predict_vcf(
        self,
        vcf_path: str | Path,
        output_path: Optional[str | Path] = None,
        genome_fasta: Optional[str] = None,
    ) -> pd.DataFrame:
        """Score all variants in a VCF file.

        Args:
            vcf_path: Path to input VCF
            output_path: If set, write results to this TSV file
            genome_fasta: Reference genome for sequence context

        Returns:
            DataFrame with variant annotations and scores
        """
        from cyvcf2 import VCF

        logger.info(f"Processing VCF: {vcf_path}")
        vcf = VCF(str(vcf_path))

        variants = []
        for record in vcf:
            for alt in record.ALT:
                v = Variant(
                    chrom=record.CHROM,
                    pos=record.POS,
                    ref=record.REF,
                    alt=alt,
                )
                variants.append(v)

        logger.info(f"Found {len(variants)} variants")

        # Extract features
        X, feature_names = self.feature_extractor.extract_batch(variants)

        # Predict
        predictions = self.predict_from_features(X, variants)

        # Build output DataFrame
        rows = [pred.to_dict() for pred in predictions]
        df = pd.DataFrame(rows)

        if output_path:
            df.to_csv(output_path, sep="\t", index=False)
            logger.info(f"Results written to {output_path}")

        return df

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[list[str]] = None,
        cross_validate: bool = True,
        n_folds: int = 5,
    ) -> dict:
        """Train the meta-learner.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Binary labels (1=pathogenic, 0=benign)
            feature_names: Feature names
            cross_validate: Whether to use cross-validation
            n_folds: Number of CV folds

        Returns:
            Training metrics dictionary
        """
        if cross_validate:
            return self.meta_learner.fit_cv(X, y, feature_names, n_folds)
        else:
            return self.meta_learner.fit(X, y, feature_names)

    def save(self, model_dir: str | Path) -> None:
        """Save the trained predictor."""
        self.meta_learner.save(model_dir)

    def teardown(self) -> None:
        """Clean up all resources."""
        self.feature_extractor.teardown()
