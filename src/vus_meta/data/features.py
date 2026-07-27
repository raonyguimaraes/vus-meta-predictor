"""Feature extraction pipeline.

Coordinates all predictors to build a unified feature vector for each variant.
Handles:
- Running each predictor (or looking up precomputed scores)
- Assembling features into a consistent vector
- Deriving additional features (interactions, transformations)
- Caching results for efficiency
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from vus_meta.schemas import Variant, PredictorScore


# Master feature list: all features that go into the meta-learner
FEATURE_DEFINITIONS = {
    # ESM-2 protein language model features
    "esm2_score": "ESM-2 pathogenicity score (sigmoid of LLR)",
    "esm2_llr": "ESM-2 log-likelihood ratio (mut vs wt)",
    "esm2_wt_prob": "ESM-2 wildtype amino acid probability",
    "esm2_mut_prob": "ESM-2 mutant amino acid probability",
    "esm2_wt_rank": "ESM-2 wildtype AA rank among all AAs",
    "esm2_entropy": "ESM-2 positional entropy",
    # DNABERT-2 genomic language model features
    "dnabert2_score": "DNABERT-2 pathogenicity score",
    "dnabert2_llr": "DNABERT-2 sequence log-likelihood ratio",
    "dnabert2_embedding_dist": "DNABERT-2 ref/alt embedding distance",
    # Nucleotide Transformer features
    "nt_score": "Nucleotide Transformer pathogenicity score",
    "nt_l2_distance": "Nucleotide Transformer ref/alt L2 distance",
    # dbNSFP precomputed scores
    "cadd_phred": "CADD Phred-scaled score",
    "revel": "REVEL score",
    "alphamissense": "AlphaMissense score",
    "metarnn": "MetaRNN score",
    "bayesdel_addaf": "BayesDel (with AF) score",
    "vest4": "VEST4 score",
    "polyphen2_hdiv": "PolyPhen-2 HDIV score",
    "polyphen2_hvar": "PolyPhen-2 HVAR score",
    "sift": "SIFT score (lower = more damaging)",
    "mpc": "MPC (Missense badness, PolyPhen-2, Constraint) score",
    "primate_ai": "PrimateAI score",
    # Conservation
    "gerp_rs": "GERP++ rejected substitutions score",
    "phylop100way": "phyloP 100-way vertebrate conservation",
    "phylop30way": "phyloP 30-way mammalian conservation",
    "phastcons100way": "phastCons 100-way vertebrate conservation",
    "phastcons30way": "phastCons 30-way mammalian conservation",
    # Splicing
    "spliceai_max": "SpliceAI maximum delta score",
    # Population frequency
    "gnomad_af": "gnomAD allele frequency",
    "gnomad_af_popmax": "gnomAD population-max allele frequency",
    "gnomad_log10_af": "log10(gnomAD AF)",
    "gnomad_nhomalt": "gnomAD number of homozygotes",
    # Derived features
    "sift_inverted": "1 - SIFT (so higher = more damaging, consistent direction)",
    "conservation_mean": "Mean of phyloP and phastCons scores",
    "lm_agreement": "Agreement between ESM-2 and DNABERT-2 scores",
}


class FeatureExtractor:
    """Extracts and assembles feature vectors from multiple predictors.

    Usage:
        extractor = FeatureExtractor(predictors=[esm2, dbnsfp, gnomad])
        extractor.setup()
        features, names = extractor.extract(variant)
        # or batch:
        feature_matrix, names = extractor.extract_batch(variants)
    """

    def __init__(
        self,
        predictors: Optional[list] = None,
        feature_names: Optional[list[str]] = None,
        use_derived_features: bool = True,
    ):
        self.predictors = predictors or []
        self.use_derived_features = use_derived_features
        self._feature_names = feature_names

    def setup(self) -> None:
        """Initialize all predictors."""
        for pred in self.predictors:
            logger.info(f"Setting up predictor: {pred.name}")
            pred.setup()

    def _collect_raw_scores(
        self, variant: Variant, **kwargs
    ) -> dict[str, PredictorScore]:
        """Run all predictors on a variant."""
        results = {}
        for pred in self.predictors:
            if pred.can_score(variant):
                try:
                    score = pred.predict(variant, **kwargs)
                    results[pred.name] = score
                except Exception as e:
                    logger.warning(f"Predictor {pred.name} failed on {variant}: {e}")
                    results[pred.name] = PredictorScore(name=pred.name, score=None)
            else:
                results[pred.name] = PredictorScore(name=pred.name, score=None)
        return results

    def _scores_to_features(
        self, scores: dict[str, PredictorScore]
    ) -> dict[str, Optional[float]]:
        """Convert predictor scores to a flat feature dictionary."""
        features: dict[str, Optional[float]] = {}

        # ESM-2 features
        esm2 = scores.get("esm2")
        if esm2 and esm2.raw_output:
            features["esm2_score"] = esm2.score
            features["esm2_llr"] = esm2.raw_output.get("llr")
            features["esm2_wt_prob"] = esm2.raw_output.get("wt_prob")
            features["esm2_mut_prob"] = esm2.raw_output.get("mut_prob")
            features["esm2_wt_rank"] = esm2.raw_output.get("wt_rank")
            features["esm2_entropy"] = esm2.raw_output.get("entropy")
        else:
            for key in ["esm2_score", "esm2_llr", "esm2_wt_prob", "esm2_mut_prob",
                         "esm2_wt_rank", "esm2_entropy"]:
                features[key] = None

        # DNABERT-2 features
        dnabert2 = scores.get("dnabert2")
        if dnabert2 and dnabert2.raw_output:
            features["dnabert2_score"] = dnabert2.score
            features["dnabert2_llr"] = dnabert2.raw_output.get("llr")
            features["dnabert2_embedding_dist"] = dnabert2.raw_output.get("embedding_distance")
        else:
            for key in ["dnabert2_score", "dnabert2_llr", "dnabert2_embedding_dist"]:
                features[key] = None

        # Nucleotide Transformer features
        nt = scores.get("nucleotide_transformer")
        if nt and nt.raw_output:
            features["nt_score"] = nt.score
            features["nt_l2_distance"] = nt.raw_output.get("l2_distance")
        else:
            features["nt_score"] = None
            features["nt_l2_distance"] = None

        # dbNSFP features (these come as a big dict in raw_output)
        dbnsfp = scores.get("dbnsfp")
        if dbnsfp and dbnsfp.raw_output:
            for key in [
                "cadd_phred", "revel", "alphamissense", "metarnn",
                "bayesdel_addaf", "vest4", "polyphen2_hdiv", "polyphen2_hvar",
                "sift", "mpc", "primate_ai",
                "gerp_rs", "phylop100way", "phylop30way",
                "phastcons100way", "phastcons30way", "spliceai_max",
            ]:
                features[key] = dbnsfp.raw_output.get(key)
        else:
            for key in [
                "cadd_phred", "revel", "alphamissense", "metarnn",
                "bayesdel_addaf", "vest4", "polyphen2_hdiv", "polyphen2_hvar",
                "sift", "mpc", "primate_ai",
                "gerp_rs", "phylop100way", "phylop30way",
                "phastcons100way", "phastcons30way", "spliceai_max",
            ]:
                features[key] = None

        # gnomAD features
        gnomad = scores.get("gnomad")
        if gnomad and gnomad.raw_output:
            features["gnomad_af"] = gnomad.raw_output.get("af")
            features["gnomad_af_popmax"] = gnomad.raw_output.get("af_popmax")
            features["gnomad_log10_af"] = gnomad.raw_output.get("log10_af")
            features["gnomad_nhomalt"] = gnomad.raw_output.get("nhomalt")
        else:
            for key in ["gnomad_af", "gnomad_af_popmax", "gnomad_log10_af", "gnomad_nhomalt"]:
                features[key] = None

        # Derived features
        if self.use_derived_features:
            features.update(self._compute_derived_features(features))

        return features

    def _compute_derived_features(
        self, features: dict[str, Optional[float]]
    ) -> dict[str, Optional[float]]:
        """Compute interaction and transformation features."""
        derived = {}

        # Invert SIFT so all scores have same direction (higher = more damaging)
        sift = features.get("sift")
        derived["sift_inverted"] = (1.0 - sift) if sift is not None else None

        # Mean conservation score
        cons_scores = [
            features.get("phylop100way"),
            features.get("phylop30way"),
            features.get("phastcons100way"),
            features.get("phastcons30way"),
        ]
        valid_cons = [s for s in cons_scores if s is not None]
        derived["conservation_mean"] = (
            sum(valid_cons) / len(valid_cons) if valid_cons else None
        )

        # Agreement between language model scores
        esm2 = features.get("esm2_score")
        dnabert2 = features.get("dnabert2_score")
        if esm2 is not None and dnabert2 is not None:
            derived["lm_agreement"] = 1.0 - abs(esm2 - dnabert2)
        else:
            derived["lm_agreement"] = None

        return derived

    def extract(self, variant: Variant, **kwargs) -> tuple[np.ndarray, list[str]]:
        """Extract feature vector for a single variant.

        Returns:
            (feature_array, feature_names) — NaN for missing values
        """
        scores = self._collect_raw_scores(variant, **kwargs)
        feature_dict = self._scores_to_features(scores)

        if self._feature_names is None:
            self._feature_names = sorted(feature_dict.keys())

        values = np.array(
            [
                feature_dict.get(name, np.nan)
                if feature_dict.get(name) is not None
                else np.nan
                for name in self._feature_names
            ],
            dtype=np.float32,
        )

        return values, self._feature_names

    def extract_batch(
        self,
        variants: list[Variant],
        show_progress: bool = True,
        **kwargs,
    ) -> tuple[np.ndarray, list[str]]:
        """Extract feature matrix for a batch of variants.

        Returns:
            (feature_matrix, feature_names) — shape (n_variants, n_features)
        """
        all_features = []
        iterator = tqdm(variants, desc="Extracting features") if show_progress else variants

        for variant in iterator:
            feats, names = self.extract(variant, **kwargs)
            all_features.append(feats)

        feature_matrix = np.stack(all_features)
        return feature_matrix, self._feature_names

    def extract_from_dbnsfp_df(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """Fast path: extract features directly from a pre-fetched dbNSFP DataFrame.

        This avoids per-variant tabix queries when scores are already in a DataFrame
        (e.g., from a bulk dbNSFP annotation run).
        """
        feature_cols = [
            col for col in df.columns
            if col in FEATURE_DEFINITIONS
        ]

        if not feature_cols:
            logger.warning("No recognized feature columns in DataFrame")
            feature_cols = list(FEATURE_DEFINITIONS.keys())

        X = df[feature_cols].values.astype(np.float32)
        return X, feature_cols

    @property
    def feature_names(self) -> list[str]:
        if self._feature_names is None:
            return sorted(FEATURE_DEFINITIONS.keys())
        return self._feature_names

    def teardown(self) -> None:
        """Clean up all predictors."""
        for pred in self.predictors:
            pred.teardown()
