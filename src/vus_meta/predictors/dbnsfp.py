"""dbNSFP-based predictor scores.

dbNSFP (database of Non-Synonymous Functional Predictions) is a comprehensive
database of precomputed scores from 30+ prediction tools. We extract scores for:

- CADD (Combined Annotation Dependent Depletion)
- REVEL (Rare Exome Variant Ensemble Learner)
- AlphaMissense
- MetaRNN
- BayesDel
- VEST4
- Polyphen2 (HDIV, HVAR)
- SIFT
- MutationTaster
- GERP++
- phyloP (100way, 30way)
- phastCons (100way, 30way)
- SpliceAI (max delta score)

This module provides:
1. A tabix-indexed query interface for dbNSFP
2. Individual score extractors
3. A combined feature vector builder
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from vus_meta.predictors.base import BasePredictor
from vus_meta.schemas import PredictorScore, Variant


# Column name mappings in dbNSFP4
DBNSFP_SCORE_COLUMNS = {
    # In-silico predictors
    "cadd_phred": "CADD_phred",
    "cadd_raw": "CADD_raw",
    "revel": "REVEL_score",
    "alphamissense": "AlphaMissense_score",
    "metarnn": "MetaRNN_score",
    "bayesdel_addaf": "BayesDel_addAF_score",
    "vest4": "VEST4_score",
    "polyphen2_hdiv": "Polyphen2_HDIV_score",
    "polyphen2_hvar": "Polyphen2_HVAR_score",
    "sift": "SIFT_score",
    "mutation_taster": "MutationTaster_score",
    "provean": "PROVEAN_score",
    "fathmm": "FATHMM_score",
    "mpc": "MPC_score",
    "primate_ai": "PrimateAI_score",
    # Conservation scores
    "gerp_rs": "GERP++_RS",
    "phylop100way": "phyloP100way_vertebrate",
    "phylop30way": "phyloP30way_mammalian",
    "phastcons100way": "phastCons100way_vertebrate",
    "phastcons30way": "phastCons30way_mammalian",
    # Splicing
    "spliceai_max": "SpliceAI_max",  # We'll compute this from individual deltas
}

# Rank score columns (already scaled to [0,1])
DBNSFP_RANK_COLUMNS = {
    "cadd_phred_rank": "CADD_phred_rankscore",
    "revel_rank": "REVEL_rankscore",
    "metarnn_rank": "MetaRNN_rankscore",
    "bayesdel_rank": "BayesDel_addAF_rankscore",
    "vest4_rank": "VEST4_rankscore",
}


class DbNSFPPredictor(BasePredictor):
    """Query precomputed scores from dbNSFP.

    dbNSFP must be downloaded and tabix-indexed. See:
    https://sites.google.com/site/jpaborern/dbnsfp

    Usage:
        predictor = DbNSFPPredictor("/path/to/dbNSFP4.7a.gz")
        predictor.setup()
        scores = predictor.predict(variant)
    """

    name = "dbnsfp"
    version = "4.7"
    supported_variant_types = ["missense"]  # Primary support; some scores work for others

    def __init__(self, dbnsfp_path: Optional[str] = None):
        self.dbnsfp_path = dbnsfp_path
        self._tabix = None
        self._header = None
        self._col_indices: dict[str, int] = {}

    def setup(self) -> None:
        """Open tabix-indexed dbNSFP file and parse header."""
        if self.dbnsfp_path is None:
            logger.warning("No dbNSFP path provided. DbNSFP predictor will return None scores.")
            return

        import pysam

        path = Path(self.dbnsfp_path)
        if not path.exists():
            raise FileNotFoundError(f"dbNSFP file not found: {path}")

        self._tabix = pysam.TabixFile(str(path))
        # Parse header to get column indices
        header_line = self._tabix.header[0] if self._tabix.header else ""
        if header_line.startswith("#"):
            header_line = header_line[1:]
        columns = header_line.strip().split("\t")
        self._col_indices = {col: i for i, col in enumerate(columns)}

        logger.info(f"dbNSFP loaded: {len(self._col_indices)} columns")

    def _query_variant(self, variant: Variant) -> Optional[dict[str, str]]:
        """Query dbNSFP for a specific variant. Returns raw column values."""
        if self._tabix is None:
            return None

        chrom = variant.chrom.replace("chr", "")
        try:
            records = self._tabix.fetch(chrom, variant.pos - 1, variant.pos)
        except ValueError:
            return None

        for record in records:
            fields = record.strip().split("\t")
            # Match ref and alt alleles
            ref_col = self._col_indices.get("ref", self._col_indices.get("Ref", -1))
            alt_col = self._col_indices.get("alt", self._col_indices.get("Alt", -1))

            if ref_col >= 0 and alt_col >= 0:
                if fields[ref_col] == variant.ref and fields[alt_col] == variant.alt:
                    return {col: fields[idx] for col, idx in self._col_indices.items()}

        return None

    def _parse_score(self, raw: Optional[dict], col_name: str) -> Optional[float]:
        """Parse a numeric score from dbNSFP, handling missing values."""
        if raw is None or col_name not in raw:
            return None
        val = raw[col_name]
        if val in (".", "", "NA", "-"):
            return None
        try:
            # dbNSFP sometimes has multiple values separated by ";"
            # Take the first (or max for some scores)
            parts = val.split(";")
            values = [float(v) for v in parts if v not in (".", "", "NA", "-")]
            return max(values) if values else None
        except (ValueError, TypeError):
            return None

    def extract_all_scores(self, variant: Variant) -> dict[str, Optional[float]]:
        """Extract all available scores for a variant from dbNSFP.

        Returns a dictionary of score_name -> value.
        """
        raw = self._query_variant(variant)
        scores = {}

        for score_name, col_name in DBNSFP_SCORE_COLUMNS.items():
            scores[score_name] = self._parse_score(raw, col_name)

        # Also extract rank scores
        for score_name, col_name in DBNSFP_RANK_COLUMNS.items():
            scores[score_name] = self._parse_score(raw, col_name)

        # Compute SpliceAI max from individual delta scores if available
        if raw is not None:
            splice_deltas = []
            for col in [
                "SpliceAI_DS_AG", "SpliceAI_DS_AL",
                "SpliceAI_DS_DG", "SpliceAI_DS_DL",
            ]:
                val = self._parse_score(raw, col)
                if val is not None:
                    splice_deltas.append(val)
            scores["spliceai_max"] = max(splice_deltas) if splice_deltas else None

        return scores

    def predict(self, variant: Variant) -> PredictorScore:
        """Return combined dbNSFP scores as a PredictorScore."""
        scores = self.extract_all_scores(variant)

        # Use REVEL as the primary score (it's itself a meta-predictor and well-calibrated)
        primary_score = scores.get("revel")

        return PredictorScore(
            name=self.name,
            score=primary_score,
            raw_output=scores,
        )

    def get_feature_vector(self, variant: Variant) -> tuple[np.ndarray, list[str]]:
        """Build a numeric feature vector from all dbNSFP scores.

        Missing values are encoded as NaN (handled by the meta-learner).
        Returns: (feature_array, feature_names)
        """
        scores = self.extract_all_scores(variant)
        feature_names = sorted(scores.keys())
        feature_values = np.array(
            [scores[name] if scores[name] is not None else np.nan for name in feature_names],
            dtype=np.float32,
        )
        return feature_values, feature_names

    def teardown(self) -> None:
        if self._tabix is not None:
            self._tabix.close()
            self._tabix = None


class GnomADPredictor(BasePredictor):
    """Population allele frequency features from gnomAD.

    Higher allele frequency in gnomAD → more likely benign.
    Features extracted:
    - gnomAD exome AF (all, popmax)
    - gnomAD genome AF (all, popmax)
    - Number of homozygotes
    - log10(AF) for better scaling
    """

    name = "gnomad"
    version = "4.0"
    supported_variant_types = []

    def __init__(self, gnomad_vcf_path: Optional[str] = None):
        self.gnomad_vcf_path = gnomad_vcf_path
        self._vcf = None

    def setup(self) -> None:
        """Open gnomAD VCF for querying."""
        if self.gnomad_vcf_path is None:
            logger.warning("No gnomAD VCF path provided.")
            return

        try:
            from cyvcf2 import VCF
            self._vcf = VCF(self.gnomad_vcf_path)
            logger.info("gnomAD VCF loaded")
        except Exception as e:
            logger.error(f"Failed to load gnomAD VCF: {e}")

    def predict(self, variant: Variant) -> PredictorScore:
        """Extract gnomAD allele frequency features."""
        if self._vcf is None:
            return PredictorScore(
                name=self.name,
                score=None,
                raw_output={"af": None, "af_popmax": None, "nhomalt": None},
            )

        try:
            region = f"{variant.chrom}:{variant.pos}-{variant.pos}"
            for record in self._vcf(region):
                if record.REF == variant.ref and variant.alt in record.ALT:
                    alt_idx = list(record.ALT).index(variant.alt)
                    af = record.INFO.get("AF", [0.0])
                    af = af[alt_idx] if isinstance(af, (list, tuple)) else af
                    af_popmax = record.INFO.get("AF_popmax", af)
                    nhomalt = record.INFO.get("nhomalt", [0])
                    nhomalt = nhomalt[alt_idx] if isinstance(nhomalt, (list, tuple)) else nhomalt

                    # Convert AF to a benignity score (higher AF = more benign)
                    log_af = np.log10(af + 1e-10)
                    # Rare variants (AF < 1e-4) get high score; common variants low
                    benign_score = min(1.0, max(0.0, (log_af + 2) / 4))

                    return PredictorScore(
                        name=self.name,
                        score=1.0 - benign_score,  # pathogenicity score
                        raw_output={
                            "af": float(af),
                            "af_popmax": float(af_popmax) if af_popmax else None,
                            "nhomalt": int(nhomalt),
                            "log10_af": float(log_af),
                        },
                    )
        except Exception as e:
            logger.debug(f"gnomAD lookup failed for {variant}: {e}")

        # Variant not found in gnomAD → likely very rare → slightly pathogenic signal
        return PredictorScore(
            name=self.name,
            score=0.6,  # Mild pathogenic prior for absent variants
            raw_output={"af": 0.0, "af_popmax": 0.0, "nhomalt": 0, "log10_af": -10.0},
        )

    def teardown(self) -> None:
        self._vcf = None
