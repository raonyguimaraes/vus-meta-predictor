"""Tests for data schemas."""

import numpy as np
import pytest
from vus_meta.schemas import (
    ACMGClassification,
    PredictorScore,
    Variant,
    VariantType,
    VUSPrediction,
    score_to_classification,
)


class TestVariant:
    def test_basic_creation(self):
        v = Variant(chrom="17", pos=7674220, ref="C", alt="T", gene="TP53")
        assert v.chrom == "17"
        assert v.pos == 7674220
        assert v.gene == "TP53"

    def test_variant_id(self):
        v = Variant(chrom="chr17", pos=100, ref="A", alt="G")
        assert v.variant_id == "chr17-100-A-G"

    def test_repr_with_hgvs(self):
        v = Variant(
            chrom="17", pos=100, ref="C", alt="T",
            gene="TP53", hgvs_p="p.R72P",
        )
        assert "R72P" in repr(v)
        assert "TP53" in repr(v)


class TestClassification:
    @pytest.mark.parametrize("score,expected", [
        (0.995, ACMGClassification.PATHOGENIC),
        (0.95, ACMGClassification.LIKELY_PATHOGENIC),
        (0.5, ACMGClassification.VUS),
        (0.05, ACMGClassification.LIKELY_BENIGN),
        (0.005, ACMGClassification.BENIGN),
    ])
    def test_score_to_classification(self, score, expected):
        assert score_to_classification(score) == expected

    def test_boundary_values(self):
        assert score_to_classification(0.99) == ACMGClassification.PATHOGENIC
        assert score_to_classification(0.90) == ACMGClassification.LIKELY_PATHOGENIC
        assert score_to_classification(0.10 + 1e-10) == ACMGClassification.VUS
        assert score_to_classification(0.10) == ACMGClassification.LIKELY_BENIGN


class TestVUSPrediction:
    def test_to_dict(self):
        v = Variant(chrom="1", pos=100, ref="A", alt="G", gene="BRCA1")
        pred = VUSPrediction(
            variant=v,
            meta_score=0.85,
            classification=ACMGClassification.LIKELY_PATHOGENIC,
            confidence=0.9,
            predictor_scores={"esm2": PredictorScore(name="esm2", score=0.8)},
            feature_importances={"revel": 0.3, "cadd_phred": 0.2},
        )
        d = pred.to_dict()
        assert d["meta_score"] == 0.85
        assert d["gene"] == "BRCA1"
        assert "esm2" in d["predictor_scores"]
        assert len(d["top_features"]) == 2

    def test_repr(self):
        v = Variant(chrom="1", pos=100, ref="A", alt="G")
        pred = VUSPrediction(
            variant=v,
            meta_score=0.85,
            classification=ACMGClassification.LIKELY_PATHOGENIC,
            confidence=0.9,
        )
        r = repr(pred)
        assert "0.850" in r
        assert "Likely Pathogenic" in r
