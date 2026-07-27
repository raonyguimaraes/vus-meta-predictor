"""Tests for ACMG classifier."""
import sys
sys.path.insert(0, "src")

from vus_meta.acmg import ACMGClassifier, ACMGClassificationResult


class TestACMGClassifier:
    def setup_method(self):
        self.classifier = ACMGClassifier()
    
    def test_nonsense_in_lof_gene(self):
        """PVS1 should fire for nonsense in BRCA1."""
        result = self.classifier.classify(
            gene="BRCA1", variant_type="nonsense",
            gnomad_af=0.0, computational_score=0.99,
        )
        assert result.classification in ("Pathogenic", "Likely Pathogenic")
        pvs1 = [e for e in result.evidence if e.criterion == "PVS1"][0]
        assert pvs1.met is True
    
    def test_common_variant_benign(self):
        """BA1 should classify common variants as benign."""
        result = self.classifier.classify(
            gene="TP53", variant_type="missense",
            gnomad_af=0.10,
        )
        assert result.classification == "Benign"
    
    def test_missense_with_computational_support(self):
        """PP3 + PM2 should give at least LP for missense."""
        result = self.classifier.classify(
            gene="SCN1A", variant_type="missense",
            ref_aa="R", alt_aa="C", aa_pos=1648,
            gnomad_af=0.0,
            computational_score=0.95,
            alphamissense_score=0.92,
            phylop_score=7.5,
        )
        assert result.pathogenic_points > 0
        pp3 = [e for e in result.evidence if e.criterion == "PP3"][0]
        assert pp3.met is True
    
    def test_bp4_benign_evidence(self):
        """Low computational scores should trigger BP4."""
        result = self.classifier.classify(
            gene="BRCA2", variant_type="missense",
            computational_score=0.05,
            alphamissense_score=0.10,
            phylop_score=0.2,
        )
        bp4 = [e for e in result.evidence if e.criterion == "BP4"][0]
        assert bp4.met is True
    
    def test_to_dict(self):
        result = self.classifier.classify(
            gene="TP53", variant_type="missense",
            gnomad_af=0.001,
        )
        d = result.to_dict()
        assert "classification" in d
        assert "evidence" in d
        assert len(d["evidence"]) > 0


class TestACMGResult:
    def test_summary(self):
        result = ACMGClassificationResult()
        text = result.summary()
        assert "Uncertain Significance" in text
