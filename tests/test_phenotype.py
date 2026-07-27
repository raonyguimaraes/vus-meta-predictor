"""Tests for phenotype matching."""
import sys
import os
sys.path.insert(0, "src")

import pytest
from vus_meta.phenotype import HPOOntology, PatientVariantMatcher, CandidateVariant


class TestHPOOntology:
    @pytest.fixture
    def hpo(self):
        ont = HPOOntology()
        if os.path.exists("data/raw/hp.obo"):
            ont.load("data/raw/hp.obo")
        return ont
    
    def test_load(self, hpo):
        if not hpo.terms:
            pytest.skip("HPO OBO not available")
        assert len(hpo.terms) > 10000
    
    def test_search(self, hpo):
        if not hpo.terms:
            pytest.skip("HPO OBO not available")
        results = hpo.search("seizure")
        assert len(results) > 0
        assert any("Seizure" in name for _, name in results)
    
    def test_ancestors(self, hpo):
        if not hpo.terms:
            pytest.skip("HPO OBO not available")
        ancestors = hpo.get_ancestors("HP:0001250")  # Seizures
        assert "HP:0001250" in ancestors
        assert "HP:0000001" in ancestors  # Root


class TestPatientMatcher:
    @pytest.fixture
    def matcher(self):
        m = PatientVariantMatcher()
        if os.path.exists("data/raw/hp.obo") and os.path.exists("data/raw/genes_to_phenotype.txt"):
            m.load_data()
            return m
        return None
    
    def test_ranking(self, matcher):
        if matcher is None:
            pytest.skip("HPO data not available")

        patient_hpo = ["HP:0001250"]  # Seizures
        variants = [
            CandidateVariant("SCN1A", "p.R1648C", "missense", 0.95),
            CandidateVariant("BRCA1", "p.C61G", "missense", 0.99),
        ]

        ranked = matcher.rank_variants(patient_hpo, variants)
        assert len(ranked) == 2
        assert ranked[0].rank == 1
        # Both should get some ranking; SCN1A should have higher phenotype score
        # for a seizure patient since it's a known epilepsy gene
        scn1a_rank = [r for r in ranked if r.variant.gene == "SCN1A"][0]
        brca1_rank = [r for r in ranked if r.variant.gene == "BRCA1"][0]
        assert scn1a_rank.phenotype_score >= brca1_rank.phenotype_score
