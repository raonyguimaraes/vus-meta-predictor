#!/usr/bin/env python3
"""Generate a complete clinical report for a simulated patient case.

This demonstrates the full pipeline:
  Patient phenotype → Variant ranking → ACMG classification → Report
"""

import sys
sys.path.insert(0, "src")

from vus_meta.acmg import ACMGClassifier
from vus_meta.phenotype import PatientVariantMatcher, CandidateVariant
from vus_meta.clinical_report import (
    ClinicalReportGenerator, PatientCase, VariantInterpretation,
)


def main():
    # ── Setup ──
    matcher = PatientVariantMatcher()
    matcher.load_data()
    classifier = ACMGClassifier()
    reporter = ClinicalReportGenerator()
    
    # ── Case 1: Pediatric Epilepsy ──
    patient = PatientCase(
        patient_id="CASE-2026-0042",
        age="3 years",
        sex="Male",
        indication="Epileptic encephalopathy, developmental regression",
        hpo_terms=["HP:0001250", "HP:0001249", "HP:0001252", "HP:0001344", "HP:0002376"],
        hpo_names=["Seizure", "Intellectual disability", "Muscular hypotonia",
                   "Absent speech", "Developmental regression"],
        family_history="No family history of epilepsy. Non-consanguineous parents.",
        test_type="Clinical Exome Sequencing",
    )
    
    # Candidate variants (simulated exome findings)
    candidates = [
        CandidateVariant("SCN1A", "p.Arg1648Cys", "missense", 0.95),
        CandidateVariant("MECP2", "p.Arg133Cys", "missense", 0.88),
        CandidateVariant("CACNA1A", "p.Arg1667Trp", "missense", 0.82),
        CandidateVariant("BRCA2", "p.Ala2466Thr", "missense", 0.45),
        CandidateVariant("TTN", "p.Val12356Ile", "missense", 0.20),
    ]
    
    # Rank by phenotype
    ranked = matcher.rank_variants(patient.hpo_terms, candidates)
    
    # Build full interpretations with ACMG
    interpretations = []
    for rv in ranked[:3]:  # Top 3 variants
        v = rv.variant
        
        # ACMG classification
        acmg = classifier.classify(
            gene=v.gene,
            variant_type=v.variant_type,
            gnomad_af=0.0,
            computational_score=v.pathogenicity_score,
            alphamissense_score=v.alphamissense or v.pathogenicity_score * 0.95,
            phylop_score=7.0 if v.pathogenicity_score > 0.8 else 2.0,
        )
        
        interp = VariantInterpretation(
            gene=v.gene,
            variant=v.variant_description,
            variant_type=v.variant_type,
            pathogenicity_score=v.pathogenicity_score,
            alphamissense_score=v.alphamissense or v.pathogenicity_score * 0.95,
            esm2_llr=-6.5 if v.pathogenicity_score > 0.9 else -2.0,
            phylop_score=7.0 if v.pathogenicity_score > 0.8 else 2.0,
            gnomad_af=0.0,
            acmg_result=acmg,
            phenotype_rank=rv.rank,
            phenotype_score=rv.phenotype_score,
            matched_phenotypes=rv.matched_hpo_terms[:5],
        )
        interpretations.append(interp)
    
    # ── Generate Report ──
    report = reporter.generate_report(patient, interpretations)
    print(report)
    
    # ── Save report ──
    with open("data/processed/clinical_report_case1.txt", "w") as f:
        f.write(report)
    print(f"\n📄 Report saved to data/processed/clinical_report_case1.txt")


if __name__ == "__main__":
    main()
