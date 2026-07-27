#!/usr/bin/env python3
"""Demo: Full clinical workflow — ACMG classification + phenotype matching.

Simulates a real clinical case:
  Patient with seizures + intellectual disability
  → Exome sequencing finds 5 candidate variants
  → System ranks them by phenotype match + pathogenicity
  → Produces ACMG classification for top candidate
"""

import sys
sys.path.insert(0, "src")

from vus_meta.acmg import ACMGClassifier
from vus_meta.phenotype import PatientVariantMatcher, CandidateVariant


def main():
    print("=" * 70)
    print("  Clinical Workflow Demo: Phenotype-Aware Variant Interpretation")
    print("=" * 70)
    
    # ── Step 1: Load knowledge bases ──
    print("\n📚 Loading knowledge bases...")
    
    matcher = PatientVariantMatcher()
    matcher.load_data(
        hpo_obo_path="data/raw/hp.obo",
        genes_to_phenotype_path="data/raw/genes_to_phenotype.txt",
    )
    
    classifier = ACMGClassifier()
    
    # ── Step 2: Define patient ──
    print("\n👤 PATIENT CASE:")
    print("   3-year-old with:")
    print("   • Seizures (HP:0001250)")
    print("   • Intellectual disability (HP:0001249)")  
    print("   • Muscular hypotonia (HP:0001252)")
    print("   • Absent speech (HP:0001344)")
    
    patient_hpo = [
        "HP:0001250",  # Seizures
        "HP:0001249",  # Intellectual disability
        "HP:0001252",  # Muscular hypotonia
        "HP:0001344",  # Absent speech
    ]
    
    # ── Step 3: Candidate variants from exome ──
    print("\n🧬 CANDIDATE VARIANTS FROM EXOME:")
    
    candidates = [
        CandidateVariant(
            gene="SCN1A",
            variant_description="p.Arg1648Cys",
            variant_type="missense",
            pathogenicity_score=0.95,
            alphamissense=0.92,
        ),
        CandidateVariant(
            gene="BRCA2",
            variant_description="p.Ala2466Thr",
            variant_type="missense",
            pathogenicity_score=0.45,
            alphamissense=0.30,
        ),
        CandidateVariant(
            gene="MECP2",
            variant_description="p.Arg133Cys",
            variant_type="missense",
            pathogenicity_score=0.88,
            alphamissense=0.85,
        ),
        CandidateVariant(
            gene="TTN",
            variant_description="p.Val12356Ile",
            variant_type="missense",
            pathogenicity_score=0.20,
            alphamissense=0.15,
        ),
        CandidateVariant(
            gene="CACNA1A",
            variant_description="p.Arg1667Trp",
            variant_type="missense",
            pathogenicity_score=0.82,
            alphamissense=0.78,
        ),
    ]
    
    for v in candidates:
        print(f"   • {v.gene:8s} {v.variant_description:15s} ({v.variant_type}) "
              f"path_score={v.pathogenicity_score:.2f}")
    
    # ── Step 4: Rank by phenotype match + pathogenicity ──
    print("\n📊 PHENOTYPE-AWARE RANKING:")
    
    ranked = matcher.rank_variants(patient_hpo, candidates)
    
    print(f"\n   {'Rank':>4s}  {'Gene':8s} {'Variant':15s} {'Combined':>8s} "
          f"{'Pheno':>7s} {'Path':>6s} {'Gene-Dis':>8s}")
    print(f"   {'─'*65}")
    
    for rv in ranked:
        v = rv.variant
        marker = " ← TOP CANDIDATE" if rv.rank == 1 else ""
        print(f"   {rv.rank:4d}  {v.gene:8s} {v.variant_description:15s} "
              f"{rv.combined_score:8.3f} {rv.phenotype_score:7.3f} "
              f"{rv.pathogenicity_score:6.2f} {rv.gene_disease_score:8.3f}{marker}")
        if rv.matched_hpo_terms:
            print(f"         Matched: {', '.join(rv.matched_hpo_terms[:3])}")
    
    # ── Step 5: ACMG classification for top candidate ──
    top = ranked[0]
    print(f"\n{'='*70}")
    print(f"  ACMG CLASSIFICATION: {top.variant.gene} {top.variant.variant_description}")
    print(f"{'='*70}")
    
    acmg_result = classifier.classify(
        gene=top.variant.gene,
        variant_type=top.variant.variant_type,
        ref_aa="R" if "Arg" in top.variant.variant_description else None,
        alt_aa="C" if "Cys" in top.variant.variant_description else None,
        aa_pos=1648,
        gnomad_af=0.0,  # Not in gnomAD
        computational_score=top.variant.pathogenicity_score,
        alphamissense_score=top.variant.alphamissense,
        phylop_score=7.5,  # Highly conserved
        spliceai_score=0.01,  # No splice impact
    )
    
    print(f"\n{acmg_result.summary()}")
    
    # ── Step 6: Clinical report ──
    print(f"\n{'='*70}")
    print("  CLINICAL REPORT SUMMARY")
    print(f"{'='*70}")
    print(f"""
   Patient: 3-year-old with epileptic encephalopathy
   
   TOP CANDIDATE: {top.variant.gene} {top.variant.variant_description}
   
   ACMG Classification: {acmg_result.classification}
   Evidence: +{acmg_result.pathogenic_points} pathogenic / -{acmg_result.benign_points} benign points
   
   Phenotype Match: {top.phenotype_score:.1%}
     The gene {top.variant.gene} is associated with phenotypes matching
     this patient's presentation (seizures, intellectual disability).
   
   Pathogenicity: {top.variant.pathogenicity_score:.2f}
     Multiple computational tools predict this variant is deleterious:
     • Meta-predictor score: {top.variant.pathogenicity_score:.3f}
     • AlphaMissense: {top.variant.alphamissense:.3f} (pathogenic)
   
   Recommendation: This variant in {top.variant.gene} is the most likely 
   cause of the patient's presentation. Confirm with functional studies
   and family segregation analysis.
""")
    
    # ── Also show example 2: cancer patient ──
    print(f"\n{'='*70}")
    print("  CASE 2: Hereditary Cancer Syndrome")
    print(f"{'='*70}")
    
    print("\n👤 PATIENT: 35-year-old female with:")
    print("   • Breast carcinoma (HP:0003002)")
    print("   • Family history of breast cancer")
    
    cancer_hpo = ["HP:0003002", "HP:0006721"]  # Breast carcinoma, Pancreatic adenocarcinoma
    
    cancer_variants = [
        CandidateVariant("BRCA1", "p.Cys61Gly", "missense", 0.997, alphamissense=0.95),
        CandidateVariant("TP53", "p.Arg175His", "missense", 0.93, alphamissense=0.99),
        CandidateVariant("ATM", "p.Val2424Gly", "missense", 0.55, alphamissense=0.40),
        CandidateVariant("MYH7", "p.Arg403Gln", "missense", 0.88, alphamissense=0.90),
    ]
    
    ranked_cancer = matcher.rank_variants(cancer_hpo, cancer_variants)
    
    print(f"\n   {'Rank':>4s}  {'Gene':8s} {'Variant':15s} {'Combined':>8s} "
          f"{'Pheno':>7s} {'Path':>6s}")
    print(f"   {'─'*55}")
    for rv in ranked_cancer:
        v = rv.variant
        print(f"   {rv.rank:4d}  {v.gene:8s} {v.variant_description:15s} "
              f"{rv.combined_score:8.3f} {rv.phenotype_score:7.3f} {rv.pathogenicity_score:6.2f}")
    
    # ACMG for BRCA1
    brca1_acmg = classifier.classify(
        gene="BRCA1", variant_type="missense",
        ref_aa="C", alt_aa="G", aa_pos=61,
        gnomad_af=0.0,
        computational_score=0.997,
        alphamissense_score=0.95,
        phylop_score=8.2,
    )
    
    print(f"\n   BRCA1 p.Cys61Gly ACMG: {brca1_acmg.classification}")
    print(f"   Points: +{brca1_acmg.pathogenic_points} / -{brca1_acmg.benign_points}")
    
    for ev in brca1_acmg.evidence:
        if ev.met:
            icon = "🔴" if ev.direction.value == "Pathogenic" else "🟢"
            print(f"   {icon} {ev.criterion}: {ev.description}")
    
    print(f"\n{'='*70}")
    print(f"  Demo complete! 🏥🧬")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
