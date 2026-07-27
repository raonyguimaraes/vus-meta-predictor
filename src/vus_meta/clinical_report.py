"""Clinical Report Generator.

Generates natural language clinical reports from structured variant
interpretation data. Combines:
- ACMG evidence assessment
- Phenotype matching results
- Computational predictions (ESM-2, AlphaMissense, conservation)
- Gene-disease associations

Can optionally use an LLM (Claude, GPT) to generate polished reports,
or falls back to template-based generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from loguru import logger

from vus_meta.acmg import ACMGClassificationResult, ACMGClassifier, EvidenceDirection
from vus_meta.phenotype import RankedVariant, PatientVariantMatcher, CandidateVariant


@dataclass
class PatientCase:
    """A clinical case for variant interpretation."""
    patient_id: str = "Anonymous"
    age: Optional[str] = None
    sex: Optional[str] = None
    indication: str = ""
    hpo_terms: list[str] = field(default_factory=list)
    hpo_names: list[str] = field(default_factory=list)
    family_history: str = ""
    test_type: str = "Exome Sequencing"


@dataclass
class VariantInterpretation:
    """Complete interpretation for a single variant."""
    gene: str
    variant: str  # e.g., "p.Arg1648Cys"
    variant_type: str
    chromosome: str = ""
    position: int = 0
    ref: str = ""
    alt: str = ""
    
    # Scores
    pathogenicity_score: float = 0.5
    alphamissense_score: Optional[float] = None
    esm2_llr: Optional[float] = None
    phylop_score: Optional[float] = None
    gnomad_af: Optional[float] = None
    
    # ACMG result
    acmg_result: Optional[ACMGClassificationResult] = None
    
    # Phenotype match
    phenotype_rank: Optional[int] = None
    phenotype_score: float = 0.0
    matched_phenotypes: list[str] = field(default_factory=list)


class ClinicalReportGenerator:
    """Generate clinical interpretation reports.
    
    Usage:
        generator = ClinicalReportGenerator()
        
        # Option 1: Template-based (no API needed)
        report = generator.generate_report(patient, variants)
        
        # Option 2: LLM-enhanced (requires API key)
        report = generator.generate_report(patient, variants, use_llm=True)
    """
    
    def __init__(self, llm_api_key: Optional[str] = None):
        self.llm_api_key = llm_api_key
        self.classifier = ACMGClassifier()
    
    def generate_report(
        self,
        patient: PatientCase,
        variants: list[VariantInterpretation],
        use_llm: bool = False,
    ) -> str:
        """Generate a clinical interpretation report.
        
        Args:
            patient: Patient case information
            variants: List of interpreted variants, sorted by priority
            use_llm: Whether to use an LLM for natural language generation
        
        Returns:
            Formatted clinical report text
        """
        if use_llm and self.llm_api_key:
            return self._generate_llm_report(patient, variants)
        return self._generate_template_report(patient, variants)
    
    def _generate_template_report(
        self,
        patient: PatientCase,
        variants: list[VariantInterpretation],
    ) -> str:
        """Generate report using templates (no LLM needed)."""
        
        sections = []
        
        # ── Header ──
        sections.append(self._header(patient))
        
        # ── Clinical Summary ──
        sections.append(self._clinical_summary(patient))
        
        # ── Results Summary ──
        sections.append(self._results_summary(variants))
        
        # ── Detailed Variant Interpretations ──
        for i, var in enumerate(variants):
            sections.append(self._variant_detail(var, i + 1))
        
        # ── Methodology ──
        sections.append(self._methodology())
        
        # ── Disclaimer ──
        sections.append(self._disclaimer())
        
        return "\n".join(sections)
    
    def _header(self, patient: PatientCase) -> str:
        return f"""
╔══════════════════════════════════════════════════════════════════╗
║              CLINICAL VARIANT INTERPRETATION REPORT              ║
╠══════════════════════════════════════════════════════════════════╣
║  Patient:    {patient.patient_id:<50s}║
║  Test:       {patient.test_type:<50s}║
║  Date:       {date.today().isoformat():<50s}║
║  Indication: {patient.indication[:50]:<50s}║
╚══════════════════════════════════════════════════════════════════╝"""
    
    def _clinical_summary(self, patient: PatientCase) -> str:
        lines = ["\n══ CLINICAL INDICATION ══\n"]
        
        if patient.age or patient.sex:
            demo = ", ".join(filter(None, [patient.age, patient.sex]))
            lines.append(f"  Patient demographics: {demo}")
        
        lines.append(f"  Indication: {patient.indication}")
        
        if patient.hpo_names:
            lines.append(f"\n  Phenotype (HPO terms):")
            for hpo_id, name in zip(patient.hpo_terms, patient.hpo_names):
                lines.append(f"    • {name} ({hpo_id})")
        
        if patient.family_history:
            lines.append(f"\n  Family history: {patient.family_history}")
        
        return "\n".join(lines)
    
    def _results_summary(self, variants: list[VariantInterpretation]) -> str:
        lines = ["\n══ RESULTS SUMMARY ══\n"]
        
        pathogenic = [v for v in variants if v.acmg_result and 
                      v.acmg_result.classification in ("Pathogenic", "Likely Pathogenic")]
        vus = [v for v in variants if v.acmg_result and 
               v.acmg_result.classification == "Uncertain Significance"]
        benign = [v for v in variants if v.acmg_result and 
                  v.acmg_result.classification in ("Benign", "Likely Benign")]
        
        lines.append(f"  Variants analyzed: {len(variants)}")
        lines.append(f"  Pathogenic/Likely Pathogenic: {len(pathogenic)}")
        lines.append(f"  Uncertain Significance: {len(vus)}")
        lines.append(f"  Benign/Likely Benign: {len(benign)}")
        
        if pathogenic:
            lines.append(f"\n  ⚠️  REPORTABLE FINDINGS:")
            for v in pathogenic:
                lines.append(
                    f"    → {v.gene} {v.variant} — "
                    f"{v.acmg_result.classification}"
                )
                if v.matched_phenotypes:
                    lines.append(
                        f"      Phenotype match: {', '.join(v.matched_phenotypes[:3])}"
                    )
        
        return "\n".join(lines)
    
    def _variant_detail(self, var: VariantInterpretation, num: int) -> str:
        lines = [f"\n══ VARIANT {num}: {var.gene} {var.variant} ══\n"]
        
        # Basic info
        classification = var.acmg_result.classification if var.acmg_result else "Not classified"
        
        lines.append(f"  Gene:           {var.gene}")
        lines.append(f"  Variant:        {var.variant} ({var.variant_type})")
        if var.chromosome:
            lines.append(f"  Genomic:        chr{var.chromosome}:{var.position} {var.ref}>{var.alt}")
        lines.append(f"  Classification: {classification}")
        
        # Computational evidence
        lines.append(f"\n  Computational Evidence:")
        if var.pathogenicity_score is not None:
            lines.append(f"    Meta-predictor score: {var.pathogenicity_score:.3f}")
        if var.alphamissense_score is not None:
            am_class = "pathogenic" if var.alphamissense_score > 0.564 else \
                       "benign" if var.alphamissense_score < 0.340 else "ambiguous"
            lines.append(f"    AlphaMissense: {var.alphamissense_score:.3f} ({am_class})")
        if var.esm2_llr is not None:
            direction = "disfavored" if var.esm2_llr < -2 else \
                        "tolerated" if var.esm2_llr > 0 else "mildly disfavored"
            lines.append(f"    ESM-2 LLR: {var.esm2_llr:.2f} ({direction} by evolution)")
        if var.phylop_score is not None:
            cons = "highly conserved" if var.phylop_score > 4 else \
                   "conserved" if var.phylop_score > 2 else "not conserved"
            lines.append(f"    phyloP conservation: {var.phylop_score:.2f} ({cons})")
        
        # Population frequency
        if var.gnomad_af is not None:
            if var.gnomad_af == 0:
                lines.append(f"    gnomAD: Not observed (absent from population databases)")
            else:
                lines.append(f"    gnomAD AF: {var.gnomad_af:.2e}")
        
        # ACMG evidence
        if var.acmg_result:
            lines.append(f"\n  ACMG Evidence (points: +{var.acmg_result.pathogenic_points} / "
                        f"-{var.acmg_result.benign_points}):")
            
            met = [e for e in var.acmg_result.evidence if e.met]
            for e in met:
                icon = "🔴" if e.direction == EvidenceDirection.PATHOGENIC else "🟢"
                lines.append(f"    {icon} {e.criterion} ({e.strength.value}): {e.description}")
                if e.details:
                    lines.append(f"       Detail: {e.details}")
        
        # Phenotype match
        if var.phenotype_rank is not None:
            lines.append(f"\n  Phenotype Relevance:")
            lines.append(f"    Rank: #{var.phenotype_rank} of candidates")
            lines.append(f"    Phenotype match score: {var.phenotype_score:.1%}")
            if var.matched_phenotypes:
                lines.append(f"    Matching phenotypes:")
                for p in var.matched_phenotypes[:5]:
                    lines.append(f"      • {p}")
        
        # Interpretation
        lines.append(f"\n  Interpretation:")
        lines.append(f"    {self._interpretation_text(var)}")
        
        return "\n".join(lines)
    
    def _interpretation_text(self, var: VariantInterpretation) -> str:
        """Generate interpretation paragraph for a variant."""
        cls = var.acmg_result.classification if var.acmg_result else "Uncertain Significance"
        gene = var.gene
        variant = var.variant
        
        if cls == "Pathogenic":
            text = (
                f"The {var.variant_type} variant {variant} in {gene} is classified as "
                f"PATHOGENIC based on the following evidence: "
            )
            if var.acmg_result:
                met_criteria = [e.criterion for e in var.acmg_result.evidence if e.met 
                               and e.direction == EvidenceDirection.PATHOGENIC]
                text += ", ".join(met_criteria) + ". "
            
            if var.phenotype_score > 0.5:
                text += (
                    f"This variant is in a gene known to be associated with the patient's "
                    f"clinical presentation, supporting its role as the causative variant."
                )
            return text
        
        elif cls == "Likely Pathogenic":
            text = (
                f"The {var.variant_type} variant {variant} in {gene} is classified as "
                f"LIKELY PATHOGENIC. Multiple lines of computational evidence support a "
                f"deleterious effect"
            )
            if var.pathogenicity_score and var.pathogenicity_score > 0.9:
                text += f" (meta-predictor score: {var.pathogenicity_score:.3f})"
            text += ". "
            
            if var.gnomad_af is not None and var.gnomad_af < 1e-4:
                text += "This variant is absent or extremely rare in population databases. "
            
            text += (
                "Clinical correlation and potentially functional studies are recommended "
                "to confirm pathogenicity."
            )
            return text
        
        elif cls in ("Benign", "Likely Benign"):
            text = (
                f"The {var.variant_type} variant {variant} in {gene} is classified as "
                f"{cls.upper()}. "
            )
            if var.gnomad_af and var.gnomad_af > 0.01:
                text += f"This variant is observed at appreciable frequency in the population (AF={var.gnomad_af:.4f}). "
            if var.alphamissense_score and var.alphamissense_score < 0.34:
                text += "Computational tools predict this variant to be tolerated. "
            text += "This variant is not expected to be disease-causing."
            return text
        
        else:  # VUS
            text = (
                f"The {var.variant_type} variant {variant} in {gene} is classified as a "
                f"VARIANT OF UNCERTAIN SIGNIFICANCE. There is insufficient evidence to "
                f"determine whether this variant is pathogenic or benign. "
            )
            if var.pathogenicity_score and var.pathogenicity_score > 0.7:
                text += (
                    f"Computational prediction suggests this variant may be damaging "
                    f"(score: {var.pathogenicity_score:.3f}), but this alone is insufficient "
                    f"for clinical classification. "
                )
            text += (
                "Periodic re-evaluation is recommended as new evidence becomes available. "
                "Functional studies and family segregation data may help reclassify this variant."
            )
            return text
    
    def _methodology(self) -> str:
        return """
══ METHODOLOGY ══

  Variant interpretation was performed using the VUS Meta-Predictor pipeline:
  
  • ESM-2 protein language model (35M parameters) for evolutionary fitness
  • AlphaMissense (DeepMind) pathogenicity scores
  • phyloP 100-way vertebrate conservation scores
  • Amino acid physicochemical property analysis (Grantham distance)
  • Gene-level constraint metrics from ClinVar
  • ACMG/AMP criteria applied per Richards et al. (2015) guidelines
  • Phenotype matching using Human Phenotype Ontology (HPO) with
    Resnik semantic similarity
  
  Classification follows the ACMG/AMP 5-tier system with Bayesian
  point-based evidence combining (Tavtigian et al. 2018)."""
    
    def _disclaimer(self) -> str:
        return """
══ DISCLAIMER ══

  This report is generated by an automated computational pipeline and 
  is intended to assist clinical interpretation. It does NOT constitute 
  a clinical diagnosis. All findings should be reviewed by a qualified 
  molecular geneticist or genetic counselor. Variants of clinical 
  significance should be confirmed by an independent method.
  
  Report generated by VUS Meta-Predictor v1.0
  https://github.com/raonyguimaraes/vus-meta-predictor
"""

    def _generate_llm_report(
        self,
        patient: PatientCase,
        variants: list[VariantInterpretation],
    ) -> str:
        """Generate report using LLM for natural language polish.
        
        Falls back to template if LLM not available.
        """
        # Build structured data for LLM prompt
        template_report = self._generate_template_report(patient, variants)
        
        try:
            import anthropic
            
            client = anthropic.Anthropic(api_key=self.llm_api_key)
            
            prompt = f"""You are a clinical geneticist writing a variant interpretation report.
Below is a structured report with all the data. Rewrite it as a polished clinical 
report that a referring physician would receive. Keep all the data and evidence but 
make the language clear and professional. Keep ACMG criteria references.

Structured data:
{template_report}

Write the polished clinical report:"""
            
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}],
            )
            
            return message.content[0].text
            
        except Exception as e:
            logger.warning(f"LLM generation failed ({e}), using template")
            return template_report
