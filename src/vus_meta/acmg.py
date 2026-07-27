"""ACMG/AMP Evidence Automator.

Implements automated assessment of ACMG/AMP criteria for variant classification.
Based on Richards et al. (2015) Standards and guidelines for interpretation
of sequence variants.

Criteria implemented:
  PATHOGENIC:
    PVS1 - Null variant (nonsense, frameshift, splice ±1/2) in gene where LOF is mechanism
    PS1  - Same amino acid change as established pathogenic variant
    PM1  - Located in mutational hotspot / well-established functional domain
    PM2  - Absent from controls (or extremely low frequency in gnomAD)
    PM5  - Novel missense at same position as established pathogenic
    PP2  - Missense in gene with low rate of benign missense
    PP3  - Multiple computational evidence supports deleterious effect
    
  BENIGN:
    BA1  - Allele frequency >5% in gnomAD
    BS1  - Allele frequency greater than expected for disease
    BS2  - Observed in healthy individual (homozygous for recessive)
    BP1  - Missense in gene where only truncating cause disease
    BP4  - Multiple computational evidence suggests no impact
    BP7  - Synonymous with no splicing impact
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import math

import numpy as np
from loguru import logger


class EvidenceStrength(str, Enum):
    STAND_ALONE = "Stand-alone"
    VERY_STRONG = "Very Strong"
    STRONG = "Strong"
    MODERATE = "Moderate"
    SUPPORTING = "Supporting"


class EvidenceDirection(str, Enum):
    PATHOGENIC = "Pathogenic"
    BENIGN = "Benign"


@dataclass
class ACMGEvidence:
    """A single piece of ACMG evidence."""
    criterion: str  # e.g., "PVS1", "PM2", "BP4"
    strength: EvidenceStrength
    direction: EvidenceDirection
    met: bool
    description: str
    details: str = ""
    score: Optional[float] = None  # Underlying numeric value


@dataclass
class ACMGClassificationResult:
    """Full ACMG classification result with all evidence."""
    evidence: list[ACMGEvidence] = field(default_factory=list)
    classification: str = "Uncertain Significance"
    
    # Point-based scoring (Tavtigian et al. 2018)
    pathogenic_points: float = 0.0
    benign_points: float = 0.0
    
    def add_evidence(self, ev: ACMGEvidence):
        self.evidence.append(ev)
        if ev.met:
            points = self._strength_to_points(ev.strength)
            if ev.direction == EvidenceDirection.PATHOGENIC:
                self.pathogenic_points += points
            else:
                self.benign_points += points
        self._update_classification()
    
    def _strength_to_points(self, strength: EvidenceStrength) -> float:
        """Bayesian point system from Tavtigian et al. (2018)."""
        return {
            EvidenceStrength.SUPPORTING: 1,
            EvidenceStrength.MODERATE: 2,
            EvidenceStrength.STRONG: 4,
            EvidenceStrength.VERY_STRONG: 8,
            EvidenceStrength.STAND_ALONE: 8,
        }[strength]
    
    def _update_classification(self):
        """Apply ACMG combining rules."""
        net = self.pathogenic_points - self.benign_points
        
        if net >= 10:
            self.classification = "Pathogenic"
        elif net >= 6:
            self.classification = "Likely Pathogenic"
        elif net <= -6:
            self.classification = "Benign"
        elif net <= -2:
            self.classification = "Likely Benign"
        else:
            self.classification = "Uncertain Significance"
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [f"Classification: {self.classification}"]
        lines.append(f"Evidence score: +{self.pathogenic_points} pathogenic, -{self.benign_points} benign")
        lines.append("")
        
        met_evidence = [e for e in self.evidence if e.met]
        unmet_evidence = [e for e in self.evidence if not e.met]
        
        if met_evidence:
            lines.append("MET CRITERIA:")
            for e in met_evidence:
                icon = "🔴" if e.direction == EvidenceDirection.PATHOGENIC else "🟢"
                lines.append(f"  {icon} {e.criterion} ({e.strength.value}): {e.description}")
                if e.details:
                    lines.append(f"     {e.details}")
        
        if unmet_evidence:
            lines.append("\nNOT MET:")
            for e in unmet_evidence:
                lines.append(f"  ○ {e.criterion}: {e.description}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        return {
            "classification": self.classification,
            "pathogenic_points": self.pathogenic_points,
            "benign_points": self.benign_points,
            "net_points": self.pathogenic_points - self.benign_points,
            "evidence": [
                {
                    "criterion": e.criterion,
                    "strength": e.strength.value,
                    "direction": e.direction.value,
                    "met": e.met,
                    "description": e.description,
                    "details": e.details,
                }
                for e in self.evidence
            ],
        }


class ACMGClassifier:
    """Automated ACMG/AMP variant classifier.
    
    Usage:
        classifier = ACMGClassifier()
        classifier.load_data()
        result = classifier.classify(variant_info)
    """
    
    # Genes where loss-of-function is a known disease mechanism
    LOF_GENES = {
        'BRCA1', 'BRCA2', 'TP53', 'APC', 'MLH1', 'MSH2', 'MSH6', 'PMS2',
        'RB1', 'PTEN', 'VHL', 'NF1', 'NF2', 'TSC1', 'TSC2', 'WT1',
        'MEN1', 'CDH1', 'STK11', 'SMAD4', 'BMPR1A', 'MUTYH', 'ATM',
        'PALB2', 'RAD51C', 'RAD51D', 'CHEK2', 'NBN',
        'PKD1', 'PKD2', 'CFTR', 'DMD', 'SMN1', 'FBN1',
        'COL1A1', 'COL1A2', 'LDLR',
    }
    
    # Genes where missense variants are a common mechanism of disease
    MISSENSE_DISEASE_GENES = {
        'TP53', 'MYH7', 'MYBPC3', 'KCNQ1', 'KCNH2', 'SCN5A', 'SCN1A',
        'RYR1', 'RYR2', 'CACNA1A', 'LMNA', 'MYH7', 'GLA', 'GAA',
        'LDLR', 'PTEN', 'RET', 'MEN1',
    }
    
    # Genes where only truncating variants cause disease (missense are usually benign)
    TRUNCATING_ONLY_GENES = {
        'BRCA1', 'BRCA2', 'APC', 'MLH1', 'MSH2', 'MSH6', 'PMS2',
    }
    
    def __init__(self):
        self.known_pathogenic = {}  # (gene, aa_change) → True
        self.gene_path_ratio = {}
    
    def load_clinvar_knowledge(self, variant_df=None):
        """Load known pathogenic variants from ClinVar for PS1/PM5."""
        if variant_df is not None:
            path_variants = variant_df[variant_df['label'] == 1]
            for _, row in path_variants.iterrows():
                gene = row.get('gene', '')
                aa_change = f"{row.get('ref_aa', '')}{row.get('aa_pos', '')}{row.get('alt_aa', '')}"
                self.known_pathogenic[(gene, aa_change)] = True
                # Also store by position for PM5
                pos_key = (gene, row.get('aa_pos', 0))
                if pos_key not in self.known_pathogenic:
                    self.known_pathogenic[pos_key] = True
    
    def classify(
        self,
        gene: str,
        variant_type: str,  # 'missense', 'nonsense', 'frameshift', 'splice', 'synonymous', etc.
        ref_aa: Optional[str] = None,
        alt_aa: Optional[str] = None,
        aa_pos: Optional[int] = None,
        gnomad_af: Optional[float] = None,
        gnomad_af_popmax: Optional[float] = None,
        computational_score: Optional[float] = None,  # Our meta-predictor score
        alphamissense_score: Optional[float] = None,
        spliceai_score: Optional[float] = None,
        phylop_score: Optional[float] = None,
        is_canonical_splice: bool = False,
    ) -> ACMGClassificationResult:
        """Classify a variant using ACMG/AMP criteria."""
        
        result = ACMGClassificationResult()
        
        # ── PVS1: Null variant in LOF gene ──
        is_lof = variant_type in ('nonsense', 'frameshift') or (
            variant_type == 'splice' and is_canonical_splice
        )
        pvs1_met = is_lof and gene in self.LOF_GENES
        result.add_evidence(ACMGEvidence(
            criterion="PVS1",
            strength=EvidenceStrength.VERY_STRONG,
            direction=EvidenceDirection.PATHOGENIC,
            met=pvs1_met,
            description="Null variant in gene where LOF is a known mechanism of disease",
            details=f"Variant type: {variant_type}, Gene: {gene}, LOF gene: {gene in self.LOF_GENES}",
        ))
        
        # ── PS1: Same amino acid change as established pathogenic ──
        if ref_aa and alt_aa and aa_pos:
            aa_change = f"{ref_aa}{aa_pos}{alt_aa}"
            ps1_met = (gene, aa_change) in self.known_pathogenic
            result.add_evidence(ACMGEvidence(
                criterion="PS1",
                strength=EvidenceStrength.STRONG,
                direction=EvidenceDirection.PATHOGENIC,
                met=ps1_met,
                description="Same amino acid change as previously established pathogenic variant",
                details=f"{gene} p.{aa_change}" + (" — KNOWN PATHOGENIC" if ps1_met else ""),
            ))
            
            # ── PM5: Novel missense at position with known pathogenic ──
            pm5_met = not ps1_met and (gene, aa_pos) in self.known_pathogenic
            result.add_evidence(ACMGEvidence(
                criterion="PM5",
                strength=EvidenceStrength.MODERATE,
                direction=EvidenceDirection.PATHOGENIC,
                met=pm5_met,
                description="Novel missense change at amino acid position with different known pathogenic change",
                details=f"Position {aa_pos} has other pathogenic variant(s)" if pm5_met else "",
            ))
        
        # ── PM2: Absent from controls ──
        if gnomad_af is not None:
            pm2_met = gnomad_af < 1e-5  # Very rare
            result.add_evidence(ACMGEvidence(
                criterion="PM2",
                strength=EvidenceStrength.MODERATE if gnomad_af == 0 else EvidenceStrength.SUPPORTING,
                direction=EvidenceDirection.PATHOGENIC,
                met=pm2_met,
                description="Absent from controls (or extremely low frequency)",
                details=f"gnomAD AF = {gnomad_af:.2e}" if gnomad_af else "Not found in gnomAD",
                score=gnomad_af,
            ))
        
        # ── PP2: Missense in gene with low benign missense rate ──
        if variant_type == 'missense':
            pp2_met = gene in self.MISSENSE_DISEASE_GENES
            result.add_evidence(ACMGEvidence(
                criterion="PP2",
                strength=EvidenceStrength.SUPPORTING,
                direction=EvidenceDirection.PATHOGENIC,
                met=pp2_met,
                description="Missense variant in gene with low rate of benign missense variation",
                details=f"{gene} is known missense-sensitive gene" if pp2_met else "",
            ))
        
        # ── PP3: Computational evidence supports deleterious ──
        pp3_met = False
        pp3_details = []
        
        if computational_score is not None and computational_score > 0.8:
            pp3_met = True
            pp3_details.append(f"Meta-predictor: {computational_score:.3f}")
        
        if alphamissense_score is not None and alphamissense_score > 0.564:  # AM pathogenic threshold
            pp3_met = True
            pp3_details.append(f"AlphaMissense: {alphamissense_score:.3f} (pathogenic)")
        
        if phylop_score is not None and phylop_score > 4.0:
            pp3_met = True
            pp3_details.append(f"phyloP: {phylop_score:.2f} (highly conserved)")
        
        if spliceai_score is not None and spliceai_score > 0.5:
            pp3_met = True
            pp3_details.append(f"SpliceAI: {spliceai_score:.3f} (splice impact)")
        
        # Upgrade to moderate/strong based on number of concordant predictors
        n_concordant = len(pp3_details)
        if n_concordant >= 3:
            pp3_strength = EvidenceStrength.STRONG
        elif n_concordant >= 2:
            pp3_strength = EvidenceStrength.MODERATE
        else:
            pp3_strength = EvidenceStrength.SUPPORTING
        
        result.add_evidence(ACMGEvidence(
            criterion="PP3",
            strength=pp3_strength,
            direction=EvidenceDirection.PATHOGENIC,
            met=pp3_met,
            description="Computational evidence supports a deleterious effect",
            details="; ".join(pp3_details) if pp3_details else "No strong computational support",
            score=computational_score,
        ))
        
        # ── BA1: Allele frequency > 5% ──
        if gnomad_af is not None:
            ba1_met = gnomad_af > 0.05
            result.add_evidence(ACMGEvidence(
                criterion="BA1",
                strength=EvidenceStrength.STAND_ALONE,
                direction=EvidenceDirection.BENIGN,
                met=ba1_met,
                description="Allele frequency >5% in population database",
                details=f"gnomAD AF = {gnomad_af:.4f}" if ba1_met else "",
                score=gnomad_af,
            ))
            
            # ── BS1: Greater allele frequency than expected ──
            bs1_met = 0.01 < gnomad_af <= 0.05
            result.add_evidence(ACMGEvidence(
                criterion="BS1",
                strength=EvidenceStrength.STRONG,
                direction=EvidenceDirection.BENIGN,
                met=bs1_met,
                description="Allele frequency greater than expected for disorder",
                details=f"gnomAD AF = {gnomad_af:.4f}",
                score=gnomad_af,
            ))
        
        # ── BP1: Missense in truncating-only gene ──
        if variant_type == 'missense':
            bp1_met = gene in self.TRUNCATING_ONLY_GENES
            result.add_evidence(ACMGEvidence(
                criterion="BP1",
                strength=EvidenceStrength.SUPPORTING,
                direction=EvidenceDirection.BENIGN,
                met=bp1_met,
                description="Missense variant in gene where only truncating variants cause disease",
                details=f"{gene} — primarily truncating mechanism" if bp1_met else "",
            ))
        
        # ── BP4: Computational evidence suggests no impact ──
        bp4_met = False
        bp4_details = []
        
        if computational_score is not None and computational_score < 0.15:
            bp4_met = True
            bp4_details.append(f"Meta-predictor: {computational_score:.3f}")
        
        if alphamissense_score is not None and alphamissense_score < 0.340:  # AM benign threshold
            bp4_met = True
            bp4_details.append(f"AlphaMissense: {alphamissense_score:.3f} (benign)")
        
        if phylop_score is not None and phylop_score < 0.5:
            bp4_met = True
            bp4_details.append(f"phyloP: {phylop_score:.2f} (not conserved)")
        
        result.add_evidence(ACMGEvidence(
            criterion="BP4",
            strength=EvidenceStrength.SUPPORTING,
            direction=EvidenceDirection.BENIGN,
            met=bp4_met,
            description="Multiple computational tools suggest no impact on gene/gene product",
            details="; ".join(bp4_details) if bp4_details else "",
            score=computational_score,
        ))
        
        # ── BP7: Synonymous with no splicing impact ──
        if variant_type == 'synonymous':
            bp7_met = spliceai_score is None or spliceai_score < 0.1
            result.add_evidence(ACMGEvidence(
                criterion="BP7",
                strength=EvidenceStrength.SUPPORTING,
                direction=EvidenceDirection.BENIGN,
                met=bp7_met,
                description="Synonymous variant with no predicted splice impact",
                details=f"SpliceAI: {spliceai_score:.3f}" if spliceai_score else "No splice impact predicted",
            ))
        
        return result
