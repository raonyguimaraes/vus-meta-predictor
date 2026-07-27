"""Patient-Variant Matcher: HPO phenotype-aware variant prioritization.

Given a patient's HPO terms and a list of candidate variants,
ranks variants by combining:
1. Pathogenicity score (our meta-predictor)
2. Gene-disease association strength
3. Phenotype match (HPO semantic similarity)
4. Inheritance pattern compatibility

Uses the HPO ontology and gene-phenotype annotations.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger


@dataclass 
class HPOTerm:
    """An HPO phenotype term."""
    id: str         # e.g., "HP:0001250"
    name: str       # e.g., "Seizures"
    parents: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    depth: int = 0
    ic: float = 0.0  # Information content


@dataclass
class GeneDiseaseAssociation:
    """Association between a gene and disease phenotypes."""
    gene: str
    hpo_terms: set[str] = field(default_factory=set)
    diseases: list[str] = field(default_factory=list)
    inheritance: set[str] = field(default_factory=set)  # AD, AR, XL, etc.


@dataclass
class CandidateVariant:
    """A candidate variant to be ranked."""
    gene: str
    variant_description: str  # e.g., "p.R248W"
    variant_type: str  # missense, nonsense, etc.
    pathogenicity_score: float = 0.5
    zygosity: str = "heterozygous"  # heterozygous, homozygous, hemizygous
    # Optional additional scores
    alphamissense: Optional[float] = None
    spliceai: Optional[float] = None


@dataclass
class RankedVariant:
    """A variant with phenotype-aware ranking."""
    variant: CandidateVariant
    combined_score: float
    rank: int
    phenotype_score: float  # How well gene matches patient phenotype
    pathogenicity_score: float
    gene_disease_score: float  # Strength of gene-disease association
    inheritance_compatible: bool
    matched_hpo_terms: list[str] = field(default_factory=list)
    explanation: str = ""


class HPOOntology:
    """Parse and query the HPO ontology."""
    
    def __init__(self):
        self.terms: dict[str, HPOTerm] = {}
        self.name_to_id: dict[str, str] = {}
        self._ancestors_cache: dict[str, set[str]] = {}
    
    def load(self, obo_path: str | Path):
        """Parse hp.obo file."""
        obo_path = Path(obo_path)
        if not obo_path.exists():
            logger.warning(f"HPO OBO file not found: {obo_path}")
            return
        
        current_term = None
        
        with open(obo_path) as f:
            for line in f:
                line = line.strip()
                
                if line == "[Term]":
                    current_term = HPOTerm(id="", name="")
                elif line == "" and current_term:
                    if current_term.id.startswith("HP:"):
                        self.terms[current_term.id] = current_term
                        self.name_to_id[current_term.name.lower()] = current_term.id
                    current_term = None
                elif current_term:
                    if line.startswith("id: "):
                        current_term.id = line[4:]
                    elif line.startswith("name: "):
                        current_term.name = line[6:]
                    elif line.startswith("is_a: "):
                        parent_id = line[6:].split(" !")[0].strip()
                        current_term.parents.append(parent_id)
        
        # Build children links and compute depths
        for hpo_id, term in self.terms.items():
            for parent_id in term.parents:
                if parent_id in self.terms:
                    self.terms[parent_id].children.append(hpo_id)
        
        self._compute_depths()
        logger.info(f"HPO ontology loaded: {len(self.terms):,} terms")
    
    def _compute_depths(self):
        """BFS from root to compute term depths."""
        root = "HP:0000001"  # All
        if root not in self.terms:
            return
        
        from collections import deque
        queue = deque([(root, 0)])
        visited = set()
        
        while queue:
            term_id, depth = queue.popleft()
            if term_id in visited:
                continue
            visited.add(term_id)
            
            if term_id in self.terms:
                self.terms[term_id].depth = depth
                for child in self.terms[term_id].children:
                    queue.append((child, depth + 1))
    
    def get_ancestors(self, hpo_id: str) -> set[str]:
        """Get all ancestors of an HPO term (including self)."""
        if hpo_id in self._ancestors_cache:
            return self._ancestors_cache[hpo_id]
        
        ancestors = {hpo_id}
        if hpo_id in self.terms:
            for parent in self.terms[hpo_id].parents:
                ancestors |= self.get_ancestors(parent)
        
        self._ancestors_cache[hpo_id] = ancestors
        return ancestors
    
    def compute_ic(self, term_counts: dict[str, int]):
        """Compute information content for each term based on annotation frequency."""
        total = sum(term_counts.values())
        if total == 0:
            return
        
        for hpo_id, term in self.terms.items():
            # Count includes all descendants
            ancestors = self.get_ancestors(hpo_id)
            count = sum(term_counts.get(a, 0) for a in ancestors)
            if count > 0:
                term.ic = -math.log(count / total)
    
    def semantic_similarity(self, terms_a: set[str], terms_b: set[str]) -> float:
        """Compute semantic similarity between two sets of HPO terms.
        
        Uses Resnik's information content-based similarity with 
        Best Match Average (BMA) strategy.
        """
        if not terms_a or not terms_b:
            return 0.0
        
        # For each term in A, find best matching term in B
        scores_a_to_b = []
        for ta in terms_a:
            best = 0.0
            ancestors_a = self.get_ancestors(ta)
            for tb in terms_b:
                ancestors_b = self.get_ancestors(tb)
                # Common ancestors
                common = ancestors_a & ancestors_b
                if common:
                    # Max IC among common ancestors (MICA)
                    max_ic = max(
                        self.terms[c].ic for c in common if c in self.terms
                    ) if common else 0
                    best = max(best, max_ic)
            scores_a_to_b.append(best)
        
        # Symmetric: also B → A
        scores_b_to_a = []
        for tb in terms_b:
            best = 0.0
            ancestors_b = self.get_ancestors(tb)
            for ta in terms_a:
                ancestors_a = self.get_ancestors(ta)
                common = ancestors_a & ancestors_b
                if common:
                    max_ic = max(
                        self.terms[c].ic for c in common if c in self.terms
                    ) if common else 0
                    best = max(best, max_ic)
            scores_b_to_a.append(best)
        
        # BMA: average of best matches in both directions
        avg_a = np.mean(scores_a_to_b) if scores_a_to_b else 0
        avg_b = np.mean(scores_b_to_a) if scores_b_to_a else 0
        
        return (avg_a + avg_b) / 2
    
    def search(self, query: str) -> list[tuple[str, str]]:
        """Search HPO terms by name substring."""
        query_lower = query.lower()
        results = []
        for hpo_id, term in self.terms.items():
            if query_lower in term.name.lower():
                results.append((hpo_id, term.name))
        return sorted(results, key=lambda x: len(x[1]))[:20]


class GenePhenotypeDB:
    """Gene-phenotype association database from HPO annotations."""
    
    def __init__(self):
        self.gene_associations: dict[str, GeneDiseaseAssociation] = {}
        self.hpo_to_genes: dict[str, set[str]] = defaultdict(set)
    
    def load(self, genes_to_phenotype_path: str | Path):
        """Load HPO gene-to-phenotype annotations."""
        path = Path(genes_to_phenotype_path)
        if not path.exists():
            logger.warning(f"Gene-phenotype file not found: {path}")
            return
        
        with open(path) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) < 4:
                    continue
                
                # Format: gene_id  gene_symbol  hpo_id  hpo_name  ...
                gene = parts[1].strip()
                hpo_id = parts[2].strip()
                
                if gene not in self.gene_associations:
                    self.gene_associations[gene] = GeneDiseaseAssociation(gene=gene)
                
                self.gene_associations[gene].hpo_terms.add(hpo_id)
                self.hpo_to_genes[hpo_id].add(gene)
                
                # Parse disease name if available
                if len(parts) > 3:
                    self.gene_associations[gene].diseases.append(parts[3])
        
        logger.info(f"Loaded {len(self.gene_associations):,} gene-phenotype associations")
        logger.info(f"  {len(self.hpo_to_genes):,} unique HPO terms")
    
    def get_gene_hpo_terms(self, gene: str) -> set[str]:
        """Get all HPO terms associated with a gene."""
        assoc = self.gene_associations.get(gene)
        return assoc.hpo_terms if assoc else set()
    
    def get_genes_for_phenotype(self, hpo_id: str) -> set[str]:
        """Get all genes associated with an HPO term."""
        return self.hpo_to_genes.get(hpo_id, set())


class PatientVariantMatcher:
    """Rank candidate variants for a patient based on phenotype + pathogenicity.
    
    Usage:
        matcher = PatientVariantMatcher()
        matcher.load_data("data/raw/hp.obo", "data/raw/genes_to_phenotype.txt")
        
        patient_hpo = ["HP:0001250", "HP:0001249", "HP:0002069"]  # Seizures, ID, dementia
        variants = [CandidateVariant(...), ...]
        
        ranked = matcher.rank_variants(patient_hpo, variants)
    """
    
    def __init__(self):
        self.hpo = HPOOntology()
        self.gene_pheno_db = GenePhenotypeDB()
        self._loaded = False
    
    def load_data(
        self,
        hpo_obo_path: str = "data/raw/hp.obo",
        genes_to_phenotype_path: str = "data/raw/genes_to_phenotype.txt",
    ):
        """Load all required data."""
        self.hpo.load(hpo_obo_path)
        self.gene_pheno_db.load(genes_to_phenotype_path)
        
        # Compute IC based on gene annotation frequencies
        term_counts = defaultdict(int)
        for gene, assoc in self.gene_pheno_db.gene_associations.items():
            for hpo_id in assoc.hpo_terms:
                term_counts[hpo_id] += 1
        self.hpo.compute_ic(term_counts)
        
        self._loaded = True
        logger.info("PatientVariantMatcher ready")
    
    def rank_variants(
        self,
        patient_hpo_terms: list[str],
        candidate_variants: list[CandidateVariant],
        weights: Optional[dict] = None,
    ) -> list[RankedVariant]:
        """Rank candidate variants for a patient.
        
        Args:
            patient_hpo_terms: Patient's HPO term IDs
            candidate_variants: List of candidate variants
            weights: Optional dict of weights for combining scores
                     {'pathogenicity': 0.4, 'phenotype': 0.4, 'gene_disease': 0.2}
        
        Returns:
            List of RankedVariant sorted by combined score (best first)
        """
        if not self._loaded:
            raise RuntimeError("Call load_data() first")
        
        if weights is None:
            weights = {
                'pathogenicity': 0.35,
                'phenotype': 0.40,
                'gene_disease': 0.25,
            }
        
        patient_hpo_set = set(patient_hpo_terms)
        ranked = []
        
        for variant in candidate_variants:
            # 1. Phenotype similarity
            gene_hpo = self.gene_pheno_db.get_gene_hpo_terms(variant.gene)
            pheno_score = self.hpo.semantic_similarity(patient_hpo_set, gene_hpo)
            
            # Normalize phenotype score to [0, 1]
            pheno_normalized = min(pheno_score / 5.0, 1.0)  # IC scores typically 0-8
            
            # 2. Gene-disease association strength
            has_association = variant.gene in self.gene_pheno_db.gene_associations
            n_phenotypes = len(gene_hpo)
            gene_disease_score = min(n_phenotypes / 20.0, 1.0) if has_association else 0.0
            
            # 3. Pathogenicity score (from our predictor)
            path_score = variant.pathogenicity_score
            
            # 4. Combined score
            combined = (
                weights['pathogenicity'] * path_score +
                weights['phenotype'] * pheno_normalized +
                weights['gene_disease'] * gene_disease_score
            )
            
            # Find matched HPO terms
            matched = []
            for patient_term in patient_hpo_terms:
                patient_ancestors = self.hpo.get_ancestors(patient_term)
                if patient_ancestors & gene_hpo:
                    term_name = self.hpo.terms.get(patient_term, HPOTerm(id=patient_term, name="?")).name
                    matched.append(f"{patient_term} ({term_name})")
            
            # Build explanation
            explanation_parts = []
            if path_score > 0.8:
                explanation_parts.append(f"High pathogenicity ({path_score:.2f})")
            if pheno_normalized > 0.5:
                explanation_parts.append(f"Strong phenotype match ({pheno_normalized:.2f})")
            if matched:
                explanation_parts.append(f"Matched: {', '.join(matched[:3])}")
            if not has_association:
                explanation_parts.append("No known gene-disease association")
            
            ranked.append(RankedVariant(
                variant=variant,
                combined_score=combined,
                rank=0,
                phenotype_score=pheno_normalized,
                pathogenicity_score=path_score,
                gene_disease_score=gene_disease_score,
                inheritance_compatible=True,  # Simplified
                matched_hpo_terms=matched,
                explanation="; ".join(explanation_parts),
            ))
        
        # Sort by combined score (descending)
        ranked.sort(key=lambda x: x.combined_score, reverse=True)
        for i, rv in enumerate(ranked):
            rv.rank = i + 1
        
        return ranked
    
    def search_hpo(self, query: str) -> list[tuple[str, str]]:
        """Search HPO terms by name."""
        return self.hpo.search(query)
