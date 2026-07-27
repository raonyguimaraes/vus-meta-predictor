# VUS Meta-Predictor 🧬

**A meta-learning pipeline for Variant of Uncertain Significance (VUS) interpretation**

Combines multiple variant effect predictors — including genomic language models (DNABERT-2, Nucleotide Transformer), protein language models (ESM-2), and established tools (CADD, AlphaMissense, SpliceAI, REVEL) — into a calibrated meta-predictor that outputs interpretable pathogenicity scores.

## Architecture

```
                    ┌─────────────────────┐
                    │   Input: VCF/HGVS   │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │  Variant Normalizer  │
                    │  (hgvs → genomic)   │
                    └─────────┬───────────┘
                              │
            ┌─────────────────┼─────────────────┐
            │                 │                 │
   ┌────────▼───────┐ ┌──────▼───────┐ ┌───────▼──────┐
   │  Sequence-based │ │ Structure-   │ │ Conservation │
   │  Predictors     │ │ based Pred.  │ │ & Population │
   ├─────────────────┤ ├──────────────┤ ├──────────────┤
   │ • DNABERT-2     │ │ • ESM-2      │ │ • phyloP     │
   │ • Nucleotide    │ │   (embeddings│ │ • phastCons  │
   │   Transformer   │ │   + LLR)     │ │ • GERP++     │
   │ • SpliceAI      │ │ • AlphaMiss. │ │ • gnomAD AF  │
   │ • CADD          │ │   scores     │ │ • REVEL      │
   └────────┬────────┘ └──────┬───────┘ └───────┬──────┘
            │                 │                 │
            └─────────────────┼─────────────────┘
                              │
                    ┌─────────▼───────────┐
                    │   Feature Vector    │
                    │   (N features)      │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │   Meta-Learner      │
                    │  (Calibrated        │
                    │   Gradient Boosting │
                    │   + Isotonic Reg.)  │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │  Calibrated Score   │
                    │  + Confidence       │
                    │  + Explanations     │
                    │  (SHAP)             │
                    └─────────────────────┘
```

## Features

- **Multi-source scoring**: Ensembles 10+ individual predictors
- **Genomic Language Models**: Direct inference with DNABERT-2 and Nucleotide Transformer for zero-shot variant effect prediction
- **Protein Language Models**: ESM-2 log-likelihood ratios for missense variant scoring
- **Calibrated outputs**: Isotonic regression calibration against ClinVar gold-standard labels
- **Explainability**: SHAP-based feature importance for each prediction
- **ClinVar training**: Trained on ClinVar Pathogenic/Benign variants, evaluated on held-out VUS reclassifications
- **Batch & single-variant modes**: VCF batch processing or single HGVS query

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from vus_meta import VUSPredictor

predictor = VUSPredictor.from_pretrained("models/meta_v1")

# Single variant
result = predictor.predict("NM_000546.6:c.215C>G")  # TP53 variant
print(result)
# VUSPrediction(score=0.92, classification='Likely Pathogenic',
#               confidence=0.87, top_features={'esm2_llr': -3.2, ...})

# Batch from VCF
results = predictor.predict_vcf("input.vcf")
results.to_csv("scored_variants.tsv")
```

## Data Sources

- **ClinVar** (2024+): Training labels (Pathogenic/Likely Pathogenic vs Benign/Likely Benign)
- **gnomAD v4**: Population allele frequencies
- **dbNSFP v4.7+**: Pre-computed scores (CADD, REVEL, MetaRNN, AlphaMissense, etc.)
- **UCSC Genome Browser**: Conservation scores (phyloP, phastCons)
- **Ensembl/RefSeq**: Transcript annotations

## License

MIT
