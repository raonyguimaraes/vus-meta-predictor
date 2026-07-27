# VUS Meta-Predictor 🧬

**A meta-learning pipeline for Variant of Uncertain Significance (VUS) interpretation using protein language models (ESM-2), amino acid physicochemistry, and gene-level constraint.**

Combines multiple variant effect predictors into a calibrated meta-predictor that outputs interpretable pathogenicity scores with SHAP explanations.

## Results

### Model Comparison (5-fold CV, 10,461 clinical gene variants)

| Model | AUROC | AUPRC | Brier Score |
|-------|-------|-------|-------------|
| A: ESM-2 only | 0.788 | 0.923 | 0.154 |
| B: AA features only | 0.946 | 0.978 | 0.074 |
| **C: Combined (ESM-2 + AA + Gene)** | **0.962** | **0.985** | **0.063** |

### All-Variant Model (1.43M ClinVar variants, 27 features)

| Metric | Value |
|--------|-------|
| CV AUROC | **0.9955** |
| CV AUPRC | 0.9790 |
| Mean Calibration Error | 0.0006 |

### Clinical Gene Highlights

| Gene | VUS Scored | → P/LP | → B/LB | ESM-2 AUROC |
|------|-----------|--------|--------|-------------|
| BRCA1 | 862 | 14 | 627 | 0.939 |
| MSH2 | 1,662 | 105 | 812 | 0.932 |
| MYH7 | 1,167 | 830 | 0 | — |
| SCN1A | 634 | 253 | 0 | — |
| STK11 | — | — | — | 0.975 |
| PTEN | — | — | — | 0.937 |

## Architecture

```
                    ┌─────────────────────┐
                    │   Input: VCF/HGVS   │
                    └─────────┬───────────┘
                              │
            ┌─────────────────┼─────────────────┐
            │                 │                 │
   ┌────────▼───────┐ ┌──────▼───────┐ ┌───────▼──────┐
   │  ESM-2 Protein │ │  Amino Acid  │ │ Gene-Level   │
   │  Language Model │ │  Properties  │ │ Constraints  │
   ├─────────────────┤ ├──────────────┤ ├──────────────┤
   │ • LLR score     │ │ • Grantham   │ │ • Path ratio │
   │ • WT/Mut prob   │ │ • Hydrophobi.│ │ • Has known  │
   │ • WT rank       │ │ • Volume     │ │   pathogenic │
   │ • Entropy       │ │ • Charge     │ │ • Variant    │
   │ • Mut rank      │ │ • Polarity   │ │   count      │
   │                 │ │ • To/from    │ │              │
   │                 │ │   Pro/Cys/Gly│ │              │
   └────────┬────────┘ └──────┬───────┘ └───────┬──────┘
            │                 │                 │
            └─────────────────┼─────────────────┘
                              │
                    ┌─────────▼───────────┐
                    │   Meta-Learner      │
                    │  (XGBoost +         │
                    │   Isotonic Calib.)  │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │  Calibrated Score   │
                    │  + Classification   │
                    │  + SHAP Explanation │
                    └─────────────────────┘
```

## Top Features (Combined Model)

| Rank | Feature | Source | Importance |
|------|---------|--------|------------|
| 1 | gene_path_ratio | Gene | 0.205 |
| 2 | esm2_score | ESM-2 | 0.078 |
| 3 | esm2_llr | ESM-2 | 0.045 |
| 4 | from_cysteine | AA | 0.042 |
| 5 | to_proline | AA | 0.037 |
| 6 | aromatic_change | AA | 0.037 |
| 7 | grantham | AA | 0.031 |
| 8 | esm2_mut_prob | ESM-2 | 0.031 |

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

### Score variants with the trained model

```python
from vus_meta import VUSPredictor

predictor = VUSPredictor.from_pretrained("data/models/integrated_v1")
result = predictor.predict_variant(variant)
print(result)
# VUSPrediction(score=0.92, class='Likely Pathogenic', confidence=0.87)
```

### Train from scratch

```bash
# 1. Download ClinVar
python scripts/train_real_clinvar.py

# 2. Score with ESM-2
python scripts/esm2_scoring.py

# 3. Train integrated model
python scripts/train_integrated_model.py
```

### Run demo with synthetic data

```bash
python scripts/demo_training.py
```

## Data Sources

- **ClinVar** (latest): 4.45M variants, training labels
- **UniProt**: Protein sequences for ESM-2 inference
- **ESM-2** (35M params): Facebook AI protein language model

## Project Structure

```
├── src/vus_meta/
│   ├── schemas.py              # Variant, VUSPrediction data models
│   ├── core.py                 # VUSPredictor orchestrator
│   ├── cli.py                  # Command-line interface
│   ├── predictors/
│   │   ├── esm2.py             # ESM-2 protein LM predictor
│   │   ├── dnabert.py          # DNABERT-2 / Nucleotide Transformer
│   │   └── dbnsfp.py           # dbNSFP + gnomAD lookups
│   ├── data/
│   │   ├── clinvar.py          # ClinVar parser
│   │   └── features.py         # Feature extraction pipeline
│   └── models/
│       └── meta_learner.py     # XGBoost + calibration + SHAP
├── scripts/
│   ├── demo_training.py        # Synthetic data demo
│   ├── train_real_clinvar.py   # All-variant ClinVar training
│   ├── train_missense_v2.py    # Missense-specific model
│   ├── esm2_scoring.py         # ESM-2 batch scoring pipeline
│   └── train_integrated_model.py # Combined model
├── tests/                      # 18 tests, all passing
└── configs/default.yaml        # Configuration
```

## License

MIT
