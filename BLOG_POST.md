# I Built a Variant Interpretation Pipeline With ESM-2 in One Session — Here's What I Learned

**TL;DR:** I built a complete VUS (Variant of Uncertain Significance) interpretation pipeline that combines protein language models, ACMG automation, and phenotype-aware variant ranking. The most surprising finding? AlphaMissense alone (AUROC 0.961) is incredibly hard to beat with an ensemble. But the real value isn't in the score — it's in everything around it.

**Repo:** [github.com/raonyguimaraes/vus-meta-predictor](https://github.com/raonyguimaraes/vus-meta-predictor)

---

## The Problem

Over 2.3 million variants in ClinVar are classified as "Uncertain Significance." For patients and clinicians, this means: *we found something, but we don't know if it matters.* Every year, labs spend enormous effort manually reviewing these variants, and most patients just wait.

I wanted to see how far you could get building a complete interpretation system from scratch — not just a pathogenicity score, but the full clinical workflow: phenotype matching, ACMG evidence assembly, and report generation.

## What I Built

The pipeline has four layers:

**1. Pathogenicity Scoring**
- ESM-2 protein language model (masked marginal log-likelihood ratios)
- AlphaMissense, phyloP conservation, phastCons
- Amino acid physicochemistry (Grantham distance, hydrophobicity, charge)
- Gene-level constraint (pathogenic ratio, variant burden)
- XGBoost meta-learner with isotonic calibration

**2. ACMG/AMP Evidence Automator**
- Implements 11 ACMG criteria (PVS1, PS1, PM2, PM5, PP2, PP3, BA1, BS1, BP1, BP4, BP7)
- Bayesian point-based scoring per Tavtigian et al. 2018
- Auto-generates evidence chains with citations

**3. Phenotype-Aware Variant Ranking**
- HPO ontology (20,413 terms) with Resnik semantic similarity
- 5,269 gene-phenotype associations
- Ranks variants by: pathogenicity (35%) + phenotype match (40%) + gene-disease association (25%)

**4. Clinical Report Generator**
- Template-based professional reports
- Optional LLM integration for natural language polish
- Full methodology and disclaimer sections

Plus: FastAPI web app, Docker deployment, GitHub Actions CI, 28 tests.

## The Benchmarks (Honest Numbers)

I used a proper temporal hold-out: train on ClinVar submissions before 2024, test on 123,419 variants from 2024+.

| Model | AUROC | AUPRC |
|-------|-------|-------|
| AA features only | 0.888 | 0.665 |
| AA + Conservation | 0.914 | 0.745 |
| AA + AlphaMissense | 0.931 | 0.793 |
| All features combined | **0.932** | **0.811** |
| AlphaMissense standalone | **0.961** | — |

**The humbling result:** AlphaMissense alone gets 0.961. My ensemble of everything gets 0.932. Google wins on raw missense prediction, and it's not even close.

But here's what matters: **when my system says a variant is benign, it's right 97.3% of the time** (NPV). That's clinically useful.

## What I Actually Learned

### 1. AlphaMissense is unreasonably effective
It's trained on evolutionary alignment + protein structure + a massive LM. Trying to beat it with feature engineering on the same task is like bringing a knife to a gunfight. If you need a missense pathogenicity score, just use AlphaMissense. Seriously.

### 2. The real gap is everything AlphaMissense can't do
AlphaMissense covers 71 million missense variants. But **79.7% of pathogenic ClinVar variants are NOT missense** — frameshift (37%), nonsense (24%), splice (15%). These have no AlphaMissense score. Conservation and variant-type features are what matter here.

### 3. ESM-2 (small) adds marginal value to an ensemble
The 35M-parameter ESM-2 gets AUROC 0.788 standalone — decent but not amazing. The 650M model would likely do much better, and with proper masked marginal scoring rather than teacher forcing. But when AlphaMissense is in the feature set, ESM-2 is largely redundant — AlphaMissense already encodes this signal.

### 4. The phenotype layer is the most underserved
When I showed the demo to colleagues, the feature they found most compelling wasn't the pathogenicity score — it was the phenotype-aware ranking. Given a patient with seizures, the system correctly prioritizes SCN1A over BRCA1 even when BRCA1 has a higher pathogenicity score. This seems obvious, but most variant interpretation tools don't do it.

### 5. ACMG automation is harder than it looks
The ACMG framework has 28 criteria. I implemented 11. Each one has edge cases, gene-specific exceptions, and nuances that require domain expertise. PS1 (same amino acid change as known pathogenic) sounds simple until you realize you need to match across different transcript isoforms, handle compound heterozygosity, and account for variants that are pathogenic in one gene but benign in another.

## The Stack

- **Python 3.12**, XGBoost, scikit-learn
- **ESM-2** (Facebook AI) for protein language model inference
- **AlphaMissense** (DeepMind) for pre-computed missense scores
- **ClinVar** (4.45M variants) for training and evaluation
- **HPO ontology** for phenotype matching
- **FastAPI** for the web interface
- **Docker** for deployment

8,774 lines of Python, 51 files, 28 tests, one afternoon.

## What Would I Do Differently?

1. **Start with AlphaMissense as the baseline.** I spent time building an ESM-2 pipeline that ended up being redundant. If you're working on missense variants, benchmark against AlphaMissense first.

2. **Focus on non-missense variants earlier.** That's where the gap actually is. SpliceAI for splice variants, conservation for regulatory, loss-of-function annotation for truncating.

3. **Build the clinical workflow first, score second.** Clinicians don't need another number. They need automated evidence assembly, phenotype context, and a draft report they can review and sign. The score is just one input.

## Try It

```bash
git clone https://github.com/raonyguimaraes/vus-meta-predictor
cd vus-meta-predictor
docker compose run --rm download   # Get data (~1.5 GB)
docker compose up                   # Web app on localhost:8000
```

Or hit the API directly:
```
GET /api/predict/BRCA1/C61G
GET /api/predict/TP53/R248W
POST /api/phenotype-match  (with HPO terms + candidate variants)
POST /api/acmg             (automated ACMG classification)
```

The code is MIT licensed. PRs welcome, especially for:
- SpliceAI integration for non-coding variants
- More ACMG criteria (PS2, PM3, PM4, PP1, BS3, BS4, BP2, BP5)
- gnomAD v4 allele frequency integration
- Per-gene calibration models

---

*Built by [Raony Guimaraes](https://github.com/raonyguimaraes). If you work in clinical genomics and have thoughts on this, I'd love to hear from you.*
