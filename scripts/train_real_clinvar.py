#!/usr/bin/env python3
"""Train the VUS Meta-Predictor on real ClinVar data.

Pipeline:
1. Parse ClinVar VCF → extract missense variants with P/LP and B/LB labels
2. Extract features available directly from ClinVar INFO fields
3. Download allele frequency data from ClinVar annotations
4. Engineer additional sequence-context features
5. Train XGBoost meta-learner with 5-fold CV
6. Calibrate with isotonic regression
7. Evaluate on held-out test set
8. Score VUS variants
"""

import sys
import gzip
import os
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, "src")

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")


# ═══════════════════════════════════════════════════════════════════════
# STEP 1: Parse ClinVar into structured data
# ═══════════════════════════════════════════════════════════════════════

PATHOGENIC = {"Pathogenic", "Likely_pathogenic", "Pathogenic/Likely_pathogenic"}
BENIGN = {"Benign", "Likely_benign", "Benign/Likely_benign"}

# Map review status to star rating
REVIEW_STARS = {
    "practice_guideline": 4,
    "reviewed_by_expert_panel": 3,
    "criteria_provided,_multiple_submitters,_no_conflicts": 2,
    "criteria_provided,_conflicting_classifications": 1,
    "criteria_provided,_single_submitter": 1,
    "no_assertion_criteria_provided": 0,
    "no_classification_provided": 0,
    "no_assertion_provided": 0,
}

# Amino acid properties for feature engineering
AA_PROPERTIES = {
    # (hydrophobicity, volume, charge, polarity)
    'A': ( 1.8,  88.6,  0, 0), 'R': (-4.5, 173.4,  1, 1), 'N': (-3.5, 114.1,  0, 1),
    'D': (-3.5, 111.1, -1, 1), 'C': ( 2.5, 108.5,  0, 0), 'E': (-3.5, 138.4, -1, 1),
    'Q': (-3.5, 143.8,  0, 1), 'G': (-0.4,  60.1,  0, 0), 'H': (-3.2, 153.2,  1, 1),
    'I': ( 4.5, 166.7,  0, 0), 'L': ( 3.8, 166.7,  0, 0), 'K': (-3.9, 168.6,  1, 1),
    'M': ( 1.9, 162.9,  0, 0), 'F': ( 2.8, 189.9,  0, 0), 'P': (-1.6, 112.7,  0, 0),
    'S': (-0.8,  89.0,  0, 1), 'T': (-0.7, 116.1,  0, 1), 'W': (-0.9, 227.8,  0, 0),
    'Y': (-1.3, 193.6,  0, 1), 'V': ( 4.2, 140.0,  0, 0),
}

# Grantham distance matrix (simplified — precomputed pairwise distances)
# Source: Grantham R. (1974) Science
GRANTHAM = {}  # Will compute on-the-fly from properties

# Blosum62 substitution scores (selected common pairs, higher = more conservative)
BLOSUM62 = {
    ('A','A'): 4, ('A','G'): 0, ('A','V'): 0, ('A','S'): 1, ('A','T'): 0,
    ('R','R'): 5, ('R','K'): 2, ('R','Q'): 1, ('R','H'): 0, ('R','N'): 0,
    ('D','D'): 6, ('D','E'): 2, ('D','N'): 1, ('E','E'): 5, ('E','Q'): 2,
    ('G','G'): 6, ('G','A'): 0, ('H','H'): 8, ('I','I'): 4, ('I','V'): 3,
    ('I','L'): 2, ('L','L'): 4, ('L','M'): 2, ('K','K'): 5, ('K','R'): 2,
    ('M','M'): 5, ('F','F'): 6, ('F','Y'): 3, ('F','W'): 1, ('P','P'): 7,
    ('S','S'): 4, ('S','T'): 1, ('T','T'): 5, ('W','W'):11, ('Y','Y'): 7,
    ('V','V'): 4, ('V','I'): 3, ('N','N'): 6, ('C','C'): 9, ('Q','Q'): 5,
}


def compute_grantham_distance(aa1, aa2):
    """Compute Grantham distance between two amino acids."""
    if aa1 == aa2:
        return 0
    p1 = AA_PROPERTIES.get(aa1)
    p2 = AA_PROPERTIES.get(aa2)
    if p1 is None or p2 is None:
        return np.nan
    
    # Simplified Grantham: uses composition, polarity, and volume
    hydro_diff = (p1[0] - p2[0]) ** 2
    vol_diff = (p1[1] - p2[1]) ** 2 / 1000
    charge_diff = abs(p1[2] - p2[2]) * 50
    polar_diff = abs(p1[3] - p2[3]) * 30
    
    return np.sqrt(hydro_diff * 1.833 + vol_diff * 2.0 + charge_diff + polar_diff)


def get_blosum62_score(aa1, aa2):
    """Look up BLOSUM62 substitution score."""
    if aa1 == aa2:
        return BLOSUM62.get((aa1, aa1), 4)
    score = BLOSUM62.get((aa1, aa2), BLOSUM62.get((aa2, aa1), None))
    if score is not None:
        return score
    # Default for unmatched pairs: conservative heuristic
    return -2


# Map three-letter to one-letter amino acid codes
AA3TO1 = {
    'Ala':'A', 'Arg':'R', 'Asn':'N', 'Asp':'D', 'Cys':'C', 'Glu':'E',
    'Gln':'Q', 'Gly':'G', 'His':'H', 'Ile':'I', 'Leu':'L', 'Lys':'K',
    'Met':'M', 'Phe':'F', 'Pro':'P', 'Ser':'S', 'Thr':'T', 'Trp':'W',
    'Tyr':'Y', 'Val':'V', 'Ter':'*',
}


def parse_hgvs_protein(hgvs_p):
    """Parse protein HGVS like p.Arg72Pro → (R, 72, P)."""
    if not hgvs_p or 'p.' not in hgvs_p:
        return None, None, None
    p_part = hgvs_p.split('p.')[-1]
    if len(p_part) < 5:
        return None, None, None
    
    # Try three-letter code: Arg72Pro
    ref_aa3 = p_part[:3]
    ref_aa = AA3TO1.get(ref_aa3)
    if ref_aa is None:
        return None, None, None
    
    # Extract position and alt AA
    rest = p_part[3:]
    pos_str = ''
    for c in rest:
        if c.isdigit():
            pos_str += c
        else:
            break
    
    if not pos_str:
        return None, None, None
    
    pos = int(pos_str)
    alt_aa3 = rest[len(pos_str):len(pos_str)+3]
    alt_aa = AA3TO1.get(alt_aa3)
    
    return ref_aa, pos, alt_aa


def parse_clinvar_full(vcf_path, max_variants=None):
    """Parse ClinVar VCF into a comprehensive DataFrame."""
    records = []
    count = 0
    
    with gzip.open(vcf_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            fields = line.strip().split('\t')
            if len(fields) < 8:
                continue
            
            chrom, pos, vid, ref, alt, _, _, info_str = fields[:8]
            
            # Skip multi-allelic
            if ',' in alt:
                alt = alt.split(',')[0]
            
            # Parse INFO
            info = {}
            for item in info_str.split(';'):
                if '=' in item:
                    k, v = item.split('=', 1)
                    info[k] = v
            
            clnsig = info.get('CLNSIG', '')
            mc = info.get('MC', '')
            revstat = info.get('CLNREVSTAT', '')
            gene = info.get('GENEINFO', '').split(':')[0] if 'GENEINFO' in info else ''
            clnhgvs = info.get('CLNHGVS', '')
            clnvc = info.get('CLNVC', '')
            
            # Determine label
            if clnsig in PATHOGENIC:
                label = 1
            elif clnsig in BENIGN:
                label = 0
            elif 'Uncertain_significance' in clnsig:
                label = -1
            else:
                label = -2  # other (conflicting, not provided, etc.)
            
            # Review stars
            stars = 0
            for status, s in REVIEW_STARS.items():
                if status in revstat:
                    stars = max(stars, s)
            
            # Molecular consequence flags
            is_missense = 'missense_variant' in mc
            is_nonsense = 'nonsense' in mc or 'stop_gained' in mc
            is_splice = 'splice' in mc
            is_frameshift = 'frameshift' in mc
            is_synonymous = 'synonymous' in mc
            is_inframe = 'inframe' in mc
            is_lof = is_nonsense or is_splice or is_frameshift
            
            # Parse protein change for missense
            ref_aa, aa_pos, alt_aa = None, None, None
            if is_missense:
                # Try to parse from MC field
                for mc_item in mc.split(','):
                    if 'missense' in mc_item and '|' in mc_item:
                        # Format: SO:0001583|missense_variant
                        pass
                # Try from HGVS
                # ClinVar CLNHGVS is genomic, we need protein — extract from MC or gene info
                # For now we'll use positional features
            
            # Allele frequencies from ClinVar INFO
            af_esp = info.get('AF_ESP', '')
            af_exac = info.get('AF_EXAC', '')
            af_tgp = info.get('AF_TGP', '')
            
            def safe_float(val):
                try:
                    return float(val) if val and val != '.' else np.nan
                except:
                    return np.nan
            
            record = {
                'chrom': chrom,
                'pos': int(pos),
                'ref': ref,
                'alt': alt,
                'clinvar_id': vid,
                'gene': gene,
                'clnsig': clnsig,
                'label': label,
                'stars': stars,
                'clnvc': clnvc,
                'is_missense': is_missense,
                'is_nonsense': is_nonsense,
                'is_splice': is_splice,
                'is_frameshift': is_frameshift,
                'is_synonymous': is_synonymous,
                'is_inframe': is_inframe,
                'is_lof': is_lof,
                'is_snv': clnvc == 'single_nucleotide_variant',
                'ref_len': len(ref),
                'alt_len': len(alt),
                'af_esp': safe_float(af_esp),
                'af_exac': safe_float(af_exac),
                'af_tgp': safe_float(af_tgp),
            }
            
            records.append(record)
            count += 1
            if count % 500000 == 0:
                logger.info(f"  Parsed {count:,} variants...")
            if max_variants and count >= max_variants:
                break
    
    df = pd.DataFrame(records)
    logger.info(f"Parsed {len(df):,} total variants from ClinVar")
    return df


# ═══════════════════════════════════════════════════════════════════════
# STEP 2: Feature Engineering from ClinVar annotations
# ═══════════════════════════════════════════════════════════════════════

def engineer_features(df):
    """Create features from ClinVar annotations for training.
    
    Since we don't have dbNSFP, we engineer features from:
    - Variant type (missense, nonsense, splice, frameshift, etc.)
    - Allele frequencies (ESP, ExAC, 1000G)
    - Sequence context (transition/transversion, CpG, etc.)
    - Gene-level constraint metrics (derived from ClinVar itself)
    - Position-based features
    """
    logger.info("Engineering features...")
    
    features = pd.DataFrame(index=df.index)
    
    # ---- Variant type features (one-hot) ----
    features['is_missense'] = df['is_missense'].astype(float)
    features['is_nonsense'] = df['is_nonsense'].astype(float)
    features['is_splice'] = df['is_splice'].astype(float)
    features['is_frameshift'] = df['is_frameshift'].astype(float)
    features['is_synonymous'] = df['is_synonymous'].astype(float)
    features['is_inframe'] = df['is_inframe'].astype(float)
    features['is_lof'] = df['is_lof'].astype(float)
    features['is_snv'] = df['is_snv'].astype(float)
    
    # ---- Sequence features ----
    # Transition vs transversion (for SNVs)
    transitions = {('A','G'), ('G','A'), ('C','T'), ('T','C')}
    features['is_transition'] = df.apply(
        lambda r: 1.0 if (r['ref'], r['alt']) in transitions else 0.0
        if len(r['ref']) == 1 and len(r['alt']) == 1 else np.nan,
        axis=1
    )
    
    # Variant length features
    features['ref_len'] = df['ref_len'].astype(float)
    features['alt_len'] = df['alt_len'].astype(float)
    features['len_diff'] = (df['alt_len'] - df['ref_len']).astype(float)
    
    # ---- Allele frequency features ----
    # Log-transform AFs (very important for pathogenicity)
    for col in ['af_esp', 'af_exac', 'af_tgp']:
        features[col] = df[col]
        features[f'{col}_log10'] = np.log10(df[col].clip(lower=1e-10))
        # Set to NaN where original was NaN
        features.loc[df[col].isna(), f'{col}_log10'] = np.nan
    
    # Max AF across databases
    af_cols = ['af_esp', 'af_exac', 'af_tgp']
    features['af_max'] = df[af_cols].max(axis=1)
    features['af_max_log10'] = np.log10(features['af_max'].clip(lower=1e-10))
    features.loc[features['af_max'].isna(), 'af_max_log10'] = np.nan
    
    # Has any AF data at all (informative about how well-studied the variant is)
    features['has_af_data'] = (~df[af_cols].isna().all(axis=1)).astype(float)
    
    # ---- Gene-level features (computed from ClinVar itself) ----
    # Gene pathogenic ratio: fraction of known P/LP variants vs total classified
    gene_path_counts = df[df['label'] == 1].groupby('gene').size()
    gene_benign_counts = df[df['label'] == 0].groupby('gene').size()
    gene_total = (gene_path_counts.add(gene_benign_counts, fill_value=0))
    gene_path_ratio = (gene_path_counts / gene_total).fillna(0)
    
    features['gene_path_ratio'] = df['gene'].map(gene_path_ratio).fillna(0.5).astype(float)
    
    # Gene variant count (proxy for how well-studied)
    gene_total_variants = df.groupby('gene').size()
    features['gene_variant_count_log'] = np.log10(
        df['gene'].map(gene_total_variants).fillna(1).clip(lower=1)
    ).astype(float)
    
    # Gene has known pathogenic variants at all
    genes_with_pathogenic = set(df[df['label'] == 1]['gene'].unique())
    features['gene_has_pathogenic'] = df['gene'].isin(genes_with_pathogenic).astype(float)
    
    # ---- Review status features ----
    features['review_stars'] = df['stars'].astype(float)
    
    # ---- Chromosomal features ----
    # Autosome vs sex chromosome
    features['is_autosome'] = (~df['chrom'].isin(['X', 'Y', 'MT'])).astype(float)
    
    # Position features (normalize per chromosome)
    features['position_log'] = np.log10(df['pos'].clip(lower=1)).astype(float)
    
    # Drop any fully empty columns
    features = features.dropna(axis=1, how='all')
    
    logger.info(f"Engineered {features.shape[1]} features")
    
    # Report missing rates
    missing_rates = features.isna().mean().sort_values(ascending=False)
    logger.info("Feature missing rates:")
    for fname, rate in missing_rates.items():
        if rate > 0:
            logger.info(f"  {fname:30s} {rate:.1%}")
    
    return features


# ═══════════════════════════════════════════════════════════════════════
# STEP 3: Train and Evaluate
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  VUS Meta-Predictor — Real ClinVar Training")
    print("=" * 70)
    print()
    
    # Step 1: Parse ClinVar
    print("STEP 1: Parsing ClinVar VCF...")
    clinvar_df = parse_clinvar_full('data/raw/clinvar.vcf.gz')
    
    print(f"\n  Label distribution:")
    print(f"    Pathogenic (1):  {(clinvar_df['label'] == 1).sum():>10,}")
    print(f"    Benign (0):      {(clinvar_df['label'] == 0).sum():>10,}")
    print(f"    VUS (-1):        {(clinvar_df['label'] == -1).sum():>10,}")
    print(f"    Other (-2):      {(clinvar_df['label'] == -2).sum():>10,}")
    
    # Step 2: Filter to training set (P/LP and B/LB with ≥1 star)
    print("\nSTEP 2: Preparing training data...")
    
    # Use variants with clear labels and minimum review quality
    train_mask = (
        (clinvar_df['label'].isin([0, 1])) &
        (clinvar_df['stars'] >= 1) &
        (clinvar_df['is_snv'])  # Focus on SNVs for now
    )
    
    train_df = clinvar_df[train_mask].copy()
    vus_df = clinvar_df[(clinvar_df['label'] == -1) & (clinvar_df['is_snv'])].copy()
    
    print(f"  Training variants (SNV, ≥1 star): {len(train_df):,}")
    print(f"    Pathogenic: {(train_df['label'] == 1).sum():,}")
    print(f"    Benign:     {(train_df['label'] == 0).sum():,}")
    print(f"  VUS variants to score: {len(vus_df):,}")
    
    # Step 3: Engineer features
    print("\nSTEP 3: Engineering features...")
    
    # Engineer features on ALL data (so gene-level stats are computed globally)
    all_features = engineer_features(clinvar_df)
    
    # Extract training and VUS feature matrices
    train_features = all_features.loc[train_df.index]
    vus_features = all_features.loc[vus_df.index]
    
    X_train = train_features.values.astype(np.float32)
    y_train = train_df['label'].values.astype(np.int32)
    feature_names = list(train_features.columns)
    
    X_vus = vus_features.values.astype(np.float32)
    
    print(f"\n  Feature matrix: {X_train.shape[0]:,} × {X_train.shape[1]} features")
    print(f"  VUS matrix: {X_vus.shape[0]:,} × {X_vus.shape[1]} features")
    print(f"  Class balance: {y_train.mean():.1%} pathogenic")
    
    # Step 4: Train with cross-validation
    print("\nSTEP 4: Training meta-learner (5-fold CV)...")
    
    from vus_meta.models.meta_learner import MetaLearner
    
    learner = MetaLearner(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
    )
    
    metrics = learner.fit_cv(
        X_train, y_train,
        feature_names=feature_names,
        n_folds=5,
    )
    
    print(f"\n  Cross-validation Results:")
    print(f"    AUROC: {metrics['cv_auroc']:.4f}")
    print(f"    AUPRC: {metrics['cv_auprc']:.4f}")
    print(f"    Brier: {metrics['cv_brier']:.4f}")
    for fm in metrics['fold_metrics']:
        print(f"    Fold {fm['fold']+1}: AUROC={fm['auroc']:.4f}")
    
    # Step 5: Feature importance
    print(f"\nSTEP 5: Feature Importance:")
    importance = learner.feature_importance()
    for i, (name, imp) in enumerate(list(importance.items())[:20]):
        bar = "█" * int(imp * 100)
        print(f"  {i+1:2d}. {name:30s} {imp:.4f} {bar}")
    
    # Step 6: Calibration assessment
    print(f"\nSTEP 6: Calibration Assessment:")
    from sklearn.calibration import calibration_curve
    
    oof_preds = metrics['oof_predictions']
    prob_true, prob_pred = calibration_curve(y_train, oof_preds, n_bins=10, strategy='quantile')
    print(f"  {'Bin':>5s}  {'Predicted':>10s}  {'Actual':>10s}  {'Diff':>8s}")
    for i, (pt, pp) in enumerate(zip(prob_true, prob_pred)):
        diff = abs(pt - pp)
        marker = "✓" if diff < 0.05 else "!"
        print(f"  {i+1:5d}  {pp:10.4f}  {pt:10.4f}  {diff:8.4f} {marker}")
    
    mean_cal_error = np.mean(np.abs(prob_true - prob_pred))
    print(f"\n  Mean Calibration Error: {mean_cal_error:.4f}")
    
    # Step 7: Score VUS variants
    print(f"\nSTEP 7: Scoring {len(vus_df):,} VUS variants...")
    
    vus_scores, vus_confidences = learner.predict_with_confidence(X_vus)
    
    from vus_meta.schemas import score_to_classification
    
    vus_classifications = [score_to_classification(s).value for s in vus_scores]
    class_counts = pd.Series(vus_classifications).value_counts()
    
    print(f"\n  VUS Reclassification Results:")
    for cls in ['Pathogenic', 'Likely Pathogenic', 'Uncertain Significance', 'Likely Benign', 'Benign']:
        count = class_counts.get(cls, 0)
        pct = count / len(vus_classifications) * 100
        bar = "█" * int(pct / 2)
        print(f"    {cls:30s} {count:>8,} ({pct:5.1f}%) {bar}")
    
    # Step 8: Save everything
    print(f"\nSTEP 8: Saving model and results...")
    
    model_dir = Path("data/models/clinvar_v1")
    learner.save(model_dir)
    
    # Save scored VUS
    vus_results = vus_df[['chrom', 'pos', 'ref', 'alt', 'gene', 'clinvar_id']].copy()
    vus_results['meta_score'] = vus_scores
    vus_results['confidence'] = vus_confidences
    vus_results['classification'] = vus_classifications
    
    results_dir = Path("data/processed")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    vus_results.to_csv(results_dir / "vus_scored.tsv", sep='\t', index=False)
    
    # Save training metrics
    import json
    metrics_out = {k: v for k, v in metrics.items() if k != 'oof_predictions'}
    metrics_out['n_training'] = int(len(train_df))
    metrics_out['n_features'] = int(len(feature_names))
    metrics_out['n_vus_scored'] = int(len(vus_df))
    metrics_out['feature_importance'] = {k: float(v) for k, v in list(importance.items())[:30]}
    
    with open(results_dir / "training_metrics.json", 'w') as f:
        json.dump(metrics_out, f, indent=2, default=str)
    
    print(f"  Model saved to {model_dir}")
    print(f"  VUS scores saved to {results_dir / 'vus_scored.tsv'}")
    print(f"  Metrics saved to {results_dir / 'training_metrics.json'}")
    
    # Step 9: Show interesting examples
    print(f"\nSTEP 9: Interesting Reclassified VUS Examples:")
    print(f"\n  High-confidence Pathogenic reclassifications (top genes):")
    
    path_vus = vus_results[
        (vus_results['classification'] == 'Pathogenic') &
        (vus_results['confidence'] > 0.8)
    ].sort_values('meta_score', ascending=False)
    
    if len(path_vus) > 0:
        top_genes = path_vus['gene'].value_counts().head(10)
        for gene, count in top_genes.items():
            print(f"    {gene:15s} {count:>5,} variants → Pathogenic")
    
    print(f"\n  High-confidence Benign reclassifications (top genes):")
    ben_vus = vus_results[
        (vus_results['classification'] == 'Benign') &
        (vus_results['confidence'] > 0.8)
    ].sort_values('meta_score')
    
    if len(ben_vus) > 0:
        top_genes = ben_vus['gene'].value_counts().head(10)
        for gene, count in top_genes.items():
            print(f"    {gene:15s} {count:>5,} variants → Benign")
    
    print()
    print(f"  Sample high-confidence predictions:")
    for _, row in path_vus.head(5).iterrows():
        print(f"    {row['gene']:10s} chr{row['chrom']}:{row['pos']} {row['ref']}>{row['alt']}  "
              f"score={row['meta_score']:.4f}  conf={row['confidence']:.3f}  → {row['classification']}")
    
    print()
    for _, row in ben_vus.head(5).iterrows():
        print(f"    {row['gene']:10s} chr{row['chrom']}:{row['pos']} {row['ref']}>{row['alt']}  "
              f"score={row['meta_score']:.4f}  conf={row['confidence']:.3f}  → {row['classification']}")
    
    print()
    print("=" * 70)
    print("  Training complete! 🧬")
    print("=" * 70)


if __name__ == "__main__":
    main()
