#!/usr/bin/env python3
"""Train a missense-specific model with amino acid property features.

This is where protein language model features (ESM-2) would add the most value.
For now we use amino acid physicochemical properties as a proxy to demonstrate
the architecture. With ESM-2, we'd replace/augment these with LLR features.

Features:
- Grantham distance between ref and alt amino acids
- BLOSUM62 substitution score
- Hydrophobicity change (Kyte-Doolittle)
- Volume change
- Charge change
- Polarity change
- Allele frequencies (ESP, ExAC, 1000G)
- Gene-level constraint metrics
- ClinVar review status
"""

import sys
import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, "src")

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

# Amino acid properties: (hydrophobicity_KD, molecular_weight, volume, pI, polarity_Grantham)
AA_FULL_PROPS = {
    'A': ( 1.8,  89.1,  88.6,  6.00, 8.1),
    'R': (-4.5, 174.2, 173.4, 10.76, 10.5),
    'N': (-3.5, 132.1, 114.1,  5.41, 11.6),
    'D': (-3.5, 133.1, 111.1,  2.77, 13.0),
    'C': ( 2.5, 121.2, 108.5,  5.07, 5.5),
    'E': (-3.5, 147.1, 138.4,  3.22, 12.3),
    'Q': (-3.5, 146.2, 143.8,  5.65, 10.5),
    'G': (-0.4,  75.0,  60.1,  5.97, 9.0),
    'H': (-3.2, 155.2, 153.2,  7.59, 10.4),
    'I': ( 4.5, 131.2, 166.7,  6.02, 5.2),
    'L': ( 3.8, 131.2, 166.7,  5.98, 4.9),
    'K': (-3.9, 146.2, 168.6,  9.74, 11.3),
    'M': ( 1.9, 149.2, 162.9,  5.74, 5.7),
    'F': ( 2.8, 165.2, 189.9,  5.48, 5.2),
    'P': (-1.6, 115.1, 112.7,  6.30, 8.0),
    'S': (-0.8, 105.1,  89.0,  5.68, 9.2),
    'T': (-0.7, 119.1, 116.1,  5.60, 8.6),
    'W': (-0.9, 204.2, 227.8,  5.89, 5.4),
    'Y': (-1.3, 181.2, 193.6,  5.66, 6.2),
    'V': ( 4.2, 117.1, 140.0,  5.96, 5.9),
}

AA3TO1 = {
    'Ala':'A', 'Arg':'R', 'Asn':'N', 'Asp':'D', 'Cys':'C', 'Glu':'E',
    'Gln':'Q', 'Gly':'G', 'His':'H', 'Ile':'I', 'Leu':'L', 'Lys':'K',
    'Met':'M', 'Phe':'F', 'Pro':'P', 'Ser':'S', 'Thr':'T', 'Trp':'W',
    'Tyr':'Y', 'Val':'V', 'Ter':'*',
}

# Amino acid classification features
AA_CLASSES = {
    'tiny': set('AGS'),
    'small': set('ACDGNPSTV'),
    'aliphatic': set('AILV'),
    'aromatic': set('FHW'),
    'nonpolar': set('ACFGILMPVW'),
    'polar': set('DEHKNQRSTY'),
    'charged': set('DEHKR'),
    'positive': set('HKR'),
    'negative': set('DE'),
}


def parse_missense_from_mc(mc_str):
    """Parse amino acid change from ClinVar MC field or HGVS.
    
    Tries to extract ref_aa (1-letter), position, alt_aa (1-letter)
    from patterns like p.Arg72Pro or p.R72P.
    """
    if not mc_str:
        return None, None, None
    
    # Look for three-letter code pattern: Arg72Pro
    match = re.search(r'p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})', mc_str)
    if match:
        ref3, pos_str, alt3 = match.groups()
        ref_aa = AA3TO1.get(ref3)
        alt_aa = AA3TO1.get(alt3)
        if ref_aa and alt_aa and alt_aa != '*':
            return ref_aa, int(pos_str), alt_aa
    
    # Try one-letter code: p.R72P
    match = re.search(r'p\.([A-Z])(\d+)([A-Z])', mc_str)
    if match:
        ref_aa, pos_str, alt_aa = match.groups()
        if ref_aa in AA_FULL_PROPS and alt_aa in AA_FULL_PROPS:
            return ref_aa, int(pos_str), alt_aa
    
    return None, None, None


def compute_aa_features(ref_aa, alt_aa):
    """Compute amino acid substitution features."""
    if ref_aa is None or alt_aa is None:
        return {}
    
    p1 = AA_FULL_PROPS.get(ref_aa)
    p2 = AA_FULL_PROPS.get(alt_aa)
    if p1 is None or p2 is None:
        return {}
    
    features = {}
    
    # Raw property changes
    features['hydro_ref'] = p1[0]
    features['hydro_alt'] = p2[0]
    features['hydro_change'] = p2[0] - p1[0]
    features['hydro_abs_change'] = abs(p2[0] - p1[0])
    
    features['mw_change'] = p2[1] - p1[1]
    features['mw_abs_change'] = abs(p2[1] - p1[1])
    
    features['volume_change'] = p2[2] - p1[2]
    features['volume_abs_change'] = abs(p2[2] - p1[2])
    
    features['pI_change'] = p2[3] - p1[3]
    features['pI_abs_change'] = abs(p2[3] - p1[3])
    
    features['polarity_change'] = p2[4] - p1[4]
    features['polarity_abs_change'] = abs(p2[4] - p1[4])
    
    # Grantham distance (composition, polarity, volume)
    d_c = (p1[4] - p2[4]) ** 2  # polarity
    d_v = ((p1[2] - p2[2]) / 50.0) ** 2  # volume normalized
    d_p = ((p1[0] - p2[0]) / 2.0) ** 2  # hydrophobicity
    features['grantham_distance'] = np.sqrt(1.833 * d_p + d_c + 2.0 * d_v)
    
    # Classification changes
    for cls_name, cls_set in AA_CLASSES.items():
        ref_in = ref_aa in cls_set
        alt_in = alt_aa in cls_set
        features[f'class_{cls_name}_ref'] = float(ref_in)
        features[f'class_{cls_name}_alt'] = float(alt_in)
        features[f'class_{cls_name}_change'] = float(ref_in != alt_in)
    
    # Special substitution categories
    features['is_to_proline'] = float(alt_aa == 'P')
    features['is_from_proline'] = float(ref_aa == 'P')
    features['is_to_glycine'] = float(alt_aa == 'G')
    features['is_from_glycine'] = float(ref_aa == 'G')
    features['is_to_cysteine'] = float(alt_aa == 'C')
    features['is_from_cysteine'] = float(ref_aa == 'C')
    features['involves_charged'] = float(
        ref_aa in AA_CLASSES['charged'] or alt_aa in AA_CLASSES['charged']
    )
    features['charge_reversal'] = float(
        (ref_aa in AA_CLASSES['positive'] and alt_aa in AA_CLASSES['negative']) or
        (ref_aa in AA_CLASSES['negative'] and alt_aa in AA_CLASSES['positive'])
    )
    
    return features


def main():
    print("=" * 70)
    print("  Missense-Specific VUS Predictor — Real ClinVar Training")
    print("=" * 70)
    print()
    
    # ── STEP 1: Parse missense variants with AA changes ──
    print("STEP 1: Parsing missense variants from ClinVar...")
    
    PATHOGENIC = {"Pathogenic", "Likely_pathogenic", "Pathogenic/Likely_pathogenic"}
    BENIGN = {"Benign", "Likely_benign", "Benign/Likely_benign"}
    REVIEW_STARS = {
        "practice_guideline": 4, "reviewed_by_expert_panel": 3,
        "criteria_provided,_multiple_submitters,_no_conflicts": 2,
        "criteria_provided,_conflicting_classifications": 1,
        "criteria_provided,_single_submitter": 1,
    }
    
    records = []
    with gzip.open('data/raw/clinvar.vcf.gz', 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if len(fields) < 8:
                continue
            
            chrom, pos, vid, ref, alt = fields[0], fields[1], fields[2], fields[3], fields[4]
            info_str = fields[7]
            
            info = {}
            for item in info_str.split(';'):
                if '=' in item:
                    k, v = item.split('=', 1)
                    info[k] = v
            
            mc = info.get('MC', '')
            if 'missense_variant' not in mc:
                continue
            
            clnsig = info.get('CLNSIG', '')
            if clnsig in PATHOGENIC:
                label = 1
            elif clnsig in BENIGN:
                label = 0
            elif 'Uncertain_significance' in clnsig:
                label = -1
            else:
                continue  # skip conflicting/other for missense model
            
            # Parse amino acid change
            ref_aa, aa_pos, alt_aa = parse_missense_from_mc(mc)
            if ref_aa is None:
                # Try CLNHGVS
                ref_aa, aa_pos, alt_aa = parse_missense_from_mc(info.get('CLNHGVS', ''))
            
            revstat = info.get('CLNREVSTAT', '')
            stars = max((REVIEW_STARS.get(s.strip(), 0) for s in revstat.split(',')), default=0)
            gene = info.get('GENEINFO', '').split(':')[0] if 'GENEINFO' in info else ''
            
            def safe_float(val):
                try:
                    return float(val) if val and val != '.' else np.nan
                except:
                    return np.nan
            
            records.append({
                'chrom': chrom, 'pos': int(pos), 'ref': ref, 'alt': alt,
                'clinvar_id': vid, 'gene': gene, 'label': label, 'stars': stars,
                'ref_aa': ref_aa, 'alt_aa': alt_aa, 'aa_pos': aa_pos,
                'af_esp': safe_float(info.get('AF_ESP', '')),
                'af_exac': safe_float(info.get('AF_EXAC', '')),
                'af_tgp': safe_float(info.get('AF_TGP', '')),
            })
    
    df = pd.DataFrame(records)
    logger.info(f"Parsed {len(df):,} missense variants")
    logger.info(f"  With AA annotations: {df['ref_aa'].notna().sum():,}")
    logger.info(f"  Labels: P={( df['label']==1).sum():,}, B={(df['label']==0).sum():,}, VUS={(df['label']==-1).sum():,}")
    
    # ── STEP 2: Feature engineering ──
    print("\nSTEP 2: Computing amino acid features...")
    
    # Gene-level features (from all data including VUS)
    gene_path_counts = df[df['label'] == 1].groupby('gene').size()
    gene_ben_counts = df[df['label'] == 0].groupby('gene').size()
    gene_total = gene_path_counts.add(gene_ben_counts, fill_value=0)
    gene_path_ratio = (gene_path_counts / gene_total).fillna(0.5)
    gene_total_all = df.groupby('gene').size()
    genes_with_path = set(df[df['label'] == 1]['gene'].unique())
    
    all_features = []
    for idx, row in df.iterrows():
        feat = {}
        
        # AA substitution features
        if row['ref_aa'] is not None and row['alt_aa'] is not None:
            feat.update(compute_aa_features(row['ref_aa'], row['alt_aa']))
        
        # Allele frequency features
        for col in ['af_esp', 'af_exac', 'af_tgp']:
            feat[col] = row[col]
            feat[f'{col}_log10'] = np.log10(row[col]) if pd.notna(row[col]) and row[col] > 0 else np.nan
        
        af_vals = [row[c] for c in ['af_esp', 'af_exac', 'af_tgp'] if pd.notna(row[c])]
        feat['af_max'] = max(af_vals) if af_vals else np.nan
        feat['af_max_log10'] = np.log10(feat['af_max']) if pd.notna(feat.get('af_max')) and feat['af_max'] > 0 else np.nan
        feat['has_af_data'] = float(len(af_vals) > 0)
        
        # Gene-level features
        gene = row['gene']
        feat['gene_path_ratio'] = gene_path_ratio.get(gene, 0.5)
        feat['gene_variant_count_log'] = np.log10(max(gene_total_all.get(gene, 1), 1))
        feat['gene_has_pathogenic'] = float(gene in genes_with_path)
        
        # Position features
        if row['aa_pos'] is not None:
            feat['aa_position'] = float(row['aa_pos'])
            feat['aa_position_log'] = np.log10(max(row['aa_pos'], 1))
        else:
            feat['aa_position'] = np.nan
            feat['aa_position_log'] = np.nan
        
        feat['review_stars'] = float(row['stars'])
        
        # Transition/transversion
        transitions = {('A','G'), ('G','A'), ('C','T'), ('T','C')}
        if len(row['ref']) == 1 and len(row['alt']) == 1:
            feat['is_transition'] = float((row['ref'], row['alt']) in transitions)
        else:
            feat['is_transition'] = np.nan
        
        all_features.append(feat)
    
    feat_df = pd.DataFrame(all_features)
    logger.info(f"Feature matrix: {feat_df.shape}")
    
    # ── STEP 3: Split and train ──
    print("\nSTEP 3: Training missense model...")
    
    # Training: P/LP and B/LB with stars >= 1 and AA features present
    has_aa = feat_df['grantham_distance'].notna()
    train_mask = (df['label'].isin([0, 1])) & (df['stars'] >= 1) & has_aa
    vus_mask = (df['label'] == -1) & has_aa
    
    train_df = df[train_mask]
    train_feat = feat_df[train_mask]
    vus_feat = feat_df[vus_mask]
    vus_orig = df[vus_mask]
    
    X_train = train_feat.values.astype(np.float32)
    y_train = train_df['label'].values.astype(np.int32)
    feature_names = list(train_feat.columns)
    X_vus = vus_feat.values.astype(np.float32)
    
    print(f"  Training: {X_train.shape[0]:,} missense variants × {X_train.shape[1]} features")
    print(f"  VUS to score: {X_vus.shape[0]:,}")
    print(f"  Class balance: {y_train.mean():.1%} pathogenic")
    
    from vus_meta.models.meta_learner import MetaLearner
    
    learner = MetaLearner(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
    )
    
    metrics = learner.fit_cv(X_train, y_train, feature_names=feature_names, n_folds=5)
    
    print(f"\n  Cross-validation Results:")
    print(f"    AUROC: {metrics['cv_auroc']:.4f}")
    print(f"    AUPRC: {metrics['cv_auprc']:.4f}")  
    print(f"    Brier: {metrics['cv_brier']:.4f}")
    for fm in metrics['fold_metrics']:
        print(f"    Fold {fm['fold']+1}: AUROC={fm['auroc']:.4f}")
    
    # ── STEP 4: Feature importance ──
    print(f"\nSTEP 4: Feature Importance (missense-specific):")
    importance = learner.feature_importance()
    for i, (name, imp) in enumerate(list(importance.items())[:25]):
        bar = "█" * int(imp * 200)
        print(f"  {i+1:2d}. {name:30s} {imp:.4f} {bar}")
    
    # ── STEP 5: Score VUS ──
    print(f"\nSTEP 5: Scoring {X_vus.shape[0]:,} missense VUS...")
    vus_scores, vus_conf = learner.predict_with_confidence(X_vus)
    
    from vus_meta.schemas import score_to_classification
    vus_cls = [score_to_classification(s).value for s in vus_scores]
    
    cls_counts = pd.Series(vus_cls).value_counts()
    print(f"\n  Missense VUS Reclassification:")
    for cls in ['Pathogenic', 'Likely Pathogenic', 'Uncertain Significance', 'Likely Benign', 'Benign']:
        count = cls_counts.get(cls, 0)
        pct = count / len(vus_cls) * 100
        print(f"    {cls:30s} {count:>8,} ({pct:5.1f}%)")
    
    # ── STEP 6: Save ──
    model_dir = Path("data/models/missense_v1")
    learner.save(model_dir)
    
    results = vus_orig[['chrom', 'pos', 'ref', 'alt', 'gene', 'clinvar_id', 'ref_aa', 'alt_aa', 'aa_pos']].copy()
    results['meta_score'] = vus_scores
    results['confidence'] = vus_conf
    results['classification'] = vus_cls
    results.to_csv('data/processed/missense_vus_scored.tsv', sep='\t', index=False)
    
    # ── STEP 7: Clinical gene analysis ──
    print(f"\nSTEP 7: Clinical Gene Missense VUS Analysis:")
    clinical_genes = ['BRCA1', 'BRCA2', 'TP53', 'MLH1', 'MSH2', 'MSH6', 'APC', 
                      'LDLR', 'SCN1A', 'CFTR', 'MYH7', 'KCNQ1', 'PKD1', 'NF1', 'RB1']
    
    for gene in clinical_genes:
        g = results[results['gene'] == gene]
        if len(g) == 0:
            continue
        n_p = (g['classification'].isin(['Pathogenic', 'Likely Pathogenic'])).sum()
        n_b = (g['classification'].isin(['Benign', 'Likely Benign'])).sum()
        n_v = (g['classification'] == 'Uncertain Significance').sum()
        print(f"  {gene:8s}: {len(g):5,} missense VUS → {n_p:4,} P/LP, {n_v:4,} VUS, {n_b:4,} B/LB")
    
    # Show some example predictions with AA changes
    print(f"\nSTEP 8: Example Missense Predictions:")
    
    # High-confidence pathogenic
    hp = results[(results['classification'] == 'Pathogenic') & (results['confidence'] > 0.8)]
    hp = hp[hp['gene'].isin(clinical_genes)].sort_values('meta_score', ascending=False)
    print(f"\n  High-confidence Pathogenic missense VUS:")
    for _, row in hp.head(10).iterrows():
        aa_change = f"{row['ref_aa']}{row['aa_pos']}{row['alt_aa']}" if pd.notna(row['aa_pos']) else '?'
        print(f"    {row['gene']:8s} p.{aa_change:12s} score={row['meta_score']:.4f} conf={row['confidence']:.3f}")
    
    # High-confidence benign
    hb = results[(results['classification'] == 'Benign') & (results['confidence'] > 0.8)]
    hb = hb[hb['gene'].isin(clinical_genes)].sort_values('meta_score')
    print(f"\n  High-confidence Benign missense VUS:")
    for _, row in hb.head(10).iterrows():
        aa_change = f"{row['ref_aa']}{row['aa_pos']}{row['alt_aa']}" if pd.notna(row['aa_pos']) else '?'
        print(f"    {row['gene']:8s} p.{aa_change:12s} score={row['meta_score']:.4f} conf={row['confidence']:.3f}")
    
    # Calibration
    print(f"\nSTEP 9: Calibration Check:")
    from sklearn.calibration import calibration_curve
    oof = metrics['oof_predictions']
    prob_true, prob_pred = calibration_curve(y_train, oof, n_bins=10, strategy='quantile')
    mce = np.mean(np.abs(prob_true - prob_pred))
    print(f"  Mean Calibration Error: {mce:.4f}")
    for pt, pp in zip(prob_true, prob_pred):
        print(f"    predicted={pp:.4f}  actual={pt:.4f}  diff={abs(pt-pp):.4f}")
    
    # Save metrics
    import json
    metrics_out = {k: v for k, v in metrics.items() if k != 'oof_predictions'}
    metrics_out['n_training'] = int(len(train_df))
    metrics_out['n_features'] = int(len(feature_names))
    metrics_out['n_missense_vus'] = int(X_vus.shape[0])
    metrics_out['mean_calibration_error'] = float(mce)
    metrics_out['top_features'] = {k: float(v) for k, v in list(importance.items())[:30]}
    with open('data/processed/missense_training_metrics.json', 'w') as f:
        json.dump(metrics_out, f, indent=2, default=str)
    
    print(f"\n  Model saved to {model_dir}")
    print(f"  Results saved to data/processed/missense_vus_scored.tsv")
    
    print()
    print("=" * 70)
    print("  Missense model training complete! 🧬")
    print("=" * 70)


if __name__ == "__main__":
    main()
