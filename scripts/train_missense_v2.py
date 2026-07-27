#!/usr/bin/env python3
"""Train missense model using ClinVar variant_summary (with protein annotations)."""

import sys, gzip, re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, "src")
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

# ── Amino acid data ──
AA3TO1 = {
    'Ala':'A','Arg':'R','Asn':'N','Asp':'D','Cys':'C','Glu':'E','Gln':'Q','Gly':'G',
    'His':'H','Ile':'I','Leu':'L','Lys':'K','Met':'M','Phe':'F','Pro':'P','Ser':'S',
    'Thr':'T','Trp':'W','Tyr':'Y','Val':'V','Ter':'*',
}

# Kyte-Doolittle hydrophobicity, volume (Å³), molecular weight, pI, Grantham polarity
AA_PROPS = {
    'A': ( 1.8,  88.6,  89.1,  6.00,  8.1), 'R': (-4.5, 173.4, 174.2, 10.76, 10.5),
    'N': (-3.5, 114.1, 132.1,  5.41, 11.6), 'D': (-3.5, 111.1, 133.1,  2.77, 13.0),
    'C': ( 2.5, 108.5, 121.2,  5.07,  5.5), 'E': (-3.5, 138.4, 147.1,  3.22, 12.3),
    'Q': (-3.5, 143.8, 146.2,  5.65, 10.5), 'G': (-0.4,  60.1,  75.0,  5.97,  9.0),
    'H': (-3.2, 153.2, 155.2,  7.59, 10.4), 'I': ( 4.5, 166.7, 131.2,  6.02,  5.2),
    'L': ( 3.8, 166.7, 131.2,  5.98,  4.9), 'K': (-3.9, 168.6, 146.2,  9.74, 11.3),
    'M': ( 1.9, 162.9, 149.2,  5.74,  5.7), 'F': ( 2.8, 189.9, 165.2,  5.48,  5.2),
    'P': (-1.6, 112.7, 115.1,  6.30,  8.0), 'S': (-0.8,  89.0, 105.1,  5.68,  9.2),
    'T': (-0.7, 116.1, 119.1,  5.60,  8.6), 'W': (-0.9, 227.8, 204.2,  5.89,  5.4),
    'Y': (-1.3, 193.6, 181.2,  5.66,  6.2), 'V': ( 4.2, 140.0, 117.1,  5.96,  5.9),
}

AA_CLASSES = {
    'tiny': set('AGS'), 'small': set('ACDGNPSTV'), 'aliphatic': set('AILV'),
    'aromatic': set('FHWY'), 'nonpolar': set('ACFGILMPVW'), 'polar': set('DEHKNQRSTY'),
    'charged': set('DEHKR'), 'positive': set('HKR'), 'negative': set('DE'),
}

# Precomputed Grantham distance matrix
def grantham_distance(a1, a2):
    if a1 == a2: return 0.0
    p1, p2 = AA_PROPS.get(a1), AA_PROPS.get(a2)
    if not p1 or not p2: return np.nan
    return np.sqrt(1.833*((p1[0]-p2[0])/2)**2 + ((p1[4]-p2[4]))**2 + 2.0*((p1[1]-p2[1])/50)**2)


def parse_protein_change(name_str):
    """Extract ref_aa, position, alt_aa from ClinVar Name field."""
    # Pattern: (p.Xxx123Yyy) — three-letter code
    m = re.search(r'p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})', name_str)
    if m:
        ref3, pos, alt3 = m.group(1), int(m.group(2)), m.group(3)
        ref1, alt1 = AA3TO1.get(ref3), AA3TO1.get(alt3)
        if ref1 and alt1 and ref1 != '*' and alt1 != '*':
            return ref1, pos, alt1
    return None, None, None


def compute_features(ref_aa, alt_aa, aa_pos, af_val, gene_path_ratio_val,
                     gene_count_log, gene_has_path, stars, chrom, pos,
                     ref_nuc, alt_nuc):
    """Compute all features for a single missense variant."""
    f = {}
    
    # ── AA substitution features ──
    p1 = AA_PROPS.get(ref_aa)
    p2 = AA_PROPS.get(alt_aa)
    if p1 and p2:
        f['hydro_change'] = p2[0] - p1[0]
        f['hydro_abs_change'] = abs(f['hydro_change'])
        f['volume_change'] = p2[1] - p1[1]
        f['volume_abs_change'] = abs(f['volume_change'])
        f['mw_change'] = p2[2] - p1[2]
        f['mw_abs_change'] = abs(f['mw_change'])
        f['pI_change'] = p2[3] - p1[3]
        f['pI_abs_change'] = abs(f['pI_change'])
        f['polarity_change'] = p2[4] - p1[4]
        f['polarity_abs_change'] = abs(f['polarity_change'])
        f['grantham'] = grantham_distance(ref_aa, alt_aa)
    else:
        for k in ['hydro_change','hydro_abs_change','volume_change','volume_abs_change',
                   'mw_change','mw_abs_change','pI_change','pI_abs_change',
                   'polarity_change','polarity_abs_change','grantham']:
            f[k] = np.nan
    
    # AA class change features
    for cls_name, cls_set in AA_CLASSES.items():
        f[f'{cls_name}_change'] = float((ref_aa in cls_set) != (alt_aa in cls_set)) if ref_aa and alt_aa else np.nan
    
    # Special substitution flags
    f['to_proline'] = float(alt_aa == 'P') if alt_aa else 0
    f['from_proline'] = float(ref_aa == 'P') if ref_aa else 0
    f['to_glycine'] = float(alt_aa == 'G') if alt_aa else 0
    f['from_glycine'] = float(ref_aa == 'G') if ref_aa else 0
    f['to_cysteine'] = float(alt_aa == 'C') if alt_aa else 0
    f['from_cysteine'] = float(ref_aa == 'C') if ref_aa else 0
    f['charge_reversal'] = float(
        (ref_aa in AA_CLASSES['positive'] and alt_aa in AA_CLASSES['negative']) or
        (ref_aa in AA_CLASSES['negative'] and alt_aa in AA_CLASSES['positive'])
    ) if ref_aa and alt_aa else 0
    
    # ── Position features ──
    f['aa_position'] = float(aa_pos) if aa_pos else np.nan
    f['aa_position_log'] = np.log10(max(aa_pos, 1)) if aa_pos else np.nan
    
    # ── Allele frequency ──
    f['af_max'] = af_val if not np.isnan(af_val) else np.nan
    f['af_max_log10'] = np.log10(af_val) if not np.isnan(af_val) and af_val > 0 else np.nan
    f['has_af_data'] = float(not np.isnan(af_val))
    
    # ── Gene features ──
    f['gene_path_ratio'] = gene_path_ratio_val
    f['gene_variant_count_log'] = gene_count_log
    f['gene_has_pathogenic'] = gene_has_path
    
    # ── Other ──
    f['review_stars'] = float(stars)
    
    # Transition/transversion
    transitions = {('A','G'), ('G','A'), ('C','T'), ('T','C')}
    if len(ref_nuc) == 1 and len(alt_nuc) == 1:
        f['is_transition'] = float((ref_nuc, alt_nuc) in transitions)
    else:
        f['is_transition'] = np.nan
    
    f['is_autosome'] = float(chrom not in ('X', 'Y', 'MT'))
    
    return f


def main():
    print("=" * 70)
    print("  Missense VUS Predictor v2 — With Amino Acid Features")
    print("=" * 70)
    
    # ── Parse variant_summary for missense with protein changes ──
    print("\nSTEP 1: Parsing variant_summary.txt.gz for missense variants...")
    
    PATHOGENIC = {"Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic"}
    BENIGN = {"Benign", "Likely benign", "Benign/Likely benign"}
    REVIEW_STARS = {
        "practice guideline": 4, "reviewed by expert panel": 3,
        "criteria provided, multiple submitters, no conflicts": 2,
        "criteria provided, conflicting classifications": 1,
        "criteria provided, single submitter": 1,
    }
    
    records = []
    total_lines = 0
    
    with gzip.open('data/raw/variant_summary.txt.gz', 'rt') as f:
        header = f.readline().strip().split('\t')
        col_idx = {c: i for i, c in enumerate(header)}
        
        for line in f:
            total_lines += 1
            if total_lines % 1000000 == 0:
                logger.info(f"  Processed {total_lines:,} lines, kept {len(records):,}...")
            
            fields = line.strip().split('\t')
            if len(fields) < 34:
                continue
            
            assembly = fields[col_idx['Assembly']]
            if assembly != 'GRCh38':
                continue
            
            name = fields[col_idx['Name']]
            if 'p.' not in name:
                continue
            
            # Parse protein change
            ref_aa, aa_pos, alt_aa = parse_protein_change(name)
            if ref_aa is None:
                continue
            
            # Clinical significance
            clinsig = fields[col_idx['ClinicalSignificance']]
            if any(t in clinsig for t in PATHOGENIC):
                label = 1
            elif any(t in clinsig for t in BENIGN):
                label = 0
            elif 'Uncertain significance' in clinsig:
                label = -1
            else:
                continue
            
            gene = fields[col_idx['GeneSymbol']]
            chrom = fields[col_idx['Chromosome']]
            pos_str = fields[col_idx['PositionVCF']]
            ref_nuc = fields[col_idx['ReferenceAlleleVCF']]
            alt_nuc = fields[col_idx['AlternateAlleleVCF']]
            
            review = fields[col_idx['ReviewStatus']]
            stars = max((REVIEW_STARS.get(r.strip(), 0) for r in review.split(',')), default=0)
            
            try:
                pos_int = int(pos_str)
            except:
                continue
            
            records.append({
                'chrom': chrom, 'pos': pos_int, 'ref': ref_nuc, 'alt': alt_nuc,
                'gene': gene, 'name': name[:100], 'ref_aa': ref_aa, 'alt_aa': alt_aa,
                'aa_pos': aa_pos, 'label': label, 'stars': stars, 'clinsig': clinsig,
            })
    
    df = pd.DataFrame(records)
    # Drop exact duplicates (same variant appearing multiple times)
    df = df.drop_duplicates(subset=['chrom', 'pos', 'ref', 'alt'])
    
    logger.info(f"Parsed {len(df):,} unique missense variants with protein changes")
    logger.info(f"  Pathogenic: {(df['label']==1).sum():,}")
    logger.info(f"  Benign: {(df['label']==0).sum():,}")
    logger.info(f"  VUS: {(df['label']==-1).sum():,}")
    
    # ── Compute gene-level features ──
    print("\nSTEP 2: Computing gene-level features...")
    
    gp = df[df['label']==1].groupby('gene').size()
    gb = df[df['label']==0].groupby('gene').size()
    gt = gp.add(gb, fill_value=0)
    gene_path_ratio = (gp / gt).fillna(0.5)
    gene_total_all = df.groupby('gene').size()
    genes_with_path = set(df[df['label']==1]['gene'].unique())
    
    # ── Compute features ──
    print("\nSTEP 3: Computing amino acid features...")
    
    all_feats = []
    for idx, row in df.iterrows():
        gene = row['gene']
        f = compute_features(
            ref_aa=row['ref_aa'], alt_aa=row['alt_aa'], aa_pos=row['aa_pos'],
            af_val=np.nan,  # No AF in variant_summary
            gene_path_ratio_val=gene_path_ratio.get(gene, 0.5),
            gene_count_log=np.log10(max(gene_total_all.get(gene, 1), 1)),
            gene_has_path=float(gene in genes_with_path),
            stars=row['stars'], chrom=row['chrom'], pos=row['pos'],
            ref_nuc=row['ref'], alt_nuc=row['alt'],
        )
        all_feats.append(f)
        
        if (idx + 1) % 200000 == 0:
            logger.info(f"  Features computed for {idx+1:,} variants...")
    
    feat_df = pd.DataFrame(all_feats)
    logger.info(f"Feature matrix shape: {feat_df.shape}")
    
    # ── Split and train ──
    print("\nSTEP 4: Training missense model (5-fold CV)...")
    
    train_mask = (df['label'].isin([0, 1])) & (df['stars'] >= 1)
    vus_mask = (df['label'] == -1)
    
    X_train = feat_df[train_mask].values.astype(np.float32)
    y_train = df[train_mask]['label'].values.astype(np.int32)
    feature_names = list(feat_df.columns)
    X_vus = feat_df[vus_mask].values.astype(np.float32)
    
    print(f"  Training: {X_train.shape[0]:,} × {X_train.shape[1]} features")
    print(f"  VUS: {X_vus.shape[0]:,}")
    print(f"  Pathogenic rate: {y_train.mean():.1%}")
    
    from vus_meta.models.meta_learner import MetaLearner
    
    learner = MetaLearner(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
    )
    
    metrics = learner.fit_cv(X_train, y_train, feature_names=feature_names, n_folds=5)
    
    print(f"\n  ╔══════════════════════════════════╗")
    print(f"  ║  MISSENSE MODEL CV RESULTS       ║")
    print(f"  ╠══════════════════════════════════╣")
    print(f"  ║  AUROC: {metrics['cv_auroc']:.4f}                 ║")
    print(f"  ║  AUPRC: {metrics['cv_auprc']:.4f}                 ║")
    print(f"  ║  Brier: {metrics['cv_brier']:.4f}                 ║")
    print(f"  ╚══════════════════════════════════╝")
    for fm in metrics['fold_metrics']:
        print(f"    Fold {fm['fold']+1}: AUROC={fm['auroc']:.4f}")
    
    # ── Feature importance ──
    print(f"\nSTEP 5: Feature Importance (amino acid features):")
    importance = learner.feature_importance()
    for i, (name, imp) in enumerate(list(importance.items())[:25]):
        bar = "█" * int(imp * 200)
        print(f"  {i+1:2d}. {name:30s} {imp:.4f} {bar}")
    
    # ── Score VUS ──
    print(f"\nSTEP 6: Scoring {X_vus.shape[0]:,} missense VUS...")
    vus_scores, vus_conf = learner.predict_with_confidence(X_vus)
    
    from vus_meta.schemas import score_to_classification
    vus_cls = [score_to_classification(s).value for s in vus_scores]
    
    print(f"\n  Reclassification results:")
    for cls in ['Pathogenic', 'Likely Pathogenic', 'Uncertain Significance', 'Likely Benign', 'Benign']:
        n = sum(1 for c in vus_cls if c == cls)
        pct = n / len(vus_cls) * 100
        bar = "█" * int(pct / 2)
        print(f"    {cls:30s} {n:>8,} ({pct:5.1f}%) {bar}")
    
    # ── Save ──
    model_dir = Path("data/models/missense_v2")
    learner.save(model_dir)
    
    vus_df = df[vus_mask].copy()
    vus_df['meta_score'] = vus_scores
    vus_df['confidence'] = vus_conf
    vus_df['classification'] = vus_cls
    vus_df.to_csv('data/processed/missense_v2_scored.tsv', sep='\t', index=False)
    
    # ── Clinical gene detail ──
    print(f"\nSTEP 7: Clinical Gene Missense Analysis:")
    for gene in ['BRCA1','BRCA2','TP53','MLH1','MSH2','APC','LDLR','SCN1A','CFTR','MYH7','NF1','RB1','KCNQ1']:
        g = vus_df[vus_df['gene'] == gene]
        if len(g) == 0: continue
        np_ = g['classification'].isin(['Pathogenic','Likely Pathogenic']).sum()
        nb = g['classification'].isin(['Benign','Likely Benign']).sum()
        nv = (g['classification'] == 'Uncertain Significance').sum()
        print(f"  {gene:8s}: {len(g):5,} → {np_:4,} P/LP, {nv:4,} VUS, {nb:4,} B/LB")
    
    # ── Example predictions ──
    print(f"\n  Top pathogenic predictions in BRCA1/BRCA2/TP53:")
    for gene in ['BRCA1', 'BRCA2', 'TP53']:
        g = vus_df[(vus_df['gene'] == gene) & (vus_df['classification'].isin(['Pathogenic','Likely Pathogenic']))]
        g = g.sort_values('meta_score', ascending=False)
        for _, r in g.head(5).iterrows():
            print(f"    {gene:6s} p.{r['ref_aa']}{r['aa_pos']}{r['alt_aa']:4s}  score={r['meta_score']:.4f} → {r['classification']}")
    
    # ── Calibration ──
    from sklearn.calibration import calibration_curve
    oof = metrics['oof_predictions']
    prob_true, prob_pred = calibration_curve(y_train, oof, n_bins=10, strategy='quantile')
    mce = np.mean(np.abs(prob_true - prob_pred))
    print(f"\n  Calibration: MCE = {mce:.4f}")
    
    # Save metrics
    import json
    m = {k: v for k, v in metrics.items() if k != 'oof_predictions'}
    m['n_training'] = int(X_train.shape[0])
    m['n_features'] = int(X_train.shape[1])
    m['n_vus'] = int(X_vus.shape[0])
    m['mce'] = float(mce)
    m['top_features'] = {k: float(v) for k, v in list(importance.items())[:30]}
    with open('data/processed/missense_v2_metrics.json', 'w') as f:
        json.dump(m, f, indent=2, default=str)
    
    print(f"\n  Model → {model_dir}")
    print(f"  Results → data/processed/missense_v2_scored.tsv")
    print(f"\n{'='*70}")
    print(f"  Missense model v2 complete! 🧬")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
