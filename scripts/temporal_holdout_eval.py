#!/usr/bin/env python3
"""Temporal hold-out evaluation: train on pre-2024, test on 2024+ ClinVar submissions.

This is the gold-standard evaluation for variant interpretation tools.
Variants reclassified from VUS to P/B after the training cutoff date
serve as the test set.
"""

import sys, gzip, re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

sys.path.insert(0, "src")
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

AA3TO1 = {
    'Ala':'A','Arg':'R','Asn':'N','Asp':'D','Cys':'C','Glu':'E','Gln':'Q','Gly':'G',
    'His':'H','Ile':'I','Leu':'L','Lys':'K','Met':'M','Phe':'F','Pro':'P','Ser':'S',
    'Thr':'T','Trp':'W','Tyr':'Y','Val':'V',
}

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


def grantham(a1, a2):
    p1, p2 = AA_PROPS.get(a1), AA_PROPS.get(a2)
    if not p1 or not p2 or a1 == a2: return 0.0
    return float(np.sqrt(1.833*((p1[0]-p2[0])/2)**2 + (p1[4]-p2[4])**2 + 2.0*((p1[1]-p2[1])/50)**2))


def main():
    print("=" * 70)
    print("  Temporal Hold-Out Evaluation")
    print("  Train: pre-2024 | Test: 2024+ submissions")
    print("=" * 70)
    
    # ── Parse with dates ──
    logger.info("Parsing variant_summary with dates...")
    
    PATHOGENIC = {"Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic"}
    BENIGN = {"Benign", "Likely benign", "Benign/Likely benign"}
    
    records = []
    with gzip.open('data/raw/variant_summary.txt.gz', 'rt') as f:
        header = f.readline().strip().split('\t')
        col_idx = {c: i for i, c in enumerate(header)}
        
        for line in f:
            fields = line.strip().split('\t')
            if len(fields) < 34: continue
            if fields[col_idx['Assembly']] != 'GRCh38': continue
            
            name = fields[col_idx['Name']]
            if 'p.' not in name: continue
            
            m = re.search(r'p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})', name)
            if not m: continue
            
            ref_aa = AA3TO1.get(m.group(1))
            alt_aa = AA3TO1.get(m.group(3))
            if not ref_aa or not alt_aa or ref_aa == '*' or alt_aa == '*': continue
            
            clinsig = fields[col_idx['ClinicalSignificance']]
            if any(t in clinsig for t in PATHOGENIC): label = 1
            elif any(t in clinsig for t in BENIGN): label = 0
            else: continue  # Only use P/LP and B/LB for this evaluation
            
            # Parse last evaluated date
            last_eval = fields[col_idx['LastEvaluated']]
            # Format is like "Jan 01, 2024" or "2024-01-01" or "-"
            year = None
            if last_eval and last_eval != '-':
                ym = re.search(r'(20\d{2})', last_eval)
                if ym:
                    year = int(ym.group(1))
            
            gene = fields[col_idx['GeneSymbol']]
            chrom = fields[col_idx['Chromosome']]
            
            review = fields[col_idx['ReviewStatus']]
            
            records.append({
                'gene': gene, 'ref_aa': ref_aa, 'alt_aa': alt_aa,
                'aa_pos': int(m.group(2)), 'label': label,
                'chrom': chrom, 'year': year, 'review': review,
            })
    
    df = pd.DataFrame(records).drop_duplicates(subset=['gene', 'aa_pos', 'ref_aa', 'alt_aa'])
    
    logger.info(f"Total labeled missense: {len(df):,}")
    logger.info(f"  With year info: {df['year'].notna().sum():,}")
    logger.info(f"  Year distribution:")
    year_counts = df['year'].value_counts().sort_index()
    for y in range(2018, 2027):
        c = year_counts.get(y, 0)
        logger.info(f"    {y}: {c:,}")
    
    # ── Temporal split ──
    TRAIN_CUTOFF = 2024
    
    train_mask = df['year'].notna() & (df['year'] < TRAIN_CUTOFF)
    test_mask = df['year'].notna() & (df['year'] >= TRAIN_CUTOFF)
    
    train_df = df[train_mask]
    test_df = df[test_mask]
    
    print(f"\n  Temporal split:")
    print(f"    Train (pre-{TRAIN_CUTOFF}): {len(train_df):,} (P={(train_df['label']==1).sum():,}, B={(train_df['label']==0).sum():,})")
    print(f"    Test ({TRAIN_CUTOFF}+):     {len(test_df):,} (P={(test_df['label']==1).sum():,}, B={(test_df['label']==0).sum():,})")
    
    # ── Compute features ──
    logger.info("Computing features...")
    
    # Gene-level from training data only (no leakage!)
    gp = train_df[train_df['label']==1].groupby('gene').size()
    gb = train_df[train_df['label']==0].groupby('gene').size()
    gt = gp.add(gb, fill_value=0)
    gene_path_ratio = (gp / gt).fillna(0.5)
    gene_total = df.groupby('gene').size()  # total can use all data
    genes_with_path = set(train_df[train_df['label']==1]['gene'].unique())
    
    def compute_features(row_df):
        feats = []
        for _, row in row_df.iterrows():
            f = {}
            p1, p2 = AA_PROPS.get(row['ref_aa']), AA_PROPS.get(row['alt_aa'])
            if p1 and p2:
                f['hydro_abs_change'] = abs(p2[0] - p1[0])
                f['volume_abs_change'] = abs(p2[1] - p1[1])
                f['mw_abs_change'] = abs(p2[2] - p1[2])
                f['pI_abs_change'] = abs(p2[3] - p1[3])
                f['polarity_abs_change'] = abs(p2[4] - p1[4])
                f['grantham'] = grantham(row['ref_aa'], row['alt_aa'])
                
                for cn, cs in AA_CLASSES.items():
                    f[f'{cn}_change'] = float((row['ref_aa'] in cs) != (row['alt_aa'] in cs))
                
                f['to_proline'] = float(row['alt_aa'] == 'P')
                f['from_proline'] = float(row['ref_aa'] == 'P')
                f['from_cysteine'] = float(row['ref_aa'] == 'C')
                f['to_cysteine'] = float(row['alt_aa'] == 'C')
                f['charge_reversal'] = float(
                    (row['ref_aa'] in AA_CLASSES['positive'] and row['alt_aa'] in AA_CLASSES['negative']) or
                    (row['ref_aa'] in AA_CLASSES['negative'] and row['alt_aa'] in AA_CLASSES['positive'])
                )
            
            f['aa_position_log'] = np.log10(max(row['aa_pos'], 1))
            gene = row['gene']
            f['gene_path_ratio'] = gene_path_ratio.get(gene, 0.5)
            f['gene_variant_count_log'] = np.log10(max(gene_total.get(gene, 1), 1))
            f['gene_has_pathogenic'] = float(gene in genes_with_path)
            
            feats.append(f)
        return pd.DataFrame(feats, index=row_df.index)
    
    train_feats = compute_features(train_df)
    test_feats = compute_features(test_df)
    
    feature_names = list(train_feats.columns)
    X_train = train_feats.values.astype(np.float32)
    y_train = train_df['label'].values.astype(np.int32)
    X_test = test_feats.values.astype(np.float32)
    y_test = test_df['label'].values.astype(np.int32)
    
    # ── Train and evaluate ──
    from vus_meta.models.meta_learner import MetaLearner
    
    print(f"\nTraining on pre-{TRAIN_CUTOFF} data ({X_train.shape[0]:,} × {X_train.shape[1]})...")
    
    learner = MetaLearner(n_estimators=500, max_depth=6, learning_rate=0.05, min_child_weight=10)
    metrics = learner.fit(X_train, y_train, feature_names=feature_names)
    
    # Predict on held-out test set
    test_scores = learner.predict(X_test)
    
    auroc = roc_auc_score(y_test, test_scores)
    auprc = average_precision_score(y_test, test_scores)
    
    print(f"\n{'='*60}")
    print(f"  TEMPORAL HOLD-OUT RESULTS")
    print(f"  Train: pre-{TRAIN_CUTOFF} ({X_train.shape[0]:,})")
    print(f"  Test:  {TRAIN_CUTOFF}+   ({X_test.shape[0]:,})")
    print(f"{'='*60}")
    print(f"  AUROC: {auroc:.4f}")
    print(f"  AUPRC: {auprc:.4f}")
    print(f"{'='*60}")
    
    # Precision at various recall levels
    precision, recall, _ = precision_recall_curve(y_test, test_scores)
    print(f"\n  Precision @ Recall levels:")
    for target_recall in [0.95, 0.90, 0.80, 0.50]:
        idx = np.argmin(np.abs(recall - target_recall))
        print(f"    Recall={target_recall:.0%}: Precision={precision[idx]:.4f}")
    
    # Classification breakdown
    from vus_meta.schemas import score_to_classification
    test_cls = [score_to_classification(s).value for s in test_scores]
    
    # True positive / false positive analysis
    tp = sum(1 for c, l in zip(test_cls, y_test) if c in ('Pathogenic', 'Likely Pathogenic') and l == 1)
    fp = sum(1 for c, l in zip(test_cls, y_test) if c in ('Pathogenic', 'Likely Pathogenic') and l == 0)
    tn = sum(1 for c, l in zip(test_cls, y_test) if c in ('Benign', 'Likely Benign') and l == 0)
    fn = sum(1 for c, l in zip(test_cls, y_test) if c in ('Benign', 'Likely Benign') and l == 1)
    vus = sum(1 for c in test_cls if c == 'Uncertain Significance')
    
    print(f"\n  Classification on test set:")
    print(f"    True Positives (P/LP & actually path):   {tp:5,}")
    print(f"    False Positives (P/LP & actually benign): {fp:5,}")
    print(f"    True Negatives (B/LB & actually benign):  {tn:5,}")
    print(f"    False Negatives (B/LB & actually path):   {fn:5,}")
    print(f"    Uncertain (remained VUS):                 {vus:5,}")
    
    if tp + fp > 0:
        print(f"\n    Positive Predictive Value: {tp/(tp+fp):.4f}")
    if tn + fn > 0:
        print(f"    Negative Predictive Value: {tn/(tn+fn):.4f}")
    
    # Save metrics
    import json
    results = {
        'train_cutoff': TRAIN_CUTOFF,
        'n_train': int(X_train.shape[0]),
        'n_test': int(X_test.shape[0]),
        'test_auroc': float(auroc),
        'test_auprc': float(auprc),
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'vus_remaining': vus,
        'ppv': float(tp/(tp+fp)) if (tp+fp) > 0 else None,
        'npv': float(tn/(tn+fn)) if (tn+fn) > 0 else None,
    }
    with open('data/processed/temporal_holdout_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Results saved to data/processed/temporal_holdout_results.json")
    print(f"\n{'='*60}")
    print(f"  Temporal evaluation complete! 📊")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
