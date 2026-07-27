#!/usr/bin/env python3
"""Train the integrated model: AA features + ESM-2 + gene features.

This is the final model that combines:
1. Amino acid physicochemical features (Grantham, hydrophobicity, etc.)
2. ESM-2 protein language model features (LLR, entropy, rank)
3. Gene-level constraint features
"""

import sys, re, math
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

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


def grantham_distance(a1, a2):
    if a1 == a2: return 0.0
    p1, p2 = AA_PROPS.get(a1), AA_PROPS.get(a2)
    if not p1 or not p2: return np.nan
    return np.sqrt(1.833*((p1[0]-p2[0])/2)**2 + ((p1[4]-p2[4]))**2 + 2.0*((p1[1]-p2[1])/50)**2)


def compute_aa_features(ref_aa, alt_aa):
    """Compute AA substitution features."""
    f = {}
    p1, p2 = AA_PROPS.get(ref_aa), AA_PROPS.get(alt_aa)
    if not p1 or not p2:
        return f
    
    f['hydro_change'] = p2[0] - p1[0]
    f['hydro_abs_change'] = abs(f['hydro_change'])
    f['volume_change'] = p2[1] - p1[1]
    f['volume_abs_change'] = abs(f['volume_change'])
    f['mw_abs_change'] = abs(p2[2] - p1[2])
    f['pI_abs_change'] = abs(p2[3] - p1[3])
    f['polarity_abs_change'] = abs(p2[4] - p1[4])
    f['grantham'] = grantham_distance(ref_aa, alt_aa)
    
    for cn, cs in AA_CLASSES.items():
        f[f'{cn}_change'] = float((ref_aa in cs) != (alt_aa in cs))
    
    f['to_proline'] = float(alt_aa == 'P')
    f['from_proline'] = float(ref_aa == 'P')
    f['to_glycine'] = float(alt_aa == 'G')
    f['from_cysteine'] = float(ref_aa == 'C')
    f['to_cysteine'] = float(alt_aa == 'C')
    f['charge_reversal'] = float(
        (ref_aa in AA_CLASSES['positive'] and alt_aa in AA_CLASSES['negative']) or
        (ref_aa in AA_CLASSES['negative'] and alt_aa in AA_CLASSES['positive'])
    )
    return f


def main():
    print("=" * 70)
    print("  Integrated Model: AA Features + ESM-2 + Gene Constraints")
    print("=" * 70)
    
    # ── Load ESM-2 features ──
    print("\nSTEP 1: Loading ESM-2 features...")
    esm2_df = pd.read_csv('data/processed/esm2_features.tsv', sep='\t', index_col=0)
    logger.info(f"ESM-2 features: {len(esm2_df):,} variants, columns: {list(esm2_df.columns)}")
    
    # Parse variant info from index (gene_RefPosAlt)
    esm2_df['gene'] = esm2_df.index.map(lambda x: x.split('_')[0])
    esm2_df['aa_change'] = esm2_df.index.map(lambda x: x.split('_')[1] if '_' in x else '')
    esm2_df['ref_aa'] = esm2_df['aa_change'].map(lambda x: x[0] if len(x) > 0 else '')
    esm2_df['alt_aa'] = esm2_df['aa_change'].map(lambda x: x[-1] if len(x) > 0 else '')
    esm2_df['aa_pos'] = esm2_df['aa_change'].map(
        lambda x: int(re.search(r'(\d+)', x).group(1)) if re.search(r'(\d+)', x) else 0
    )
    
    # ── Compute combined features ──
    print("\nSTEP 2: Computing combined feature vectors...")
    
    # Gene-level features
    labeled = esm2_df[esm2_df['label'].isin([0, 1])]
    gp = labeled[labeled['label']==1].groupby('gene').size()
    gb = labeled[labeled['label']==0].groupby('gene').size()
    gt = gp.add(gb, fill_value=0)
    gene_path_ratio = (gp / gt).fillna(0.5)
    gene_total_all = esm2_df.groupby('gene').size()
    genes_with_path = set(esm2_df[esm2_df['label']==1]['gene'].unique())
    
    all_features = []
    for idx, row in esm2_df.iterrows():
        f = {}
        
        # ESM-2 features
        for col in ['esm2_llr', 'esm2_wt_logp', 'esm2_mut_logp', 'esm2_wt_prob',
                     'esm2_mut_prob', 'esm2_wt_rank', 'esm2_mut_rank',
                     'esm2_entropy', 'esm2_score']:
            f[col] = row.get(col, np.nan)
        
        # AA features
        if row['ref_aa'] and row['alt_aa']:
            aa_feats = compute_aa_features(row['ref_aa'], row['alt_aa'])
            f.update(aa_feats)
        
        # Gene features
        gene = row['gene']
        f['gene_path_ratio'] = gene_path_ratio.get(gene, 0.5)
        f['gene_variant_count_log'] = np.log10(max(gene_total_all.get(gene, 1), 1))
        f['gene_has_pathogenic'] = float(gene in genes_with_path)
        
        # Position features
        if row['aa_pos'] > 0:
            f['aa_position_log'] = np.log10(row['aa_pos'])
        else:
            f['aa_position_log'] = np.nan
        
        all_features.append(f)
    
    feat_df = pd.DataFrame(all_features, index=esm2_df.index)
    logger.info(f"Combined feature matrix: {feat_df.shape}")
    
    # ── Train/test split ──
    print("\nSTEP 3: Training integrated model...")
    
    train_mask = esm2_df['label'].isin([0, 1])
    vus_mask = esm2_df['label'] == -1
    
    X_train = feat_df[train_mask].values.astype(np.float32)
    y_train = esm2_df[train_mask]['label'].values.astype(np.int32)
    feature_names = list(feat_df.columns)
    X_vus = feat_df[vus_mask].values.astype(np.float32)
    
    print(f"  Training: {X_train.shape[0]:,} variants × {X_train.shape[1]} features")
    print(f"  VUS: {X_vus.shape[0]:,} variants")
    print(f"  P={y_train.sum()}, B={(1-y_train).sum()} ({y_train.mean():.1%} pathogenic)")
    
    from vus_meta.models.meta_learner import MetaLearner
    
    # ── Model A: ESM-2 only ──
    print("\n  --- Model A: ESM-2 features only ---")
    esm2_cols = [c for c in feature_names if c.startswith('esm2_')]
    esm2_col_idx = [feature_names.index(c) for c in esm2_cols]
    
    learner_esm = MetaLearner(n_estimators=300, max_depth=4, learning_rate=0.05, min_child_weight=5)
    metrics_esm = learner_esm.fit_cv(
        X_train[:, esm2_col_idx], y_train,
        feature_names=esm2_cols, n_folds=5
    )
    
    # ── Model B: AA features only ──
    print("\n  --- Model B: AA features only ---")
    aa_cols = [c for c in feature_names if not c.startswith('esm2_')]
    aa_col_idx = [feature_names.index(c) for c in aa_cols]
    
    learner_aa = MetaLearner(n_estimators=300, max_depth=4, learning_rate=0.05, min_child_weight=5)
    metrics_aa = learner_aa.fit_cv(
        X_train[:, aa_col_idx], y_train,
        feature_names=aa_cols, n_folds=5
    )
    
    # ── Model C: Combined (ESM-2 + AA + gene) ──
    print("\n  --- Model C: Combined (ESM-2 + AA + gene) ---")
    learner_combined = MetaLearner(n_estimators=500, max_depth=6, learning_rate=0.05, min_child_weight=5)
    metrics_combined = learner_combined.fit_cv(
        X_train, y_train,
        feature_names=feature_names, n_folds=5
    )
    
    # ── Compare models ──
    print("\n" + "=" * 70)
    print("  MODEL COMPARISON")
    print("=" * 70)
    print(f"\n  {'Model':30s} {'AUROC':>8s} {'AUPRC':>8s} {'Brier':>8s}")
    print(f"  {'-'*56}")
    print(f"  {'A: ESM-2 only':30s} {metrics_esm['cv_auroc']:8.4f} {metrics_esm['cv_auprc']:8.4f} {metrics_esm['cv_brier']:8.4f}")
    print(f"  {'B: AA features only':30s} {metrics_aa['cv_auroc']:8.4f} {metrics_aa['cv_auprc']:8.4f} {metrics_aa['cv_brier']:8.4f}")
    print(f"  {'C: Combined (ESM-2 + AA)':30s} {metrics_combined['cv_auroc']:8.4f} {metrics_combined['cv_auprc']:8.4f} {metrics_combined['cv_brier']:8.4f}")
    
    # Improvement
    delta_auroc = metrics_combined['cv_auroc'] - max(metrics_esm['cv_auroc'], metrics_aa['cv_auroc'])
    print(f"\n  Combined improvement over best individual: +{delta_auroc:.4f} AUROC")
    
    # ── Feature importance for combined model ──
    print(f"\n  Combined Model — Feature Importance:")
    importance = learner_combined.feature_importance()
    for i, (name, imp) in enumerate(list(importance.items())[:20]):
        source = "ESM-2" if name.startswith('esm2') else ("Gene" if 'gene' in name else "AA")
        bar = "█" * int(imp * 150)
        print(f"  {i+1:2d}. [{source:5s}] {name:28s} {imp:.4f} {bar}")
    
    # ── Score VUS with combined model ──
    print(f"\nSTEP 4: Scoring {X_vus.shape[0]:,} VUS with combined model...")
    vus_scores, vus_conf = learner_combined.predict_with_confidence(X_vus)
    
    from vus_meta.schemas import score_to_classification
    vus_cls = [score_to_classification(s).value for s in vus_scores]
    
    print(f"\n  VUS Reclassification (Combined Model):")
    for cls in ['Pathogenic', 'Likely Pathogenic', 'Uncertain Significance', 'Likely Benign', 'Benign']:
        n = sum(1 for c in vus_cls if c == cls)
        pct = n / len(vus_cls) * 100
        bar = "█" * int(pct / 2)
        print(f"    {cls:30s} {n:>6,} ({pct:5.1f}%) {bar}")
    
    # ── Per-gene VUS results ──
    print(f"\n  Per-Gene VUS Reclassification:")
    vus_results = esm2_df[vus_mask].copy()
    vus_results['meta_score'] = vus_scores
    vus_results['confidence'] = vus_conf
    vus_results['classification'] = vus_cls
    
    for gene in ['BRCA1', 'BRCA2', 'TP53', 'MLH1', 'MSH2', 'MYH7', 'SCN1A', 'CFTR', 'LDLR', 'NF1']:
        g = vus_results[vus_results['gene'] == gene]
        if len(g) == 0: continue
        np_ = g['classification'].isin(['Pathogenic','Likely Pathogenic']).sum()
        nb = g['classification'].isin(['Benign','Likely Benign']).sum()
        nv = (g['classification'] == 'Uncertain Significance').sum()
        print(f"    {gene:8s}: {len(g):5,} → {np_:4,} P/LP, {nv:4,} VUS, {nb:4,} B/LB")
    
    # ── Example high-confidence predictions ──
    print(f"\n  Example High-Confidence Pathogenic Predictions:")
    hp = vus_results[vus_results['classification'].isin(['Pathogenic', 'Likely Pathogenic'])].sort_values('meta_score', ascending=False)
    for _, row in hp.head(15).iterrows():
        aa_change = row.get('aa_change', '?')
        print(f"    {row['gene']:10s} p.{aa_change:12s} score={row['meta_score']:.4f}  "
              f"esm2_llr={row.get('esm2_llr', 0):.3f}  conf={row['confidence']:.3f}")
    
    print(f"\n  Example High-Confidence Benign Predictions:")
    hb = vus_results[vus_results['classification'].isin(['Benign', 'Likely Benign'])].sort_values('meta_score')
    for _, row in hb.head(15).iterrows():
        aa_change = row.get('aa_change', '?')
        print(f"    {row['gene']:10s} p.{aa_change:12s} score={row['meta_score']:.4f}  "
              f"esm2_llr={row.get('esm2_llr', 0):.3f}  conf={row['confidence']:.3f}")
    
    # ── Save ──
    model_dir = Path("data/models/integrated_v1")
    learner_combined.save(model_dir)
    vus_results.to_csv('data/processed/integrated_vus_scored.tsv', sep='\t')
    
    import json
    metrics_out = {
        'model_a_esm2_only': {k: v for k, v in metrics_esm.items() if k != 'oof_predictions'},
        'model_b_aa_only': {k: v for k, v in metrics_aa.items() if k != 'oof_predictions'},
        'model_c_combined': {k: v for k, v in metrics_combined.items() if k != 'oof_predictions'},
        'n_training': int(X_train.shape[0]),
        'n_features': int(X_train.shape[1]),
        'n_vus': int(X_vus.shape[0]),
        'top_features': {k: float(v) for k, v in list(importance.items())[:30]},
    }
    with open('data/processed/integrated_metrics.json', 'w') as f:
        json.dump(metrics_out, f, indent=2, default=str)
    
    print(f"\n  Model → {model_dir}")
    print(f"  Results → data/processed/integrated_vus_scored.tsv")
    
    print(f"\n{'='*70}")
    print(f"  Integrated model training complete! 🧬🔬")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
