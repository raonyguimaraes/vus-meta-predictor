#!/usr/bin/env python3
"""Final benchmark: Train the best model with ALL features and do temporal evaluation.

Features:
- ESM-2 LLR (from pre-computed features)
- AlphaMissense scores (lookup by protein variant)
- Conservation (phyloP, phastCons)
- Amino acid properties
- Gene-level constraints

Evaluation: Temporal hold-out (pre-2024 train, 2024+ test)
"""

import sys, gzip, re, math, time
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import roc_auc_score, average_precision_score

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


def build_alphamissense_index():
    """Build an index: (uniprot_id, protein_variant) → am_pathogenicity.
    Stream through the file and only keep entries we need."""
    logger.info("Building AlphaMissense index (streaming)...")
    
    am = {}
    count = 0
    with gzip.open('data/raw/AlphaMissense_hg38.tsv.gz', 'rt') as f:
        for line in f:
            if line.startswith('#') or line.startswith('CHROM'): continue
            parts = line.split('\t')
            if len(parts) >= 9:
                # Index by (uniprot_id, protein_variant) for accurate matching
                # Also index by (chrom, pos, ref, alt) for genomic matching
                pv = parts[7].strip()  # e.g., "V2L"
                uniprot = parts[5].strip()
                try:
                    score = float(parts[8])
                    am[(uniprot, pv)] = score
                    # Also genomic index
                    chrom = parts[0].replace('chr', '')
                    pos = parts[1]
                    am[(chrom, pos, parts[2], parts[3])] = score
                    count += 1
                    if count % 10_000_000 == 0:
                        logger.info(f"  Indexed {count/1e6:.0f}M AlphaMissense entries...")
                except (ValueError, IndexError):
                    pass
    
    logger.info(f"AlphaMissense index: {count:,} entries ({sys.getsizeof(am)/(1024**3):.1f} GB)")
    return am


def main():
    print("=" * 70)
    print("  FINAL BENCHMARK: All Features + Temporal Hold-Out")
    print("=" * 70)
    
    # ── Load pre-computed ESM-2 features ──
    logger.info("Loading pre-computed ESM-2 features...")
    esm2_path = Path("data/processed/esm2_features.tsv")
    esm2_features = {}
    if esm2_path.exists():
        edf = pd.read_csv(esm2_path, sep='\t', index_col=0)
        for idx, row in edf.iterrows():
            gene = str(idx).split('_')[0]
            aa_change = str(idx).split('_')[1] if '_' in str(idx) else ''
            esm2_features[(gene, aa_change)] = {
                'esm2_llr': row.get('esm2_llr', np.nan),
                'esm2_wt_prob': row.get('esm2_wt_prob', np.nan),
                'esm2_entropy': row.get('esm2_entropy', np.nan),
                'esm2_score': row.get('esm2_score', np.nan),
            }
        logger.info(f"ESM-2 features: {len(esm2_features):,}")
    
    # ── Load AlphaMissense ──
    am_index = build_alphamissense_index()
    
    # ── Load conservation ──
    bw_files = {}
    try:
        import pyBigWig
        for name, path in [("phyloP", "data/raw/hg38.phyloP100way.bw"),
                           ("phastCons", "data/raw/hg38.phastCons100way.bw")]:
            if Path(path).exists():
                bw_files[name] = pyBigWig.open(path)
                logger.info(f"Loaded {name}")
    except ImportError:
        logger.warning("pyBigWig not available")
    
    # ── Parse variants ──
    logger.info("Parsing variant_summary...")
    
    PATHOGENIC = {"Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic"}
    BENIGN = {"Benign", "Likely benign", "Benign/Likely benign"}
    
    records = []
    with gzip.open('data/raw/variant_summary.txt.gz', 'rt') as f:
        header = f.readline().strip().split('\t')
        ci = {c: i for i, c in enumerate(header)}
        
        for line in f:
            fields = line.strip().split('\t')
            if len(fields) < 34 or fields[ci['Assembly']] != 'GRCh38': continue
            
            name = fields[ci['Name']]
            m = re.search(r'p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})', name)
            if not m: continue
            
            ref_aa, alt_aa = AA3TO1.get(m.group(1)), AA3TO1.get(m.group(3))
            if not ref_aa or not alt_aa or ref_aa == '*' or alt_aa == '*': continue
            
            clinsig = fields[ci['ClinicalSignificance']]
            if any(t in clinsig for t in PATHOGENIC): label = 1
            elif any(t in clinsig for t in BENIGN): label = 0
            else: continue
            
            last_eval = fields[ci['LastEvaluated']]
            year = None
            ym = re.search(r'(20\d{2})', last_eval) if last_eval != '-' else None
            if ym: year = int(ym.group(1))
            
            gene = fields[ci['GeneSymbol']]
            chrom = fields[ci['Chromosome']]
            pos_str = fields[ci['PositionVCF']]
            ref_nuc = fields[ci['ReferenceAlleleVCF']]
            alt_nuc = fields[ci['AlternateAlleleVCF']]
            
            try: pos = int(pos_str)
            except: continue
            
            aa_pos = int(m.group(2))
            pv = f"{ref_aa}{aa_pos}{alt_aa}"
            
            records.append({
                'gene': gene, 'ref_aa': ref_aa, 'alt_aa': alt_aa, 'aa_pos': aa_pos,
                'label': label, 'year': year, 'chrom': chrom, 'pos': pos,
                'ref': ref_nuc, 'alt': alt_nuc, 'pv': pv,
            })
    
    df = pd.DataFrame(records).drop_duplicates(subset=['gene', 'aa_pos', 'ref_aa', 'alt_aa'])
    logger.info(f"Labeled missense: {len(df):,}")
    
    # ── Compute ALL features ──
    logger.info("Computing features...")
    
    # Gene stats from pre-2024 only
    pre2024 = df[df['year'].notna() & (df['year'] < 2024)]
    gp = pre2024[pre2024['label']==1].groupby('gene').size()
    gb = pre2024[pre2024['label']==0].groupby('gene').size()
    gt = gp.add(gb, fill_value=0)
    gene_pr = (gp / gt).fillna(0.5)
    gene_total = df.groupby('gene').size()
    genes_path = set(pre2024[pre2024['label']==1]['gene'].unique())
    
    features_list = []
    for _, row in df.iterrows():
        f = {}
        
        # AA properties
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
            f['from_cysteine'] = float(row['ref_aa'] == 'C')
            f['charge_reversal'] = float(
                (row['ref_aa'] in AA_CLASSES['positive'] and row['alt_aa'] in AA_CLASSES['negative']) or
                (row['ref_aa'] in AA_CLASSES['negative'] and row['alt_aa'] in AA_CLASSES['positive'])
            )
        
        f['aa_position_log'] = np.log10(max(row['aa_pos'], 1))
        f['gene_path_ratio'] = gene_pr.get(row['gene'], 0.5)
        f['gene_variant_count_log'] = np.log10(max(gene_total.get(row['gene'], 1), 1))
        f['gene_has_pathogenic'] = float(row['gene'] in genes_path)
        
        # ESM-2
        key = (row['gene'], row['pv'])
        esm = esm2_features.get(key, {})
        for col in ['esm2_llr', 'esm2_wt_prob', 'esm2_entropy', 'esm2_score']:
            f[col] = esm.get(col, np.nan)
        
        # AlphaMissense
        am_key = (str(row['chrom']), str(row['pos']), row['ref'], row['alt'])
        f['alphamissense'] = am_index.get(am_key, np.nan)
        
        # Conservation
        chrom_name = f"chr{row['chrom']}" if not str(row['chrom']).startswith('chr') else str(row['chrom'])
        for bw_name, bw in bw_files.items():
            try:
                vals = bw.values(chrom_name, row['pos'] - 1, row['pos'])
                f[bw_name] = vals[0] if vals and vals[0] is not None else np.nan
            except: f[bw_name] = np.nan
        
        features_list.append(f)
    
    feat_df = pd.DataFrame(features_list, index=df.index)
    feature_names = list(feat_df.columns)
    
    # Feature coverage
    logger.info(f"Feature matrix: {feat_df.shape}")
    for col in ['esm2_llr', 'alphamissense', 'phyloP', 'phastCons']:
        if col in feat_df.columns:
            cov = feat_df[col].notna().mean() * 100
            logger.info(f"  {col}: {cov:.1f}% coverage")
    
    # ── Temporal split ──
    train_mask = df['year'].notna() & (df['year'] < 2024)
    test_mask = df['year'].notna() & (df['year'] >= 2024)
    
    X_train = feat_df[train_mask].values.astype(np.float32)
    y_train = df[train_mask]['label'].values.astype(np.int32)
    X_test = feat_df[test_mask].values.astype(np.float32)
    y_test = df[test_mask]['label'].values.astype(np.int32)
    
    # ── Train models and compare ──
    from vus_meta.models.meta_learner import MetaLearner
    
    results = {}
    
    # Model configs to compare
    models = {
        'AA only': [c for c in feature_names if not c.startswith('esm2') and c != 'alphamissense' and c not in ('phyloP', 'phastCons')],
        'AA + Conservation': [c for c in feature_names if not c.startswith('esm2') and c != 'alphamissense'],
        'AA + AlphaMissense': [c for c in feature_names if not c.startswith('esm2') and c not in ('phyloP', 'phastCons')],
        'AA + ESM-2': [c for c in feature_names if c != 'alphamissense' and c not in ('phyloP', 'phastCons')],
        'ALL FEATURES': feature_names,
    }
    
    for model_name, cols in models.items():
        col_idx = [feature_names.index(c) for c in cols]
        
        learner = MetaLearner(n_estimators=500, max_depth=6, learning_rate=0.05, min_child_weight=10)
        learner.fit(X_train[:, col_idx], y_train, feature_names=cols)
        
        test_scores = learner.predict(X_test[:, col_idx])
        auroc = roc_auc_score(y_test, test_scores)
        auprc = average_precision_score(y_test, test_scores)
        
        results[model_name] = {'auroc': auroc, 'auprc': auprc, 'n_features': len(cols)}
        logger.info(f"  {model_name:25s}: AUROC={auroc:.4f}, AUPRC={auprc:.4f} ({len(cols)} features)")
    
    # ── Print final comparison ──
    print(f"\n{'='*70}")
    print(f"  FINAL BENCHMARK — Temporal Hold-Out (pre-2024 → 2024+)")
    print(f"  Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
    print(f"{'='*70}")
    print(f"\n  {'Model':25s} {'AUROC':>8s} {'AUPRC':>8s} {'Feats':>6s}")
    print(f"  {'-'*49}")
    
    for name, r in results.items():
        marker = " ← best" if r['auroc'] == max(v['auroc'] for v in results.values()) else ""
        print(f"  {name:25s} {r['auroc']:8.4f} {r['auprc']:8.4f} {r['n_features']:6d}{marker}")
    
    # ── Individual score AUROC ──
    print(f"\n  Individual Score AUROC (test set):")
    for col in ['alphamissense', 'phyloP', 'phastCons', 'esm2_score', 'grantham', 'gene_path_ratio']:
        if col in feature_names:
            valid = feat_df[test_mask][col].notna()
            if valid.sum() > 100:
                try:
                    auc = roc_auc_score(y_test[valid], feat_df[test_mask].loc[valid.values, col])
                    print(f"    {col:25s}: {auc:.4f} (n={valid.sum():,})")
                except: pass
    
    # Save final results
    import json
    with open('data/processed/final_benchmark.json', 'w') as f:
        json.dump({
            'models': {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                          for kk, vv in v.items()} for k, v in results.items()},
            'n_train': int(X_train.shape[0]),
            'n_test': int(X_test.shape[0]),
        }, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"  Final benchmark complete! 🏆")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
