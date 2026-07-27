#!/usr/bin/env python3
"""Score ALL ClinVar missense variants with ESM-2 + AlphaMissense + conservation."""

import sys, gzip, re, math, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import esm as esm_module
from loguru import logger

sys.path.insert(0, "src")
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

AA3TO1 = {
    'Ala':'A','Arg':'R','Asn':'N','Asp':'D','Cys':'C','Glu':'E','Gln':'Q','Gly':'G',
    'His':'H','Ile':'I','Leu':'L','Lys':'K','Met':'M','Phe':'F','Pro':'P','Ser':'S',
    'Thr':'T','Trp':'W','Tyr':'Y','Val':'V',
}
AA = "ACDEFGHIKLMNPQRSTVWY"


def load_alphamissense():
    """Load AlphaMissense scores into a lookup dict: (chrom, pos, ref, alt) → score."""
    path = Path("data/raw/AlphaMissense_hg38.tsv.gz")
    if not path.exists():
        logger.warning("AlphaMissense not found")
        return {}
    
    logger.info("Loading AlphaMissense scores...")
    am = {}
    count = 0
    with gzip.open(path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            if line.startswith('CHROM'):  # header
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 5:
                chrom, pos, ref, alt = parts[0], parts[1], parts[2], parts[3]
                try:
                    score = float(parts[4])
                    # Normalize chrom
                    chrom_clean = chrom.replace('chr', '')
                    am[(chrom_clean, int(pos), ref, alt)] = score
                    count += 1
                    if count % 10_000_000 == 0:
                        logger.info(f"  Loaded {count/1e6:.0f}M AlphaMissense scores...")
                except (ValueError, IndexError):
                    pass
    
    logger.info(f"AlphaMissense: {count:,} scores loaded")
    return am


def load_conservation_scores():
    """Load phyloP and phastCons bigWig files."""
    try:
        import pyBigWig
        bw_files = {}
        for name, path in [
            ("phyloP100way", "data/raw/hg38.phyloP100way.bw"),
            ("phastCons100way", "data/raw/hg38.phastCons100way.bw"),
        ]:
            if Path(path).exists():
                bw_files[name] = pyBigWig.open(path)
                logger.info(f"Loaded {name}")
        return bw_files
    except ImportError:
        logger.warning("pyBigWig not installed, skipping conservation scores")
        return {}


def query_conservation(bw_files, chrom, pos):
    """Query conservation scores at a position."""
    scores = {}
    chrom_name = f"chr{chrom}" if not chrom.startswith("chr") else chrom
    for name, bw in bw_files.items():
        try:
            vals = bw.values(chrom_name, pos - 1, pos)
            if vals and vals[0] is not None:
                scores[name] = vals[0]
        except Exception:
            pass
    return scores


def main():
    print("=" * 70)
    print("  Full-Scale Scoring: ESM-2 + AlphaMissense + Conservation")
    print("=" * 70)
    
    # ── Load all variant data from variant_summary ──
    logger.info("Loading variant_summary...")
    
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
            elif 'Uncertain significance' in clinsig: label = -1
            else: continue
            
            gene = fields[col_idx['GeneSymbol']]
            chrom = fields[col_idx['Chromosome']]
            pos_str = fields[col_idx['PositionVCF']]
            ref_nuc = fields[col_idx['ReferenceAlleleVCF']]
            alt_nuc = fields[col_idx['AlternateAlleleVCF']]
            
            try: pos_int = int(pos_str)
            except: continue
            
            records.append({
                'gene': gene, 'ref_aa': ref_aa, 'alt_aa': alt_aa,
                'aa_pos': int(m.group(2)), 'label': label,
                'chrom': chrom, 'pos': pos_int, 'ref': ref_nuc, 'alt': alt_nuc,
            })
    
    df = pd.DataFrame(records).drop_duplicates(subset=['gene', 'aa_pos', 'ref_aa', 'alt_aa'])
    logger.info(f"Total missense: {len(df):,} (P={( df['label']==1).sum():,}, B={(df['label']==0).sum():,}, VUS={(df['label']==-1).sum():,})")
    
    # ── Load AlphaMissense ──
    am_scores = load_alphamissense()
    
    # ── Load conservation ──
    bw_files = load_conservation_scores()
    
    # ── Look up AlphaMissense and conservation for each variant ──
    logger.info("Looking up AlphaMissense + conservation scores...")
    am_vals = []
    phylop_vals = []
    phastcons_vals = []
    
    for _, row in df.iterrows():
        # AlphaMissense
        key = (str(row['chrom']), row['pos'], row['ref'], row['alt'])
        am_vals.append(am_scores.get(key, np.nan))
        
        # Conservation
        cons = query_conservation(bw_files, str(row['chrom']), row['pos'])
        phylop_vals.append(cons.get('phyloP100way', np.nan))
        phastcons_vals.append(cons.get('phastCons100way', np.nan))
    
    df['alphamissense'] = am_vals
    df['phyloP100way'] = phylop_vals
    df['phastCons100way'] = phastcons_vals
    
    am_hit_rate = df['alphamissense'].notna().mean()
    phylop_hit_rate = df['phyloP100way'].notna().mean()
    logger.info(f"AlphaMissense coverage: {am_hit_rate:.1%}")
    logger.info(f"phyloP coverage: {phylop_hit_rate:.1%}")
    
    # ── Download ALL protein sequences for ESM-2 ──
    logger.info("Downloading protein sequences for all genes...")
    
    import requests
    
    genes_needed = list(df['gene'].unique())
    logger.info(f"Need sequences for {len(genes_needed)} genes")
    
    # Load existing sequences
    fasta_path = Path("data/raw/priority_genes.fasta")
    sequences = {}
    if fasta_path.exists():
        with open(fasta_path) as f:
            g, s = None, []
            for line in f:
                if line.startswith('>'):
                    if g: sequences[g] = ''.join(s)
                    g = line[1:].strip().split()[0]; s = []
                else: s.append(line.strip())
            if g: sequences[g] = ''.join(s)
    
    # Download missing sequences
    missing_genes = [g for g in genes_needed if g not in sequences]
    logger.info(f"Have {len(sequences)} sequences, need {len(missing_genes)} more")
    
    if missing_genes:
        batch_size = 50
        for i in range(0, min(len(missing_genes), 2000), batch_size):
            batch = missing_genes[i:i+batch_size]
            gene_query = ' OR '.join(f'gene_exact:{g}' for g in batch)
            query = f'({gene_query}) AND organism_id:9606 AND reviewed:true'
            
            try:
                resp = requests.get(
                    "https://rest.uniprot.org/uniprotkb/search",
                    params={'query': query, 'format': 'fasta', 'size': 500},
                    timeout=30,
                )
                for line in resp.text.strip().split('\n'):
                    if line.startswith('>'):
                        gn_match = re.search(r'GN=(\S+)', line)
                        if gn_match: current_gene = gn_match.group(1); current_seq = []
                        continue
                    if 'current_gene' in dir():
                        current_seq.append(line.strip())
                        if current_gene not in sequences:
                            sequences[current_gene] = ''.join(current_seq)
            except Exception as e:
                if i == 0: logger.warning(f"UniProt batch failed: {e}")
            
            if (i + batch_size) % 500 == 0:
                logger.info(f"  Downloaded sequences for {len(sequences)} genes...")
    
    logger.info(f"Total sequences: {len(sequences)}")
    
    # Save all sequences
    with open("data/raw/all_genes.fasta", 'w') as f:
        for gene, seq in sorted(sequences.items()):
            f.write(f">{gene}\n{seq}\n")
    
    # ── ESM-2 scoring ──
    logger.info("Loading ESM-2 35M...")
    model, alphabet = esm_module.pretrained.esm2_t12_35M_UR50D()
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    aa_to_idx = {aa: alphabet.get_idx(aa) for aa in AA}
    
    # Group by gene and score
    gene_groups = df.groupby('gene')
    esm2_results = {}
    scored = 0
    start = time.time()
    
    for gene, group in gene_groups:
        if gene not in sequences:
            continue
        seq = sequences[gene][:1022]
        
        # One forward pass per protein (teacher forcing)
        data = [('protein', seq)]
        _, _, tokens = batch_converter(data)
        
        with torch.no_grad():
            output = model(tokens)
            logits = output['logits'][0, 1:-1]  # skip BOS/EOS
        
        aa_indices = torch.tensor([aa_to_idx[aa] for aa in AA])
        log_probs = torch.log_softmax(logits, dim=-1)[:, aa_indices].numpy()
        
        aa_idx_map = {aa: i for i, aa in enumerate(AA)}
        
        for idx, row in group.iterrows():
            pos_0 = row['aa_pos'] - 1
            if pos_0 >= len(seq) or pos_0 >= log_probs.shape[0]:
                continue
            
            wt_i = aa_idx_map.get(row['ref_aa'])
            mut_i = aa_idx_map.get(row['alt_aa'])
            if wt_i is None or mut_i is None:
                continue
            
            lp = log_probs[pos_0]
            llr = float(lp[mut_i] - lp[wt_i])
            probs = np.exp(lp)
            entropy = float(-np.sum(probs * lp))
            
            esm2_results[idx] = {
                'esm2_llr': llr,
                'esm2_wt_prob': float(probs[wt_i]),
                'esm2_mut_prob': float(probs[mut_i]),
                'esm2_entropy': entropy,
                'esm2_score': float(1.0 / (1.0 + math.exp(llr * 2.0))),
            }
            scored += 1
        
        elapsed = time.time() - start
        if scored % 50000 < len(group) and scored > 0:
            rate = scored / elapsed
            remaining = (len(df) - scored) / rate / 60
            logger.info(f"  ESM-2: {scored:,}/{len(df):,} variants ({rate:.0f}/s, ~{remaining:.0f} min left)")
    
    logger.info(f"ESM-2 scored {scored:,} variants in {time.time()-start:.0f}s")
    
    # ── Merge all features ──
    esm2_df = pd.DataFrame.from_dict(esm2_results, orient='index')
    df = df.join(esm2_df)
    
    # Save full feature dataset
    df.to_csv('data/processed/full_features.tsv', sep='\t', index=False)
    logger.info(f"Saved full features: {df.shape}")
    
    # ── Summary stats ──
    print(f"\n{'='*60}")
    print(f"  FULL-SCALE FEATURE COVERAGE")
    print(f"{'='*60}")
    print(f"  Total variants: {len(df):,}")
    for col in ['alphamissense', 'phyloP100way', 'phastCons100way', 'esm2_llr']:
        cov = df[col].notna().mean() * 100
        print(f"  {col:20s}: {cov:.1f}% coverage")
    
    # ── Discriminative power of each score ──
    from sklearn.metrics import roc_auc_score
    
    labeled = df[df['label'].isin([0, 1])].dropna(subset=['esm2_llr'])
    print(f"\n  Individual Score AUROC (on {len(labeled):,} labeled variants):")
    
    for col, negate in [
        ('esm2_llr', True), ('esm2_score', False),
        ('alphamissense', False), ('phyloP100way', False), ('phastCons100way', False),
    ]:
        valid = labeled.dropna(subset=[col])
        if len(valid) > 100:
            vals = -valid[col] if negate else valid[col]
            try:
                auc = roc_auc_score(valid['label'], vals)
                print(f"    {col:20s}: AUROC = {auc:.4f} (n={len(valid):,})")
            except: pass
    
    print(f"\n{'='*60}")
    print(f"  Full-scale scoring complete! 🧬")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
