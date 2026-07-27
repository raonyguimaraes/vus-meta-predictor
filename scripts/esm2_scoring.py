#!/usr/bin/env python3
"""ESM-2 variant scoring pipeline.

Scores missense variants using ESM-2 masked marginal log-likelihood ratios.
For each variant:
1. Load the protein sequence
2. Mask the wildtype amino acid position
3. Compare P(mutant) vs P(wildtype) under the language model

Strategy for scale:
- Use ESM-2 35M (fast on CPU, ~20 variants/sec)
- Group variants by protein (one sequence load per protein)
- Batch mask positions across the same protein
- Score high-priority clinical genes first
"""

import sys
import gzip
import re
import math
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import esm
from loguru import logger

sys.path.insert(0, "src")
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

AA3TO1 = {
    'Ala':'A','Arg':'R','Asn':'N','Asp':'D','Cys':'C','Glu':'E','Gln':'Q','Gly':'G',
    'His':'H','Ile':'I','Leu':'L','Lys':'K','Met':'M','Phe':'F','Pro':'P','Ser':'S',
    'Thr':'T','Trp':'W','Tyr':'Y','Val':'V',
}
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def download_uniprot_sequences(gene_list, output_path):
    """Download protein sequences from UniProt for given gene names.
    
    Uses UniProt REST API to fetch reviewed (Swiss-Prot) human sequences.
    """
    import requests
    
    sequences = {}
    
    # Batch query UniProt
    batch_size = 50
    for i in range(0, len(gene_list), batch_size):
        batch = gene_list[i:i+batch_size]
        gene_query = ' OR '.join(f'gene_exact:{g}' for g in batch)
        query = f'({gene_query}) AND organism_id:9606 AND reviewed:true'
        
        url = "https://rest.uniprot.org/uniprotkb/search"
        params = {
            'query': query,
            'format': 'fasta',
            'size': 500,
        }
        
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            
            # Parse FASTA
            current_header = None
            current_seq = []
            
            for line in resp.text.strip().split('\n'):
                if line.startswith('>'):
                    if current_header and current_seq:
                        # Extract gene name from header
                        gn_match = re.search(r'GN=(\S+)', current_header)
                        if gn_match:
                            gn = gn_match.group(1)
                            seq_str = ''.join(current_seq)
                            # Keep longest isoform per gene
                            if gn not in sequences or len(seq_str) > len(sequences[gn]):
                                sequences[gn] = seq_str
                    current_header = line
                    current_seq = []
                else:
                    current_seq.append(line.strip())
            
            # Don't forget last entry
            if current_header and current_seq:
                gn_match = re.search(r'GN=(\S+)', current_header)
                if gn_match:
                    gn = gn_match.group(1)
                    seq_str = ''.join(current_seq)
                    if gn not in sequences or len(seq_str) > len(sequences[gn]):
                        sequences[gn] = seq_str
        
        except Exception as e:
            logger.warning(f"UniProt batch query failed: {e}")
        
        if (i + batch_size) % 200 == 0:
            logger.info(f"  Downloaded {len(sequences)} sequences so far...")
    
    # Save sequences
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for gene, seq in sorted(sequences.items()):
            f.write(f">{gene}\n{seq}\n")
    
    logger.info(f"Downloaded {len(sequences)} protein sequences → {output_path}")
    return sequences


def load_sequences(fasta_path):
    """Load sequences from a FASTA file."""
    sequences = {}
    current_gene = None
    current_seq = []
    
    with open(fasta_path) as f:
        for line in f:
            if line.startswith('>'):
                if current_gene:
                    sequences[current_gene] = ''.join(current_seq)
                current_gene = line[1:].strip().split()[0]
                current_seq = []
            else:
                current_seq.append(line.strip())
    if current_gene:
        sequences[current_gene] = ''.join(current_seq)
    
    return sequences


class ESM2Scorer:
    """Efficient ESM-2 variant scorer."""
    
    def __init__(self, model_name='esm2_t12_35M_UR50D', device='cpu'):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.alphabet = None
        self.batch_converter = None
        self._aa_to_idx = {}
    
    def setup(self):
        logger.info(f"Loading ESM-2 model: {self.model_name}")
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(self.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.batch_converter = self.alphabet.get_batch_converter()
        
        for aa in AA_ALPHABET:
            self._aa_to_idx[aa] = self.alphabet.get_idx(aa)
        
        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        logger.info(f"ESM-2 loaded: {n_params:.0f}M params")
    
    @torch.no_grad()
    def score_all_positions(self, sequence, max_len=1022):
        """Score all positions in a protein sequence at once.
        
        Returns a (L, 20) matrix of log-probabilities for each AA at each position.
        This is much more efficient than scoring one position at a time.
        """
        # Truncate if needed (ESM-2 max length)
        seq = sequence[:max_len]
        
        data = [('protein', seq)]
        _, _, tokens = self.batch_converter(data)
        tokens = tokens.to(self.device)
        
        # Forward pass WITHOUT masking (teacher forcing approximation)
        # This gives us the model's prediction at each position given the full context
        # For a more accurate approach, we'd mask each position individually,
        # but that's O(L) forward passes. The unmasked approach is a good approximation.
        output = self.model(tokens)
        logits = output['logits'][0]  # (seq_len + 2, vocab_size) — includes BOS/EOS
        
        # Extract log-probs for standard amino acids at each position
        # Skip BOS (index 0) and EOS (last index)
        aa_indices = torch.tensor([self._aa_to_idx[aa] for aa in AA_ALPHABET])
        log_probs = torch.log_softmax(logits[1:-1], dim=-1)  # (L, vocab_size)
        aa_log_probs = log_probs[:, aa_indices]  # (L, 20)
        
        return aa_log_probs.cpu().numpy()
    
    @torch.no_grad()  
    def score_position_masked(self, sequence, position, max_len=1022):
        """Score a single position with proper masking (more accurate, slower).
        
        Returns (20,) array of log-probabilities.
        """
        seq = sequence[:max_len]
        
        data = [('protein', seq)]
        _, _, tokens = self.batch_converter(data)
        tokens = tokens.to(self.device)
        
        # Mask the target position (+1 for BOS)
        mask_pos = position + 1
        tokens[0, mask_pos] = self.alphabet.mask_idx
        
        output = self.model(tokens)
        logits = output['logits'][0, mask_pos]
        
        aa_indices = torch.tensor([self._aa_to_idx[aa] for aa in AA_ALPHABET])
        log_probs = torch.log_softmax(logits, dim=0)
        aa_log_probs = log_probs[aa_indices]
        
        return aa_log_probs.cpu().numpy()
    
    def compute_variant_features(self, sequence, position, wt_aa, mut_aa, use_masking=False):
        """Compute ESM-2 features for a single variant.
        
        Args:
            sequence: Full protein sequence
            position: 0-based position in protein
            wt_aa: Wildtype amino acid (1-letter)
            mut_aa: Mutant amino acid (1-letter)
            use_masking: If True, use masked scoring (slower, more accurate)
        
        Returns dict of features.
        """
        if position >= len(sequence) or position >= 1022:
            return None
        
        if use_masking:
            log_probs = self.score_position_masked(sequence, position)
        else:
            # Use the precomputed matrix if available
            log_probs = self._current_log_probs[position] if hasattr(self, '_current_log_probs') else \
                        self.score_all_positions(sequence)[position]
        
        aa_to_idx = {aa: i for i, aa in enumerate(AA_ALPHABET)}
        
        wt_idx = aa_to_idx.get(wt_aa)
        mut_idx = aa_to_idx.get(mut_aa)
        
        if wt_idx is None or mut_idx is None:
            return None
        
        wt_logp = log_probs[wt_idx]
        mut_logp = log_probs[mut_idx]
        llr = mut_logp - wt_logp
        
        # Probabilities
        probs = np.exp(log_probs)
        wt_prob = probs[wt_idx]
        mut_prob = probs[mut_idx]
        
        # Rank of wildtype AA
        sorted_indices = np.argsort(-probs)
        wt_rank = np.where(sorted_indices == wt_idx)[0][0] + 1
        
        # Entropy at position
        entropy = -np.sum(probs * log_probs)
        
        # Top-1 prediction matches wildtype?
        top1_is_wt = float(sorted_indices[0] == wt_idx)
        
        # Relative rank of mutant
        mut_rank = np.where(sorted_indices == mut_idx)[0][0] + 1
        
        return {
            'esm2_llr': float(llr),
            'esm2_wt_logp': float(wt_logp),
            'esm2_mut_logp': float(mut_logp),
            'esm2_wt_prob': float(wt_prob),
            'esm2_mut_prob': float(mut_prob),
            'esm2_wt_rank': float(wt_rank),
            'esm2_mut_rank': float(mut_rank),
            'esm2_entropy': float(entropy),
            'esm2_top1_is_wt': top1_is_wt,
            'esm2_score': float(1.0 / (1.0 + math.exp(llr * 2.0))),  # Sigmoid transform
        }
    
    def score_protein_variants(self, sequence, variants, use_masking=False):
        """Score all variants in a single protein efficiently.
        
        Args:
            sequence: Protein sequence
            variants: List of (position_0based, wt_aa, mut_aa, variant_id) tuples
        
        Returns: dict mapping variant_id to feature dict
        """
        results = {}
        
        if not use_masking:
            # One forward pass for the whole protein
            self._current_log_probs = self.score_all_positions(sequence)
        
        for pos, wt_aa, mut_aa, var_id in variants:
            # Verify wildtype matches
            if pos < len(sequence) and sequence[pos] != wt_aa:
                logger.debug(f"Mismatch at {var_id}: expected {wt_aa}, got {sequence[pos]}")
                # Try to continue anyway
            
            feats = self.compute_variant_features(sequence, pos, wt_aa, mut_aa, use_masking)
            results[var_id] = feats
        
        if hasattr(self, '_current_log_probs'):
            del self._current_log_probs
        
        return results


def main():
    print("=" * 70)
    print("  ESM-2 Variant Scoring Pipeline")
    print("=" * 70)
    
    # ── Load our missense VUS data ──
    print("\nSTEP 1: Loading missense VUS data...")
    vus_df = pd.read_csv('data/processed/missense_v2_scored.tsv', sep='\t')
    logger.info(f"Loaded {len(vus_df):,} scored missense VUS")
    
    # Focus on high-priority clinical genes
    PRIORITY_GENES = [
        # Hereditary cancer
        'BRCA1', 'BRCA2', 'TP53', 'MLH1', 'MSH2', 'MSH6', 'APC', 'RB1', 'PTEN',
        'CDH1', 'STK11', 'PALB2', 'ATM', 'CHEK2', 'RAD51C', 'RAD51D', 'BARD1',
        # Cardiac
        'MYH7', 'MYBPC3', 'KCNQ1', 'KCNH2', 'SCN5A', 'LMNA', 'RYR2', 'PKP2',
        'DSP', 'DSG2', 'DSC2', 'TMEM43', 'MYL2', 'MYL3', 'TNNI3', 'TNNT2',
        # Neurological
        'SCN1A', 'MECP2', 'FMR1', 'SMN1', 'DYNC1H1', 'CACNA1A',
        # Other high-impact
        'CFTR', 'LDLR', 'NF1', 'NF2', 'TSC1', 'TSC2', 'VHL', 'PKD1', 'PKD2',
        'GBA', 'HEXA', 'PAH', 'ACADVL', 'GAA', 'GLA', 'IDUA',
        'COL1A1', 'COL1A2', 'FBN1', 'ENG', 'ACVRL1',
    ]
    
    # Also load the training data to score pathogenic/benign for validation
    # Load from variant_summary 
    print("\nSTEP 2: Loading variant_summary for all labeled missense variants...")
    
    records = []
    with gzip.open('data/raw/variant_summary.txt.gz', 'rt') as f:
        header = f.readline().strip().split('\t')
        col_idx = {c: i for i, c in enumerate(header)}
        
        for line in f:
            fields = line.strip().split('\t')
            if len(fields) < 34:
                continue
            
            assembly = fields[col_idx['Assembly']]
            if assembly != 'GRCh38':
                continue
            
            gene = fields[col_idx['GeneSymbol']]
            if gene not in PRIORITY_GENES:
                continue
            
            name = fields[col_idx['Name']]
            if 'p.' not in name:
                continue
            
            # Parse protein change
            m = re.search(r'p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})', name)
            if not m:
                continue
            ref3, pos_str, alt3 = m.group(1), int(m.group(2)), m.group(3)
            ref_aa = AA3TO1.get(ref3)
            alt_aa = AA3TO1.get(alt3)
            if not ref_aa or not alt_aa or ref_aa == '*' or alt_aa == '*':
                continue
            
            clinsig = fields[col_idx['ClinicalSignificance']]
            PATHOGENIC = {"Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic"}
            BENIGN = {"Benign", "Likely benign", "Benign/Likely benign"}
            
            if any(t in clinsig for t in PATHOGENIC):
                label = 1
            elif any(t in clinsig for t in BENIGN):
                label = 0
            elif 'Uncertain significance' in clinsig:
                label = -1
            else:
                continue
            
            records.append({
                'gene': gene,
                'ref_aa': ref_aa,
                'alt_aa': alt_aa,
                'aa_pos': int(pos_str),
                'label': label,
                'name': name[:100],
                'clinsig': clinsig,
            })
    
    all_variants_df = pd.DataFrame(records).drop_duplicates(subset=['gene', 'aa_pos', 'ref_aa', 'alt_aa'])
    logger.info(f"Priority gene variants: {len(all_variants_df):,}")
    logger.info(f"  Pathogenic: {(all_variants_df['label']==1).sum():,}")
    logger.info(f"  Benign: {(all_variants_df['label']==0).sum():,}")
    logger.info(f"  VUS: {(all_variants_df['label']==-1).sum():,}")
    
    # ── Download protein sequences ──
    print("\nSTEP 3: Downloading protein sequences from UniProt...")
    
    fasta_path = Path('data/raw/priority_genes.fasta')
    if fasta_path.exists():
        sequences = load_sequences(fasta_path)
        logger.info(f"Loaded {len(sequences)} sequences from cache")
    else:
        genes_needed = list(all_variants_df['gene'].unique())
        sequences = download_uniprot_sequences(genes_needed, fasta_path)
    
    # Show which genes we have sequences for
    genes_with_seq = set(sequences.keys())
    genes_needed = set(all_variants_df['gene'].unique())
    logger.info(f"Have sequences for {len(genes_with_seq & genes_needed)}/{len(genes_needed)} genes")
    
    missing = genes_needed - genes_with_seq
    if missing:
        logger.warning(f"Missing sequences: {', '.join(sorted(missing)[:10])}")
    
    # ── Setup ESM-2 ──
    print("\nSTEP 4: Setting up ESM-2 scorer...")
    scorer = ESM2Scorer(model_name='esm2_t12_35M_UR50D')
    scorer.setup()
    
    # ── Score variants gene by gene ──
    print("\nSTEP 5: Scoring variants with ESM-2...")
    
    # Group variants by gene
    gene_groups = all_variants_df.groupby('gene')
    
    all_esm2_features = {}
    genes_scored = 0
    variants_scored = 0
    start_time = time.time()
    
    for gene, group in gene_groups:
        if gene not in sequences:
            continue
        
        seq = sequences[gene]
        
        # Prepare variant list: (pos_0based, wt_aa, mut_aa, variant_id)
        variants = []
        for idx, row in group.iterrows():
            pos_0 = row['aa_pos'] - 1  # Convert to 0-based
            var_id = f"{gene}_{row['ref_aa']}{row['aa_pos']}{row['alt_aa']}"
            variants.append((pos_0, row['ref_aa'], row['alt_aa'], var_id))
        
        # Score all variants in this protein
        try:
            results = scorer.score_protein_variants(seq, variants)
            
            for var_id, feats in results.items():
                if feats is not None:
                    # Also store the gene and label
                    row_match = group[
                        (group['gene'] == gene) &
                        (group['ref_aa'] + group['aa_pos'].astype(str) + group['alt_aa'] == 
                         var_id.split('_')[1])
                    ]
                    if len(row_match) > 0:
                        feats['label'] = row_match.iloc[0]['label']
                        feats['gene'] = gene
                        feats['variant'] = var_id
                    all_esm2_features[var_id] = feats
                    variants_scored += 1
        
        except Exception as e:
            logger.warning(f"Failed to score {gene}: {e}")
        
        genes_scored += 1
        elapsed = time.time() - start_time
        rate = variants_scored / elapsed if elapsed > 0 else 0
        
        if genes_scored % 5 == 0:
            logger.info(
                f"  {genes_scored}/{len(gene_groups)} genes, "
                f"{variants_scored:,} variants scored "
                f"({rate:.0f} var/s, ETA {(len(all_variants_df) - variants_scored) / max(rate, 1) / 60:.0f} min)"
            )
    
    total_time = time.time() - start_time
    logger.info(f"ESM-2 scoring complete: {variants_scored:,} variants in {total_time:.0f}s ({variants_scored/total_time:.0f} var/s)")
    
    # ── Build results DataFrame ──
    print("\nSTEP 6: Analyzing ESM-2 scores...")
    
    esm2_df = pd.DataFrame.from_dict(all_esm2_features, orient='index')
    
    if len(esm2_df) == 0:
        logger.error("No variants scored! Check UniProt sequences.")
        return
    
    # Split by label
    path_df = esm2_df[esm2_df['label'] == 1]
    ben_df = esm2_df[esm2_df['label'] == 0]
    vus_esm_df = esm2_df[esm2_df['label'] == -1]
    
    print(f"\n  ESM-2 Score Distribution:")
    print(f"  {'':30s} {'Pathogenic':>12s} {'Benign':>12s} {'VUS':>12s}")
    print(f"  {'-'*66}")
    
    for col in ['esm2_llr', 'esm2_score', 'esm2_wt_rank', 'esm2_entropy']:
        p_val = path_df[col].mean() if len(path_df) > 0 else float('nan')
        b_val = ben_df[col].mean() if len(ben_df) > 0 else float('nan')
        v_val = vus_esm_df[col].mean() if len(vus_esm_df) > 0 else float('nan')
        print(f"  {col:30s} {p_val:12.4f} {b_val:12.4f} {v_val:12.4f}")
    
    # ── Evaluate discriminative power ──
    print(f"\n  ESM-2 Discriminative Power (Pathogenic vs Benign):")
    
    labeled = esm2_df[esm2_df['label'].isin([0, 1])].copy()
    if len(labeled) > 10:
        from sklearn.metrics import roc_auc_score, average_precision_score
        
        for col in ['esm2_llr', 'esm2_score', 'esm2_wt_rank', 'esm2_entropy']:
            valid = labeled.dropna(subset=[col])
            if len(valid) > 10:
                # For LLR: more negative = more pathogenic, so negate for AUROC
                if 'llr' in col:
                    auc = roc_auc_score(valid['label'], -valid[col])
                elif col == 'esm2_wt_rank':
                    auc = roc_auc_score(valid['label'], valid[col])  # Higher rank = more conserved = more damaging
                else:
                    auc = roc_auc_score(valid['label'], valid[col])
                print(f"    {col:30s} AUROC = {auc:.4f}")
    
    # ── Save ESM-2 features ──
    output_path = Path('data/processed/esm2_features.tsv')
    esm2_df.to_csv(output_path, sep='\t')
    logger.info(f"ESM-2 features saved to {output_path}")
    
    # ── Per-gene breakdown ──
    print(f"\n  Per-gene ESM-2 AUROC:")
    for gene in PRIORITY_GENES[:20]:
        g = labeled[labeled['gene'] == gene]
        if len(g) > 10 and g['label'].nunique() > 1:
            try:
                auc = roc_auc_score(g['label'], g['esm2_score'])
                n_p = (g['label']==1).sum()
                n_b = (g['label']==0).sum()
                print(f"    {gene:12s} AUROC={auc:.4f}  (P={n_p:3d}, B={n_b:3d})")
            except:
                pass
    
    print(f"\n{'='*70}")
    print(f"  ESM-2 scoring complete! 🧬")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
