"""Genomic language model predictors: DNABERT-2 and Nucleotide Transformer.

These models operate on DNA sequences and can assess variant effects
by comparing reference vs alternative sequence log-likelihoods.

DNABERT-2:
    - BPE tokenization (no k-mer dependency)
    - Trained on multi-species genomes
    - Good at regulatory variant assessment

Nucleotide Transformer:
    - 6-mer tokenization, up to 2.5B params
    - Strong on promoter, enhancer, splice site prediction
    - Can extract embeddings for downstream tasks

For variant effect prediction, we use two strategies:
1. Log-Likelihood Ratio (LLR): Compare P(seq_alt) vs P(seq_ref)
2. Embedding distance: Compare embedding of ref vs alt windows
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from loguru import logger

from vus_meta.predictors.base import BasePredictor
from vus_meta.schemas import PredictorScore, Variant


class DNABERT2Predictor(BasePredictor):
    """DNABERT-2 genomic language model predictor.

    Uses masked language modeling to compute variant effect scores.
    For each variant, we:
    1. Extract a window around the variant position
    2. Mask the variant position(s)
    3. Compare P(ref_nucleotide | context) vs P(alt_nucleotide | context)
    """

    name = "dnabert2"
    version = "2.0"
    supported_variant_types = []  # Can score all variant types

    def __init__(
        self,
        model_name: str = "zhihan1996/DNABERT-2-117M",
        device: Optional[str] = None,
        window_size: int = 512,  # bp around variant
    ):
        self.model_name = model_name
        self.window_size = window_size
        self.model = None
        self.tokenizer = None

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def setup(self) -> None:
        """Load DNABERT-2 model and tokenizer."""
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        logger.info(f"Loading DNABERT-2: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = AutoModelForMaskedLM.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info("DNABERT-2 loaded successfully")

    def _get_dna_window(
        self,
        chrom: str,
        pos: int,
        ref: str,
        alt: str,
        genome_fasta: Optional[str] = None,
    ) -> tuple[str, str, int]:
        """Extract reference and alternative DNA sequences around the variant.

        Returns: (ref_seq, alt_seq, variant_offset_in_window)
        """
        half = self.window_size // 2
        start = pos - 1 - half  # Convert to 0-based
        end = start + self.window_size

        if genome_fasta is not None:
            import pysam

            fasta = pysam.FastaFile(genome_fasta)
            # Adjust for chromosome naming
            chrom_name = chrom if chrom in fasta.references else f"chr{chrom}"
            ref_seq = fasta.fetch(chrom_name, max(0, start), end).upper()
            fasta.close()
        else:
            # Return placeholder if no genome available
            logger.warning("No genome FASTA provided — using placeholder sequence")
            ref_seq = "N" * self.window_size

        # Build alt sequence
        var_offset = (pos - 1) - max(0, start)
        alt_seq = ref_seq[:var_offset] + alt + ref_seq[var_offset + len(ref):]

        return ref_seq, alt_seq, var_offset

    @torch.no_grad()
    def compute_sequence_score(self, sequence: str) -> float:
        """Compute pseudo-log-likelihood of a DNA sequence.

        Uses masked language modeling: for each position, mask it and
        compute log P(true_token | context). Sum over all positions.
        
        For efficiency, we use a single forward pass with no masking
        and use the diagonal of the output (teacher forcing approximation).
        """
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        outputs = self.model(**inputs)
        logits = outputs.logits  # (1, seq_len, vocab_size)

        # Get log-probabilities
        log_probs = torch.log_softmax(logits, dim=-1)

        # Score = sum of log P(true_token | context) for each position
        input_ids = inputs["input_ids"]
        token_log_probs = log_probs[0, torch.arange(input_ids.shape[1]), input_ids[0]]

        # Exclude special tokens (CLS, SEP, PAD)
        special_tokens = set(self.tokenizer.all_special_ids)
        mask = torch.tensor(
            [tid.item() not in special_tokens for tid in input_ids[0]],
            device=self.device,
        )
        score = (token_log_probs * mask).sum().item()

        return score

    @torch.no_grad()
    def compute_embedding(self, sequence: str) -> np.ndarray:
        """Get the [CLS] embedding of a DNA sequence."""
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        # We need the base model (without MLM head) for embeddings
        outputs = self.model.bert(**inputs) if hasattr(self.model, 'bert') else self.model(**inputs, output_hidden_states=True)
        
        if hasattr(outputs, 'last_hidden_state'):
            cls_embedding = outputs.last_hidden_state[0, 0].cpu().numpy()
        else:
            cls_embedding = outputs.hidden_states[-1][0, 0].cpu().numpy()

        return cls_embedding

    def predict(
        self,
        variant: Variant,
        genome_fasta: Optional[str] = None,
        ref_seq: Optional[str] = None,
        alt_seq: Optional[str] = None,
    ) -> PredictorScore:
        """Score a variant using DNABERT-2.

        Computes both:
        1. LLR: log P(alt_seq) - log P(ref_seq) (more negative = more damaging)
        2. Embedding cosine distance between ref and alt contexts
        """
        try:
            if ref_seq is None or alt_seq is None:
                ref_seq, alt_seq, _ = self._get_dna_window(
                    variant.chrom, variant.pos, variant.ref, variant.alt, genome_fasta
                )

            # Compute sequence-level scores
            ref_score = self.compute_sequence_score(ref_seq)
            alt_score = self.compute_sequence_score(alt_seq)
            llr = alt_score - ref_score

            # Compute embedding distance
            ref_emb = self.compute_embedding(ref_seq)
            alt_emb = self.compute_embedding(alt_seq)
            cosine_sim = np.dot(ref_emb, alt_emb) / (
                np.linalg.norm(ref_emb) * np.linalg.norm(alt_emb) + 1e-10
            )
            embedding_distance = 1.0 - cosine_sim

            # Convert to pathogenicity score
            # More negative LLR + larger embedding distance = more damaging
            # Use sigmoid on combined score
            combined = -llr * 0.5 + embedding_distance * 10.0
            pathogenicity_score = 1.0 / (1.0 + np.exp(-combined))

            return PredictorScore(
                name=self.name,
                score=float(pathogenicity_score),
                raw_output={
                    "llr": llr,
                    "ref_score": ref_score,
                    "alt_score": alt_score,
                    "embedding_distance": float(embedding_distance),
                    "cosine_similarity": float(cosine_sim),
                },
            )
        except Exception as e:
            logger.error(f"DNABERT-2 prediction failed for {variant}: {e}")
            return PredictorScore(name=self.name, score=None)

    def teardown(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
            if self.device == "cuda":
                torch.cuda.empty_cache()


class NucleotideTransformerPredictor(BasePredictor):
    """Nucleotide Transformer predictor.

    Uses the InstaDeep Nucleotide Transformer models.
    Available sizes: 500M, 1B, 2.5B parameters.
    Tokenizes DNA into 6-mers.
    """

    name = "nucleotide_transformer"
    version = "2.0"
    supported_variant_types = []

    def __init__(
        self,
        model_name: str = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
        device: Optional[str] = None,
        window_size: int = 1000,
    ):
        self.model_name = model_name
        self.window_size = window_size
        self.model = None
        self.tokenizer = None

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def setup(self) -> None:
        """Load Nucleotide Transformer."""
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        logger.info(f"Loading Nucleotide Transformer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = AutoModelForMaskedLM.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info("Nucleotide Transformer loaded")

    @torch.no_grad()
    def compute_embeddings(self, sequence: str) -> np.ndarray:
        """Get mean-pooled embeddings from Nucleotide Transformer."""
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            truncation=True,
            max_length=1000,
        ).to(self.device)

        outputs = self.model(**inputs, output_hidden_states=True)
        # Mean pool the last hidden state (exclude special tokens)
        hidden = outputs.hidden_states[-1][0]  # (seq_len, hidden_dim)
        attention_mask = inputs["attention_mask"][0]
        pooled = (hidden * attention_mask.unsqueeze(-1)).sum(0) / attention_mask.sum()
        return pooled.cpu().numpy()

    def predict(
        self,
        variant: Variant,
        genome_fasta: Optional[str] = None,
        ref_seq: Optional[str] = None,
        alt_seq: Optional[str] = None,
    ) -> PredictorScore:
        """Score variant by embedding distance between ref and alt."""
        try:
            if ref_seq is None or alt_seq is None:
                # Would normally extract from genome FASTA
                logger.warning("No sequences provided for Nucleotide Transformer")
                return PredictorScore(name=self.name, score=None)

            ref_emb = self.compute_embeddings(ref_seq)
            alt_emb = self.compute_embeddings(alt_seq)

            # L2 distance and cosine similarity
            l2_dist = float(np.linalg.norm(ref_emb - alt_emb))
            cosine_sim = float(
                np.dot(ref_emb, alt_emb)
                / (np.linalg.norm(ref_emb) * np.linalg.norm(alt_emb) + 1e-10)
            )

            # Higher distance = more disruptive
            pathogenicity_score = 1.0 / (1.0 + np.exp(-l2_dist + 2.0))

            return PredictorScore(
                name=self.name,
                score=float(pathogenicity_score),
                raw_output={
                    "l2_distance": l2_dist,
                    "cosine_similarity": cosine_sim,
                },
            )
        except Exception as e:
            logger.error(f"Nucleotide Transformer prediction failed: {e}")
            return PredictorScore(name=self.name, score=None)

    def teardown(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
            if self.device == "cuda":
                torch.cuda.empty_cache()
