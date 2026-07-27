"""ESM-2 protein language model for variant effect prediction.

Uses log-likelihood ratios (LLR) to score missense variants:
    LLR = log P(mutant_aa | context) - log P(wt_aa | context)

A negative LLR indicates the mutation is disfavored by the evolutionary
language model, suggesting pathogenicity.

Reference:
    Brandes et al. (2023) "Genome-wide prediction of disease variant
    effects with a deep protein language model" Nature Genetics.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
from loguru import logger

from vus_meta.predictors.base import BasePredictor
from vus_meta.schemas import PredictorScore, Variant, VariantType


# Standard amino acid vocabulary for ESM-2
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


class ESM2Predictor(BasePredictor):
    """ESM-2 log-likelihood ratio predictor for missense variants.

    This predictor computes the masked marginal log-likelihood ratio:
    For each position, mask the token, predict, and compare the probability
    of wildtype vs mutant amino acid.

    Supports ESM-2 models of various sizes:
    - esm2_t6_8M_UR50D (8M params, fast)
    - esm2_t12_35M_UR50D (35M params)
    - esm2_t30_150M_UR50D (150M params)
    - esm2_t33_650M_UR50D (650M params, recommended)
    - esm2_t36_3B_UR50D (3B params, best quality)
    """

    name = "esm2"
    version = "2.0"
    supported_variant_types = [VariantType.MISSENSE.value]

    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
        device: Optional[str] = None,
        context_window: int = 1022,  # ESM-2 max is 1022 tokens
    ):
        self.model_name = model_name
        self.context_window = context_window
        self.model = None
        self.alphabet = None
        self.batch_converter = None

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def setup(self) -> None:
        """Load ESM-2 model from torch hub or HuggingFace."""
        logger.info(f"Loading ESM-2 model: {self.model_name} on {self.device}")

        try:
            import esm as esm_module

            self.model, self.alphabet = esm_module.pretrained.load_model_and_alphabet(
                self.model_name
            )
        except ImportError:
            logger.info("esm package not found, falling back to torch.hub")
            self.model, self.alphabet = torch.hub.load(
                "facebookresearch/esm:main", self.model_name
            )

        self.model = self.model.to(self.device)
        self.model.eval()
        self.batch_converter = self.alphabet.get_batch_converter()

        # Build AA token index map
        self._aa_to_idx = {}
        for aa in AA_ALPHABET:
            idx = self.alphabet.get_idx(aa)
            self._aa_to_idx[aa] = idx

        logger.info(f"ESM-2 loaded: {sum(p.numel() for p in self.model.parameters())/1e6:.0f}M params")

    def _get_sequence_context(self, sequence: str, position: int) -> tuple[str, int]:
        """Extract a window around the mutation position.

        Returns the subsequence and the adjusted position index.
        """
        seq_len = len(sequence)
        if seq_len <= self.context_window:
            return sequence, position

        half_window = self.context_window // 2
        start = max(0, position - half_window)
        end = min(seq_len, start + self.context_window)
        start = max(0, end - self.context_window)

        new_pos = position - start
        return sequence[start:end], new_pos

    @torch.no_grad()
    def compute_llr(
        self,
        sequence: str,
        position: int,  # 0-based position in sequence
        wt_aa: str,
        mut_aa: str,
    ) -> dict:
        """Compute masked marginal log-likelihood ratio.

        Returns dict with:
            - llr: log P(mut|context) - log P(wt|context)
            - wt_prob: P(wt_aa | masked context)
            - mut_prob: P(mut_aa | masked context)
            - wt_rank: rank of wt_aa among all AAs
            - entropy: Shannon entropy at the position
        """
        seq_context, adj_pos = self._get_sequence_context(sequence, position)

        # Verify wildtype amino acid matches
        if seq_context[adj_pos] != wt_aa:
            logger.warning(
                f"Sequence mismatch at pos {position}: expected {wt_aa}, "
                f"found {seq_context[adj_pos]}"
            )

        # Create masked sequence (replace position with <mask> token)
        data = [("variant", seq_context)]
        _, _, tokens = self.batch_converter(data)
        tokens = tokens.to(self.device)

        # Mask the target position (+1 for BOS token)
        mask_pos = adj_pos + 1
        tokens[0, mask_pos] = self.alphabet.mask_idx

        # Forward pass
        output = self.model(tokens)
        logits = output["logits"][0, mask_pos]  # shape: (vocab_size,)

        # Convert to probabilities over standard amino acids
        aa_indices = torch.tensor(
            [self._aa_to_idx[aa] for aa in AA_ALPHABET], device=self.device
        )
        aa_logits = logits[aa_indices]
        aa_probs = torch.softmax(aa_logits, dim=0)

        # Build AA probability dict
        prob_dict = {aa: aa_probs[i].item() for i, aa in enumerate(AA_ALPHABET)}

        wt_prob = prob_dict.get(wt_aa, 1e-10)
        mut_prob = prob_dict.get(mut_aa, 1e-10)

        llr = math.log(mut_prob) - math.log(wt_prob)

        # Rank of wildtype AA (1 = most probable)
        sorted_probs = sorted(prob_dict.values(), reverse=True)
        wt_rank = sorted_probs.index(wt_prob) + 1 if wt_prob in sorted_probs else 20

        # Shannon entropy at the position
        probs_array = np.array(list(prob_dict.values()))
        entropy = -np.sum(probs_array * np.log(probs_array + 1e-10))

        return {
            "llr": llr,
            "wt_prob": wt_prob,
            "mut_prob": mut_prob,
            "wt_rank": wt_rank,
            "entropy": entropy,
            "all_probs": prob_dict,
        }

    def predict(self, variant: Variant, sequence: Optional[str] = None) -> PredictorScore:
        """Score a missense variant using ESM-2 LLR.

        Args:
            variant: Variant object with aa_ref, aa_alt, aa_pos populated
            sequence: Full protein sequence. If None, will try to fetch it.
        """
        if variant.variant_type != VariantType.MISSENSE:
            return PredictorScore(name=self.name, score=None)

        if not all([variant.aa_ref, variant.aa_alt, variant.aa_pos]):
            logger.warning(f"Missing AA annotations for {variant}")
            return PredictorScore(name=self.name, score=None)

        if sequence is None:
            # In a real pipeline, fetch from UniProt or RefSeq
            logger.warning("No protein sequence provided, cannot compute ESM-2 LLR")
            return PredictorScore(name=self.name, score=None)

        try:
            result = self.compute_llr(
                sequence=sequence,
                position=variant.aa_pos - 1,  # Convert to 0-based
                wt_aa=variant.aa_ref,
                mut_aa=variant.aa_alt,
            )

            # Convert LLR to a 0-1 score using sigmoid transform
            # More negative LLR → more likely pathogenic → higher score
            pathogenicity_score = 1.0 / (1.0 + math.exp(result["llr"] * 2.0))

            return PredictorScore(
                name=self.name,
                score=pathogenicity_score,
                rank_score=None,
                raw_output={
                    "llr": result["llr"],
                    "wt_prob": result["wt_prob"],
                    "mut_prob": result["mut_prob"],
                    "wt_rank": result["wt_rank"],
                    "entropy": result["entropy"],
                },
            )
        except Exception as e:
            logger.error(f"ESM-2 prediction failed for {variant}: {e}")
            return PredictorScore(name=self.name, score=None)

    @torch.no_grad()
    def predict_all_mutations(
        self, sequence: str, position: int, wt_aa: str
    ) -> dict[str, float]:
        """Score all possible amino acid substitutions at a position.

        Useful for generating saturation mutagenesis profiles.
        Returns dict mapping each AA to its LLR.
        """
        results = {}
        seq_context, adj_pos = self._get_sequence_context(sequence, position)

        data = [("seq", seq_context)]
        _, _, tokens = self.batch_converter(data)
        tokens = tokens.to(self.device)

        mask_pos = adj_pos + 1
        tokens[0, mask_pos] = self.alphabet.mask_idx

        output = self.model(tokens)
        logits = output["logits"][0, mask_pos]

        aa_indices = torch.tensor(
            [self._aa_to_idx[aa] for aa in AA_ALPHABET], device=self.device
        )
        aa_logits = logits[aa_indices]
        aa_probs = torch.softmax(aa_logits, dim=0)

        wt_idx_in_alphabet = AA_ALPHABET.index(wt_aa)
        wt_log_prob = torch.log(aa_probs[wt_idx_in_alphabet] + 1e-10).item()

        for i, aa in enumerate(AA_ALPHABET):
            if aa == wt_aa:
                results[aa] = 0.0
            else:
                mut_log_prob = torch.log(aa_probs[i] + 1e-10).item()
                results[aa] = mut_log_prob - wt_log_prob

        return results

    def teardown(self) -> None:
        """Free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
            if self.device == "cuda":
                torch.cuda.empty_cache()
