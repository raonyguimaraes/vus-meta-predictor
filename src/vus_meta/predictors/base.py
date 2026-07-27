"""Base class for all variant effect predictors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from vus_meta.schemas import PredictorScore, Variant


class BasePredictor(ABC):
    """Abstract base class for variant effect predictors.

    Each predictor takes a Variant and returns a PredictorScore.
    Predictors can be:
    - Local models (ESM-2, DNABERT-2) that run inference directly
    - Lookup-based (dbNSFP, precomputed scores)
    - API-based (external services)
    """

    name: str = "base"
    version: str = "0.0.0"
    supported_variant_types: list[str] = []

    @abstractmethod
    def predict(self, variant: Variant) -> PredictorScore:
        """Score a single variant."""
        ...

    def predict_batch(self, variants: list[Variant]) -> list[PredictorScore]:
        """Score a batch of variants. Override for efficiency."""
        return [self.predict(v) for v in variants]

    def can_score(self, variant: Variant) -> bool:
        """Check if this predictor can score the given variant type."""
        if not self.supported_variant_types:
            return True
        return variant.variant_type.value in self.supported_variant_types

    @abstractmethod
    def setup(self) -> None:
        """Initialize model / load data. Called once before predictions."""
        ...

    def teardown(self) -> None:
        """Clean up resources."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, v={self.version})"
