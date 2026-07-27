"""Meta-learner: calibrated ensemble of variant effect predictors.

Architecture:
1. Feature vector from all predictors (scores, sub-scores, derived features)
2. XGBoost gradient boosting classifier (handles missing values natively)
3. Isotonic regression calibration (maps raw scores to calibrated probabilities)
4. SHAP explainability for each prediction

Training:
- ClinVar Pathogenic/Likely Pathogenic → label 1
- ClinVar Benign/Likely Benign → label 0
- Held-out evaluation on recently reclassified VUS
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import shap
except (ImportError, Exception):
    shap = None


class MetaLearner:
    """Calibrated gradient boosting meta-learner for VUS interpretation.

    Features:
    - Handles missing predictor scores (NaN) natively via XGBoost
    - Isotonic regression calibration for well-calibrated probabilities
    - SHAP values for per-prediction explainability
    - Cross-validated training with stratified splits

    Parameters:
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Boosting learning rate
        min_child_weight: Minimum sum of instance weight in a child
        subsample: Fraction of samples used per tree
        colsample_bytree: Fraction of features used per tree
        scale_pos_weight: Ratio of negative to positive samples (for imbalance)
        calibration_method: 'isotonic' or 'sigmoid'
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        min_child_weight: int = 5,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: Optional[float] = None,
        calibration_method: str = "isotonic",
        random_state: int = 42,
    ):
        if xgb is None:
            raise ImportError("xgboost is required: pip install xgboost")

        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "min_child_weight": min_child_weight,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "use_label_encoder": False,
            "random_state": random_state,
            "tree_method": "hist",  # Fast, supports missing values
        }
        if scale_pos_weight is not None:
            self.params["scale_pos_weight"] = scale_pos_weight

        self.calibration_method = calibration_method
        self.random_state = random_state

        self.model: Optional[xgb.XGBClassifier] = None
        self.calibrator: Optional[IsotonicRegression] = None
        self.feature_names: Optional[list[str]] = None
        self.explainer = None
        self._is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[list[str]] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        calibration_fraction: float = 0.2,
    ) -> dict:
        """Train the meta-learner with calibration.

        Args:
            X: Feature matrix (n_samples, n_features). NaN = missing.
            y: Binary labels (1 = pathogenic, 0 = benign)
            feature_names: Names for each feature column
            X_val: Optional validation set features
            y_val: Optional validation set labels
            calibration_fraction: Fraction of training data for calibration

        Returns:
            Dictionary with training metrics
        """
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        logger.info(f"Training meta-learner: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"  Class balance: {y.sum()}/{len(y)} pathogenic ({y.mean():.1%})")

        # Split data: training + calibration
        from sklearn.model_selection import train_test_split

        X_train, X_cal, y_train, y_cal = train_test_split(
            X, y,
            test_size=calibration_fraction,
            stratify=y,
            random_state=self.random_state,
        )

        # Auto-compute class weight if not set
        if "scale_pos_weight" not in self.params:
            n_neg = (y_train == 0).sum()
            n_pos = (y_train == 1).sum()
            self.params["scale_pos_weight"] = n_neg / max(n_pos, 1)
            logger.info(f"  Auto scale_pos_weight: {self.params['scale_pos_weight']:.2f}")

        # Train XGBoost
        self.model = xgb.XGBClassifier(**self.params)

        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=50,
        )

        # Calibrate on held-out calibration set
        raw_probs_cal = self.model.predict_proba(X_cal)[:, 1]
        self.calibrator = IsotonicRegression(
            y_min=0.001, y_max=0.999, out_of_bounds="clip"
        )
        self.calibrator.fit(raw_probs_cal, y_cal)

        # Initialize SHAP explainer
        if shap is not None:
            self.explainer = shap.TreeExplainer(self.model)

        self._is_fitted = True

        # Compute metrics
        metrics = self._evaluate(X_cal, y_cal, "calibration")
        if X_val is not None and y_val is not None:
            val_metrics = self._evaluate(X_val, y_val, "validation")
            metrics.update(val_metrics)

        logger.info(f"Training complete. Calibration AUC: {metrics.get('calibration_auroc', 'N/A'):.4f}")
        return metrics

    def fit_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[list[str]] = None,
        n_folds: int = 5,
    ) -> dict:
        """Cross-validated training with per-fold calibration.

        Returns out-of-fold predictions and aggregated metrics.
        """
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        oof_raw = np.zeros(len(y))
        oof_cal = np.zeros(len(y))
        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            logger.info(f"  Fold {fold + 1}/{n_folds}")
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Split train into train + calibration
            from sklearn.model_selection import train_test_split
            X_tr, X_cal, y_tr, y_cal = train_test_split(
                X_train, y_train, test_size=0.15, stratify=y_train,
                random_state=self.random_state + fold,
            )

            model = xgb.XGBClassifier(**self.params)
            model.fit(X_tr, y_tr, eval_set=[(X_cal, y_cal)], verbose=0)

            raw_probs = model.predict_proba(X_val)[:, 1]
            oof_raw[val_idx] = raw_probs

            # Calibrate
            cal_probs = model.predict_proba(X_cal)[:, 1]
            calibrator = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
            calibrator.fit(cal_probs, y_cal)
            oof_cal[val_idx] = calibrator.predict(raw_probs)

            fold_auc = roc_auc_score(y_val, oof_cal[val_idx])
            fold_metrics.append({"fold": fold, "auroc": fold_auc})
            logger.info(f"    Fold {fold + 1} AUROC: {fold_auc:.4f}")

        # Train final model on all data
        self.fit(X, y, feature_names=feature_names)

        # Overall OOF metrics
        overall_auc = roc_auc_score(y, oof_cal)
        overall_ap = average_precision_score(y, oof_cal)
        overall_brier = brier_score_loss(y, oof_cal)

        logger.info(f"CV Results — AUROC: {overall_auc:.4f}, AUPRC: {overall_ap:.4f}, Brier: {overall_brier:.4f}")

        return {
            "cv_auroc": overall_auc,
            "cv_auprc": overall_ap,
            "cv_brier": overall_brier,
            "fold_metrics": fold_metrics,
            "oof_predictions": oof_cal,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict calibrated pathogenicity probabilities.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Calibrated probability array of shape (n_samples,)
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        raw_probs = self.model.predict_proba(X)[:, 1]

        if self.calibrator is not None:
            return self.calibrator.predict(raw_probs)
        return raw_probs

    def predict_single(self, features: np.ndarray) -> tuple[float, dict[str, float]]:
        """Predict a single variant and return score + SHAP explanations.

        Returns:
            (calibrated_score, {feature_name: shap_value})
        """
        X = features.reshape(1, -1)
        score = self.predict(X)[0]

        explanations = {}
        if self.explainer is not None and self.feature_names is not None:
            shap_values = self.explainer.shap_values(X)
            for i, name in enumerate(self.feature_names):
                explanations[name] = float(shap_values[0, i])

        return float(score), explanations

    def predict_with_confidence(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict with confidence estimates.

        Confidence is based on:
        1. Distance from the 0.5 decision boundary (certainty)
        2. Feature completeness (fewer missing values = higher confidence)

        Returns:
            (scores, confidences) — both of shape (n_samples,)
        """
        scores = self.predict(X)

        # Certainty: how far from 0.5 (rescaled to [0, 1])
        certainty = np.abs(scores - 0.5) * 2.0

        # Completeness: fraction of non-missing features
        completeness = 1.0 - np.isnan(X).mean(axis=1)

        # Combined confidence
        confidence = certainty * 0.7 + completeness * 0.3

        return scores, np.clip(confidence, 0.0, 1.0)

    def _evaluate(self, X: np.ndarray, y: np.ndarray, prefix: str) -> dict:
        """Evaluate model performance."""
        raw_probs = self.model.predict_proba(X)[:, 1]
        cal_probs = self.calibrator.predict(raw_probs) if self.calibrator else raw_probs

        return {
            f"{prefix}_auroc": roc_auc_score(y, cal_probs),
            f"{prefix}_auprc": average_precision_score(y, cal_probs),
            f"{prefix}_brier": brier_score_loss(y, cal_probs),
        }

    def feature_importance(self, importance_type: str = "gain") -> dict[str, float]:
        """Get feature importance from the XGBoost model."""
        if self.model is None or self.feature_names is None:
            return {}
        importances = self.model.feature_importances_
        return dict(
            sorted(
                zip(self.feature_names, importances.tolist()),
                key=lambda x: x[1],
                reverse=True,
            )
        )

    def save(self, path: str | Path) -> None:
        """Save model, calibrator, and metadata."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save XGBoost model
        self.model.save_model(str(path / "xgb_model.json"))

        # Save calibrator
        with open(path / "calibrator.pkl", "wb") as f:
            pickle.dump(self.calibrator, f)

        # Save metadata
        metadata = {
            "feature_names": self.feature_names,
            "params": self.params,
            "calibration_method": self.calibration_method,
            "version": "1.0",
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> MetaLearner:
        """Load a saved meta-learner."""
        path = Path(path)

        with open(path / "metadata.json") as f:
            metadata = json.load(f)

        learner = cls(**{k: v for k, v in metadata["params"].items()
                         if k in cls.__init__.__code__.co_varnames})
        learner.feature_names = metadata["feature_names"]

        # Load XGBoost model
        learner.model = xgb.XGBClassifier()
        learner.model.load_model(str(path / "xgb_model.json"))

        # Load calibrator
        with open(path / "calibrator.pkl", "rb") as f:
            learner.calibrator = pickle.load(f)

        # Initialize SHAP
        if shap is not None:
            learner.explainer = shap.TreeExplainer(learner.model)

        learner._is_fitted = True
        logger.info(f"Model loaded from {path}")
        return learner
