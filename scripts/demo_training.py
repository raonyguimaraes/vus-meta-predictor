#!/usr/bin/env python3
"""Demo: Train and evaluate the VUS Meta-Predictor with synthetic data.

This script demonstrates the full pipeline:
1. Generate synthetic feature vectors that mimic real ClinVar + dbNSFP data
2. Train the XGBoost meta-learner with cross-validation
3. Calibrate predictions with isotonic regression
4. Generate SHAP explanations
5. Evaluate on held-out "VUS" variants

Run: python scripts/demo_training.py
"""

import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")


def generate_synthetic_data(
    n_pathogenic: int = 5000,
    n_benign: int = 5000,
    n_vus: int = 2000,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Generate synthetic variant features mimicking real predictor score distributions.

    We model each feature with realistic distributions for pathogenic vs benign variants,
    including correlated features and missing data patterns.

    Returns:
        X_train, y_train, X_vus, feature_names
    """
    rng = np.random.RandomState(random_state)

    feature_configs = {
        # (name, path_mean, path_std, ben_mean, ben_std, missing_rate)
        # Language model features
        "esm2_score":         (0.75, 0.15, 0.30, 0.18, 0.15),
        "esm2_llr":           (-2.5, 1.5, -0.3, 0.8, 0.15),
        "esm2_wt_prob":       (0.35, 0.20, 0.55, 0.20, 0.15),
        "esm2_mut_prob":      (0.05, 0.04, 0.15, 0.10, 0.15),
        "esm2_wt_rank":       (3.0, 2.0, 1.5, 1.0, 0.15),
        "esm2_entropy":       (1.8, 0.6, 2.5, 0.5, 0.15),
        "dnabert2_score":     (0.70, 0.18, 0.35, 0.20, 0.25),
        "dnabert2_llr":       (-1.5, 1.0, -0.2, 0.5, 0.25),
        "dnabert2_embedding_dist": (0.15, 0.08, 0.05, 0.03, 0.25),
        "nt_score":           (0.68, 0.20, 0.33, 0.18, 0.30),
        "nt_l2_distance":     (2.5, 1.2, 0.8, 0.5, 0.30),
        # dbNSFP scores
        "cadd_phred":         (28.0, 5.0, 12.0, 6.0, 0.05),
        "revel":              (0.80, 0.15, 0.20, 0.15, 0.08),
        "alphamissense":      (0.85, 0.12, 0.15, 0.12, 0.10),
        "metarnn":            (0.82, 0.14, 0.18, 0.14, 0.10),
        "bayesdel_addaf":     (0.40, 0.20, -0.20, 0.18, 0.10),
        "vest4":              (0.85, 0.12, 0.30, 0.18, 0.12),
        "polyphen2_hdiv":     (0.95, 0.08, 0.20, 0.22, 0.08),
        "polyphen2_hvar":     (0.90, 0.12, 0.18, 0.20, 0.08),
        "sift":               (0.02, 0.03, 0.40, 0.25, 0.08),
        "mpc":                (2.5, 0.8, 0.8, 0.6, 0.15),
        "primate_ai":         (0.80, 0.14, 0.35, 0.18, 0.12),
        # Conservation
        "gerp_rs":            (5.0, 1.5, 1.5, 2.0, 0.05),
        "phylop100way":       (7.0, 2.0, 1.5, 2.5, 0.03),
        "phylop30way":        (1.8, 0.5, 0.3, 0.6, 0.03),
        "phastcons100way":    (0.98, 0.05, 0.50, 0.30, 0.03),
        "phastcons30way":     (0.95, 0.08, 0.40, 0.28, 0.03),
        # Splicing
        "spliceai_max":       (0.15, 0.20, 0.02, 0.04, 0.05),
        # Population frequency
        "gnomad_af":          (1e-6, 5e-6, 5e-4, 1e-3, 0.20),
        "gnomad_af_popmax":   (5e-6, 1e-5, 1e-3, 2e-3, 0.20),
        "gnomad_log10_af":    (-6.5, 1.0, -3.5, 1.0, 0.20),
        "gnomad_nhomalt":     (0.0, 0.1, 2.0, 5.0, 0.20),
        # Derived
        "sift_inverted":      (0.98, 0.03, 0.60, 0.25, 0.08),
        "conservation_mean":  (3.5, 1.0, 0.8, 1.0, 0.03),
        "lm_agreement":       (0.85, 0.10, 0.80, 0.12, 0.30),
    }

    feature_names = list(feature_configs.keys())
    n_features = len(feature_names)
    n_total = n_pathogenic + n_benign

    # Generate pathogenic features
    X_path = np.zeros((n_pathogenic, n_features))
    for i, (name, (p_mean, p_std, _, _, _)) in enumerate(feature_configs.items()):
        X_path[:, i] = rng.normal(p_mean, p_std, n_pathogenic)

    # Generate benign features
    X_ben = np.zeros((n_benign, n_features))
    for i, (name, (_, _, b_mean, b_std, _)) in enumerate(feature_configs.items()):
        X_ben[:, i] = rng.normal(b_mean, b_std, n_benign)

    # Combine
    X_train = np.vstack([X_path, X_ben])
    y_train = np.concatenate([np.ones(n_pathogenic), np.zeros(n_benign)])

    # Add correlated noise (features should be somewhat correlated)
    noise = rng.normal(0, 0.05, X_train.shape)
    X_train += noise

    # Clip values to realistic ranges
    for i, name in enumerate(feature_names):
        if "prob" in name or "phastcons" in name or name in ("revel", "alphamissense", "polyphen2_hdiv", "polyphen2_hvar", "vest4", "lm_agreement"):
            X_train[:, i] = np.clip(X_train[:, i], 0, 1)
        if name == "gnomad_af":
            X_train[:, i] = np.clip(X_train[:, i], 0, 1)
        if name == "sift":
            X_train[:, i] = np.clip(X_train[:, i], 0, 1)

    # Introduce missing values
    for i, (name, (_, _, _, _, miss_rate)) in enumerate(feature_configs.items()):
        mask = rng.random(n_total) < miss_rate
        X_train[mask, i] = np.nan

    # Generate VUS data (intermediate distributions — between pathogenic and benign)
    X_vus = np.zeros((n_vus, n_features))
    for i, (name, (p_mean, p_std, b_mean, b_std, miss_rate)) in enumerate(feature_configs.items()):
        # VUS are a mix: some are truly pathogenic, some benign, some ambiguous
        vus_mean = (p_mean + b_mean) / 2
        vus_std = max(p_std, b_std) * 1.2  # Wider distribution
        X_vus[:, i] = rng.normal(vus_mean, vus_std, n_vus)
        # Higher missing rate for VUS
        mask = rng.random(n_vus) < miss_rate * 1.5
        X_vus[mask, i] = np.nan

    # Shuffle training data
    shuffle_idx = rng.permutation(n_total)
    X_train = X_train[shuffle_idx]
    y_train = y_train[shuffle_idx]

    logger.info(f"Generated synthetic data:")
    logger.info(f"  Training: {X_train.shape[0]} variants × {X_train.shape[1]} features")
    logger.info(f"  VUS: {X_vus.shape[0]} variants")
    logger.info(f"  Missing rate: {np.isnan(X_train).mean():.1%}")

    return X_train, y_train, X_vus, feature_names


def main():
    """Run the full demo pipeline."""
    print("=" * 70)
    print("  VUS Meta-Predictor — Training Demo")
    print("=" * 70)
    print()

    # 1. Generate synthetic data
    print("Step 1: Generating synthetic ClinVar-like data...")
    X_train, y_train, X_vus, feature_names = generate_synthetic_data(
        n_pathogenic=2000, n_benign=2000, n_vus=500
    )
    print()

    # 2. Train meta-learner with cross-validation
    print("Step 2: Training meta-learner (5-fold CV)...")
    from vus_meta.models.meta_learner import MetaLearner

    learner = MetaLearner(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
    )

    metrics = learner.fit_cv(X_train, y_train, feature_names=feature_names, n_folds=5)
    print(f"\n  Cross-validation results:")
    print(f"    AUROC: {metrics['cv_auroc']:.4f}")
    print(f"    AUPRC: {metrics['cv_auprc']:.4f}")
    print(f"    Brier: {metrics['cv_brier']:.4f}")
    print()

    # 3. Feature importance
    print("Step 3: Feature importance (XGBoost gain):")
    importance = learner.feature_importance()
    for i, (name, imp) in enumerate(list(importance.items())[:15]):
        bar = "█" * int(imp * 200)
        print(f"  {i+1:2d}. {name:30s} {imp:.4f} {bar}")
    print()

    # 4. Score VUS variants
    print("Step 4: Scoring VUS variants...")
    from vus_meta.schemas import score_to_classification

    vus_scores, vus_confidences = learner.predict_with_confidence(X_vus)

    # Classification distribution
    classifications = [score_to_classification(s) for s in vus_scores]
    class_counts = pd.Series([c.value for c in classifications]).value_counts()
    print(f"\n  VUS reclassification distribution:")
    for cls, count in class_counts.items():
        pct = count / len(classifications) * 100
        bar = "█" * int(pct)
        print(f"    {cls:30s} {count:5d} ({pct:5.1f}%) {bar}")
    print()

    # 5. Example single prediction with SHAP
    print("Step 5: Example prediction with explanations:")
    score, explanations = learner.predict_single(X_vus[0])
    cls = score_to_classification(score)
    print(f"\n  Score: {score:.4f} → {cls.value}")
    print(f"  Top contributing features:")
    sorted_exp = sorted(explanations.items(), key=lambda x: abs(x[1]), reverse=True)
    for name, shap_val in sorted_exp[:8]:
        direction = "→ pathogenic" if shap_val > 0 else "→ benign"
        print(f"    {name:30s} SHAP={shap_val:+.4f} {direction}")
    print()

    # 6. Save model
    model_dir = Path("data/models/demo_v1")
    print(f"Step 6: Saving model to {model_dir}...")
    learner.save(model_dir)
    print()

    # 7. Load and verify
    print("Step 7: Loading saved model and verifying...")
    loaded = MetaLearner.load(model_dir)
    verify_scores = loaded.predict(X_vus[:10])
    original_scores = learner.predict(X_vus[:10])
    max_diff = np.max(np.abs(verify_scores - original_scores))
    print(f"  Max score difference after save/load: {max_diff:.10f}")
    print(f"  ✅ Model save/load verified!" if max_diff < 1e-6 else "  ❌ Mismatch detected!")
    print()

    # 8. Calibration assessment
    print("Step 8: Calibration assessment:")
    from sklearn.calibration import calibration_curve

    # Use OOF predictions for calibration assessment
    oof_preds = metrics["oof_predictions"]
    prob_true, prob_pred = calibration_curve(y_train, oof_preds, n_bins=10)
    print(f"  {'Predicted':>12s}  {'Actual':>12s}  {'Diff':>8s}")
    for pt, pp in zip(prob_true, prob_pred):
        diff = abs(pt - pp)
        marker = "✓" if diff < 0.05 else "!"
        print(f"  {pp:12.3f}  {pt:12.3f}  {diff:8.3f} {marker}")

    mean_cal_error = np.mean(np.abs(prob_true - prob_pred))
    print(f"\n  Mean calibration error: {mean_cal_error:.4f}")
    print()

    print("=" * 70)
    print("  Demo complete! 🧬")
    print("=" * 70)


if __name__ == "__main__":
    main()
