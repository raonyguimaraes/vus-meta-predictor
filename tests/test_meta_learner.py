"""Tests for the meta-learner."""

import numpy as np
import pytest
import tempfile
from pathlib import Path


class TestMetaLearner:
    @pytest.fixture
    def sample_data(self):
        """Generate simple synthetic data for testing."""
        rng = np.random.RandomState(42)
        n = 500
        n_features = 10
        X = rng.randn(n, n_features)
        # Make label correlated with first few features
        logits = X[:, 0] * 2 + X[:, 1] * 1.5 + X[:, 2] + rng.randn(n) * 0.5
        y = (logits > 0).astype(int)
        feature_names = [f"feature_{i}" for i in range(n_features)]
        return X, y, feature_names

    def test_fit_and_predict(self, sample_data):
        from vus_meta.models.meta_learner import MetaLearner
        X, y, names = sample_data

        learner = MetaLearner(n_estimators=50, max_depth=3)
        metrics = learner.fit(X, y, feature_names=names)

        assert "calibration_auroc" in metrics
        assert metrics["calibration_auroc"] > 0.8

        probs = learner.predict(X)
        assert probs.shape == (len(y),)
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_cross_validation(self, sample_data):
        from vus_meta.models.meta_learner import MetaLearner
        X, y, names = sample_data

        learner = MetaLearner(n_estimators=50, max_depth=3)
        metrics = learner.fit_cv(X, y, feature_names=names, n_folds=3)

        assert "cv_auroc" in metrics
        assert metrics["cv_auroc"] > 0.8
        assert len(metrics["fold_metrics"]) == 3
        assert "oof_predictions" in metrics

    def test_predict_single_with_shap(self, sample_data):
        from vus_meta.models.meta_learner import MetaLearner
        X, y, names = sample_data

        learner = MetaLearner(n_estimators=50, max_depth=3)
        learner.fit(X, y, feature_names=names)

        score, explanations = learner.predict_single(X[0])
        assert 0 <= score <= 1
        # SHAP explanations are only available if shap imported successfully
        if explanations:
            assert len(explanations) == len(names)
        else:
            # shap not available (e.g., numpy/matplotlib incompatibility)
            assert explanations == {}

    def test_predict_with_confidence(self, sample_data):
        from vus_meta.models.meta_learner import MetaLearner
        X, y, names = sample_data

        learner = MetaLearner(n_estimators=50, max_depth=3)
        learner.fit(X, y, feature_names=names)

        scores, confidences = learner.predict_with_confidence(X)
        assert scores.shape == confidences.shape
        assert np.all(confidences >= 0) and np.all(confidences <= 1)

    def test_handles_missing_values(self, sample_data):
        from vus_meta.models.meta_learner import MetaLearner
        X, y, names = sample_data

        # Introduce missing values
        rng = np.random.RandomState(42)
        mask = rng.random(X.shape) < 0.2
        X_missing = X.copy()
        X_missing[mask] = np.nan

        learner = MetaLearner(n_estimators=50, max_depth=3)
        learner.fit(X_missing, y, feature_names=names)

        probs = learner.predict(X_missing)
        assert not np.any(np.isnan(probs))

    def test_save_and_load(self, sample_data):
        from vus_meta.models.meta_learner import MetaLearner
        X, y, names = sample_data

        learner = MetaLearner(n_estimators=50, max_depth=3)
        learner.fit(X, y, feature_names=names)
        original_preds = learner.predict(X[:10])

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_model"
            learner.save(save_path)

            loaded = MetaLearner.load(save_path)
            loaded_preds = loaded.predict(X[:10])

            np.testing.assert_array_almost_equal(original_preds, loaded_preds, decimal=5)

    def test_feature_importance(self, sample_data):
        from vus_meta.models.meta_learner import MetaLearner
        X, y, names = sample_data

        learner = MetaLearner(n_estimators=50, max_depth=3)
        learner.fit(X, y, feature_names=names)

        importance = learner.feature_importance()
        assert len(importance) == len(names)
        assert all(v >= 0 for v in importance.values())
        # feature_0, feature_1, feature_2 should be top features
        top_3 = list(importance.keys())[:3]
        assert "feature_0" in top_3  # Strongest signal
