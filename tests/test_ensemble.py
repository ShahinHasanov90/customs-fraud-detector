"""Tests for the ensemble fraud detection model."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_detector.config import EnsembleConfig, IsolationForestConfig, XGBoostConfig
from fraud_detector.models.ensemble import FraudEnsemble


@pytest.fixture
def synthetic_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Generate synthetic training data with a clear fraud signal."""
    rng = np.random.RandomState(42)
    n_legitimate = 500
    n_fraud = 50

    # Legitimate declarations
    legit = pd.DataFrame(
        {
            "price_per_kg": rng.normal(50, 10, n_legitimate),
            "weight_log": rng.normal(3, 0.5, n_legitimate),
            "value_ratio": rng.normal(1.0, 0.2, n_legitimate),
            "origin_risk": rng.uniform(0.05, 0.25, n_legitimate),
            "importer_fraud_rate": rng.uniform(0.0, 0.05, n_legitimate),
        }
    )

    # Fraudulent declarations (shifted distributions)
    fraud = pd.DataFrame(
        {
            "price_per_kg": rng.normal(10, 5, n_fraud),
            "weight_log": rng.normal(5, 1.0, n_fraud),
            "value_ratio": rng.normal(0.2, 0.1, n_fraud),
            "origin_risk": rng.uniform(0.3, 0.6, n_fraud),
            "importer_fraud_rate": rng.uniform(0.1, 0.5, n_fraud),
        }
    )

    X = pd.concat([legit, fraud], ignore_index=True)
    y = np.array([0] * n_legitimate + [1] * n_fraud)

    return X, y


@pytest.fixture
def small_config() -> tuple[IsolationForestConfig, XGBoostConfig, EnsembleConfig]:
    """Lightweight model configs for fast testing."""
    iso = IsolationForestConfig(n_estimators=20, contamination=0.05, random_state=42)
    xgb = XGBoostConfig(
        n_estimators=20,
        max_depth=3,
        learning_rate=0.1,
        scale_pos_weight=10,
        random_state=42,
    )
    ens = EnsembleConfig(
        isolation_forest_weight=0.3,
        xgboost_weight=0.7,
        fraud_threshold=0.5,
    )
    return iso, xgb, ens


class TestFraudEnsemble:
    """Tests for the FraudEnsemble model."""

    def test_not_fitted_raises(self) -> None:
        """Calling predict on an unfitted model should raise RuntimeError."""
        model = FraudEnsemble()
        X = np.random.rand(5, 3)
        with pytest.raises(RuntimeError, match="has not been fitted"):
            model.predict_proba(X)

    def test_fit_and_predict(
        self,
        synthetic_data: tuple[pd.DataFrame, np.ndarray],
        small_config: tuple[IsolationForestConfig, XGBoostConfig, EnsembleConfig],
    ) -> None:
        """Model should fit and produce valid probability scores."""
        X, y = synthetic_data
        iso, xgb, ens = small_config

        model = FraudEnsemble(iso_config=iso, xgb_config=xgb, ensemble_config=ens)
        model.fit(X, y)

        assert model.is_fitted
        scores = model.predict_proba(X)
        assert scores.shape == (len(X),)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_binary_predictions(
        self,
        synthetic_data: tuple[pd.DataFrame, np.ndarray],
        small_config: tuple[IsolationForestConfig, XGBoostConfig, EnsembleConfig],
    ) -> None:
        """Binary predictions should be 0 or 1."""
        X, y = synthetic_data
        iso, xgb, ens = small_config

        model = FraudEnsemble(iso_config=iso, xgb_config=xgb, ensemble_config=ens)
        model.fit(X, y)
        preds = model.predict(X)

        assert set(np.unique(preds)).issubset({0, 1})

    def test_custom_threshold(
        self,
        synthetic_data: tuple[pd.DataFrame, np.ndarray],
        small_config: tuple[IsolationForestConfig, XGBoostConfig, EnsembleConfig],
    ) -> None:
        """A very low threshold should flag most samples."""
        X, y = synthetic_data
        iso, xgb, ens = small_config

        model = FraudEnsemble(iso_config=iso, xgb_config=xgb, ensemble_config=ens)
        model.fit(X, y)

        preds_strict = model.predict(X, threshold=0.9)
        preds_loose = model.predict(X, threshold=0.1)

        assert preds_loose.sum() >= preds_strict.sum()

    def test_save_and_load(
        self,
        synthetic_data: tuple[pd.DataFrame, np.ndarray],
        small_config: tuple[IsolationForestConfig, XGBoostConfig, EnsembleConfig],
    ) -> None:
        """A saved model should produce identical scores after loading."""
        X, y = synthetic_data
        iso, xgb, ens = small_config

        model = FraudEnsemble(iso_config=iso, xgb_config=xgb, ensemble_config=ens)
        model.fit(X, y)
        original_scores = model.predict_proba(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir)
            loaded = FraudEnsemble.load(tmpdir)

        loaded_scores = loaded.predict_proba(X)
        np.testing.assert_allclose(original_scores, loaded_scores, rtol=1e-5)

    def test_get_params(
        self,
        small_config: tuple[IsolationForestConfig, XGBoostConfig, EnsembleConfig],
    ) -> None:
        """get_params should return a dictionary of hyperparameters."""
        iso, xgb, ens = small_config
        model = FraudEnsemble(iso_config=iso, xgb_config=xgb, ensemble_config=ens)
        params = model.get_params()

        assert isinstance(params, dict)
        assert params["iso_n_estimators"] == 20
        assert params["xgb_max_depth"] == 3
        assert params["ensemble_iso_weight"] == 0.3

    def test_feature_names_preserved(
        self,
        synthetic_data: tuple[pd.DataFrame, np.ndarray],
        small_config: tuple[IsolationForestConfig, XGBoostConfig, EnsembleConfig],
    ) -> None:
        """Feature names should be stored when training with a DataFrame."""
        X, y = synthetic_data
        iso, xgb, ens = small_config

        model = FraudEnsemble(iso_config=iso, xgb_config=xgb, ensemble_config=ens)
        model.fit(X, y)

        assert model.feature_names == list(X.columns)

    def test_fraud_detection_quality(
        self,
        synthetic_data: tuple[pd.DataFrame, np.ndarray],
        small_config: tuple[IsolationForestConfig, XGBoostConfig, EnsembleConfig],
    ) -> None:
        """Model should achieve reasonable separation on synthetic data."""
        X, y = synthetic_data
        iso, xgb, ens = small_config

        model = FraudEnsemble(iso_config=iso, xgb_config=xgb, ensemble_config=ens)
        model.fit(X, y)
        scores = model.predict_proba(X)

        # Fraud samples should generally score higher than legitimate
        fraud_mean = scores[y == 1].mean()
        legit_mean = scores[y == 0].mean()
        assert fraud_mean > legit_mean
