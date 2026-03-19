"""Ensemble fraud detection model combining Isolation Forest and XGBoost.

The ensemble uses a weighted average of an unsupervised anomaly detector
(Isolation Forest) and a supervised classifier (XGBoost) to produce a
unified fraud probability score.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier

from fraud_detector.config import (
    EnsembleConfig,
    IsolationForestConfig,
    XGBoostConfig,
)

logger = logging.getLogger(__name__)


class FraudEnsemble:
    """Ensemble model for customs fraud detection.

    Combines an Isolation Forest (unsupervised anomaly scoring) with an
    XGBoost binary classifier (supervised fraud prediction). The final
    fraud score is a weighted blend of both model outputs.

    Args:
        iso_config: Isolation Forest hyperparameters.
        xgb_config: XGBoost hyperparameters.
        ensemble_config: Ensemble blending weights and threshold.
    """

    def __init__(
        self,
        iso_config: IsolationForestConfig | None = None,
        xgb_config: XGBoostConfig | None = None,
        ensemble_config: EnsembleConfig | None = None,
    ) -> None:
        self.iso_config = iso_config or IsolationForestConfig()
        self.xgb_config = xgb_config or XGBoostConfig()
        self.ensemble_config = ensemble_config or EnsembleConfig()

        self.isolation_forest: IsolationForest | None = None
        self.xgb_classifier: XGBClassifier | None = None
        self.feature_names: list[str] = []
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        """Whether the ensemble has been fitted on training data."""
        return self._is_fitted

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray | pd.Series,
        eval_set: list[tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> FraudEnsemble:
        """Train both component models.

        Args:
            X: Feature matrix.
            y: Binary fraud labels (1 = fraud, 0 = legitimate).
            eval_set: Optional evaluation set(s) for XGBoost early stopping.

        Returns:
            Self, for method chaining.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.int32)

        logger.info(
            "Training Isolation Forest (n_estimators=%d, contamination=%.3f)",
            self.iso_config.n_estimators,
            self.iso_config.contamination,
        )
        self.isolation_forest = IsolationForest(
            n_estimators=self.iso_config.n_estimators,
            contamination=self.iso_config.contamination,
            max_samples=self.iso_config.max_samples,
            max_features=self.iso_config.max_features,
            random_state=self.iso_config.random_state,
            n_jobs=self.iso_config.n_jobs,
        )
        self.isolation_forest.fit(X_arr)

        logger.info(
            "Training XGBoost (n_estimators=%d, max_depth=%d, lr=%.4f)",
            self.xgb_config.n_estimators,
            self.xgb_config.max_depth,
            self.xgb_config.learning_rate,
        )
        self.xgb_classifier = XGBClassifier(
            n_estimators=self.xgb_config.n_estimators,
            max_depth=self.xgb_config.max_depth,
            learning_rate=self.xgb_config.learning_rate,
            subsample=self.xgb_config.subsample,
            colsample_bytree=self.xgb_config.colsample_bytree,
            min_child_weight=self.xgb_config.min_child_weight,
            gamma=self.xgb_config.gamma,
            reg_alpha=self.xgb_config.reg_alpha,
            reg_lambda=self.xgb_config.reg_lambda,
            scale_pos_weight=self.xgb_config.scale_pos_weight,
            eval_metric=self.xgb_config.eval_metric,
            random_state=self.xgb_config.random_state,
            n_jobs=self.xgb_config.n_jobs,
            use_label_encoder=False,
        )

        fit_params: dict[str, Any] = {}
        if eval_set is not None:
            fit_params["eval_set"] = eval_set
            fit_params["verbose"] = False

        self.xgb_classifier.fit(X_arr, y_arr, **fit_params)

        self._is_fitted = True
        logger.info("Ensemble training complete.")
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Compute blended fraud probability scores.

        Args:
            X: Feature matrix with the same schema as training data.

        Returns:
            1-D array of fraud probability scores in [0, 1].

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._check_fitted()
        X_arr = np.asarray(X, dtype=np.float32)

        # Isolation Forest: convert anomaly score to [0, 1]
        iso_raw = self.isolation_forest.decision_function(X_arr)
        iso_scores = 1.0 / (1.0 + np.exp(iso_raw))  # sigmoid transform

        # XGBoost: probability of class 1 (fraud)
        xgb_scores = self.xgb_classifier.predict_proba(X_arr)[:, 1]

        # Weighted blend
        w_iso = self.ensemble_config.isolation_forest_weight
        w_xgb = self.ensemble_config.xgboost_weight
        blended = (w_iso * iso_scores + w_xgb * xgb_scores) / (w_iso + w_xgb)

        return blended

    def predict(
        self, X: pd.DataFrame | np.ndarray, threshold: float | None = None
    ) -> np.ndarray:
        """Produce binary fraud predictions.

        Args:
            X: Feature matrix.
            threshold: Decision threshold. Defaults to the configured value.

        Returns:
            1-D array of binary predictions (1 = fraud, 0 = legitimate).
        """
        if threshold is None:
            threshold = self.ensemble_config.fraud_threshold
        scores = self.predict_proba(X)
        return (scores >= threshold).astype(np.int32)

    def save(self, path: str | Path) -> None:
        """Persist the ensemble to disk.

        Args:
            path: Directory where model artifacts will be saved.
        """
        self._check_fitted()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.isolation_forest, path / "isolation_forest.joblib")
        joblib.dump(self.xgb_classifier, path / "xgb_classifier.joblib")
        joblib.dump(
            {
                "feature_names": self.feature_names,
                "iso_config": self.iso_config,
                "xgb_config": self.xgb_config,
                "ensemble_config": self.ensemble_config,
            },
            path / "metadata.joblib",
        )
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> FraudEnsemble:
        """Load a persisted ensemble from disk.

        Args:
            path: Directory containing model artifacts.

        Returns:
            A fitted ``FraudEnsemble`` instance.
        """
        path = Path(path)
        metadata = joblib.load(path / "metadata.joblib")

        instance = cls(
            iso_config=metadata["iso_config"],
            xgb_config=metadata["xgb_config"],
            ensemble_config=metadata["ensemble_config"],
        )
        instance.isolation_forest = joblib.load(path / "isolation_forest.joblib")
        instance.xgb_classifier = joblib.load(path / "xgb_classifier.joblib")
        instance.feature_names = metadata["feature_names"]
        instance._is_fitted = True

        logger.info("Model loaded from %s", path)
        return instance

    def get_params(self) -> dict[str, Any]:
        """Return a dictionary of all model parameters for experiment tracking."""
        return {
            "iso_n_estimators": self.iso_config.n_estimators,
            "iso_contamination": self.iso_config.contamination,
            "xgb_n_estimators": self.xgb_config.n_estimators,
            "xgb_max_depth": self.xgb_config.max_depth,
            "xgb_learning_rate": self.xgb_config.learning_rate,
            "xgb_subsample": self.xgb_config.subsample,
            "xgb_colsample_bytree": self.xgb_config.colsample_bytree,
            "xgb_scale_pos_weight": self.xgb_config.scale_pos_weight,
            "ensemble_iso_weight": self.ensemble_config.isolation_forest_weight,
            "ensemble_xgb_weight": self.ensemble_config.xgboost_weight,
            "ensemble_threshold": self.ensemble_config.fraud_threshold,
        }

    def _check_fitted(self) -> None:
        """Raise if the model has not been fitted."""
        if not self._is_fitted:
            raise RuntimeError(
                "FraudEnsemble has not been fitted. Call .fit() first or "
                "load a persisted model with FraudEnsemble.load()."
            )
