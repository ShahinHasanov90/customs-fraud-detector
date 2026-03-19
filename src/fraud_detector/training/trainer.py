"""Model training pipeline with MLflow experiment tracking.

Orchestrates data loading, feature engineering, model training, evaluation,
and artifact persistence -- with full experiment tracking via MLflow.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from fraud_detector.config import AppConfig, load_config
from fraud_detector.features.engineering import FeatureEngineer
from fraud_detector.features.selection import FeatureSelector
from fraud_detector.models.ensemble import FraudEnsemble

logger = logging.getLogger(__name__)


class FraudModelTrainer:
    """End-to-end training pipeline for the customs fraud detection model.

    Handles data splitting, feature engineering, model training with
    cross-validation, evaluation, and MLflow experiment logging.

    Args:
        config_path: Path to the YAML configuration file.
        config: Pre-loaded ``AppConfig`` (takes precedence over config_path).
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        config: AppConfig | None = None,
    ) -> None:
        self.config = config or load_config(config_path)
        self.feature_engineer = FeatureEngineer()
        self.feature_selector = FeatureSelector()
        self.model: FraudEnsemble | None = None
        self._metrics: dict[str, float] = {}

        # Configure MLflow
        mlflow.set_tracking_uri(self.config.mlflow.tracking_uri)
        mlflow.set_experiment(self.config.mlflow.experiment_name)

    def train(
        self,
        data_path: str | Path | None = None,
        df: pd.DataFrame | None = None,
    ) -> FraudEnsemble:
        """Train the fraud detection ensemble.

        Either ``data_path`` or ``df`` must be provided.

        Args:
            data_path: Path to a Parquet file containing labeled declarations.
            df: Pre-loaded DataFrame of labeled declarations.

        Returns:
            The fitted ``FraudEnsemble`` model.
        """
        if df is None:
            if data_path is None:
                raise ValueError("Provide either data_path or df.")
            data_path = Path(data_path)
            logger.info("Loading data from %s", data_path)
            df = pd.read_parquet(data_path)

        logger.info("Training data shape: %s", df.shape)

        # Feature engineering
        df = self.feature_engineer.transform(df)

        # Separate features and target
        target_col = "is_fraud"
        feature_cols = self.feature_engineer.get_feature_columns()
        available_cols = [c for c in feature_cols if c in df.columns]

        X = df[available_cols].select_dtypes(include=[np.number])
        y = df[target_col].astype(int)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.training.test_size,
            stratify=y if self.config.training.stratify else None,
            random_state=self.config.training.random_state,
        )

        logger.info(
            "Split: train=%d, test=%d, fraud_rate=%.3f",
            len(X_train),
            len(X_test),
            y_train.mean(),
        )

        # Feature selection
        self.feature_selector.fit(X_train, y=y_train)
        X_train = self.feature_selector.transform(X_train)
        X_test = self.feature_selector.transform(X_test)

        # Train the ensemble
        self.model = FraudEnsemble(
            iso_config=self.config.isolation_forest,
            xgb_config=self.config.xgboost,
            ensemble_config=self.config.ensemble,
        )

        with mlflow.start_run() as run:
            logger.info("MLflow run: %s", run.info.run_id)

            # Log parameters
            mlflow.log_params(self.model.get_params())
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("n_train_samples", len(X_train))
            mlflow.log_param("n_test_samples", len(X_test))
            mlflow.log_param("fraud_prevalence", float(y_train.mean()))

            # Fit
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(np.asarray(X_test), np.asarray(y_test))],
            )

            # Evaluate
            self._metrics = self._evaluate(X_test, y_test)
            mlflow.log_metrics(self._metrics)

            # Cross-validation
            cv_scores = self._cross_validate(X, y)
            mlflow.log_metric("cv_f1_mean", float(np.mean(cv_scores)))
            mlflow.log_metric("cv_f1_std", float(np.std(cv_scores)))

            # Log model artifact
            if self.config.mlflow.log_models:
                model_path = f"artifacts/model_{run.info.run_id[:8]}"
                self.model.save(model_path)
                mlflow.log_artifacts(model_path, artifact_path="model")

            logger.info("Training complete. Metrics: %s", self._metrics)

        return self.model

    def evaluate(
        self,
        data_path: str | Path | None = None,
        df: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """Evaluate the trained model on a held-out dataset.

        Args:
            data_path: Path to evaluation Parquet file.
            df: Pre-loaded evaluation DataFrame.

        Returns:
            Dictionary of evaluation metrics.
        """
        if self.model is None or not self.model.is_fitted:
            raise RuntimeError("Model has not been trained. Call train() first.")

        if df is None:
            if data_path is None:
                raise ValueError("Provide either data_path or df.")
            df = pd.read_parquet(Path(data_path))

        df = self.feature_engineer.transform(df)
        feature_cols = self.feature_selector.selected_features
        X = df[feature_cols]
        y = df["is_fraud"].astype(int)

        return self._evaluate(X, y)

    def _evaluate(
        self, X: pd.DataFrame | np.ndarray, y: np.ndarray | pd.Series
    ) -> dict[str, float]:
        """Compute evaluation metrics."""
        y_arr = np.asarray(y)
        y_scores = self.model.predict_proba(X)
        y_pred = self.model.predict(X)

        metrics = {
            "precision": precision_score(y_arr, y_pred, zero_division=0),
            "recall": recall_score(y_arr, y_pred, zero_division=0),
            "f1": f1_score(y_arr, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_arr, y_scores),
            "avg_precision": average_precision_score(y_arr, y_scores),
        }

        logger.info(
            "Evaluation -- P=%.3f R=%.3f F1=%.3f AUC=%.3f AP=%.3f",
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            metrics["roc_auc"],
            metrics["avg_precision"],
        )
        return metrics

    def _cross_validate(
        self, X: pd.DataFrame, y: pd.Series
    ) -> list[float]:
        """Run stratified k-fold cross-validation and return F1 scores."""
        skf = StratifiedKFold(
            n_splits=self.config.training.cv_folds,
            shuffle=True,
            random_state=self.config.training.random_state,
        )

        fold_scores: list[float] = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            fold_model = FraudEnsemble(
                iso_config=self.config.isolation_forest,
                xgb_config=self.config.xgboost,
                ensemble_config=self.config.ensemble,
            )
            fold_model.fit(X_tr, y_tr)
            y_pred = fold_model.predict(X_val)
            score = f1_score(y_val, y_pred, zero_division=0)
            fold_scores.append(score)

            logger.info("CV fold %d/%d: F1=%.3f", fold_idx + 1, self.config.training.cv_folds, score)

        logger.info(
            "CV F1: mean=%.3f, std=%.3f",
            np.mean(fold_scores),
            np.std(fold_scores),
        )
        return fold_scores

    @property
    def metrics(self) -> dict[str, float]:
        """Most recent evaluation metrics."""
        return self._metrics
