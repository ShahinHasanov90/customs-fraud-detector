"""Configuration management for the fraud detection system."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RuleEngineConfig:
    """Configuration for the rule-based screening engine."""

    price_deviation_threshold: float = 0.30
    weight_anomaly_sigma: float = 3.0
    high_risk_origins: list[str] = field(default_factory=list)
    high_risk_hs_chapters: list[str] = field(default_factory=list)
    min_declared_value_usd: float = 10.0
    max_price_per_kg_usd: float = 50_000.0


@dataclass
class IsolationForestConfig:
    """Configuration for the Isolation Forest model."""

    n_estimators: int = 200
    contamination: float = 0.02
    max_samples: str | int = "auto"
    max_features: float = 1.0
    random_state: int = 42
    n_jobs: int = -1


@dataclass
class XGBoostConfig:
    """Configuration for the XGBoost classifier."""

    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 5
    gamma: float = 0.1
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    scale_pos_weight: float = 43.0
    eval_metric: str = "aucpr"
    early_stopping_rounds: int = 30
    random_state: int = 42
    n_jobs: int = -1


@dataclass
class EnsembleConfig:
    """Configuration for the ensemble model."""

    isolation_forest_weight: float = 0.3
    xgboost_weight: float = 0.7
    fraud_threshold: float = 0.5


@dataclass
class TrainingConfig:
    """Configuration for the training pipeline."""

    test_size: float = 0.2
    stratify: bool = True
    cv_folds: int = 5
    random_state: int = 42


@dataclass
class MLflowConfig:
    """Configuration for MLflow experiment tracking."""

    experiment_name: str = "customs-fraud-detector"
    tracking_uri: str = "sqlite:///mlflow.db"
    registry_uri: str = "sqlite:///mlflow.db"
    log_models: bool = True


@dataclass
class APIConfig:
    """Configuration for the FastAPI serving layer."""

    host: str = "0.0.0.0"
    port: int = 8000
    model_path: str = "artifacts/model_latest"
    log_level: str = "info"
    cors_origins: list[str] = field(
        default_factory=lambda: ["http://localhost:3000"]
    )


@dataclass
class AppConfig:
    """Top-level application configuration."""

    rule_engine: RuleEngineConfig = field(default_factory=RuleEngineConfig)
    isolation_forest: IsolationForestConfig = field(
        default_factory=IsolationForestConfig
    )
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    api: APIConfig = field(default_factory=APIConfig)


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load application configuration from a YAML file.

    Args:
        config_path: Path to the YAML config file. If None, uses the
            ``FRAUD_DETECTOR_CONFIG`` environment variable or falls back
            to ``config/model_config.yaml``.

    Returns:
        Populated ``AppConfig`` instance.
    """
    if config_path is None:
        config_path = os.environ.get(
            "FRAUD_DETECTOR_CONFIG", "config/model_config.yaml"
        )

    path = Path(config_path)
    if not path.exists():
        return AppConfig()

    with open(path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    return AppConfig(
        rule_engine=RuleEngineConfig(**raw.get("rule_engine", {})),
        isolation_forest=IsolationForestConfig(
            **raw.get("isolation_forest", {})
        ),
        xgboost=XGBoostConfig(**raw.get("xgboost", {})),
        ensemble=EnsembleConfig(**raw.get("ensemble", {})),
        training=TrainingConfig(**raw.get("training", {})),
        mlflow=MLflowConfig(**raw.get("mlflow", {})),
        api=APIConfig(**raw.get("api", {})),
    )
