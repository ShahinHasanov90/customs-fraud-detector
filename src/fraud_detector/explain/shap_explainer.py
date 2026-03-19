"""SHAP-based model explanations for fraud predictions.

Provides feature-level attributions for individual predictions, enabling
audit transparency and human review of flagged declarations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import shap

from fraud_detector.models.ensemble import FraudEnsemble

logger = logging.getLogger(__name__)


@dataclass
class FeatureAttribution:
    """A single feature's contribution to the fraud score."""

    feature: str
    shap_value: float
    feature_value: float | str


@dataclass
class Explanation:
    """Full explanation for one declaration's fraud score."""

    declaration_id: str
    fraud_score: float
    base_value: float
    attributions: list[FeatureAttribution]

    @property
    def top_features(self, n: int = 5) -> list[FeatureAttribution]:
        """Return the top-N features by absolute SHAP value."""
        sorted_attrs = sorted(
            self.attributions, key=lambda a: abs(a.shap_value), reverse=True
        )
        return sorted_attrs[:n]

    def to_dict(self, top_n: int = 5) -> dict[str, Any]:
        """Serialize the explanation to a JSON-friendly dictionary."""
        sorted_attrs = sorted(
            self.attributions, key=lambda a: abs(a.shap_value), reverse=True
        )
        return {
            "declaration_id": self.declaration_id,
            "fraud_score": round(self.fraud_score, 4),
            "base_value": round(self.base_value, 4),
            "top_features": [
                {
                    "feature": a.feature,
                    "shap_value": round(a.shap_value, 4),
                    "feature_value": a.feature_value,
                }
                for a in sorted_attrs[:top_n]
            ],
        }


class FraudExplainer:
    """Generate SHAP explanations for fraud ensemble predictions.

    Uses ``shap.TreeExplainer`` on the XGBoost component of the ensemble
    to produce per-feature attributions.

    Args:
        model: A fitted ``FraudEnsemble`` instance.
        background_data: A sample of training data used as the SHAP
            reference distribution. 100-500 rows is usually sufficient.
    """

    def __init__(
        self,
        model: FraudEnsemble,
        background_data: pd.DataFrame | np.ndarray | None = None,
    ) -> None:
        if not model.is_fitted:
            raise ValueError("Model must be fitted before creating explainer.")

        self.model = model
        self.feature_names = model.feature_names

        # TreeExplainer is deterministic and fast for gradient-boosted models
        self._explainer = shap.TreeExplainer(
            model.xgb_classifier,
            data=background_data,
            feature_perturbation="interventional"
            if background_data is not None
            else "tree_path_dependent",
        )
        logger.info("SHAP TreeExplainer initialized.")

    def explain(
        self,
        X: pd.DataFrame | np.ndarray,
        declaration_ids: list[str] | None = None,
    ) -> list[Explanation]:
        """Compute SHAP explanations for a set of declarations.

        Args:
            X: Feature matrix (same schema as training data).
            declaration_ids: Optional list of declaration identifiers.

        Returns:
            List of ``Explanation`` objects, one per row in X.
        """
        X_arr = np.asarray(X, dtype=np.float32)
        shap_values = self._explainer.shap_values(X_arr)
        base_value = float(self._explainer.expected_value)

        # Compute fraud scores from the ensemble for context
        fraud_scores = self.model.predict_proba(X)

        if declaration_ids is None:
            declaration_ids = [f"row_{i}" for i in range(len(X_arr))]

        feature_names = (
            list(X.columns) if isinstance(X, pd.DataFrame) else self.feature_names
        )

        explanations: list[Explanation] = []
        for i in range(len(X_arr)):
            attributions = [
                FeatureAttribution(
                    feature=feature_names[j],
                    shap_value=float(shap_values[i, j]),
                    feature_value=float(X_arr[i, j]),
                )
                for j in range(len(feature_names))
            ]

            explanations.append(
                Explanation(
                    declaration_id=declaration_ids[i],
                    fraud_score=float(fraud_scores[i]),
                    base_value=base_value,
                    attributions=attributions,
                )
            )

        logger.info("Generated SHAP explanations for %d declarations.", len(X_arr))
        return explanations

    def explain_single(
        self,
        X_row: pd.DataFrame | np.ndarray,
        declaration_id: str = "unknown",
    ) -> Explanation:
        """Explain a single declaration.

        Args:
            X_row: Single-row feature matrix.
            declaration_id: Identifier for the declaration.

        Returns:
            An ``Explanation`` for the given declaration.
        """
        if isinstance(X_row, pd.DataFrame):
            X_row = X_row.iloc[:1]
        else:
            X_row = np.atleast_2d(X_row)

        results = self.explain(X_row, declaration_ids=[declaration_id])
        return results[0]

    def feature_importance_summary(
        self, X: pd.DataFrame | np.ndarray
    ) -> pd.DataFrame:
        """Compute mean absolute SHAP values across the dataset.

        Args:
            X: Feature matrix.

        Returns:
            DataFrame with columns ``feature`` and ``mean_abs_shap``,
            sorted descending.
        """
        X_arr = np.asarray(X, dtype=np.float32)
        shap_values = self._explainer.shap_values(X_arr)
        mean_abs = np.abs(shap_values).mean(axis=0)

        feature_names = (
            list(X.columns) if isinstance(X, pd.DataFrame) else self.feature_names
        )

        summary = pd.DataFrame(
            {"feature": feature_names, "mean_abs_shap": mean_abs}
        ).sort_values("mean_abs_shap", ascending=False)

        return summary.reset_index(drop=True)
