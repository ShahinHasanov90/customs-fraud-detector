"""Feature selection utilities for the fraud detection pipeline.

Provides methods to identify and remove low-signal features using
variance thresholds, correlation analysis, and model-based importance.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

logger = logging.getLogger(__name__)


class FeatureSelector:
    """Select informative features and drop redundant or low-signal columns.

    The selector applies three filtering stages in sequence:

    1. **Variance threshold** -- Drop features with near-zero variance.
    2. **Correlation filter** -- Among highly correlated pairs, keep the
       feature with higher univariate importance.
    3. **Importance filter** -- After model training, drop features whose
       importance falls below a percentile cutoff.

    Args:
        variance_threshold: Minimum variance required to keep a feature.
        correlation_threshold: Maximum Pearson correlation allowed between
            any pair of retained features.
        importance_percentile: Features below this percentile of importance
            scores are dropped (0 - 100).
    """

    def __init__(
        self,
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.95,
        importance_percentile: float = 10.0,
    ) -> None:
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.importance_percentile = importance_percentile

        self._selected_features: list[str] | None = None
        self._variance_selector: VarianceThreshold | None = None
        self._dropped_variance: list[str] = []
        self._dropped_correlation: list[str] = []
        self._dropped_importance: list[str] = []

    @property
    def selected_features(self) -> list[str]:
        """Feature names retained after selection."""
        if self._selected_features is None:
            raise RuntimeError("FeatureSelector has not been fitted.")
        return self._selected_features

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | None = None,
        feature_importances: dict[str, float] | None = None,
    ) -> FeatureSelector:
        """Fit the feature selector on training data.

        Args:
            X: Feature DataFrame.
            y: Target array (unused directly, kept for API consistency).
            feature_importances: Optional pre-computed importance scores
                keyed by feature name.

        Returns:
            Self, for method chaining.
        """
        remaining = list(X.columns)

        # Stage 1: Variance filter
        remaining = self._fit_variance(X[remaining])

        # Stage 2: Correlation filter
        remaining = self._fit_correlation(X[remaining])

        # Stage 3: Importance filter (if importances provided)
        if feature_importances is not None:
            remaining = self._fit_importance(remaining, feature_importances)

        self._selected_features = remaining
        logger.info(
            "Feature selection: %d -> %d features retained "
            "(dropped %d variance, %d correlation, %d importance)",
            len(X.columns),
            len(remaining),
            len(self._dropped_variance),
            len(self._dropped_correlation),
            len(self._dropped_importance),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted selection to new data.

        Args:
            X: DataFrame with at least the columns seen during ``fit()``.

        Returns:
            Filtered DataFrame containing only selected features.
        """
        if self._selected_features is None:
            raise RuntimeError("FeatureSelector has not been fitted.")
        return X[self._selected_features].copy()

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | None = None,
        feature_importances: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """Fit and transform in a single call."""
        self.fit(X, y=y, feature_importances=feature_importances)
        return self.transform(X)

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _fit_variance(self, X: pd.DataFrame) -> list[str]:
        """Drop features with variance below the threshold."""
        self._variance_selector = VarianceThreshold(
            threshold=self.variance_threshold
        )
        self._variance_selector.fit(X)
        mask = self._variance_selector.get_support()
        kept = [col for col, keep in zip(X.columns, mask) if keep]
        self._dropped_variance = [
            col for col, keep in zip(X.columns, mask) if not keep
        ]
        return kept

    def _fit_correlation(self, X: pd.DataFrame) -> list[str]:
        """Among highly correlated pairs, drop one feature."""
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
        )

        to_drop: set[str] = set()
        for col in upper.columns:
            correlated = upper.index[
                upper[col] > self.correlation_threshold
            ].tolist()
            if correlated:
                to_drop.add(col)

        self._dropped_correlation = sorted(to_drop)
        return [c for c in X.columns if c not in to_drop]

    def _fit_importance(
        self,
        features: list[str],
        importances: dict[str, float],
    ) -> list[str]:
        """Drop features below an importance percentile."""
        scores = [importances.get(f, 0.0) for f in features]
        cutoff = np.percentile(scores, self.importance_percentile)
        kept = [f for f, s in zip(features, scores) if s >= cutoff]
        self._dropped_importance = [
            f for f, s in zip(features, scores) if s < cutoff
        ]
        return kept

    def get_report(self) -> dict[str, Any]:
        """Return a summary of the feature selection process."""
        return {
            "features_retained": len(self._selected_features or []),
            "dropped_variance": self._dropped_variance,
            "dropped_correlation": self._dropped_correlation,
            "dropped_importance": self._dropped_importance,
        }
