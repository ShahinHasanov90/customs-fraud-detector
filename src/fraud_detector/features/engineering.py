"""Feature engineering for customs declaration data.

Derives numerical features from raw declaration fields, including price
ratios, statistical deviations from historical norms, and trade-partner
risk scores.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default risk scores by country (illustrative; in production these would
# come from an external reference table updated quarterly).
_DEFAULT_COUNTRY_RISK: dict[str, float] = {
    "CN": 0.35,
    "HK": 0.30,
    "VN": 0.25,
    "TH": 0.20,
    "MY": 0.18,
    "IN": 0.22,
    "PK": 0.28,
    "TR": 0.20,
    "AE": 0.24,
    "NG": 0.32,
}

_DEFAULT_RISK_BASELINE: float = 0.10


class FeatureEngineer:
    """Transforms raw customs declarations into ML-ready feature vectors.

    This class encapsulates all feature derivation logic so that the same
    transformations can be applied consistently during training and inference.

    Args:
        commodity_stats: DataFrame with per-HS-code aggregated statistics.
            Expected columns: ``hs_code``, ``median_value_usd``,
            ``std_value_usd``, ``median_weight_kg``, ``std_weight_kg``,
            ``median_price_per_kg``.
        importer_stats: DataFrame with per-importer historical statistics.
            Expected columns: ``importer_id``, ``fraud_rate_30d``,
            ``declaration_count_90d``, ``avg_value_usd``.
        country_risk: Optional mapping of ISO-2 country codes to risk scores.
    """

    def __init__(
        self,
        commodity_stats: pd.DataFrame | None = None,
        importer_stats: pd.DataFrame | None = None,
        country_risk: dict[str, float] | None = None,
    ) -> None:
        self.commodity_stats = commodity_stats
        self.importer_stats = importer_stats
        self.country_risk = country_risk or _DEFAULT_COUNTRY_RISK

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps to a declarations DataFrame.

        Args:
            df: Raw declarations DataFrame.

        Returns:
            DataFrame with original columns plus all derived features.
        """
        df = df.copy()
        df = self._add_basic_ratios(df)
        df = self._add_log_transforms(df)
        df = self._add_commodity_deviations(df)
        df = self._add_trade_partner_risk(df)
        df = self._add_importer_features(df)
        df = self._add_hs_code_features(df)

        logger.info(
            "Feature engineering complete: %d rows, %d columns",
            len(df),
            len(df.columns),
        )
        return df

    # ------------------------------------------------------------------
    # Feature groups
    # ------------------------------------------------------------------

    @staticmethod
    def _add_basic_ratios(df: pd.DataFrame) -> pd.DataFrame:
        """Derive price-per-kg, price-per-unit, and weight-per-unit."""
        weight = df["weight_kg"].replace(0, np.nan)
        quantity = df["quantity"].replace(0, np.nan)

        df["price_per_kg"] = df["declared_value_usd"] / weight
        df["price_per_unit"] = df["declared_value_usd"] / quantity
        df["weight_per_unit"] = df["weight_kg"] / quantity

        # Fill NaN from zero-division with column median (safe default)
        for col in ["price_per_kg", "price_per_unit", "weight_per_unit"]:
            df[col] = df[col].fillna(df[col].median())

        return df

    @staticmethod
    def _add_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
        """Log-transform skewed monetary and weight columns."""
        df["declared_value_log"] = np.log1p(
            df["declared_value_usd"].clip(lower=0)
        )
        df["weight_log"] = np.log1p(df["weight_kg"].clip(lower=0))
        df["quantity_log"] = np.log1p(df["quantity"].clip(lower=0))
        return df

    def _add_commodity_deviations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute z-scores relative to HS-code commodity statistics."""
        if self.commodity_stats is None:
            df["value_to_median_ratio"] = 1.0
            df["historical_value_zscore"] = 0.0
            df["historical_weight_zscore"] = 0.0
            df["hs_chapter_avg_value_ratio"] = 1.0
            return df

        merged = df.merge(
            self.commodity_stats,
            on="hs_code",
            how="left",
            suffixes=("", "_stat"),
        )

        median_val = merged["median_value_usd_stat"].fillna(
            df["declared_value_usd"].median()
        )
        std_val = merged["std_value_usd"].fillna(1.0)
        median_wt = merged["median_weight_kg_stat"].fillna(
            df["weight_kg"].median()
        )
        std_wt = merged["std_weight_kg"].fillna(1.0)

        df["value_to_median_ratio"] = (
            df["declared_value_usd"] / median_val.replace(0, np.nan)
        ).fillna(1.0)

        df["historical_value_zscore"] = (
            (df["declared_value_usd"] - median_val) / std_val.replace(0, 1.0)
        )

        df["historical_weight_zscore"] = (
            (df["weight_kg"] - median_wt) / std_wt.replace(0, 1.0)
        )

        df["hs_chapter_avg_value_ratio"] = df["value_to_median_ratio"]
        return df

    def _add_trade_partner_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map origin and destination countries to risk scores."""
        df["origin_risk_score"] = (
            df["origin_country"]
            .map(self.country_risk)
            .fillna(_DEFAULT_RISK_BASELINE)
        )
        df["destination_risk_score"] = (
            df["destination_country"]
            .map(self.country_risk)
            .fillna(_DEFAULT_RISK_BASELINE)
        )
        df["trade_partner_risk_score"] = (
            0.7 * df["origin_risk_score"] + 0.3 * df["destination_risk_score"]
        )
        df["route_risk_product"] = (
            df["origin_risk_score"] * df["destination_risk_score"]
        )
        return df

    def _add_importer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Join importer-level historical statistics."""
        if self.importer_stats is None:
            df["importer_fraud_rate_30d"] = 0.0
            df["importer_declaration_count_90d"] = 0
            df["importer_avg_value_ratio"] = 1.0
            return df

        merged = df.merge(
            self.importer_stats,
            on="importer_id",
            how="left",
            suffixes=("", "_imp"),
        )

        df["importer_fraud_rate_30d"] = merged["fraud_rate_30d"].fillna(0.0)
        df["importer_declaration_count_90d"] = (
            merged["declaration_count_90d"].fillna(0).astype(int)
        )

        avg_val = merged["avg_value_usd"].fillna(df["declared_value_usd"].median())
        df["importer_avg_value_ratio"] = (
            df["declared_value_usd"] / avg_val.replace(0, np.nan)
        ).fillna(1.0)

        return df

    @staticmethod
    def _add_hs_code_features(df: pd.DataFrame) -> pd.DataFrame:
        """Extract chapter and heading from HS code string."""
        hs = df["hs_code"].astype(str)
        df["hs_code_chapter"] = hs.str.split(".").str[0].str[:2]
        df["hs_code_heading"] = hs.str.replace(".", "", regex=False).str[:4]
        return df

    def get_feature_columns(self) -> list[str]:
        """Return the list of derived feature column names."""
        return [
            "price_per_kg",
            "price_per_unit",
            "weight_per_unit",
            "declared_value_log",
            "weight_log",
            "quantity_log",
            "value_to_median_ratio",
            "historical_value_zscore",
            "historical_weight_zscore",
            "hs_chapter_avg_value_ratio",
            "origin_risk_score",
            "destination_risk_score",
            "trade_partner_risk_score",
            "route_risk_product",
            "importer_fraud_rate_30d",
            "importer_declaration_count_90d",
            "importer_avg_value_ratio",
        ]
