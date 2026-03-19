"""Rule-based fraud screening engine.

Applies deterministic rules to customs declarations to flag obvious fraud
patterns before ML scoring. Each rule returns a flag name and a severity
weight that feeds into the final risk score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from fraud_detector.config import RuleEngineConfig

logger = logging.getLogger(__name__)


@dataclass
class RuleResult:
    """Result of applying a single rule to a declaration."""

    rule_name: str
    fired: bool
    severity: float  # 0.0 - 1.0
    detail: str = ""


@dataclass
class ScreeningResult:
    """Aggregated result of all rules applied to a declaration."""

    declaration_id: str
    flags: list[RuleResult] = field(default_factory=list)
    rule_score: float = 0.0

    @property
    def is_flagged(self) -> bool:
        return any(r.fired for r in self.flags)

    @property
    def fired_rules(self) -> list[str]:
        return [r.rule_name for r in self.flags if r.fired]


class RuleEngine:
    """Deterministic rule-based screening for customs declarations.

    The rule engine evaluates each declaration against a set of configurable
    rules. It is designed to catch straightforward fraud patterns that do not
    require ML -- e.g., declared value far below commodity median, impossible
    weight-to-volume ratios, or known high-risk origin/HS-code combinations.

    Args:
        config: Rule engine configuration parameters.
        commodity_stats: DataFrame with columns ``hs_code``, ``median_value_usd``,
            ``std_value_usd``, ``median_weight_kg``, ``std_weight_kg``.
    """

    def __init__(
        self,
        config: RuleEngineConfig | None = None,
        commodity_stats: pd.DataFrame | None = None,
    ) -> None:
        self.config = config or RuleEngineConfig()
        self.commodity_stats = commodity_stats
        self._rules = [
            self._check_price_deviation,
            self._check_weight_anomaly,
            self._check_origin_risk,
            self._check_hs_code_risk,
            self._check_minimum_value,
            self._check_price_per_kg_ceiling,
        ]

    def screen(self, declaration: dict[str, Any]) -> ScreeningResult:
        """Screen a single declaration against all rules.

        Args:
            declaration: Dictionary containing declaration fields.

        Returns:
            ScreeningResult with flags and an aggregated rule score.
        """
        declaration_id = declaration.get("declaration_id", "unknown")
        results: list[RuleResult] = []

        for rule_fn in self._rules:
            try:
                result = rule_fn(declaration)
                results.append(result)
            except Exception:
                logger.warning(
                    "Rule %s failed for declaration %s",
                    rule_fn.__name__,
                    declaration_id,
                    exc_info=True,
                )

        fired = [r for r in results if r.fired]
        rule_score = min(sum(r.severity for r in fired), 1.0) if fired else 0.0

        return ScreeningResult(
            declaration_id=declaration_id,
            flags=results,
            rule_score=rule_score,
        )

    def screen_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Screen a batch of declarations.

        Args:
            df: DataFrame where each row is a declaration.

        Returns:
            DataFrame with added columns: ``rule_score``, ``rule_flags``,
            ``is_rule_flagged``.
        """
        results = df.apply(
            lambda row: self.screen(row.to_dict()), axis=1
        )
        df = df.copy()
        df["rule_score"] = [r.rule_score for r in results]
        df["rule_flags"] = [r.fired_rules for r in results]
        df["is_rule_flagged"] = [r.is_flagged for r in results]
        return df

    # ------------------------------------------------------------------
    # Individual rules
    # ------------------------------------------------------------------

    def _check_price_deviation(self, decl: dict[str, Any]) -> RuleResult:
        """Flag if declared value deviates significantly from commodity median."""
        if self.commodity_stats is None:
            return RuleResult("price_deviation", False, 0.0, "no stats")

        hs_code = decl.get("hs_code", "")
        declared_value = decl.get("declared_value_usd", 0.0)

        stats_row = self.commodity_stats[
            self.commodity_stats["hs_code"] == hs_code
        ]
        if stats_row.empty:
            return RuleResult("price_deviation", False, 0.0, "hs_code not found")

        median_val = float(stats_row.iloc[0]["median_value_usd"])
        if median_val <= 0:
            return RuleResult("price_deviation", False, 0.0, "invalid median")

        ratio = declared_value / median_val
        threshold = self.config.price_deviation_threshold
        fired = ratio < threshold
        severity = min((1.0 - ratio) * 0.8, 0.6) if fired else 0.0

        return RuleResult(
            rule_name="price_deviation",
            fired=fired,
            severity=severity,
            detail=f"ratio={ratio:.2f}, threshold={threshold}",
        )

    def _check_weight_anomaly(self, decl: dict[str, Any]) -> RuleResult:
        """Flag if weight deviates more than N sigma from commodity norm."""
        if self.commodity_stats is None:
            return RuleResult("weight_anomaly", False, 0.0, "no stats")

        hs_code = decl.get("hs_code", "")
        weight_kg = decl.get("weight_kg", 0.0)
        quantity = max(decl.get("quantity", 1), 1)
        weight_per_unit = weight_kg / quantity

        stats_row = self.commodity_stats[
            self.commodity_stats["hs_code"] == hs_code
        ]
        if stats_row.empty:
            return RuleResult("weight_anomaly", False, 0.0, "hs_code not found")

        median_w = float(stats_row.iloc[0]["median_weight_kg"])
        std_w = float(stats_row.iloc[0]["std_weight_kg"])

        if std_w <= 0:
            return RuleResult("weight_anomaly", False, 0.0, "invalid std")

        z_score = abs(weight_per_unit - median_w) / std_w
        fired = z_score > self.config.weight_anomaly_sigma
        severity = min(z_score / 10.0, 0.5) if fired else 0.0

        return RuleResult(
            rule_name="weight_anomaly",
            fired=fired,
            severity=severity,
            detail=f"z={z_score:.2f}, sigma_threshold={self.config.weight_anomaly_sigma}",
        )

    def _check_origin_risk(self, decl: dict[str, Any]) -> RuleResult:
        """Flag if origin country is in the high-risk list."""
        origin = decl.get("origin_country", "")
        fired = origin in self.config.high_risk_origins
        severity = 0.3 if fired else 0.0

        return RuleResult(
            rule_name="origin_risk",
            fired=fired,
            severity=severity,
            detail=f"origin={origin}",
        )

    def _check_hs_code_risk(self, decl: dict[str, Any]) -> RuleResult:
        """Flag if the HS code chapter is in the high-risk category."""
        hs_code = str(decl.get("hs_code", ""))
        chapter = hs_code.split(".")[0][:2] if hs_code else ""
        fired = chapter in self.config.high_risk_hs_chapters
        severity = 0.25 if fired else 0.0

        return RuleResult(
            rule_name="hs_code_risk",
            fired=fired,
            severity=severity,
            detail=f"chapter={chapter}",
        )

    def _check_minimum_value(self, decl: dict[str, Any]) -> RuleResult:
        """Flag declarations below minimum plausible value."""
        declared_value = decl.get("declared_value_usd", 0.0)
        fired = 0 < declared_value < self.config.min_declared_value_usd
        severity = 0.4 if fired else 0.0

        return RuleResult(
            rule_name="minimum_value",
            fired=fired,
            severity=severity,
            detail=f"value={declared_value}",
        )

    def _check_price_per_kg_ceiling(self, decl: dict[str, Any]) -> RuleResult:
        """Flag if price per kg exceeds the maximum plausible ceiling."""
        declared_value = decl.get("declared_value_usd", 0.0)
        weight_kg = decl.get("weight_kg", 0.0)

        if weight_kg <= 0:
            return RuleResult("price_per_kg_ceiling", False, 0.0, "no weight")

        price_per_kg = declared_value / weight_kg
        fired = price_per_kg > self.config.max_price_per_kg_usd
        severity = 0.2 if fired else 0.0

        return RuleResult(
            rule_name="price_per_kg_ceiling",
            fired=fired,
            severity=severity,
            detail=f"price_per_kg={price_per_kg:.2f}",
        )
