"""Tests for the rule-based fraud screening engine."""

from __future__ import annotations

import pandas as pd
import pytest

from fraud_detector.config import RuleEngineConfig
from fraud_detector.models.rule_engine import RuleEngine, ScreeningResult


@pytest.fixture
def commodity_stats() -> pd.DataFrame:
    """Minimal commodity statistics for testing."""
    return pd.DataFrame(
        {
            "hs_code": ["8471.30", "6109.10", "2402.20"],
            "median_value_usd": [5000.0, 200.0, 800.0],
            "std_value_usd": [2000.0, 100.0, 300.0],
            "median_weight_kg": [2.5, 0.2, 0.5],
            "std_weight_kg": [1.0, 0.1, 0.2],
        }
    )


@pytest.fixture
def config() -> RuleEngineConfig:
    return RuleEngineConfig(
        price_deviation_threshold=0.30,
        weight_anomaly_sigma=3.0,
        high_risk_origins=["XX", "YY"],
        high_risk_hs_chapters=["24", "93"],
        min_declared_value_usd=10.0,
        max_price_per_kg_usd=50_000.0,
    )


@pytest.fixture
def engine(config: RuleEngineConfig, commodity_stats: pd.DataFrame) -> RuleEngine:
    return RuleEngine(config=config, commodity_stats=commodity_stats)


class TestRuleEngine:
    """Tests for individual rule logic and batch screening."""

    def test_legitimate_declaration_not_flagged(self, engine: RuleEngine) -> None:
        """A normal declaration should not trigger any rules."""
        decl = {
            "declaration_id": "DCL-001",
            "hs_code": "8471.30",
            "declared_value_usd": 4500.0,
            "weight_kg": 3.0,
            "quantity": 10,
            "origin_country": "DE",
            "destination_country": "US",
        }
        result = engine.screen(decl)
        assert not result.is_flagged
        assert result.rule_score == 0.0

    def test_price_deviation_flags_undervalued(self, engine: RuleEngine) -> None:
        """A declaration far below median value should be flagged."""
        decl = {
            "declaration_id": "DCL-002",
            "hs_code": "8471.30",
            "declared_value_usd": 100.0,  # median is 5000
            "weight_kg": 2.0,
            "quantity": 10,
            "origin_country": "DE",
            "destination_country": "US",
        }
        result = engine.screen(decl)
        assert result.is_flagged
        assert "price_deviation" in result.fired_rules

    def test_weight_anomaly_flags_outlier(self, engine: RuleEngine) -> None:
        """An extreme weight deviation should fire the weight anomaly rule."""
        decl = {
            "declaration_id": "DCL-003",
            "hs_code": "6109.10",
            "declared_value_usd": 200.0,
            "weight_kg": 50.0,  # way above median of 0.2 per unit
            "quantity": 1,
            "origin_country": "DE",
            "destination_country": "US",
        }
        result = engine.screen(decl)
        assert "weight_anomaly" in result.fired_rules

    def test_origin_risk_flags_high_risk_country(self, engine: RuleEngine) -> None:
        """A high-risk origin country should be flagged."""
        decl = {
            "declaration_id": "DCL-004",
            "hs_code": "8471.30",
            "declared_value_usd": 5000.0,
            "weight_kg": 2.5,
            "quantity": 10,
            "origin_country": "XX",
            "destination_country": "US",
        }
        result = engine.screen(decl)
        assert "origin_risk" in result.fired_rules

    def test_hs_code_risk_flags_tobacco(self, engine: RuleEngine) -> None:
        """Tobacco HS chapter (24) should fire the HS code risk rule."""
        decl = {
            "declaration_id": "DCL-005",
            "hs_code": "2402.20",
            "declared_value_usd": 800.0,
            "weight_kg": 0.5,
            "quantity": 100,
            "origin_country": "DE",
            "destination_country": "US",
        }
        result = engine.screen(decl)
        assert "hs_code_risk" in result.fired_rules

    def test_minimum_value_flags_near_zero(self, engine: RuleEngine) -> None:
        """A declared value near zero should be flagged."""
        decl = {
            "declaration_id": "DCL-006",
            "hs_code": "8471.30",
            "declared_value_usd": 5.0,
            "weight_kg": 2.0,
            "quantity": 10,
            "origin_country": "DE",
            "destination_country": "US",
        }
        result = engine.screen(decl)
        assert "minimum_value" in result.fired_rules

    def test_rule_score_capped_at_one(self, engine: RuleEngine) -> None:
        """Aggregated rule score should never exceed 1.0."""
        decl = {
            "declaration_id": "DCL-007",
            "hs_code": "2402.20",
            "declared_value_usd": 5.0,  # triggers minimum_value + price_deviation
            "weight_kg": 0.0001,
            "quantity": 1,
            "origin_country": "XX",  # triggers origin_risk
            "destination_country": "US",
        }
        result = engine.screen(decl)
        assert result.rule_score <= 1.0

    def test_screening_result_properties(self, engine: RuleEngine) -> None:
        """Verify ScreeningResult properties work correctly."""
        decl = {
            "declaration_id": "DCL-008",
            "hs_code": "8471.30",
            "declared_value_usd": 5000.0,
            "weight_kg": 2.5,
            "quantity": 10,
            "origin_country": "DE",
            "destination_country": "US",
        }
        result = engine.screen(decl)
        assert isinstance(result, ScreeningResult)
        assert result.declaration_id == "DCL-008"
        assert isinstance(result.fired_rules, list)

    def test_batch_screening(self, engine: RuleEngine) -> None:
        """Batch screening should add expected columns."""
        df = pd.DataFrame(
            [
                {
                    "declaration_id": "DCL-010",
                    "hs_code": "8471.30",
                    "declared_value_usd": 5000.0,
                    "weight_kg": 2.5,
                    "quantity": 10,
                    "origin_country": "DE",
                    "destination_country": "US",
                },
                {
                    "declaration_id": "DCL-011",
                    "hs_code": "8471.30",
                    "declared_value_usd": 50.0,
                    "weight_kg": 2.5,
                    "quantity": 10,
                    "origin_country": "XX",
                    "destination_country": "US",
                },
            ]
        )
        result_df = engine.screen_batch(df)
        assert "rule_score" in result_df.columns
        assert "rule_flags" in result_df.columns
        assert "is_rule_flagged" in result_df.columns
        assert result_df.iloc[1]["is_rule_flagged"] is True

    def test_engine_without_commodity_stats(self, config: RuleEngineConfig) -> None:
        """Engine should still work when no commodity stats are provided."""
        engine = RuleEngine(config=config, commodity_stats=None)
        decl = {
            "declaration_id": "DCL-012",
            "hs_code": "8471.30",
            "declared_value_usd": 100.0,
            "weight_kg": 2.0,
            "quantity": 10,
            "origin_country": "XX",
            "destination_country": "US",
        }
        result = engine.screen(decl)
        # Should still detect origin risk even without commodity stats
        assert "origin_risk" in result.fired_rules
