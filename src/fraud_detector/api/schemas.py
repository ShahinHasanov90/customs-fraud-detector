"""Pydantic request and response schemas for the fraud scoring API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DeclarationRequest(BaseModel):
    """Incoming customs declaration to be scored for fraud risk."""

    declaration_id: str = Field(
        ..., description="Unique identifier for the customs declaration."
    )
    hs_code: str = Field(
        ..., description="Harmonized System commodity code (e.g., '8471.30')."
    )
    declared_value_usd: float = Field(
        ..., gt=0, description="Declared customs value in USD."
    )
    weight_kg: float = Field(
        ..., gt=0, description="Gross weight of the shipment in kilograms."
    )
    quantity: int = Field(
        ..., gt=0, description="Number of items in the shipment."
    )
    origin_country: str = Field(
        ..., min_length=2, max_length=2, description="ISO 3166-1 alpha-2 origin country code."
    )
    destination_country: str = Field(
        ..., min_length=2, max_length=2, description="ISO 3166-1 alpha-2 destination country code."
    )
    importer_id: str = Field(
        ..., description="Registered importer identifier."
    )
    transport_mode: str = Field(
        default="sea",
        description="Mode of transport: sea, air, road, rail.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "declaration_id": "DCL-2024-00451",
                    "hs_code": "8471.30",
                    "declared_value_usd": 1200.00,
                    "weight_kg": 15.5,
                    "quantity": 50,
                    "origin_country": "CN",
                    "destination_country": "US",
                    "importer_id": "IMP-9382",
                    "transport_mode": "sea",
                }
            ]
        }
    }


class FeatureContribution(BaseModel):
    """A single feature's contribution to the fraud score."""

    feature: str = Field(..., description="Feature name.")
    shap_value: float = Field(..., description="SHAP attribution value.")


class PredictionResponse(BaseModel):
    """Fraud scoring response for a single declaration."""

    declaration_id: str = Field(..., description="Declaration identifier.")
    fraud_score: float = Field(
        ..., ge=0.0, le=1.0, description="Fraud probability score (0-1)."
    )
    is_flagged: bool = Field(
        ..., description="Whether the declaration exceeds the fraud threshold."
    )
    risk_level: str = Field(
        ..., description="Risk category: low, medium, high, critical."
    )
    rule_flags: list[str] = Field(
        default_factory=list,
        description="Names of deterministic rules that fired.",
    )
    top_features: list[FeatureContribution] = Field(
        default_factory=list,
        description="Top contributing features with SHAP values.",
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(default="ok")
    version: str = Field(..., description="Application version.")
    model_loaded: bool = Field(
        ..., description="Whether the ML model is loaded and ready."
    )


class BatchRequest(BaseModel):
    """Batch scoring request containing multiple declarations."""

    declarations: list[DeclarationRequest] = Field(
        ..., min_length=1, max_length=1000,
        description="List of declarations to score.",
    )


class BatchResponse(BaseModel):
    """Batch scoring response."""

    predictions: list[PredictionResponse]
    total: int = Field(..., description="Total number of declarations scored.")
    flagged: int = Field(
        ..., description="Number of declarations flagged as suspicious."
    )
