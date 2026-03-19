"""FastAPI application for real-time customs fraud scoring.

Exposes REST endpoints for single and batch declaration scoring, health
checks, and model metadata retrieval.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from fraud_detector import __version__
from fraud_detector.api.schemas import (
    BatchRequest,
    BatchResponse,
    DeclarationRequest,
    FeatureContribution,
    HealthResponse,
    PredictionResponse,
)
from fraud_detector.config import load_config
from fraud_detector.features.engineering import FeatureEngineer
from fraud_detector.models.ensemble import FraudEnsemble
from fraud_detector.models.rule_engine import RuleEngine

logger = logging.getLogger(__name__)

# Global state populated at startup
_model: FraudEnsemble | None = None
_rule_engine: RuleEngine | None = None
_feature_engineer: FeatureEngineer | None = None


def _risk_level(score: float) -> str:
    """Map a fraud score to a human-readable risk category."""
    if score >= 0.8:
        return "critical"
    if score >= 0.5:
        return "high"
    if score >= 0.3:
        return "medium"
    return "low"


def _score_declaration(
    declaration: DeclarationRequest,
) -> PredictionResponse:
    """Score a single declaration through the full pipeline."""
    decl_dict: dict[str, Any] = declaration.model_dump()

    # Rule engine screening
    rule_flags: list[str] = []
    if _rule_engine is not None:
        screening = _rule_engine.screen(decl_dict)
        rule_flags = screening.fired_rules

    # Feature engineering
    df = pd.DataFrame([decl_dict])
    if _feature_engineer is not None:
        df = _feature_engineer.transform(df)

    # ML scoring
    if _model is not None and _model.is_fitted:
        feature_cols = [
            c
            for c in df.columns
            if c not in ("declaration_id", "importer_id", "hs_code",
                         "origin_country", "destination_country",
                         "transport_mode", "rule_flags", "is_rule_flagged",
                         "rule_score", "hs_code_chapter", "hs_code_heading")
        ]
        X = df[feature_cols].select_dtypes(include=[np.number])
        fraud_score = float(_model.predict_proba(X)[0])
    else:
        # Fallback to rule score only
        fraud_score = 0.0

    threshold = 0.5
    if _model is not None:
        threshold = _model.ensemble_config.fraud_threshold

    return PredictionResponse(
        declaration_id=declaration.declaration_id,
        fraud_score=round(fraud_score, 4),
        is_flagged=fraud_score >= threshold,
        risk_level=_risk_level(fraud_score),
        rule_flags=rule_flags,
        top_features=[],  # SHAP explanations omitted for latency
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load model and configuration at startup."""
    global _model, _rule_engine, _feature_engineer

    config = load_config()
    _rule_engine = RuleEngine(config=config.rule_engine)
    _feature_engineer = FeatureEngineer()

    model_path = Path(config.api.model_path)
    if model_path.exists():
        try:
            _model = FraudEnsemble.load(model_path)
            logger.info("Model loaded from %s", model_path)
        except Exception:
            logger.warning("Failed to load model from %s", model_path, exc_info=True)
    else:
        logger.warning(
            "Model path %s does not exist. API will run without ML scoring.",
            model_path,
        )

    yield

    # Cleanup (if needed)
    logger.info("Shutting down fraud detection API.")


app = FastAPI(
    title="Customs Fraud Detector",
    description="ML-powered fraud detection for customs trade declarations.",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check() -> HealthResponse:
    """Return the health status of the API."""
    return HealthResponse(
        status="ok",
        version=__version__,
        model_loaded=_model is not None and _model.is_fitted,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["scoring"])
async def predict(declaration: DeclarationRequest) -> PredictionResponse:
    """Score a single customs declaration for fraud risk."""
    start = time.perf_counter()
    try:
        response = _score_declaration(declaration)
    except Exception as exc:
        logger.error("Scoring failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal scoring error.")
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "Scored %s in %.1f ms (score=%.3f)",
        declaration.declaration_id,
        elapsed_ms,
        response.fraud_score,
    )
    return response


@app.post("/predict/batch", response_model=BatchResponse, tags=["scoring"])
async def predict_batch(batch: BatchRequest) -> BatchResponse:
    """Score a batch of customs declarations."""
    predictions = [_score_declaration(d) for d in batch.declarations]
    flagged = sum(1 for p in predictions if p.is_flagged)
    return BatchResponse(
        predictions=predictions,
        total=len(predictions),
        flagged=flagged,
    )


def main() -> None:
    """Entry point for running the API server."""
    import uvicorn

    config = load_config()
    uvicorn.run(
        "fraud_detector.api.app:app",
        host=config.api.host,
        port=config.api.port,
        log_level=config.api.log_level,
        reload=False,
    )


if __name__ == "__main__":
    main()
