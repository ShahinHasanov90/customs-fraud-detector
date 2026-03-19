# Customs Fraud Detector

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-blue)](https://mlflow.org/)

ML-powered fraud detection system for customs trade declarations. Combines rule-based screening with ensemble machine learning models to flag suspicious shipments in real time.

---

## Overview

International customs fraud -- including undervaluation, misclassification, and origin misrepresentation -- costs governments billions in lost revenue each year. This system ingests customs declaration data and produces a fraud risk score (0.0 - 1.0) along with human-readable explanations of the factors driving each score.

The pipeline is designed to operate both as a batch scoring job (for historical audits) and as a low-latency REST API (for real-time gate checks).

## Architecture

```
                        +------------------+
                        |  Customs Data    |
                        |  (declarations)  |
                        +--------+---------+
                                 |
                                 v
                     +-----------+-----------+
                     |  Feature Engineering  |
                     |  - price ratios       |
                     |  - weight anomalies   |
                     |  - partner risk       |
                     |  - historical stats   |
                     +-----------+-----------+
                                 |
                    +------------+------------+
                    |                         |
                    v                         v
          +-----------------+       +-----------------+
          |  Rule Engine    |       |  ML Ensemble    |
          |  (deterministic |       |  - Isolation    |
          |   screening)    |       |    Forest       |
          |                 |       |  - XGBoost      |
          +---------+-------+       +--------+--------+
                    |                         |
                    +------------+------------+
                                 |
                                 v
                     +-----------+-----------+
                     |   Score Aggregation   |
                     |   & SHAP Explanation  |
                     +-----------+-----------+
                                 |
                                 v
                     +-----------+-----------+
                     |   FastAPI Service     |
                     |   /predict endpoint   |
                     +-----------------------+
```

## Model Pipeline

1. **Data Ingestion** -- Raw customs declarations are loaded and validated against a Pydantic schema.
2. **Feature Engineering** -- Derived features including price-per-kg ratios, historical deviation z-scores, trade partner risk indices, and HS-code risk tiers.
3. **Rule Engine** -- A deterministic screening layer that fires on known fraud patterns (e.g., declared value < 30% of commodity median, weight-to-volume mismatch > 3 sigma).
4. **ML Ensemble** -- An Isolation Forest detects distributional anomalies; an XGBoost classifier captures complex non-linear fraud patterns. Scores are combined via weighted averaging.
5. **Explainability** -- SHAP values provide feature-level attribution for every prediction.
6. **Serving** -- Predictions are served through a FastAPI REST endpoint with sub-100ms p95 latency.

## Features

- **Rule Engine**: Configurable rule-based screening with support for price deviation, weight anomaly, origin-destination mismatch, and HS code risk thresholds.
- **Isolation Forest**: Unsupervised anomaly detection that adapts to distribution shifts without labeled data.
- **XGBoost Classifier**: Gradient-boosted ensemble trained on historical fraud labels for high-precision detection.
- **Feature Engineering**: Automated derivation of 40+ features from raw declaration fields.
- **SHAP Explanations**: Every prediction is accompanied by feature-importance attributions for audit transparency.
- **MLflow Integration**: Full experiment tracking -- metrics, parameters, and model artifacts are versioned automatically.
- **FastAPI Serving**: Production-ready REST API with request validation, health checks, and structured logging.

## Model Performance

Evaluated on a held-out test set of 48,000 declarations (fraud prevalence ~2.3%):

| Metric        | Value |
|---------------|-------|
| Precision     | 0.89  |
| Recall        | 0.84  |
| F1 Score      | 0.86  |
| AUC-ROC       | 0.94  |
| AUC-PR        | 0.81  |
| Latency (p95) | 42 ms |

## Quick Start

```bash
# Clone the repository
git clone https://github.com/ShahinHasanov90/customs-fraud-detector.git
cd customs-fraud-detector

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .

# Run the API server
uvicorn fraud_detector.api.app:app --host 0.0.0.0 --port 8000 --reload
```

## Usage

### Score a single declaration via the API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "declaration_id": "DCL-2024-00451",
    "hs_code": "8471.30",
    "declared_value_usd": 1200.00,
    "weight_kg": 15.5,
    "quantity": 50,
    "origin_country": "CN",
    "destination_country": "US",
    "importer_id": "IMP-9382",
    "transport_mode": "sea"
  }'
```

### Response

```json
{
  "declaration_id": "DCL-2024-00451",
  "fraud_score": 0.73,
  "is_flagged": true,
  "risk_level": "high",
  "rule_flags": ["price_deviation"],
  "top_features": [
    {"feature": "price_per_kg_ratio", "shap_value": 0.28},
    {"feature": "origin_risk_score", "shap_value": 0.15},
    {"feature": "historical_value_zscore", "shap_value": 0.11}
  ]
}
```

### Train the model

```python
from fraud_detector.training.trainer import FraudModelTrainer

trainer = FraudModelTrainer(config_path="config/model_config.yaml")
trainer.train(data_path="data/declarations_train.parquet")
trainer.evaluate(data_path="data/declarations_test.parquet")
```

### Batch scoring

```python
from fraud_detector.models.ensemble import FraudEnsemble

model = FraudEnsemble.load("artifacts/model_v2.1")
predictions = model.predict(df_new_declarations)
```

## Project Structure

```
customs-fraud-detector/
├── config/
│   └── model_config.yaml
├── notebooks/
│   └── 01_exploration.ipynb
├── src/
│   └── fraud_detector/
│       ├── api/
│       │   ├── app.py
│       │   └── schemas.py
│       ├── explain/
│       │   └── shap_explainer.py
│       ├── features/
│       │   ├── engineering.py
│       │   └── selection.py
│       ├── models/
│       │   ├── ensemble.py
│       │   └── rule_engine.py
│       ├── training/
│       │   └── trainer.py
│       └── config.py
├── tests/
│   ├── test_ensemble.py
│   └── test_rule_engine.py
├── setup.py
├── requirements.txt
└── README.md
```

## Development

```bash
# Run tests
pytest tests/ -v

# Lint
flake8 src/ tests/
black --check src/ tests/

# Type checking
mypy src/fraud_detector/
```

## License

MIT License. See [LICENSE](LICENSE) for details.
