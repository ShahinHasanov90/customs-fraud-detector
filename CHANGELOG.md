# Changelog

## [1.2.0] - 2026-02-10
### Added
- SHAP explainability layer for all predictions
- Batch scoring endpoint for historical audit workflows
- MLflow experiment tracking integration

### Changed
- Upgraded XGBoost ensemble with hyperparameter tuning
- Improved feature engineering: 40+ derived features

### Fixed
- Precision improvement from 0.85 to 0.89 via feature selection

## [1.0.0] - 2025-09-01
### Added
- Initial release: Rule engine + Isolation Forest ensemble
- FastAPI serving endpoint with sub-100ms latency
- Feature engineering pipeline for customs declarations
