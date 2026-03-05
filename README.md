# California Single-Family Home Price Prediction

ML pipeline using 2025 CRMLS data to predict closing prices.

## Key Results
- Best model: Weighted Stacking (LightGBM + XGBoost + Random Forest)
- R² (dollars): 0.8568
- R² (log): 0.9045
- MdAPE (dollars): 8.92%

## Structure
- `01_preprocessing.ipynb`: Data cleaning, imputation, encoding, scaling, feature engineering
- `02_modeling.ipynb`: Model development (OLS, Decision Tree, RF, XGBoost, LightGBM), tuning, evaluation

## Tech Stack
- Python, pandas, scikit-learn, XGBoost, LightGBM, joblib
- Feature engineering: ZIP-prefix grouping, target encoding, distance to coast
- Target: log-transformed ClosePrice

## How to run
1. Run `01_preprocessing.ipynb` → generates cleaned CSVs
2. Run `02_modeling.ipynb` → trains & evaluates models

See notebooks for full details.
