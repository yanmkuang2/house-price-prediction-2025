# California Single-Family Home Price Prediction

ML pipeline using 2025 California Regional Multiple Listing Servic (CRMLS) data to predict single-family home closing prices.

## Key Results
- Training set: Janurary - November 2025
- Test set: December 2025
- Best model: Weighted Stacking (LightGBM + XGBoost + Random Forest)
- R² (dollars): 0.90
- MdAPE (dollars): 7.40%

## Price Band Analysis
- The model performs the best in the **mid-price range** ($500k–$1M), where almost 50% of the sales occur and variance is moderate
- R² (dollars): 0.40
- MdAPE (dollars): 6.32%

- The model performs the worst in the **High-end** (>$5M), where relative errors increase due to outliers and unique property features
- R² (dollars): -2.41
- MdAPE (dollars): 16.10%

## Structure
- `01_preprocessing.ipynb`: Data cleaning, imputation, encoding, scaling, feature engineering
- `02_modeling.ipynb`: Model development (OLS, Decision Tree, RF, XGBoost, LightGBM), tuning, evaluation
- `03_app.py`: Interactive Streamlit web app for real-time price prediction.
  Users input key property features (Living Area, Bedrooms, Bathrooms, Lot Size, Flooring Type, etc.)
  App loads the best single model (LightGBM), applies the same preprocessing (scaling, target encoding), and returns estimated closing price in dollars.  

## Tech Stack
- Python, pandas, scikit-learn, XGBoost, LightGBM, joblib
- Feature engineering: ZIP-prefix grouping, target encoding, distance to coast
- Model evaluation: R², MAPE, MdAPE
- Interactive demo: Streamlit

## How to run
1. Run `01_preprocessing.ipynb` → generates cleaned CSVs
2. Run `02_modeling.ipynb` → trains & evaluates models
3. Launch the app: `streamlit run 03_app.py`

See notebooks for full details.
