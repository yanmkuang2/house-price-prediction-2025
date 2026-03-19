#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="CA Home Price Predictor", layout="wide")

st.title("California Single-Family Home Price Predictor")
st.markdown("Estimate closing price based on 2025 CRMLS data (trained June–Nov, tested Dec).")

# Load best model
model = joblib.load('best_lgbm_model.pkl')

# Load scalar
scaler = joblib.load('train_scaler.pkl')

# Load target encoding mappings (adjust names to match what you saved)
highschool_means = joblib.load('highschool_target_means.pkl')
highschool_global = joblib.load('highschool_global_mean.pkl')

flooring_means = joblib.load('flooring_target_means.pkl')
flooring_global = joblib.load('flooring_global_mean.pkl')

# Load zip
zip_prefix_list = joblib.load('zip_prefix_list.pkl')
if '900' not in zip_prefix_list:
    zip_prefix_list.append('900')

zip_prefix_list = sorted(set(zip_prefix_list))  # remove duplicates, sort


# In[2]:


print(zip_prefix_list)


# # 2. User input

# In[3]:


st.header("Property Details")

col1, col2, col3 = st.columns(3)

with col1:
    living_area = st.number_input("Living Area (sq ft)", min_value=500.0, max_value=15000.0, value=2000.0, step=100.0)
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3, step=1)
    bathrooms = st.number_input("Bathrooms", min_value=1.0, max_value=10.0, value=2.0, step=0.5)

with col2:
    lot_size = st.number_input("Lot Size (sq ft)", min_value=1000.0, max_value=1000000.0, value=7000.0, step=1000.0)
    property_age = st.number_input("Property Age", min_value=0, max_value=250, value=50, step=1)
    garage_spaces = st.number_input("Garage Spaces", min_value=0, max_value=10, value=2, step=1)

with col3:
    association_fee = st.number_input("Association Fee ($)", min_value=0.0, max_value=10000.0, value=0.0, step=50.0)
    new_construction = st.checkbox("New Construction?", value=False)
    pool = st.checkbox("Private Pool?", value=False)
    view = st.checkbox("Has View?", value=False)
    fireplace = st.checkbox("Has Fireplace?", value=False)

    # High-priority categoricals (target-encoded)
    highschool_district = st.selectbox(
        "High School District",
        options=["Not specified"] + list(highschool_means.keys())
    )
    flooring_type = st.selectbox(
        "Flooring Type",
        options=["Not specified"] + list(flooring_means.keys())
    )
     # ZIP prefix dropdown
    selected_zip_prefix = st.selectbox(
        "ZIP Prefix",
        options=["Not specified"] + zip_prefix_list
    )


# # 3. Prediction

# In[4]:


if st.button("Predict Closing Price", type="primary"):
    # Build input dictionary
    input_dict = {
        'LivingArea': living_area,
        'LotSizeSquareFeet': lot_size,
        'AssociationFee': association_fee,
        'BedroomsTotal': bedrooms,
        'BathroomsTotalInteger': bathrooms,
        'Age': property_age,
        'GarageSpaces': garage_spaces,
        'NewConstructionYN': 1 if new_construction else 0,
        'PoolPrivateYN': 1 if pool else 0,
        'ViewYN': 1 if view else 0,
        'FireplaceYN': 1 if fireplace else 0,
        # Target encodings
        'HighSchoolDistrict_target_mean': highschool_means.get(highschool_district, highschool_global) if highschool_district != "Not specified" else highschool_global,
        'Flooring_target_mean': flooring_means.get(flooring_type, flooring_global) if flooring_type != "Not specified" else flooring_global,
    }

    # Add all ZIP_prefix dummies (set selected one to 1, others to 0)
    for prefix in zip_prefix_list:
        input_dict[f'ZIP_prefix_{prefix}'] = 1 if selected_zip_prefix == prefix else 0

    input_df = pd.DataFrame([input_dict])

    # Scale the raw columns
    num_cols_raw = ['LivingArea', 'LotSizeSquareFeet', 'AssociationFee']
    if all(col in input_df.columns for col in num_cols_raw):
        scaled_values = scaler.transform(input_df[num_cols_raw])
        for i, raw_col in enumerate(num_cols_raw):
            input_df[f"{raw_col}_std"] = scaled_values[:, i]  # create the _std columns

    # Drop the raw columns (model doesn't expect them)
    input_df = input_df.drop(columns=num_cols_raw, errors='ignore')

    # Align columns exactly to model's expected input
    try:
        model_columns = model.booster_.feature_name()  # LightGBM attribute
        input_df = input_df.reindex(columns=model_columns, fill_value=0)
    except AttributeError:
        st.warning("Model does not expose feature names — ensure input matches training columns.")

    # Make prediction
    try:
        pred_log = model.predict(input_df)[0]
        pred_price = np.expm1(pred_log)

        st.success(f"**Estimated Closing Price: ${pred_price:,.0f}**")

        st.info(
            """
            This is an estimate based on 2025 California single-family home data.
            Actual prices may vary due to condition, market trends, and other factors.
            """
        )
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.info("Check that all required features are provided and match the model's expected columns.")


# In[ ]:




