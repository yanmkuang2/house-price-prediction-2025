#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 14:32:19 2026

@author: yanmingkuang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 12:52:32 2026

@author: yanmingkuang
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from ftplib import FTP
import os
import glob

default_args = {
    'owner': 'yanming',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    dag_id='mls_full_pipeline',
    default_args=default_args,
    description='Download new MLS files, preprocess, train & evaluate model',
    schedule=None,  # manual trigger for testing; change to '@monthly' later
    start_date=datetime(2026, 3, 1),
    catchup=False,
)

# ────────────────────────────────────────────────
# Task 1: Download only NEW MLS files
# ────────────────────────────────────────────────

def download_new_mls_data():
    # FileZilla connection details
    host = ''           
    username = ''
    password = ''                 
    remote_dir = '/raw/california/'
    local_dir = os.path.expanduser('~/Desktop/IDX/data/')

    # Find latest month already downloaded locally
    existing_files = glob.glob(os.path.join(local_dir, 'CRMLSSold*.csv'))
    latest_local_month = datetime(2025, 12, 1)  # fallback

    if existing_files:
        dates = []
        for f in existing_files:
            try:
                yyyymm = os.path.basename(f)[9:15]  # CRMLSSoldYYYYMM.csv
                dt = datetime.strptime(yyyymm, '%Y%m')
                dates.append(dt)
            except:
                continue
        if dates:
            latest_local_month = max(dates)

    print(f"Latest local month: {latest_local_month.strftime('%Y-%m')}")

    try:
        ftp = FTP(host)
        ftp.login(user=username, passwd=password)
        ftp.cwd(remote_dir)
        print(f"Connected to {host}, dir: {ftp.pwd()}")

        remote_files = ftp.nlst()
        downloaded = []

        for file_name in remote_files:
            if file_name.startswith('CRMLSSold') and file_name.endswith('.csv'):
                yyyymm = file_name[9:15]
                try:
                    file_date = datetime.strptime(yyyymm, '%Y%m')
                    if file_date > latest_local_month:
                        local_file = os.path.join(local_dir, file_name)
                        with open(local_file, 'wb') as f:
                            ftp.retrbinary(f'RETR {file_name}', f.write)
                        downloaded.append(file_name)
                        print(f"Downloaded: {file_name} ({file_date.strftime('%Y-%m')})")
                except:
                    print(f"Skipping invalid file: {file_name}")

        ftp.quit()

        if not downloaded:
            print("No new files found.")
        else:
            print(f"Downloaded {len(downloaded)} new files.")

    except Exception as e:
        print(f"FTP error: {e}")
        raise

    
# ────────────────────────────────────────────────
# Task 2: Preprocess
# ────────────────────────────────────────────────
# 1. Data Loading
def run_preprocessing():
    import pandas as pd
    import numpy as np
    import glob
    import joblib
    from scipy import stats
    from sklearn.preprocessing import StandardScaler
    import os
    from datetime import datetime

    raw_dir = os.path.expanduser('~/Desktop/IDX/data/')
    raw_files = glob.glob(os.path.join(raw_dir, 'CRMLSSold*.csv'))

    if not raw_files:
        raise FileNotFoundError("No CRMLSSold*.csv files found")

    # Parse date from filename and sort
    file_list = []
    for f in raw_files:
        try:
            yyyymm = os.path.basename(f)[9:15]  # CRMLSSoldYYYYMM.csv
            dt = datetime.strptime(yyyymm, '%Y%m')
            file_list.append((dt, f))
        except:
            continue

    if not file_list:
        raise ValueError("No valid CRMLSSold*.csv files found")

    # Sort by date (oldest to newest)
    file_list.sort(key=lambda x: x[0])

    # Latest month = most recent file
    latest_date, latest_file = file_list[-1]
    print(f"Latest month: {latest_date.strftime('%Y-%m')} ({latest_file})")

    # Train = all except the latest month
    train_files = [f for dt, f in file_list[:-1]]
    if not train_files:
        raise ValueError("Not enough data — only one month available")

    # Load and concatenate train files
    train_dfs = []
    for f in train_files:
        df = pd.read_csv(f)
        train_dfs.append(df)
    new_train_df = pd.concat(train_dfs, ignore_index=True)

    # Test = only the latest month
    new_test_df = pd.read_csv(latest_file)

    print(f"New train shape: {new_train_df.shape} ({len(train_files)} months)")
    print(f"New test shape: {new_test_df.shape} ({latest_date.strftime('%Y-%m')})")



    # ## 1.2 Data Cleaning
    
    # In[80]:
    
    
    # 2. Data cleaning
    
    # 2.1 Restrict to single-family residential homes
    # Keep backup before filtering
    train_raw = new_train_df.copy()
    
    # Apply the filter
    new_train_df = new_train_df[
        (new_train_df['PropertyType'] == 'Residential') & 
        (new_train_df['PropertySubType'] == 'SingleFamilyResidence')
    ].copy()
    
    print(f"Filtered to single-family: {new_train_df.shape[0]:,} rows")
    
    
    # 2.2 Remove top 0.5% and bottom 0.5% of ClosePrice to exclude erroneous / non-economic transactions
    
    lower_threshold = new_train_df['ClosePrice'].quantile(0.005)   # bottom 0.5%
    upper_threshold = new_train_df['ClosePrice'].quantile(0.995)   # top 0.5%
    
    print(f"Outlier removal thresholds (ClosePrice):")
    print(f"  Lower (0.5th percentile): ${lower_threshold:,.0f}")
    print(f"  Upper (99.5th percentile): ${upper_threshold:,.0f}")
    
    rows_before = len(new_train_df)
    low_outliers  = new_train_df['ClosePrice'] < lower_threshold
    high_outliers = new_train_df['ClosePrice'] > upper_threshold
    
    print(f"Rows removed (bottom 0.5%): {low_outliers.sum():,}")
    print(f"Rows removed (top 0.5%):    {high_outliers.sum():,}")
    print(f"Total rows removed:         {low_outliers.sum() + high_outliers.sum():,}")
    print(f"Percentage removed:         {((low_outliers.sum() + high_outliers.sum()) / rows_before) * 100:.3f}%")
    
    new_train_df = new_train_df[
        (new_train_df['ClosePrice'] >= lower_threshold) & 
        (new_train_df['ClosePrice'] <= upper_threshold)
    ].copy()
    
    
    # 2.3 Exclude `ListPrice` , `OriginalListPrice`, 'DaysOnMarket'
    columns_to_drop = ['ListPrice', 'OriginalListPrice','DaysOnMarket']
    new_train_df = new_train_df.drop(columns=columns_to_drop, errors='ignore')
    
    
    # 2.4 Remove rows where LivingArea is <=0
    print("Rows with LivingArea == 0:", (new_train_df['LivingArea'] <= 0).sum())
    new_train_df = new_train_df[new_train_df['LivingArea'] > 0].copy()
    
    
    # 2.5 Remove rows where Latitude or Longitude is missing or invalid
    
    print("\nInvalid Latitude (outside CA range or 0/NaN):")
    print(new_train_df[~new_train_df['Latitude'].between(32.5, 42.0)]['Latitude'].value_counts(dropna=False))
    
    print("\nInvalid Longitude (outside CA range or 0/NaN):")
    print(new_train_df[~new_train_df['Longitude'].between(-124.5, -114.0)]['Longitude'].value_counts(dropna=False))
    
    lat_min, lat_max = 32.5, 42.0
    lon_min, lon_max = -124.5, -114.0
    
    valid_location = (
        new_train_df['Latitude'].notna() &
        new_train_df['Longitude'].notna() &
        new_train_df['Latitude'].between(lat_min, lat_max) &
        new_train_df['Longitude'].between(lon_min, lon_max)
    )
    
    new_train_df = new_train_df[valid_location].copy()
    
    # 2.6 remove illogical/impossible values
    logical_values = (
        (new_train_df['BedroomsTotal'] > 0) &
        (new_train_df['BathroomsTotalInteger'] > 0) &
        (new_train_df['LotSizeAcres'] > 0) &
        (new_train_df['LotSizeArea'] > 0) &
        (new_train_df['LotSizeSquareFeet'] > 0) &
        (new_train_df['ParkingTotal'] >= 0) &  # >= 0 to allow 0, only drop negative
        (new_train_df['LotSizeSquareFeet'] <= 217800) & #realistic max (e.g., 5 acres = 217,800 sq ft)
        (new_train_df['ParkingTotal'] <= 50) &
        (new_train_df['GarageSpaces'].isna() | new_train_df['GarageSpaces'] <= 100) # keep the missing rows
    )
    
    
    # Apply the filter
    new_train_df = new_train_df[logical_values].copy()
    
    print("\nUpdated min values after removal:")
    print("BedroomsTotal min:      ", new_train_df['BedroomsTotal'].min())
    print("BathroomsTotalInteger min:", new_train_df['BathroomsTotalInteger'].min())
    print("LotSizeAcres min:       ", new_train_df['LotSizeAcres'].min())
    print("LotSizeArea min:        ", new_train_df['LotSizeArea'].min())
    print("LotSizeSquareFeet min:  ", new_train_df['LotSizeSquareFeet'].min())
    print("ParkingTotal min:       ", new_train_df['ParkingTotal'].min())
    print("LotSizeSquareFeet max:  ", new_train_df['LotSizeSquareFeet'].max())
    print("ParkingTotal max:       ", new_train_df['ParkingTotal'].max())
    print("GarageSpaces max:       ", new_train_df['GarageSpaces'].max())
    
    # 2.7 remove duplicate rows with the same ListingIDs and other features
    duplicates = new_train_df[new_train_df['ListingId'].duplicated(keep=False)].sort_values('ListingId')
    
    print(f"Found {len(duplicates):,} duplicate rows (all occurrences)")
    print("\nSample duplicate rows:")
    print(duplicates[['ListingId', 'ClosePrice', 'CloseDate', 'City', 'LivingArea', 'BedroomsTotal', 'BathroomsTotalInteger']].head(10))
    
    new_train_df = new_train_df.drop_duplicates(subset='ListingId', keep='first').copy()
    
    
    # 2.8 remove columns with >80% missing
    missing_pct = (new_train_df.isna().mean() * 100).sort_values(ascending=False).round(2)
    
    missing_cols_to_drop = missing_pct[missing_pct > 80].index.tolist()
    
    # show only columns with missing values
    print("\nFeatures with missing values (>0%):")
    print(missing_pct[missing_pct > 0])
    
    new_train_df = new_train_df.drop(columns=missing_cols_to_drop)
    
    # 2.9 log transform ClosePrice
    print(new_train_df["ClosePrice"].isna().sum())
    print((new_train_df["ClosePrice"]<=0).sum())
    new_train_df["log_ClosePrice"] = np.log(new_train_df["ClosePrice"])
    
    # After all cleaning
    print(f"\nTotal rows removed during cleaning: "
          f"{len(train_raw) - len(new_train_df):,} "
          f"({(len(train_raw) - len(new_train_df)) / len(train_raw) * 100:.1f}% of original)")
    
    sorted_cols = sorted(new_train_df.columns)
    
    new_train_df[sorted_cols].info()
    
    
    
    # ## 1.3 Exploratory Data Analysis (EDA)
    
    # In[81]:
    
    
    # 3.1 Examine the distribution of the target variable: ClosePrice
    # Summary statistics 
    print("ClosePrice Summary Statistics:")
    print(new_train_df['ClosePrice'].describe())
    
    # Histogram
   # sns.set(style="whitegrid")
   # plt.rcParams['figure.figsize'] = (12, 7)  # wider figure to fit labels
    
   # plt.figure()
   # sns.histplot(data=new_train_df, x='ClosePrice', bins=60, kde=True, color='teal')
   # plt.title('Distribution of Close Price (Single-Family Homes)')
   # plt.xlabel('Close Price ($)')
   # plt.ylabel('Count')
   # plt.ticklabel_format(style='plain', axis='x') 
   # plt.show()
    
    # 3.2 Explore the relationships between numeric features and ClosePrice
    # Correlation matrix
    
    num_cols = [
        'ClosePrice',           # target
        'LivingArea',
        'BedroomsTotal',
        'BathroomsTotalInteger',
        'LotSizeAcres',
        'LotSizeArea',
        'LotSizeSquareFeet',
        'YearBuilt',
        'ParkingTotal',
        'GarageSpaces',
        'Stories',
        'AssociationFee',
        'Latitude',
        'Longitude',
        'MainLevelBedrooms'
        # You can add others if you think they matter, but most remaining numerics are IDs or empty
    ]
    
    
    # Compute correlation matrix
    corr_matrix = new_train_df[num_cols].corr()
    
    # Correlations with ClosePrice only – sorted descending
    close_corr = corr_matrix['ClosePrice'].sort_values(ascending=False)
    
    print("\nCorrelations with ClosePrice (sorted):")
    print(close_corr.round(3))
    
    # Bar plot – strongest associations
    #plt.figure(figsize=(10, 8))
    #sns.barplot(
    #    x=close_corr.values[1:],           # exclude self-correlation
    #    y=close_corr.index[1:],
    #    palette='coolwarm'
    #)
    #plt.axvline(0, color='gray', linestyle='--')
    #plt.title('Correlation of Numeric Features with ClosePrice', fontsize=14)
    #plt.xlabel('Pearson Correlation Coefficient')
    #plt.tight_layout()
    #plt.show()
    
    
    # In[82]:
    
    
    # 3.2 Explore the relationships between non-numeric features (e.g., categorical) and ClosePrice
        
    cat_cols = [
        'CountyOrParish', 
        'Levels', 
        'NewConstructionYN',
        'PoolPrivateYN', 
        'ViewYN', 
        'FireplaceYN', 
        'AttachedGarageYN',
        'AssociationFeeFrequency', 
        'Stories', 
        'Flooring',
        'HighSchoolDistrict',
        'MLSAreaMajor',
        'City'
    ]
    
    
    print("Categorical columns being analyzed:", cat_cols)
    
    
    def kruskal_test_by_category(df, cat_col, target_col='ClosePrice', min_obs_per_group=10):
        """
        Runs Kruskal-Wallis test on ClosePrice grouped by a categorical column.
        Skips groups with too few observations.
        """
        # Drop NaN in the category or target
        df_valid = df.dropna(subset=[cat_col, target_col]).copy()
    
        # Group prices by category
        groups = []
        group_names = []
    
        for name, group in df_valid.groupby(cat_col):
            prices = group[target_col].values
            if len(prices) >= min_obs_per_group:
                groups.append(prices)
                group_names.append(name)
    
        if len(groups) < 2:
            print(f"Not enough valid groups for {cat_col} (need ≥2 groups with ≥{min_obs_per_group} obs each)")
            return None
    
        # Run Kruskal-Wallis
        # All features except AttachedGarageYN show highly significant differences in ClosePrice across categories (p < 0.05 or much lower).  
        stat, p_value = stats.kruskal(*groups)
    
        print(f"\nKruskal-Wallis Test: {cat_col} vs {target_col}")
        print(f"Number of groups tested: {len(groups)}")
        print(f"Groups tested: {', '.join(map(str, group_names[:10]))}{'...' if len(group_names) > 10 else ''}")
        print(f"Statistic: {stat:.3f}")
        print(f"p-value:   {p_value:.4e}")
    
        if p_value < 0.05:
            print("→ Statistically significant differences (p < 0.05)")
        elif p_value < 0.10:
            print("→ Marginally significant differences (p < 0.10)")
        else:
            print("→ No strong evidence of differences (p ≥ 0.10)")
    
        return stat, p_value
    
    # Run Kruskal-Wallis on each
    for col in cat_cols:
        if col in new_train_df.columns:
            kruskal_test_by_category(new_train_df, col)
        else:
            print(f"Column not found: {col}")
    
    
    # ## 1.4 Impute Missing values
    
    # In[83]:
    
    
    # 4.1 Impute median (by Zipcodes) for numeric features
    
    
    # In[84]:
    
    
    # Number of unique ZIP codes
    print("Number of unique PostalCode values:", new_train_df['PostalCode'].nunique())
    
    # Distribution of properties per ZIP code
    print("\nProperties per PostalCode (sorted descending):")
    zip_counts = new_train_df['PostalCode'].value_counts()
    print(zip_counts.head(20))   # top 20 most common
    print("\n... (total unique ZIPs:", len(zip_counts), ")")
    print("\nZIPs with very few properties:")
    print(zip_counts.tail(10))   # rarest ZIPs
    
    # Summary stats on counts
    # Some zip codes have only 1 or few propertities -> group zip codes by the first 3 digits
    print("\nSummary of ZIP code frequencies:")
    print(zip_counts.describe().round(0))
    
    # Convert PostalCode to string 
    new_train_df['PostalCode'] = new_train_df['PostalCode'].astype(str)
    
    # Extract first 3 digits as ZIP prefix
    new_train_df['ZIP_prefix'] = new_train_df['PostalCode'].str[:3]
    
    # Number of unique ZIP prefixes
    num_unique_prefixes = new_train_df['ZIP_prefix'].nunique()
    print(f"Number of unique ZIP prefixes (first 3 digits): {num_unique_prefixes}")
    
    print("Unique ZIP prefixes and their frequencies (sorted by prefix):")
    print(new_train_df['ZIP_prefix'].value_counts().sort_index())
    
    
    # In[85]:
    
    
    # Convert to numeric, invalid become NaN
    new_train_df['ZIP_prefix_num'] = pd.to_numeric(new_train_df['ZIP_prefix'], errors='coerce')
    
    # Valid CA ZIP mask
    valid_zip_mask = (
        new_train_df['ZIP_prefix_num'].notna() &
        (new_train_df['ZIP_prefix_num'] >= 900) &
        (new_train_df['ZIP_prefix_num'] <= 961)
    )
    
    # Apply filter
    new_train_df = new_train_df[valid_zip_mask].copy()
    
    # Drop the temporary column
    new_train_df = new_train_df.drop(columns=['ZIP_prefix_num'], errors='ignore')
    
    
    # In[86]:
    
    
    print(new_train_df['ZIP_prefix'].value_counts().sort_index())
    
    
    # In[87]:
    
    
    # Summary table: missing % 
    numeric_summary = pd.DataFrame({
        'Column': num_cols,
        'Non-Null Count': [new_train_df[col].notna().sum() for col in num_cols],
        'Missing %': [new_train_df[col].isna().sum() / len(new_train_df) * 100 for col in num_cols]
    }).round(2)
    
    print("\nNumerical Summary:")
    print(numeric_summary.sort_values('Missing %', ascending=False))
    
    
    # In[88]:
    
    
    # List of numeric columns we're imputing
    num_cols_to_impute = [
        'MainLevelBedrooms',
        'AssociationFee',
        'Stories',
        'GarageSpaces',
        'YearBuilt'
    ]
    
    print(new_train_df['ZIP_prefix'].dtype)
    
    
    
    # In[89]:
    
    
    # Check for each column: how many missing values are in ZIP prefixes with no non-missing data
    print("Check for ZIP_prefix median fallback need:")
    for col in num_cols_to_impute:
        # ZIP prefixes with at least one non-missing value for this column
        zip_has_data = new_train_df[new_train_df[col].notna()]['ZIP_prefix'].unique()
    
        # Missing values in ZIP prefixes that have NO data for this column
        missing_in_no_data_zips = new_train_df[
            (new_train_df[col].isna()) & 
            (~new_train_df['ZIP_prefix'].isin(zip_has_data))
        ].shape[0]
    
        total_missing = new_train_df[col].isna().sum()
    
        print(f"\n{col}:")
        print(f"  Total missing: {total_missing:,}")
        print(f"  Missing in ZIP prefixes with no data for {col}: {missing_in_no_data_zips:,}")
        print(f"  % of missing that would need global fallback: "
              f"{(missing_in_no_data_zips / total_missing * 100) if total_missing > 0 else 0:.2f}%")
    
    
    # In[90]:
    
    
    # Compute medians by ZIP_prefix
    medians_by_zip = {}
    for col in num_cols_to_impute:
        medians_by_zip[col] = new_train_df.groupby('ZIP_prefix')[col].median()
    
    # Global fallback medians (only used when ZIP has no data)
    global_medians = {col: new_train_df[col].median() for col in num_cols_to_impute}
    
    # Imputation function
    def impute_by_zip(row, col):
        if pd.isna(row[col]):
            zip_median = medians_by_zip[col].get(row['ZIP_prefix'], None)
            if pd.notna(zip_median):
                return zip_median
            return global_medians[col]
        return row[col]
    
    # Apply imputation
    for col in num_cols_to_impute:
        new_train_df[col] = new_train_df.apply(lambda row: impute_by_zip(row, col), axis=1)
    
    # Verification
    print("\nMissing values after ZIP-prefix median imputation:")
    print(new_train_df[num_cols_to_impute].isna().sum())
    
    print("\nUpdated descriptive statistics:")
    print(new_train_df[num_cols_to_impute].describe().round(2))
    
    
    # In[91]:
    
    
    # Summary table: missing % and unique count for each
    cat_summary = pd.DataFrame({
        'Column': cat_cols,
        'Non-Null Count': [new_train_df[col].notna().sum() for col in cat_cols],
        'Missing %': [new_train_df[col].isna().sum() / len(new_train_df) * 100 for col in cat_cols],
        'Unique Values': [new_train_df[col].nunique() for col in cat_cols]
    }).round(2)
    
    print("\nCategorical Summary:")
    print(cat_summary.sort_values('Missing %', ascending=False))
    
    
    # In[92]:
    
    
    cat_cols_to_impute = [
        'AssociationFeeFrequency',
        'Flooring',
        'HighSchoolDistrict',
        'MLSAreaMajor',
        'AttachedGarageYN',
        'ViewYN',
        'PoolPrivateYN',
        'NewConstructionYN',
        'Levels',
        'FireplaceYN',
        'City'
    ]
    
    
    # In[93]:
    
    
    # Compute mode (most frequent) per ZIP_prefix for each categorical column
    modes_by_zip = {}
    for col in cat_cols_to_impute:
        modes_by_zip[col] = new_train_df.groupby('ZIP_prefix')[col].agg(lambda x: x.mode()[0] if not x.mode().empty else pd.NA)
    
    # Global fallback modes
    global_modes = {col: new_train_df[col].mode()[0] if not new_train_df[col].mode().empty else pd.NA for col in cat_cols_to_impute}
    
    # Imputation function: ZIP-prefix mode first, global mode fallback
    def impute_cat_by_zip(row, col):
        if pd.isna(row[col]):
            zip_mode = modes_by_zip[col].get(row['ZIP_prefix'], None)
            if pd.notna(zip_mode):
                return zip_mode
            return global_modes[col]
        return row[col]
    
    # Apply imputation to each categorical column
    for col in cat_cols_to_impute:
        print(f"Imputing {col} using ZIP-prefix mode...")
        new_train_df[col] = new_train_df.apply(lambda row: impute_cat_by_zip(row, col), axis=1)
    
    # Verification
    print("\nMissing values after imputation:")
    print(new_train_df[cat_cols_to_impute].isna().sum())
    
    print("\nUpdated descriptive statistics:")
    print(new_train_df[cat_cols_to_impute].describe().round(2))
    
    
    # ## 1.5 Encoding
    
    # In[94]:
    
    
    categorical_summary = pd.DataFrame({
        'Column': cat_cols,
        'Non-Null Count': [new_train_df[col].notna().sum() for col in cat_cols],
        'Missing %': [new_train_df[col].isna().sum() / len(new_train_df) * 100 for col in cat_cols],
        'Unique Values': [new_train_df[col].nunique() for col in cat_cols]
    }).round(2)
    
    print("\nCategorical Summary:")
    print(categorical_summary.sort_values('Unique Values', ascending=False))
    
    
    # In[95]:
    
    
    # I don't think we need to use MLSAreaMajor, CountyOrParish, or City in our model as we already have Zip codes
    # print(new_train_df['MLSAreaMajor'].value_counts(dropna=False)) 
    # print(new_train_df['CountyOrParish'].value_counts(dropna=False)) # We might not need to use this in our model as we already have Zip code and City
    # print(new_train_df['City'].value_counts(dropna=False))
    
    
    # In[96]:
    
    
    # For these high-cardinality features (HighSchoolDistrict, Flooring), use target encoding (also called mean encoding)
    # -> reduces each category to a single numeric value (e.g., mean ClosePrice in that category)
    # -> adds 1 column per feature
    print(new_train_df['HighSchoolDistrict'].value_counts(dropna=False))
    print(new_train_df['Flooring'].value_counts(dropna=False))
    
    
    # In[97]:
    
    
    # Target encoding function (fits on train)
    def target_encode_fit(df, cat_col, target_col='ClosePrice'):
        means = df.groupby(cat_col)[target_col].mean()
        global_mean = df[target_col].mean()
        encoded_col = f'{cat_col}_target_mean'
        df[encoded_col] = df[cat_col].map(means).fillna(global_mean)
        return df, means, global_mean
    
    # Apply and save the mappings
    new_train_df, highschool_means, highschool_global = target_encode_fit(new_train_df, 'HighSchoolDistrict')
    
    
    # In[98]:
    
    
    joblib.dump(highschool_means, 'highschool_target_means.pkl')   
    joblib.dump(highschool_global, 'highschool_global_mean.pkl')  
    
    
    # In[99]:
    
    
    # Apply and save the mappings
    new_train_df, flooring_means, flooring_global = target_encode_fit(new_train_df, 'Flooring')
    
    
    # In[100]:
    
    
    joblib.dump(flooring_means, 'flooring_target_means.pkl')
    joblib.dump(flooring_global, 'flooring_global_mean.pkl')
    
    
    # In[101]:
    
    
    print("Target-encoded columns added on train:")
    print(new_train_df[['HighSchoolDistrict', 'HighSchoolDistrict_target_mean',
                    'Flooring', 'Flooring_target_mean']].head(10))
    
    
    # In[102]:
    
    
    # recode Levels
    print(new_train_df['Levels'].value_counts(dropna=False)) 
    
    # make levels variable more simple
    def recode_levels_final(x):
        if pd.isna(x):
            return x
        if "ThreeOrMore" in x:
            return "ThreeOrMore"
        elif "Two" in x:
            return "Two"
        elif "One" in x:
            return "One"
        elif "MultiSplit" in x:
            return "MultiSplit"
        else:
            return "Other"
    
    new_train_df["Levels_final"] = new_train_df["Levels"].apply(recode_levels_final)
    
    new_train_df["Levels_final"].value_counts()
    
    
    # In[103]:
    
    
    # For low-cardinality features (Levels_final, AssociationFeeFrequency, Stories)
    # use one-hot encoding
    
    # For NewConstructionYN, PoolPrivateYN, ViewYN, FireplaceYN, AttachedGarageYN, already binary, no encoding needed
    
    low_card_cols = [
        'Levels_final', 
        'AssociationFeeFrequency',
        'Stories'
    ]
    
    # Recode from float to object
    new_train_df['Stories'] = new_train_df['Stories'].astype(str).replace('nan', 'Missing')
    print("Converted Stories to object dtype. New dtype:", new_train_df['Stories'].dtype)
    print("Unique Stories values now:", new_train_df['Stories'].unique())
    
    # One-hot encode
    dummies = pd.get_dummies(
        new_train_df[low_card_cols],
        prefix=low_card_cols,
        drop_first=True,
        dtype=int
    )
    
    # Add to DataFrame
    new_train_df = pd.concat([new_train_df.drop(columns=low_card_cols), dummies], axis=1)
    
    print(f"Added {dummies.shape[1]} new binary columns")
    print("New shape:", new_train_df.shape)
    print("\nNew one-hot columns from Stories (example):")
    print([c for c in dummies.columns if 'Stories' in c])
    
    
    # ## 1.6 Scaling
    
    # In[104]:
    
    
    # Normalize/scale numerical features
    # Apply StandardScaler: Subtracts mean, divides by standard deviation
    # Narrow it down to LivingArea, LotSizeSquareFeet, AssociationFee, which spans several orders of magnitudes
    
    num_cols_to_scale = [           
        'LivingArea',
        'LotSizeSquareFeet',
        'AssociationFee'
    ]
    
    print(new_train_df[num_cols_to_scale].describe().round(2))
    
    # Choose StandardScaler: subtracts mean of data and divides by sd 
    # -> reduce the effect of outliers/extreme values
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(new_train_df[num_cols_to_scale])
    
    # Create new scaled columns
    scaled_df = pd.DataFrame(
        scaled_values,
        columns=[f"{col}_std" for col in num_cols_to_scale],
        index=new_train_df.index
    )
    
    # Add to main DataFrame
    new_train_df = pd.concat([new_train_df, scaled_df], axis=1)
    
    print("\nfirst 5 rows:")
    print(new_train_df[[f"{col}_std" for col in num_cols_to_scale]].head())
    
    print("\nScaled columns summary:")
    print(new_train_df[[f"{col}_std" for col in num_cols_to_scale]].describe().round(2))
    
    
    
    # In[105]:
    
    
    joblib.dump(scaler, 'train_scaler.pkl')
    print("Scaler saved to 'train_scaler.pkl'")
    
    
    # ## 1.7 Feature Engineering
    
    # In[106]:
    
    
    # Capture local market trend
    # i.e., recent "comps" in the same neighborhood can be a strong predictor
    
    # Convert to date time
    new_train_df['CloseDate'] = pd.to_datetime(new_train_df['CloseDate'])
    
    # Create monthly period
    new_train_df['CloseMonth'] = new_train_df['CloseDate'].dt.to_period('M')
    
    print(new_train_df['CloseMonth'].value_counts())
    
    
    # In[107]:
    
    
    # Compute median price per ZIP_prefix per month
    monthly_medians = new_train_df.groupby(['ZIP_prefix', 'CloseMonth'])['ClosePrice'].median().reset_index()
    monthly_medians = monthly_medians.sort_values(['ZIP_prefix', 'CloseMonth'])
    print(monthly_medians)
    
    
    # In[108]:
    
    
    # Compute month-over-month growth rate
    monthly_medians['prev_median'] = monthly_medians.groupby('ZIP_prefix')['ClosePrice'].shift(1)
    monthly_medians['mom_growth'] = (monthly_medians['ClosePrice'] - monthly_medians['prev_median']) / monthly_medians['prev_median']
    print(monthly_medians['mom_growth'])
    
    
    # In[109]:
    
    
    # Backfill the first month's (January) growth rate with the next available MoM (July)
    monthly_medians['mom_growth'] = monthly_medians.groupby('ZIP_prefix')['mom_growth'].bfill()
    
    print(monthly_medians['mom_growth'])
    
    # no remaining NaN values
    print(monthly_medians['mom_growth'].isna().sum())
    
    
    # In[110]:
    
    
    # Now aggregate: mean MoM growth per ZIP_prefix
    zip_growth_rate = monthly_medians.groupby('ZIP_prefix')['mom_growth'].mean().reset_index()
    zip_growth_rate = zip_growth_rate.rename(columns={'mom_growth': 'zip_growth_rate'})
    print(zip_growth_rate )
    
    
    # In[111]:
    
    
    # Merge back to train
    new_train_df = new_train_df.merge(zip_growth_rate, on='ZIP_prefix', how='left')
    
    print(new_train_df['zip_growth_rate'].head(10))
    
    # no NaN values
    print(new_train_df['zip_growth_rate'].isna().sum())
    
    print(new_train_df['zip_growth_rate'].describe().round(4))
    
    
    # In[112]:
    
    
    # Create distance to coast
    from math import radians, sin, cos, sqrt, atan2
    
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371  # Earth radius in km
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c
    
    # Define several major coastal points (lat, lon)
    coastal_points = [
        (34.0100, -118.4960),  # Santa Monica Beach (LA area)
        (37.8199, -122.4786),  # Pacific coast near Golden Gate / SF
        (32.7157, -117.1611),  # San Diego harbor/coast
        (36.6002, -121.8947),  # Monterey Bay coast
        (40.8021, -124.1637)   # Northern CA coast proxy
    ]
    
    
    # Compute min distance to any coast for each property
    def min_distance_to_coast(row):
        distances = [haversine(row['Latitude'], row['Longitude'], lat, lon) for lat, lon in coastal_points]
        return min(distances)
    
    new_train_df['dist_to_coast_km'] = new_train_df.apply(min_distance_to_coast, axis=1)
    
    print(new_train_df['dist_to_coast_km'].describe().round(2))
    
    
    # In[113]:
    
    
    # Define reference year
    REFERENCE_YEAR = 2026
    
    # Create Age column
    new_train_df['Age'] = REFERENCE_YEAR - new_train_df['YearBuilt']
    
    
    # In[114]:
    
    
    # One hot encode ZIP_prefix
    zip_dummies_train = pd.get_dummies(
        new_train_df['ZIP_prefix'],
        prefix='ZIP_prefix',
        drop_first=True,
        dtype=int
    )
    
    # Add to train
    new_train_df = pd.concat([new_train_df.drop(columns=['ZIP_prefix']), zip_dummies_train], axis=1)
    
    print(f"Added {zip_dummies_train.shape[1]} ZIP_prefix dummy columns to train")
    print("Train shape after:", new_train_df.shape)
    
    
    # ## 1.8 Train Set Cleaning Complete
    
    # In[115]:
    
    
    # Last step of cleaning: drop irrelevant columns
    cols_to_drop = [
        'ListAgentFirstName',
        'ListAgentLastName',
        'ListOfficeName',
        'BuyerOfficeName',
        'CoListOfficeName',
        'ListAgentFullName',
        'CoListAgentFirstName',
        'CoListAgentLastName',
        'BuyerAgentMlsId',
        'BuyerAgentFirstName',
        'BuyerAgentLastName',
        'ContractStatusChangeDate',
        'PurchaseContractDate',
        'ListingContractDate',
        'ListAgentAOR',
        'BuyerAgentAOR',
        'ListAgentEmail',
        'UnparsedAddress',
        'SubdivisionName',
        'BuyerOfficeAOR',
        'StreetNumberNumeric',
        'CloseMonth'
    ]
    
    # Save CLEAN data
    new_train_df_clean = new_train_df.drop(columns=cols_to_drop).copy()
    
    new_train_df_clean.info()
    
    new_train_df_clean.to_csv('~/Desktop/IDX/data/new_train_df_clean.csv', index=False)
    
    
    # # 2. Test Set Preprocessing
    
    # ## 2.1 Data Cleaning
    
    # In[116]:
    
    
    # December 2025 as the test set
    new_test_df = pd.read_csv('/Users/yanmingkuang/Desktop/IDX/CRMLSSold202512.csv')
    print("Dataset shape:", new_test_df.shape)
    
    
    # In[117]:
    
    
    # 8.1 Restrict to single-family residential homes
    # Keep backup before filtering
    test_raw = new_test_df.copy()
    
    # Apply the filter
    new_test_df = new_test_df[
        (new_test_df['PropertyType'] == 'Residential') & 
        (new_test_df['PropertySubType'] == 'SingleFamilyResidence')
    ].copy()
    
    print(f"Filtered to single-family: {new_test_df.shape[0]:,} rows")
    
    
    # 8.2 Remove top 0.5% and bottom 0.5% of ClosePrice to exclude erroneous / non-economic transactions
    
    lower_threshold = new_test_df['ClosePrice'].quantile(0.005)   # bottom 0.5%
    upper_threshold = new_test_df['ClosePrice'].quantile(0.995)   # top 0.5%
    
    print(f"\nOutlier removal thresholds (ClosePrice):")
    print(f"  Lower (0.5th percentile): ${lower_threshold:,.0f}")
    print(f"  Upper (99.5th percentile): ${upper_threshold:,.0f}")
    
    rows_before = len(new_test_df)
    low_outliers  = new_test_df['ClosePrice'] < lower_threshold
    high_outliers = new_test_df['ClosePrice'] > upper_threshold
    
    print(f"Rows removed (bottom 0.5%): {low_outliers.sum():,}")
    print(f"Rows removed (top 0.5%):    {high_outliers.sum():,}")
    print(f"Total rows removed:         {low_outliers.sum() + high_outliers.sum():,}")
    print(f"Percentage removed:         {((low_outliers.sum() + high_outliers.sum()) / rows_before) * 100:.3f}%")
    
    new_test_df = new_test_df[
        (new_test_df['ClosePrice'] >= lower_threshold) & 
        (new_test_df['ClosePrice'] <= upper_threshold)
    ].copy()
    
    
    # 8.3 Exclude `ListPrice` , `OriginalListPrice`, 'DaysOnMarket'
    columns_to_drop = ['ListPrice', 'OriginalListPrice','DaysOnMarket']
    new_test_df = new_test_df.drop(columns=columns_to_drop, errors='ignore')
    
    
    # 8.4 Remove rows where LivingArea is <=0
    print("Rows with LivingArea == 0:", (new_test_df['LivingArea'] <= 0).sum())
    new_test_df = new_test_df[new_test_df['LivingArea'] > 0].copy()
    
    
    # 8.5 Remove rows where Latitude or Longitude is missing or invalid
    
    print("\nInvalid Latitude (outside CA range or 0/NaN):")
    print(new_test_df[~new_test_df['Latitude'].between(32.5, 42.0)]['Latitude'].value_counts(dropna=False))
    
    print("\nInvalid Longitude (outside CA range or 0/NaN):")
    print(new_test_df[~new_test_df['Longitude'].between(-124.5, -114.0)]['Longitude'].value_counts(dropna=False))
    
    lat_min, lat_max = 32.5, 42.0
    lon_min, lon_max = -124.5, -114.0
    
    valid_location = (
        new_test_df['Latitude'].notna() &
        new_test_df['Longitude'].notna() &
        new_test_df['Latitude'].between(lat_min, lat_max) &
        new_test_df['Longitude'].between(lon_min, lon_max)
    )
    
    new_test_df = new_test_df[valid_location].copy()
    
    # 8.6 remove illogical/impossible values
    logical_values = (
        (new_test_df['BedroomsTotal'] > 0) &
        (new_test_df['BathroomsTotalInteger'] > 0) &
        (new_test_df['LotSizeAcres'] > 0) &
        (new_test_df['LotSizeArea'] > 0) &
        (new_test_df['LotSizeSquareFeet'] > 0) &
        (new_test_df['ParkingTotal'] >= 0) &  # >= 0 to allow 0, only drop negative
        (new_test_df['LotSizeSquareFeet'] <= 217800) & #realistic max (e.g., 5 acres = 217,800 sq ft)
        (new_test_df['ParkingTotal'] <= 50) &
        (new_test_df['GarageSpaces'].isna() | new_test_df['GarageSpaces'] <= 100) # keep the missing rows
    )
    
    
    # Apply the filter
    new_test_df = new_test_df[logical_values].copy()
    
    print("\nUpdated min values after removal:")
    print("BedroomsTotal min:      ", new_test_df['BedroomsTotal'].min())
    print("BathroomsTotalInteger min:", new_test_df['BathroomsTotalInteger'].min())
    print("LotSizeAcres min:       ", new_test_df['LotSizeAcres'].min())
    print("LotSizeArea min:        ", new_test_df['LotSizeArea'].min())
    print("LotSizeSquareFeet min:  ", new_test_df['LotSizeSquareFeet'].min())
    print("ParkingTotal min:       ", new_test_df['ParkingTotal'].min())
    print("LotSizeSquareFeet max:  ", new_test_df['LotSizeSquareFeet'].max())
    print("ParkingTotal max:       ", new_test_df['ParkingTotal'].max())
    print("GarageSpaces max:       ", new_test_df['GarageSpaces'].max())
    
    # 8.7 remove duplicate rows with the same ListingIDs and other features
    duplicates = new_test_df[new_test_df['ListingId'].duplicated(keep=False)].sort_values('ListingId')
    
    print(f"Found {len(duplicates):,} duplicate rows (all occurrences)")
    print("\nSample duplicate rows:")
    print(duplicates[['ListingId', 'ClosePrice', 'CloseDate', 'City', 'LivingArea', 'BedroomsTotal', 'BathroomsTotalInteger']].head(10))
    
    new_test_df = new_test_df.drop_duplicates(subset='ListingId', keep='first').copy()
    
    
    # 8.8 remove columns with >80% missing
    missing_pct = (new_test_df.isna().mean() * 100).sort_values(ascending=False).round(2)
    
    missing_cols_to_drop = missing_pct[missing_pct > 80].index.tolist()
    
    # show only columns with missing values
    print("\nFeatures with missing values (>0%):")
    print(missing_pct[missing_pct > 0])
    
    new_test_df = new_test_df.drop(columns=missing_cols_to_drop)
    
    # 8.9 log transform ClosePrice
    print(new_test_df["ClosePrice"].isna().sum())
    print((new_test_df["ClosePrice"]<=0).sum())
    new_test_df["log_ClosePrice"] = np.log(new_test_df["ClosePrice"])
    
    # After all cleaning
    print(f"\nTotal rows removed during cleaning: "
          f"{len(test_raw) - len(new_test_df):,} "
          f"({(len(test_raw) - len(new_test_df)) / len(test_raw) * 100:.1f}% of original)")
    
    sorted_cols = sorted(new_test_df.columns)
    
    new_test_df[sorted_cols].info()
    
    
    
    # ## 2.2 Impute Missing Values
    
    # In[118]:
    
    
    # Extract first 3 digits as ZIP prefix
    new_test_df['ZIP_prefix'] = new_test_df['PostalCode'].str[:3]
    
    # Number of unique ZIP prefixes
    num_unique_prefixes = new_test_df['ZIP_prefix'].nunique()
    print(f"Number of unique ZIP prefixes (first 3 digits): {num_unique_prefixes}")
    
    print("Unique ZIP prefixes and their frequencies (sorted by prefix):")
    print(new_test_df['ZIP_prefix'].value_counts().sort_index())
    
    # Convert to numeric, invalid become NaN
    new_test_df['ZIP_prefix_num'] = pd.to_numeric(new_test_df['ZIP_prefix'], errors='coerce')
    
    # Valid CA ZIP mask
    valid_zip_mask = (
        new_test_df['ZIP_prefix_num'].notna() &
        (new_test_df['ZIP_prefix_num'] >= 900) &
        (new_test_df['ZIP_prefix_num'] <= 961) &
        (new_test_df['ZIP_prefix_num'] != 929)
    )
    
    # Apply filter
    new_test_df = new_test_df[valid_zip_mask].copy()
    
    # Drop the temporary column
    new_test_df = new_test_df.drop(columns=['ZIP_prefix_num'], errors='ignore')
    
    
    # In[119]:
    
    
    # For numerical features: apply the same imputation function using train-computed medians
    for col in num_cols_to_impute:
        print(f"Imputing {col} on test set using train ZIP-prefix medians...")
        new_test_df[col] = new_test_df.apply(lambda row: impute_by_zip(row, col), axis=1)
    
    # Verification
    print("\nMissing values in test after imputation:")
    print(new_test_df[num_cols_to_impute].isna().sum())
    
    print("\nTest set descriptive statistics after imputation:")
    print(new_test_df[num_cols_to_impute].describe().round(2))
    
    
    # In[120]:
    
    
    # For categorical features: Apply the same imputation function using train-computed modes
    for col in cat_cols_to_impute:
        print(f"Imputing {col} on test set using train ZIP-prefix modes...")
        new_test_df[col] = new_test_df.apply(lambda row: impute_cat_by_zip(row, col), axis=1)
    
    # Verification
    print("\nMissing values in test after imputation:")
    print(new_test_df[cat_cols_to_impute].isna().sum())
    
    print("\nTest set descriptive statistics after imputation:")
    print(new_test_df[cat_cols_to_impute].describe().round(2))
    
    
    # ## 2.3 Encoding
    
    # In[121]:
    
    
    categorical_summary_test = pd.DataFrame({
        'Column': cat_cols,
        'Non-Null Count': [new_test_df[col].notna().sum() for col in cat_cols],
        'Missing %': [new_test_df[col].isna().sum() / len(new_test_df) * 100 for col in cat_cols],
        'Unique Values': [new_test_df[col].nunique() for col in cat_cols]
    }).round(2)
    
    print("\nCategorical Summary:")
    print(categorical_summary_test.sort_values('Unique Values', ascending=False))
    
    
    # In[122]:
    
    
    # Reuse the mappings computed from train
    # (highschool_means, highschool_global, flooring_means, flooring_global)
    
    # Apply mapping to test (lookup only — no groupby, no new means)
    new_test_df['HighSchoolDistrict_target_mean'] = new_test_df['HighSchoolDistrict'].map(highschool_means).fillna(highschool_global)
    new_test_df['Flooring_target_mean'] = new_test_df['Flooring'].map(flooring_means).fillna(flooring_global)
    
    print("Target-encoded columns added on test (using train mappings):")
    print(new_test_df[['HighSchoolDistrict', 'HighSchoolDistrict_target_mean',
                   'Flooring', 'Flooring_target_mean']].head(10))
    
    print("\nMissing encoded values in test (should be very few):")
    print(new_test_df[['HighSchoolDistrict_target_mean', 'Flooring_target_mean']].isna().sum())
    
    
    # In[123]:
    
    
    # recode Levels
    print(new_test_df['Levels'].value_counts(dropna=False)) 
    
    # make levels variable more simple
    def recode_levels_final(x):
        if pd.isna(x):
            return x
        if "ThreeOrMore" in x:
            return "ThreeOrMore"
        elif "Two" in x:
            return "Two"
        elif "One" in x:
            return "One"
        elif "MultiSplit" in x:
            return "MultiSplit"
        else:
            return "Other"
    
    new_test_df["Levels_final"] = new_test_df["Levels"].apply(recode_levels_final)
    
    new_test_df["Levels_final"].value_counts()
    
    
    # In[124]:
    
    
    # For low-cardinality features (Levels_final, AssociationFeeFrequency, Stories)
    # use one-hot encoding
    
    # For NewConstructionYN, PoolPrivateYN, ViewYN, FireplaceYN, AttachedGarageYN, already binary, no encoding needed
    
    low_card_cols = [
        'Levels_final', 
        'AssociationFeeFrequency',
        'Stories'
    ]
    
    # Recode from float to object
    new_test_df['Stories'] = new_test_df['Stories'].astype(str).replace('nan', 'Missing')
    print("Converted Stories to object dtype. New dtype:", new_test_df['Stories'].dtype)
    print("Unique Stories values now:", new_test_df['Stories'].unique())
    
    # One-hot encode
    dummies = pd.get_dummies(
        new_test_df[low_card_cols],
        prefix=low_card_cols,
        drop_first=True,
        dtype=int
    )
    
    # Add to DataFrame
    new_test_df = pd.concat([new_test_df.drop(columns=low_card_cols), dummies], axis=1)
    
    print(f"Added {dummies.shape[1]} new binary columns")
    print("New shape:", new_test_df.shape)
    print("\nNew one-hot columns from Stories (example):")
    print([c for c in dummies.columns if 'Stories' in c])
    
    
    # ## 2.4 Scaling
    
    # In[125]:
    
    
    # Apply transform only (using train's mean & std)
    new_test_df[[f"{col}_std" for col in num_cols_to_scale]] = scaler.transform(new_test_df[num_cols_to_scale])
    
    print("\nTest set scaled columns preview:")
    print(new_test_df[[f"{col}_std" for col in num_cols_to_scale]].head().round(3))
    
    print("\nTest scaled columns summary:")
    print(new_test_df[[f"{col}_std" for col in num_cols_to_scale]].describe().round(3))
    
    
    # ## 2.5 Feature Engineering
    
    # In[126]:
    
    
    # Merge zip_growth_rate back to train
    new_test_df = new_test_df.merge(zip_growth_rate, on='ZIP_prefix', how='left')
    
    print(new_test_df['zip_growth_rate'].head(10))
    
    # no NaN values
    print(new_test_df['zip_growth_rate'].isna().sum())
    
    print(new_test_df['zip_growth_rate'].describe().round(4))
    
    
    # In[127]:
    
    
    # Create distance to coast
    new_test_df['dist_to_coast_km'] = new_test_df.apply(min_distance_to_coast, axis=1)
    print(new_test_df['dist_to_coast_km'].describe().round(2))
    
    
    # In[128]:
    
    
    # Create Age column
    new_test_df['Age'] = REFERENCE_YEAR - new_test_df['YearBuilt']
    
    
    # In[129]:
    
    
    # One hot encode ZIP_prefix
    zip_dummies_test = pd.get_dummies(
        new_test_df['ZIP_prefix'],
        prefix='ZIP_prefix',
        drop_first=True,
        dtype=int
    )
    
    # Align test dummies to train columns (add missing as 0, drop extras)
    missing_cols = set(zip_dummies_train.columns) - set(zip_dummies_test.columns)
    for col in missing_cols:
        zip_dummies_test[col] = 0
    
    extra_cols = set(zip_dummies_test.columns) - set(zip_dummies_train.columns)
    if extra_cols:
        print("Warning: Test has extra ZIP_prefix categories:", extra_cols)
        zip_dummies_test = zip_dummies_test[zip_dummies_train.columns]
    
    # Add to test
    new_test_df = pd.concat([new_test_df.drop(columns=['ZIP_prefix']), zip_dummies_test], axis=1)
    
    print(f"Added {zip_dummies_test.shape[1]} ZIP_prefix dummy columns to test")
    print("Test shape after:", new_test_df.shape)
    
    
    # ## 2.6 Test Set Cleaning Complete
    
    # In[130]:
    
    
    # Last step of cleaning: drop irrelevant columns
    cols_to_drop = [
        'ListAgentFirstName',
        'ListAgentLastName',
        'ListOfficeName',
        'BuyerOfficeName',
        'CoListOfficeName',
        'ListAgentFullName',
        'CoListAgentFirstName',
        'CoListAgentLastName',
        'BuyerAgentMlsId',
        'BuyerAgentFirstName',
        'BuyerAgentLastName',
        'ContractStatusChangeDate',
        'PurchaseContractDate',
        'ListingContractDate',
        'ListAgentAOR',
        'BuyerAgentAOR',
        'ListAgentEmail',
        'UnparsedAddress',
        'SubdivisionName',
        'BuyerOfficeAOR',
        'StreetNumberNumeric'
    ]
    
    # Save CLEAN data
    new_test_df_clean = new_test_df.drop(columns=cols_to_drop).copy()
    
    new_test_df_clean.info()
    
    new_test_df_clean.to_csv('~/Desktop/IDX/data/new_test_df_clean.csv', index=False)
    
    


# ────────────────────────────────────────────────
# Task 3: Model training/evaluation (placeholder for 02_modeling)
# ────────────────────────────────────────────────
def run_modeling():
    import pandas as pd
    import numpy as np
    import glob
    import pickle
    import joblib
    from scipy import stats
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import RandomizedSearchCV
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from scipy.stats import randint, uniform, loguniform
    
    print("Starting modeling and evaluation...")

    # Load cleaned data from Task 2
    new_train_df_clean = pd.read_csv('~/Desktop/IDX/data/new_train_df_clean.csv')
    new_test_df_clean  = pd.read_csv('~/Desktop/IDX/data/new_test_df_clean.csv')

    # check data shape
    print("\nTraining set shape:", new_train_df_clean.shape)
    print("Test set shape:    ", new_test_df_clean.shape)
    
    # check if there is still any missing
    print("Missing values in train:", new_train_df_clean.isna().sum().sum())
    print("Missing values in test: ", new_test_df_clean.isna().sum().sum())

    # # 2. Modeling: Linear Regression

    # In[66]:


    print(new_test_df_clean.info())


    # In[96]:


    zip_dummy_cols = [col for col in new_test_df_clean.columns if col.startswith('ZIP_prefix_')]


    # In[97]:


    zip_prefix_list = [col.replace('ZIP_prefix_', '') for col in zip_dummy_cols]
    joblib.dump(zip_prefix_list, 'zip_prefix_list.pkl')


    # In[68]:


    # select features 
    selected_features = [
        'Flooring_target_mean',
        'ViewYN',
        'PoolPrivateYN',
        'LivingArea_std',
        'AttachedGarageYN',
        'ParkingTotal',
        'Age',
        'BathroomsTotalInteger',
        'BedroomsTotal',
        'FireplaceYN',
        'Levels_final_One',
        'Levels_final_Two',
        'Levels_final_ThreeOrMore',
        'MainLevelBedrooms',
        'NewConstructionYN',
        'GarageSpaces',
        'HighSchoolDistrict_target_mean',
        'LotSizeSquareFeet_std',
        'AssociationFeeFrequency_Monthly',
        'AssociationFeeFrequency_Quarterly',
        'AssociationFeeFrequency_SemiAnnually',
        'Stories_2.0',
        'AssociationFee_std',
        'dist_to_coast_km',
        'Latitude',
        'Longitude'
    ] + zip_dummy_cols


    print(f"Total selected features: {len(selected_features)}")


    # In[69]:


    # prepare train and test
    X_train = new_train_df_clean[selected_features]
    y_train = new_train_df_clean['log_ClosePrice']

    X_test = new_test_df_clean[selected_features]
    y_test = new_test_df_clean['ClosePrice']

    y_test_log  = np.log1p(y_test)

    print("Training features shape (selected only):", X_train.shape)
    print("Test features shape:                  ", X_test.shape)


    # In[70]:


    # Write function to compute MAPE and MdAPE
    def calculate_percentage_errors(y_true, y_pred):
        ape = np.abs((y_true - y_pred) / y_true) * 100
        mape = np.mean(ape) 
        mdape = np.median(ape) 

        return mape, mdape


    # In[71]:


    # fit Linear Regression model
    model_ols = LinearRegression()
    model_ols.fit(X_train, y_train)

    # predict
    y_pred_log_ols = model_ols.predict(X_test)
    y_pred_dollars_ols= np.expm1(y_pred_log_ols)  # back to dollars

    # dollars
    r2_ols_dollars = r2_score(y_test, y_pred_dollars_ols)
    mape_ols_dollars, mdape_ols_dollars = calculate_percentage_errors(y_test, y_pred_dollars_ols)



    # log
    r2_ols_log = r2_score(y_test_log, y_pred_log_ols)

    print(f"R² - dollars:     {r2_ols_dollars:.4f}")
    print(f"MAPE - dollars:   {mape_ols_dollars:.4f}%")
    print(f"MdAPE - dollars:  {mdape_ols_dollars:.4f}%")

    print(f"R² - log:         {r2_ols_log:.4f}")


    # # 3. Modeling: Decision Tree

    # In[72]:


    model_tree = DecisionTreeRegressor(
        max_depth=10,               
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=222)
    model_tree.fit(X_train, y_train)

    y_pred_log_tree = model_tree.predict(X_test)
    y_pred_dollars_tree = np.expm1(y_pred_log_tree)     

    # dollars
    r2_tree_dollars = r2_score(y_test, y_pred_dollars_tree)
    mape_tree_dollars, mdape_tree_dollars = calculate_percentage_errors(y_test, y_pred_dollars_tree)

    # log
    r2_tree_log = r2_score(y_test_log, y_pred_log_tree)  

    print(f"R² - dollars:     {r2_tree_dollars:.4f}")
    print(f"MAPE - dollars:   {mape_tree_dollars:.4f}%")
    print(f"MdAPE - dollars:  {mdape_tree_dollars:.4f}%")

    print(f"R² - log:         {r2_tree_log:.4f}")


    # # 4. Modeling: Random Forest

    # In[23]:


    # Define parameter distributions
    param_dist = {
        'n_estimators': [100, 200, 300, 400, 500, 600, 800, 1000],
        'max_depth': [8, 10, 12, 15, 18, 20, 30, None],
        'min_samples_split': [2, 5, 10, 15, 20, 30],
        'min_samples_leaf': [1, 2, 5, 10, 15, 20],
        'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7, 0.8, None],
        'bootstrap': [True, False]
    }

    # Base model
    rf_base = RandomForestRegressor(random_state=222, n_jobs=-1)

    # Randomized search
    random_search = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=param_dist,
        n_iter=10,                   # 30 random combinations
        cv=5,                        # 5-fold cross-validation
        scoring='r2',                # optimize for R²
        n_jobs=-1,                   
        random_state=222,
        verbose=1
    )

    print("Starting RandomizedSearchCV...")
    random_search.fit(X_train, y_train)

    # Best parameters and CV score
    print("\nBest parameters:")
    print(random_search.best_params_)
    print(f"Best CV R²: {random_search.best_score_:.4f}")




    # In[24]:


    # Save best parameters
    with open('best_rf_params.pkl', 'wb') as f:
        pickle.dump(random_search.best_params_, f)

    print("Best parameters saved to 'best_rf_params.pkl'")
    print("Best params:", random_search.best_params_)


    # In[73]:


    # Load best parameters
    with open('best_rf_params.pkl', 'rb') as f:
       rf_best_params = pickle.load(f)

    print("Loaded best params:", rf_best_params)


    # In[74]:


    # Train final model with best params
    best_rf = RandomForestRegressor(**rf_best_params, random_state=222, n_jobs=-1)
    best_rf.fit(X_train, y_train)

    # Predict on test
    y_pred_rf_log_best         = best_rf.predict(X_test)
    y_pred_rf_dollars_best     = np.expm1(y_pred_rf_log_best)

    # Evaluate
    # dollars
    r2_rf_dollars_best = r2_score(y_test, y_pred_rf_dollars_best)
    mape_rf_dollars, mdape_rf_dollars = calculate_percentage_errors(y_test, y_pred_rf_dollars_best)

    # log
    r2_rf_log_best     = r2_score(y_test_log, y_pred_rf_log_best)


    print(f"R² (dollars):           {r2_rf_dollars_best:.4f}")
    print(f"MAPE - dollars:         {mape_rf_dollars:.4f}%")
    print(f"MdAPE - dollars:        {mdape_rf_dollars:.4f}%")
    print(f"R² (log):               {r2_rf_log_best:.4f}")



    # # 5. Modeling: XGBoost

    # In[77]:


    # Define parameter distributions
    param_dist_xgb = {
        'n_estimators': [100, 200, 300, 400, 500, 600, 800, 1000],
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, None],
        'learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 7, 10],
        'gamma': [0, 0.1, 0.2, 0.3],
        'reg_lambda': [0, 0.01, 0.1, 1.0, 5.0, 10.0, 20.0],   
        'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
    }

    # Base XGBoost model
    xgb_base = XGBRegressor(random_state=222, n_jobs=-1, objective='reg:squarederror')

    # Randomized search
    random_search_xgb = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_dist_xgb,
        n_iter=30,                   
        cv=5,                        
        scoring='r2',                
        n_jobs=-1,
        random_state=222,
        verbose=1
    )

    print("Starting RandomizedSearchCV for XGBoost...")
    random_search_xgb.fit(X_train, y_train)

    # Best parameters and CV score
    print("\nBest parameters:")
    print(random_search_xgb.best_params_)
    print(f"Best CV R²: {random_search_xgb.best_score_:.4f}")



    # In[78]:


    # Save best parameters
    with open('best_xgb_params.pkl', 'wb') as f:
        pickle.dump(random_search_xgb.best_params_, f)

    print("Best parameters saved to 'best_xgb_params.pkl'")
    print("Best params:", random_search_xgb.best_params_)


    # In[79]:


    # Load best parameters
    with open('best_xgb_params.pkl', 'rb') as f:
       xgb_best_params = pickle.load(f)

    print("Loaded best params:", xgb_best_params)


    # In[80]:


    # Train final model with best params
    best_xgb = XGBRegressor(**xgb_best_params, random_state=222, n_jobs=-1)
    best_xgb.fit(X_train, y_train)

    # Predict on test
    y_pred_xgb_log_best         = best_xgb.predict(X_test)
    y_pred_xgb_dollars_best     = np.expm1(y_pred_xgb_log_best)

    # Evaluate
    # dollars
    r2_xgb_dollars_best = r2_score(y_test, y_pred_xgb_dollars_best)
    mape_xgb_dollars, mdape_xgb_dollars = calculate_percentage_errors(y_test, y_pred_xgb_dollars_best)

    # log
    r2_xgb_log_best     = r2_score(y_test_log, y_pred_xgb_log_best)


    print(f"R² (dollars):           {r2_xgb_dollars_best:.4f}")
    print(f"MAPE - dollars:         {mape_xgb_dollars:.4f}%")
    print(f"MdAPE - dollars:        {mdape_xgb_dollars:.4f}%")
    print(f"R² (log):               {r2_xgb_log_best:.4f}")



    # # 6. Modeling: LightGBM

    # In[54]:


    # Define parameters
    param_dist_lgbm = {
        'n_estimators': [100, 200, 300, 400, 500, 600, 800, 1000],
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, None],
        'learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3],
        'num_leaves': [10, 20, 31, 40, 50, 60, 70, 80, 90, 100],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_samples': [10, 20, 30, 40, 50],
        'reg_lambda': [0, 0.01, 0.1, 1.0, 5.0, 10.0, 20.0],   
        'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
    }

    # Base LightGBM model
    lgbm_base = LGBMRegressor(random_state=222, n_jobs=-1, objective='regression', metric='rmse', verbosity=-1)

    # Randomized search
    random_search_lgbm = RandomizedSearchCV(
        estimator=lgbm_base,
        param_distributions=param_dist_lgbm,
        n_iter=30,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        random_state=222,
        verbose=1
    )

    random_search_lgbm.fit(X_train, y_train)

    print("\nBest parameters:", random_search_lgbm.best_params_)
    print(f"Best CV R²: {random_search_lgbm.best_score_:.4f}")


    # In[81]:


    # Save best parameters
    with open('best_lgbm_params.pkl', 'wb') as f:
        pickle.dump(random_search_lgbm.best_params_, f)

    print("Best parameters saved to 'best_lgbm_params.pkl'")
    print("Best params:", random_search_lgbm.best_params_)


    # In[82]:


    # Load best parameters
    with open('best_lgbm_params.pkl', 'rb') as f:
       lgbm_best_params = pickle.load(f)

    print("Loaded best params:", lgbm_best_params)


    # In[84]:


    # Train final model
    best_lgbm = LGBMRegressor(**lgbm_best_params, random_state=222, n_jobs=-1, verbosity=-1)
    best_lgbm.fit(X_train, y_train)

    # Predict on test
    y_pred_lgbm_log_best         = best_lgbm.predict(X_test)
    y_pred_lgbm_dollars_best     = np.expm1(y_pred_lgbm_log_best )

    # Evaluate
    # dollars
    r2_lgbm_dollars_best = r2_score(y_test, y_pred_lgbm_dollars_best)
    mape_lgbm_dollars, mdape_lgbm_dollars = calculate_percentage_errors(y_test, y_pred_lgbm_dollars_best)

    # log
    r2_lgbm_log_best     = r2_score(y_test_log, y_pred_lgbm_log_best)


    print(f"R² (dollars):           {r2_lgbm_dollars_best:.4f}")
    print(f"MAPE - dollars:         {mape_lgbm_dollars:.4f}%")
    print(f"MdAPE - dollars:        {mdape_lgbm_dollars:.4f}%")
    print(f"R² (log):               {r2_lgbm_log_best:.4f}")


    # In[87]:


    # Save the full fitted model
    with open('best_lgbm_model.pkl', 'wb') as f:
        pickle.dump(best_lgbm, f)

    print("Full tuned LightGBM model saved to 'best_lgbm_model.pkl'")


    # In[88]:


    # Mean predicted vs. actual prices
    print(np.mean(y_pred_lgbm_dollars_best))
    print(np.mean(y_test))


    # In[89]:


    # Median predicted vs. actual prices
    print(np.median(y_pred_lgbm_dollars_best))
    print(np.median(y_test))


    # In[90]:


    #import matplotlib.pyplot as plt
    #import seaborn as sns

    #plt.figure(figsize=(14, 6))

    # Actual prices
    #plt.subplot(1, 2, 1)
    #sns.histplot(y_test, bins=50, kde=True, color='lightblue', alpha=0.6)
    #plt.title('Distribution of Actual Close Prices')
    #plt.xlabel('Close Price ($ in millions)')
    #plt.ylabel('Count')
    #plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1000000))


    # Predicted prices
    #plt.subplot(1, 2, 2)
    #sns.histplot(y_pred_lgbm_dollars_best, bins=50, kde=True, color='pink', alpha=0.6)
    #plt.title('Distribution of Predicted Close Prices (LightGBM)')
    #plt.xlabel('Predicted Close Price ($ in millions)')
    #plt.ylabel('Count')
    #plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1000000))


    #plt.tight_layout()
    #plt.show()


    # In[91]:


    # Feature importance from best model
    importances_best_lgbm = pd.Series(
        best_lgbm.booster_.feature_importance(importance_type='gain'),
        index=best_lgbm.booster_.feature_name()
    ).sort_values(ascending=False)

    print("Top 15 Feature Importances (LightGBM - gain):")
    print(importances_best_lgbm.head(15).round(2))


    # In[92]:


    # Evaluate which price band the model performs the best
    price_bins = pd.cut(y_test, bins=[0, 500000, 1000000, 2000000, 5000000, np.inf],
                        labels=['< $500k', '$500k–1M', '$1M–2M', '$2M–5M', '> $5M'])

    price_bins


    # In[95]:


    df_eval = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred_lgbm_dollars_best,
        'price_band': price_bins
    })

    # Function to compute metrics per group
    def compute_metrics(group):
        y_true = group['actual']
        y_pred = group['predicted']
        return pd.Series({
            'R²': r2_score(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'MdAPE': np.median(np.abs((y_true - y_pred) / y_true)) * 100,
            'Count': len(y_true)
        }).round(4)

    # Group and compute
    metrics_by_band = df_eval.groupby('price_band').apply(compute_metrics).reset_index()

    print("Performance by Price Band:")
    print(metrics_by_band)


    # # 7. Stacking

    # In[85]:


    # Simple equal-weight average
    y_pred_log_stack = (y_pred_lgbm_log_best + y_pred_xgb_log_best + y_pred_rf_log_best) / 3

    # Weighted average
    weights = [0.8, 0.1, 0.1]  # e.g., LightGBM highest weight
    y_pred_log_stack_weighted = (
        weights[0] * y_pred_lgbm_log_best +
        weights[1] * y_pred_xgb_log_best +
        weights[2] * y_pred_rf_log_best
    )

    # Convert to dollars
    y_pred_dollars_stack = np.expm1(y_pred_log_stack)
    y_pred_dollars_stack_weighted = np.expm1(y_pred_log_stack_weighted)

    # Evaluate
    # dollars
    r2_stack_dollars = r2_score(y_test, y_pred_dollars_stack)
    r2_stack_dollars_weighted = r2_score(y_test, y_pred_dollars_stack_weighted)

    mape_stack_dollars, mdape_stack_dollars = calculate_percentage_errors(y_test, y_pred_dollars_stack)
    mape_stack_dollars_weighted, mdape_stack_dollars_weighted = calculate_percentage_errors(y_test, y_pred_dollars_stack_weighted)

    #log
    r2_stack_log = r2_score(y_test_log, y_pred_log_stack)
    r2_stack_log_weighted = r2_score(y_test_log, y_pred_log_stack_weighted)


    print("Simple Averaging Stack")
    print(f"R² (dollars): {r2_stack_dollars:.4f}")
    print(f"MAPE: {mape_stack_dollars:.4f}%")
    print(f"MdAPE: {mdape_stack_dollars:.4f}%")
    print(f"R² (log): {r2_stack_log:.4f}")

    print("Weighted Averaging Stack")
    print(f"R² (dollars): {r2_stack_dollars_weighted:.4f}")
    print(f"MAPE: {mape_stack_dollars_weighted:.4f}%")
    print(f"MdAPE: {mdape_stack_dollars_weighted:.4f}%")
    print(f"R² (log): {r2_stack_log_weighted:.4f}")



    # # 8. Summary: Model Performance

    # In[99]:


    data = {
        'Model': ['OLS', 'Decision Tree', 'Random Forest', 'XGBoost', 'LightGBM', 'Stacking', 'Stacking (weighted)'],
        'R² (dollars)': [r2_ols_dollars, r2_tree_dollars, r2_rf_dollars_best, r2_xgb_dollars_best,r2_lgbm_dollars_best, r2_stack_dollars, r2_stack_dollars_weighted],
        'R² (log)': [r2_ols_log, r2_tree_log, r2_rf_log_best,r2_xgb_log_best, r2_lgbm_log_best, r2_stack_log, r2_stack_log_weighted],
        'MAPE (%)': [mape_ols_dollars, mape_tree_dollars, mape_rf_dollars, mape_xgb_dollars, mape_lgbm_dollars, mape_stack_dollars, mape_stack_dollars_weighted],
        'MdAPE (%)': [mdape_ols_dollars, mdape_tree_dollars, mdape_rf_dollars, mdape_xgb_dollars, mdape_lgbm_dollars, mdape_stack_dollars, mdape_stack_dollars_weighted]
    }

    # Create DataFrame
    metrics_table = pd.DataFrame(data)

    # Round numbers
    metrics_table['R² (dollars)'] = metrics_table['R² (dollars)'].round(4)
    metrics_table['R² (log)'] = metrics_table['R² (log)'].round(4)
    metrics_table['MAPE (%)'] = metrics_table['MAPE (%)'].round(4)
    metrics_table['MdAPE (%)'] = metrics_table['MdAPE (%)'].round(4)

    # Display table
    print("Model Comparison")
    print(metrics_table)  # renders as HTML table in Colab/Jupyter

# ────────────────────────────────────────────────
# Define tasks
# ────────────────────────────────────────────────

download_task = PythonOperator(
    task_id='download_new_mls_data',
    python_callable=download_new_mls_data,
    dag=dag,
)

preprocess_task = PythonOperator(
    task_id='run_preprocessing',
    python_callable=run_preprocessing,  # ← this is your function from 01_preprocessing.py
    dag=dag,
)

model_task = PythonOperator(
    task_id='run_modeling',
    python_callable=run_modeling,
    dag=dag,
)

# Chain the tasks: download before preprocessing starts
download_task >> preprocess_task >> model_task
    
    
    
    
