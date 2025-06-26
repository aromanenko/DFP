import pandas as pd
import numpy as np
from typing import Tuple, Optional


def safe_average(val1, val2):
    """
    Calculate average of two values while handling missing values (NaN).
    
    Returns val1 if val2 is NaN, val2 if val1 is NaN, their average if both exist,
    or NaN if both are missing. Used for combining forecasts when some may be unavailable.
    """
    val1_clean = val1 if pd.notna(val1) else np.nan
    val2_clean = val2 if pd.notna(val2) else np.nan
    
    if pd.isna(val1_clean) and pd.isna(val2_clean):
        return np.nan
    elif pd.isna(val1_clean):
        return val2_clean
    elif pd.isna(val2_clean):
        return val1_clean
    else:
        return (val1_clean + val2_clean) / 2


def coalesce(val1, val2):
    """
    Return the first non-null value between two inputs.
    
    Similar to SQL COALESCE - returns val1 if not null, otherwise val2.
    Used for filling gaps in forecast data with backup values.
    """
    return val1 if pd.notna(val1) else val2


def create_mid_term_hybrid_forecast(mid_reconciled_forecast: pd.DataFrame) -> pd.DataFrame:
    """
    Create mid-term hybrid forecast from reconciled TS forecast data.
    
    For mid-term periods (beyond short-term ML forecast coverage):
    - Use only TS forecasts as the source (ML forecasts not available)
    - Set FORECAST_SOURCE to 'ts' since only TS data is available  
    - Set ENSEMBLE_FORECAST_VALUE to NaN (no ensemble possible with single source)
    - Rename TS_FORECAST_VALUE_REC to TS_FORECAST_VALUE for consistency
    """
    if len(mid_reconciled_forecast) == 0:
        return pd.DataFrame()
    
    # Create copy to avoid modifying original data
    mid_hybrid = mid_reconciled_forecast.copy()
    
    # Rename reconciled TS value to standard TS forecast column
    mid_hybrid = mid_hybrid.rename(columns={'TS_FORECAST_VALUE_REC': 'TS_FORECAST_VALUE'})
    
    # For mid-term: hybrid forecast = TS forecast (only source available)
    mid_hybrid['HYBRID_FORECAST_VALUE'] = mid_hybrid['TS_FORECAST_VALUE']
    
    # No ensemble possible with single source, and ML not available for mid-term
    mid_hybrid['ENSEMBLE_FORECAST_VALUE'] = np.nan
    
    # Source is always TS for mid-term (ML forecasts don't extend this far)
    mid_hybrid['FORECAST_SOURCE'] = 'ts'
    
    # Ensure all required output columns are present
    required_cols = [
        'product_lvl_id', 'location_lvl_id', 'customer_lvl_id', 'distr_channel_lvl_id',
        'PERIOD_DT', 'PERIOD_END_DT', 'TS_FORECAST_VALUE', 'SEGMENT_NAME',
        'DEMAND_TYPE', 'ASSORTMENT_TYPE', 'ML_FORECAST_VALUE',
        'HYBRID_FORECAST_VALUE', 'ENSEMBLE_FORECAST_VALUE', 'FORECAST_SOURCE'
    ]
    
    # Fill in any missing columns with sensible defaults
    for col in required_cols:
        if col not in mid_hybrid.columns:
            if col in ['SEGMENT_NAME', 'DEMAND_TYPE', 'ASSORTMENT_TYPE']:
                mid_hybrid[col] = np.nan
            elif col == 'ML_FORECAST_VALUE':
                mid_hybrid[col] = np.nan  # ML forecasts not available for mid-term
            else:
                mid_hybrid[col] = np.nan
    
    return mid_hybrid[required_cols]


def join_ts_ml_forecasts(reconciled_forecast: pd.DataFrame, 
                        ib_zero_demand_threshold: float = 0.01) -> pd.DataFrame:
    """
    Join TS and ML forecasts and apply business rules for hybrid forecast generation.
    
    Implements business rules to determine which forecast source to use:
    
    ML Priority: Use ML forecast when:
    - Promotional demand (not retired products)
    - Short lifecycle segments  
    - New assortment products
    
    TS Priority: Use TS forecast when:
    - Retired products
    - Low volume segments
    - Zero/low demand cases (forecast <= threshold)
    
    Ensemble: Use average of TS and ML for all other cases
    """
    if len(reconciled_forecast) == 0:
        return pd.DataFrame()
    
    result_df = reconciled_forecast.copy()
    
    # Create filled versions of forecast values using coalesce logic
    # TS_FORECAST_VALUE_F: Use TS if available, fallback to ML
    result_df['TS_FORECAST_VALUE_F'] = result_df.apply(
        lambda row: coalesce(row.get('TS_FORECAST_VALUE_REC'), row.get('ML_FORECAST_VALUE')), 
        axis=1
    )
    # ML_FORECAST_VALUE_F: Use ML if available, fallback to TS
    result_df['ML_FORECAST_VALUE_F'] = result_df.apply(
        lambda row: coalesce(row.get('ML_FORECAST_VALUE'), row.get('TS_FORECAST_VALUE_REC')), 
        axis=1
    )
    
    def apply_business_rules(row):
        """
        Apply business rules to determine forecast source and value for each row.
        
        Returns pandas Series with hybrid forecast value, source, and ensemble value.
        """
        ts_val = row['TS_FORECAST_VALUE_F']
        ml_val = row['ML_FORECAST_VALUE_F']
        demand_type = str(row.get('DEMAND_TYPE', '')).lower()
        segment_name = str(row.get('SEGMENT_NAME', '')).lower()
        assortment_type = str(row.get('ASSORTMENT_TYPE', '')).lower()
        
        # Rule 1: ML Priority Conditions
        ml_priority = (
            (demand_type == 'promo' and segment_name != 'retired') or  # Promo but not retired
            segment_name == 'short' or                                # Short lifecycle
            assortment_type == 'new'                                   # New products
        )
        
        # Rule 2: TS Priority Conditions  
        ts_priority = (
            segment_name == 'retired' or                              # Retired products
            segment_name == 'low volume' or                           # Low volume segments
            (pd.notna(ts_val) and ts_val <= ib_zero_demand_threshold) # Zero/low demand
        )
        
        # Apply business rules in priority order
        if ml_priority:
            # Use ML forecast for promotional, short lifecycle, or new products
            hybrid_value = ml_val
            forecast_source = 'ml'
            ensemble_value = np.nan  # No ensemble when using single source
        elif ts_priority:
            # Use TS forecast for retired, low volume, or zero demand cases
            hybrid_value = ts_val
            forecast_source = 'ts' 
            ensemble_value = np.nan  # No ensemble when using single source
        else:
            # Use ensemble (average) for all other cases
            hybrid_value = safe_average(ts_val, ml_val)
            forecast_source = 'ensemble'
            ensemble_value = safe_average(ts_val, ml_val)  # Same as hybrid for ensemble
        
        return pd.Series({
            'HYBRID_FORECAST_VALUE': hybrid_value,
            'FORECAST_SOURCE': forecast_source,
            'ENSEMBLE_FORECAST_VALUE': ensemble_value
        })
    
    # Apply business rules to each row
    business_rules_result = result_df.apply(apply_business_rules, axis=1)
    result_df = pd.concat([result_df, business_rules_result], axis=1)
    
    # Clean up: rename and remove temporary columns
    result_df = result_df.rename(columns={'TS_FORECAST_VALUE_REC': 'TS_FORECAST_VALUE'})
    result_df = result_df.drop(columns=['TS_FORECAST_VALUE_F', 'ML_FORECAST_VALUE_F'], errors='ignore')
    
    return result_df


def hybrid_forecast_generation_pipeline(reconciled_forecast: pd.DataFrame,
                                       mid_reconciled_forecast: pd.DataFrame = None,
                                       ib_zero_demand_threshold: float = 0.01) -> pd.DataFrame:
    """
    Complete hybrid forecast generation pipeline combining short-term and mid-term forecasts.
    
    Pipeline process:
    1. Apply business rules to short-term data (TS + ML) 
    2. Process mid-term data (TS only, beyond ML coverage)
    3. Combine both periods into unified hybrid forecast
    4. Ensure consistent column structure across all periods
    """
    
    # Step 1: Generate mid-term hybrid forecast (TS-only, beyond ML coverage)
    mid_term_hybrid = pd.DataFrame()
    if mid_reconciled_forecast is not None and len(mid_reconciled_forecast) > 0:
        mid_term_hybrid = create_mid_term_hybrid_forecast(mid_reconciled_forecast)
    
    # Step 2: Generate short-term hybrid forecast (TS+ML with business rules)
    short_term_hybrid = join_ts_ml_forecasts(reconciled_forecast, ib_zero_demand_threshold)
    
    # Step 3: Combine short-term and mid-term forecasts if both exist
    if len(mid_term_hybrid) > 0:
        # Ensure both dataframes have the same columns for concatenation
        for col in short_term_hybrid.columns:
            if col not in mid_term_hybrid.columns:
                mid_term_hybrid[col] = np.nan
        
        for col in mid_term_hybrid.columns:
            if col not in short_term_hybrid.columns:
                short_term_hybrid[col] = np.nan
        
        # Align column order for consistent output
        column_order = short_term_hybrid.columns.tolist()
        mid_term_hybrid = mid_term_hybrid[column_order]
        
        # Combine short-term and mid-term forecasts
        hybrid_forecast = pd.concat([short_term_hybrid, mid_term_hybrid], 
                                  ignore_index=True, sort=False)
    else:
        # Use only short-term forecast if no mid-term data
        hybrid_forecast = short_term_hybrid
    
    # Step 4: Ensure all required output columns are present with proper defaults
    required_output_cols = [
        'product_lvl_id', 'location_lvl_id', 'customer_lvl_id', 'distr_channel_lvl_id',
        'PERIOD_DT', 'PERIOD_END_DT', 'TS_FORECAST_VALUE', 'SEGMENT_NAME',
        'DEMAND_TYPE', 'ASSORTMENT_TYPE', 'ML_FORECAST_VALUE',
        'HYBRID_FORECAST_VALUE', 'ENSEMBLE_FORECAST_VALUE', 'FORECAST_SOURCE'
    ]
    
    # Fill any gaps with NaN for missing columns
    for col in required_output_cols:
        if col not in hybrid_forecast.columns:
            hybrid_forecast[col] = np.nan
    
    return hybrid_forecast[required_output_cols] 