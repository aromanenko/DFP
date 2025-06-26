import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Optional


def calculate_period_end_dt(period_dt: pd.Series, time_level: str) -> pd.Series:
    """Calculate period end date based on time level."""
    if time_level == 'day':
        return period_dt
    elif time_level in ['week', 'week.2']:
        return period_dt + pd.offsets.Week(weekday=6)
    elif time_level == 'MONTH':
        return period_dt + pd.offsets.MonthEnd(0)
    else:
        raise ValueError(f"Unknown time_level: {time_level}")


def number_days(time_lvl: str, period_dt: pd.Timestamp) -> int:
    """Calculate number of days in a period based on time level."""
    if time_lvl == 'day':
        return 1
    elif time_lvl in ['week', 'week.2']:
        return 7
    elif time_lvl == 'MONTH':
        return period_dt.days_in_month
    else:
        raise ValueError(f"Unknown time_lvl: {time_lvl}")


def create_mid_term_forecast(ts_forecast: pd.DataFrame, hist_end_dt: datetime, 
                           delays_config_length: int) -> pd.DataFrame:
    """Create mid-term forecast from TS forecast data."""
    cutoff_date = hist_end_dt + timedelta(days=delays_config_length)
    
    mid_term = ts_forecast[ts_forecast['PERIOD_DT'] > cutoff_date].copy()
    
    if len(mid_term) > 0:
        mid_term['PERIOD_END_DT'] = mid_term['PERIOD_DT']  # PERIOD_DT as PERIOD_END_DT
        mid_term['TS_FORECAST_VALUE_REC'] = mid_term['FORECAST_VALUE']  # FORECAST_VALUE as TS_FORECAST_VALUE_REC
        mid_term['ML_FORECAST_VALUE'] = np.nan  # missing as ML_FORECAST_VALUE
        mid_term['DEMAND_TYPE'] = 'Regular'
        mid_term['ASSORTMENT_TYPE'] = 'old'
        
        # Add hierarchy columns if not present
        for col in ['product_lvl_id', 'location_lvl_id', 'customer_lvl_id', 'distr_channel_lvl_id']:
            if col not in mid_term.columns:
                if col == 'product_lvl_id':
                    mid_term[col] = mid_term.get('PRODUCT_ID', 1)
                elif col == 'location_lvl_id':
                    mid_term[col] = mid_term.get('LOCATION_ID', 1)
                elif col == 'customer_lvl_id':
                    mid_term[col] = mid_term.get('CUSTOMER_ID', 1)
                elif col == 'distr_channel_lvl_id':
                    mid_term[col] = mid_term.get('DISTR_CHANNEL_ID', 1)
        
        # Remove original FORECAST_VALUE column
        mid_term = mid_term.drop(columns=['FORECAST_VALUE'], errors='ignore')
    
    return mid_term


def join_ts_ml_forecasts(ts_forecast: pd.DataFrame, ml_forecast: pd.DataFrame,
                        hist_end_dt: datetime, delays_config_length: int,
                        ml_time_lvl: str = 'week.2', ts_time_lvl: str = 'day') -> pd.DataFrame:
    """Join TS and ML forecasts for the short-term horizon."""
    
    # Filter to short-term horizon
    cutoff_date = hist_end_dt + timedelta(days=delays_config_length)
    ts_short = ts_forecast[ts_forecast['PERIOD_DT'] <= cutoff_date].copy()
    ml_short = ml_forecast[ml_forecast['PERIOD_DT'] <= cutoff_date].copy()
    
    if len(ts_short) == 0 or len(ml_short) == 0:
        return pd.DataFrame()
    
    # Add PERIOD_END_DT
    ts_short['PERIOD_END_DT'] = calculate_period_end_dt(ts_short['PERIOD_DT'], ts_time_lvl)
    ml_short['PERIOD_END_DT'] = calculate_period_end_dt(ml_short['PERIOD_DT'], ml_time_lvl)
    
    # Rename forecast value columns for clarity
    ts_short = ts_short.rename(columns={'FORECAST_VALUE': 'TS_FORECAST_VALUE'})
    ml_short = ml_short.rename(columns={'FORECAST_VALUE_total': 'ML_FORECAST_VALUE'})
    
    # Perform overlap join
    joined_data = []
    
    for _, ml_row in ml_short.iterrows():
        for _, ts_row in ts_short.iterrows():
            # Check hierarchy matching and time overlap
            if (ml_row.get('PRODUCT_ID') == ts_row.get('PRODUCT_ID') and
                ml_row.get('LOCATION_ID') == ts_row.get('LOCATION_ID') and
                ml_row.get('CUSTOMER_ID', 1) == ts_row.get('CUSTOMER_ID', 1) and
                ml_row.get('DISTR_CHANNEL_ID', 1) == ts_row.get('DISTR_CHANNEL_ID', 1) and
                # Time overlap condition: ML.PERIOD_DT<=TS.PERIOD_END_DT and ML.PERIOD_END_DT>=TS.PERIOD_DT
                ml_row['PERIOD_DT'] <= ts_row['PERIOD_END_DT'] and
                ml_row['PERIOD_END_DT'] >= ts_row['PERIOD_DT']):
                
                # Calculate overlap period
                overlap_start = max(ml_row['PERIOD_DT'], ts_row['PERIOD_DT'])
                overlap_end = min(ml_row['PERIOD_END_DT'], ts_row['PERIOD_END_DT'])
                
                # Calculate proportional values
                overlap_days = (overlap_end - overlap_start).days + 1
                
                # TS_FORECAST_VALUE = TS_FORECAST_VALUE * [overlap_days] / number_days(ts_time_lvl, PERIOD_DT)
                ts_period_days = number_days(ts_time_lvl, ts_row['PERIOD_DT'])
                ts_proportional = ts_row['TS_FORECAST_VALUE'] * (overlap_days / ts_period_days)
                
                # ML_FORECAST_VALUE = ML_FORECAST_VALUE * [overlap_days] / number_days(ml_time_lvl, PERIOD_DT)
                ml_period_days = number_days(ml_time_lvl, ml_row['PERIOD_DT'])
                ml_proportional = ml_row['ML_FORECAST_VALUE'] * (overlap_days / ml_period_days)
                
                joined_data.append({
                    'product_lvl_id': ml_row.get('PRODUCT_ID'),
                    'location_lvl_id': ml_row.get('LOCATION_ID'),
                    'customer_lvl_id': ml_row.get('CUSTOMER_ID', 1),
                    'distr_channel_lvl_id': ml_row.get('DISTR_CHANNEL_ID', 1),
                    'PERIOD_DT': overlap_start,
                    'PERIOD_END_DT': overlap_end,
                    'TS_FORECAST_VALUE': ts_proportional,
                    'ML_FORECAST_VALUE': ml_proportional,
                    'DEMAND_TYPE': ml_row.get('DEMAND_TYPE', 'regular'),
                    'ASSORTMENT_TYPE': ml_row.get('ASSORTMENT_TYPE', 'old')
                })
    
    result_df = pd.DataFrame(joined_data)
    
    # Replace missing values with zero
    if len(result_df) > 0:
        result_df = result_df.copy()
        result_df['TS_FORECAST_VALUE'] = result_df['TS_FORECAST_VALUE'].fillna(0)
        result_df['ML_FORECAST_VALUE'] = result_df['ML_FORECAST_VALUE'].fillna(0)
    
    return result_df


def reconcile_ml_ts_forecast(joined_df: pd.DataFrame) -> pd.DataFrame:
    """Reconcile ML and TS Forecast using the reconciliation algorithm."""
    if len(joined_df) == 0:
        return joined_df
        
    result_df = joined_df.copy()
    
    # Group by the specified dimensions
    grouping_cols = ['product_lvl_id', 'location_lvl_id', 'customer_lvl_id', 'distr_channel_lvl_id', 'PERIOD_DT']
    
    def reconcile_group(group):
        ml_total = group['ML_FORECAST_VALUE'].sum()
        ts_total = group['TS_FORECAST_VALUE'].sum()
        
        if ts_total > 0:
            # TS_FORECAST_VALUE_REC = (TS_FORECAST_VALUE / ts_total) * ml_total
            group['TS_FORECAST_VALUE_REC'] = (group['TS_FORECAST_VALUE'] / ts_total) * ml_total
        else:
            # If no TS forecast, set reconciled value to 0
            group['TS_FORECAST_VALUE_REC'] = 0
            
        return group
    
    result_df = result_df.groupby(grouping_cols, group_keys=False).apply(reconcile_group)
    
    return result_df


def add_segment_name(reconciled_df: pd.DataFrame, segments_df: pd.DataFrame) -> pd.DataFrame:
    """Add segment name to reconciled forecast data."""
    if len(reconciled_df) == 0:
        return reconciled_df
        
    # Perform left join to add SEGMENT_NAME
    join_cols = ['product_lvl_id', 'location_lvl_id', 'customer_lvl_id', 'distr_channel_lvl_id']
    
    # Ensure segments_df has the right column names
    segments_join = segments_df.copy()
    if 'PRODUCT_ID' in segments_join.columns:
        segments_join = segments_join.rename(columns={
            'PRODUCT_ID': 'product_lvl_id',
            'LOCATION_ID': 'location_lvl_id',
            'CUSTOMER_ID': 'customer_lvl_id',
            'DISTR_CHANNEL_ID': 'distr_channel_lvl_id'
        })
    
    # Add missing hierarchy levels if not present
    for col in join_cols:
        if col not in segments_join.columns:
            if col == 'customer_lvl_id':
                segments_join[col] = 1  # Default customer level
            elif col == 'distr_channel_lvl_id':
                segments_join[col] = 1  # Default distribution channel level
    
    # Perform the join
    result_df = pd.merge(reconciled_df, segments_join[join_cols + ['SEGMENT_NAME']], 
                        on=join_cols, how='left')
    
    return result_df


def forecast_reconciliation_pipeline(ts_forecast: pd.DataFrame, 
                                   ml_forecast: pd.DataFrame,
                                   segments_df: pd.DataFrame,
                                   hist_end_dt: datetime,
                                   delays_config_length: int = 30,
                                   ml_time_lvl: str = 'week.2',
                                   ts_time_lvl: str = 'day') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Complete forecast reconciliation pipeline."""
    
    # Create mid-term forecast
    mid_term_forecast = create_mid_term_forecast(ts_forecast, hist_end_dt, delays_config_length)
    
    # Join TS and ML forecasts
    joined = join_ts_ml_forecasts(ts_forecast, ml_forecast, hist_end_dt, delays_config_length,
                                 ml_time_lvl, ts_time_lvl)
    
    # Reconcile ML and TS Forecast
    reconciled = reconcile_ml_ts_forecast(joined)
    
    # Add Segment Name
    final_reconciled = add_segment_name(reconciled, segments_df)
    
    return final_reconciled, mid_term_forecast 