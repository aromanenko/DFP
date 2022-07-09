import pandas as pd
import numpy as np
import datetime
from calendar import monthrange
import reconcilation
import warnings
warnings.filterwarnings('ignore')


def generate_data(config : dict, IB_HIST_END_DT : datetime.datetime, hierarchies : dict) -> (pd.DataFrame, pd.DataFrame):
    """
    Function generating input data
    
    Parameters
    ----------
    config : dict
        Configuration parameters used within the step
    IB_HIST_END_DT
        Last known date (i.e. sales and stock information is known)
    hierarchies
        Dictionary containg matches of key names with the relevant hierarchical tables
       
    Returns
    -------
    pd.DataFrame
        VF_FORECAST used as input data of the algorithm
    pd.DataFrame
        ML_FORECAST used as input data of the algorithm
    pd.DataFrame
        VF_TS_SEGMENTS containg information about segment names
    """
    freq = config['vf_time_lvl'][0]
    timerange = pd.date_range(IB_HIST_END_DT, IB_HIST_END_DT + datetime.timedelta(weeks=52), freq=freq)
    timerange += datetime.timedelta(1)
    VF_FORECAST = pd.DataFrame(timerange, columns=['PERIOD_DT'])
    for key in ['product', 'customer', 'location', 'distr_channel']:
        column_name = f"{key}_lvl_id{config[f'vf_{key}_lvl']}"
        keys_df = pd.DataFrame(hierarchies[key][column_name]).drop_duplicates()
        VF_FORECAST = pd.merge(VF_FORECAST, keys_df, 'cross')
    VF_FORECAST['FORECAST_VALUE'] = np.abs(np.random.normal(500, 300, VF_FORECAST.shape[0]))
    
    freq = config['ml_time_lvl'][0]
    timerange = pd.date_range(IB_HIST_END_DT, IB_HIST_END_DT + datetime.timedelta(weeks=8), freq=freq)
    timerange += datetime.timedelta(1)
    ML_FORECAST = pd.DataFrame(timerange, columns=['PERIOD_DT'])
    for key in ['product', 'customer', 'location', 'distr_channel']:
        column_name = f"{key}_lvl_id{config[f'ml_{key}_lvl']}"
        keys_df = pd.DataFrame(hierarchies[key][column_name])
        ML_FORECAST = pd.merge(ML_FORECAST, keys_df, 'cross')    
    ML_FORECAST['FORECAST_VALUE'] = np.abs(np.random.normal(500, 300, ML_FORECAST.shape[0]))
    ML_FORECAST['DEMAND_TYPE'] = np.random.choice(['promo', 'regular'], ML_FORECAST.shape[0])
    ML_FORECAST['ASSORTMENT_TYPE'] = np.random.choice(['new', 'old'], ML_FORECAST.shape[0])
    ML_FORECAST['FORECAST_VALUE'] = np.abs(np.random.normal(500, 300, ML_FORECAST.shape[0]))
    
    VF_TS_SEGMENTS = VF_FORECAST.loc[:, VF_FORECAST.columns.str.contains('lvl_id')].drop_duplicates()
    VF_TS_SEGMENTS['SEGMENT_NAME'] = "Name of segment " + VF_TS_SEGMENTS.index.astype(str)
    
    return VF_FORECAST, ML_FORECAST, VF_TS_SEGMENTS
    
    
def periods_splitting(VF_FORECAST : pd.DataFrame, IB_HIST_END_DT : datetime.datetime,
                      delays_config_length : int) -> (pd.DataFrame, pd.DataFrame):
    """
    Step 3.1.a and 3.1.b
    
    Splitting periods into short-term and mid-term
    
    Parameters
    ----------
    VF_FORECAST : pd.DataFrame
        Input table containing the keys columns, PERIOD_DT column, FORECAST_VALUE column.
    IB_HIST_END_DT : datetime.datetime
        Last known date (i.e. sales and stock information is known)
    delays_config_length : int
        Lenght of short-term forecasting period
    
    Returns
    -------
    pd.DataFrame
        Short-term period of the VF_FORECAST
    pd.DataFrame
        Mid-term period of the VF_FORECAST
    """
    mid_period = (VF_FORECAST['PERIOD_DT'] - pd.to_datetime(IB_HIST_END_DT)).dt.days > delays_config_length
    mid_reconciled_forecast = VF_FORECAST[mid_period].copy(deep=True)
    mid_reconciled_forecast = mid_reconciled_forecast.rename(
        columns={'PERIOD_DT' : 'PERIOD_END_DT', 'FORECAST_VALUE' : 'VF_FORECAST_VALUE_REC'}
    )
    mid_reconciled_forecast['ML_FORECAST_VALUE'] = None
    mid_reconciled_forecast['DEMAND_TYPE'] = 'Regular'
    mid_reconciled_forecast['ASSORTMENT_TYPE'] = 'old'
    
    return VF_FORECAST[~mid_period], mid_reconciled_forecast


def add_period_ends(VF_FORECAST : pd.DataFrame, ML_FORECAST : pd.DataFrame, config : dict) -> (pd.DataFrame, pd.DataFrame):
    """
    Step 3.2
    Adding ends of forecasting periods
    
    Parameters
    ----------
    VF_FORECAST : pd.DataFrame
        Input table containing the keys columns, PERIOD_DT column, FORECAST_VALUE column.
    ML_FORECAST : pd.DataFrame
        Input table containing the keys columns, PERIOD_DT column,
        FORECAST_VALUE column, columns with types of DEMAND and ASSORTMENT.
    config : dict
        Configuration parameters used within the step
        
    
    Returns
    -------
    pd.DataFrame
        VF_FORECAST with added periods ends
    pd.DataFrame
        MF_FORECAST with added periods ends
    
    """
    VF_FORECAST['PERIOD_END_DT'] =  VF_FORECAST['PERIOD_DT'].apply(
        lambda x : pd.date_range(x, periods=1, freq=config['vf_time_lvl'][0])[0]
    )

    ML_FORECAST['PERIOD_END_DT'] =  ML_FORECAST['PERIOD_DT'].apply(
        lambda x : pd.date_range(x, periods=1, freq=config['ml_time_lvl'][0])[0]
    )
    return VF_FORECAST, ML_FORECAST


def match_keys(VF_FORECAST : pd.DataFrame, ML_FORECAST : pd.DataFrame,
               config : dict, hierarchies : dict) -> (pd.DataFrame, pd.DataFrame):
    
    """
    Step 3.3 and 3.4
    JOIN VF and ML keys to the hierachies
    
    Parameters
    ----------
    VF_FORECAST : pd.DataFrame
        Input table containing the keys columns, PERIOD_DT column, FORECAST_VALUE column.
    ML_FORECAST : pd.DataFrame
        Input table containing the keys columns, PERIOD_DT column,
        FORECAST_VALUE column, columns with types of DEMAND and ASSORTMENT.
    config : dict
        Configuration parameters used within the step
    hierarchies : dict
        Dictionary containg matches of key names with the relevant hierarchical tables
        
    
    Returns
    -------
    pd.DataFrame
        VF_FORECAST
    pd.DataFrame
        MF_FORECAST
    
    """
    for key in ['product', 'customer', 'location', 'distr_channel']:
        ml_column_name = f"{key}_lvl_id{config[f'ml_{key}_lvl']}"
        vf_column_name = f"{key}_lvl_id{config[f'vf_{key}_lvl']}"
        if ml_column_name == vf_column_name:
            continue
        ML_FORECAST = pd.merge(
            ML_FORECAST, hierarchies[key][[ml_column_name, vf_column_name]].drop_duplicates(),
            on=ml_column_name, how='left'
        )
        VF_FORECAST = pd.merge(
            VF_FORECAST, hierarchies[key][[ml_column_name, vf_column_name]].drop_duplicates(),
            on=vf_column_name, how='left'
        )
    ML_FORECAST = ML_FORECAST.rename(columns={'FORECAST_VALUE' : 'ML_FORECAST_VALUE'})
    VF_FORECAST = VF_FORECAST.rename(columns={'FORECAST_VALUE' : 'VF_FORECAST_VALUE'})
    return VF_FORECAST, ML_FORECAST


def match_forecasts(VF_FORECAST : pd.DataFrame, ML_FORECAST : pd.DataFrame,
                    IB_HIST_END_DT : datetime.datetime) -> pd.DataFrame:
    """
    Step 3.5
    JOIN VF to ML. Transforming period columns.
    
    Parameters
    ----------
    VF_FORECAST : pd.DataFrame
        Input table containing the keys columns, PERIOD_DT column, FORECAST_VALUE column.
    IB_HIST_END_DT : datetime.datetime
        Last known date (i.e. sales and stock information is known)
    
    Returns
    -------
    pd.DataFrame
        Table containing joint vf and ml forecasts
    
    """
    #using not only max but also min because other hierarchical columns was added earlier
    merge_keys = ML_FORECAST.columns[ML_FORECAST.columns.str.contains('lvl_id')].tolist()
    df = pd.merge(ML_FORECAST, VF_FORECAST, on=merge_keys, how='left', suffixes=['_ML', '_VF'])
    df = df[(df['PERIOD_DT_ML'] <= df['PERIOD_END_DT_VF']) & (df['PERIOD_END_DT_ML'] >= df['PERIOD_DT_VF'])]
    df = df[(df['PERIOD_DT_ML'] > IB_HIST_END_DT) & (df['PERIOD_DT_VF'] > IB_HIST_END_DT)]
    df['PERIOD_DT'] = np.maximum(df['PERIOD_DT_ML'], df['PERIOD_DT_VF'])
    df['PERIOD_END_DT'] = np.minimum(df['PERIOD_END_DT_ML'], df['PERIOD_END_DT_VF'])
    df['VF_FORECAST_VALUE'] = df['VF_FORECAST_VALUE'].fillna(0)
    df = df.drop(['PERIOD_DT_ML', 'PERIOD_DT_VF', 'PERIOD_END_DT_ML', 'PERIOD_END_DT_VF'], axis=1)
    df = df.reset_index(drop=True)
    return df


def number_days(time_lvl : str, period_dt : datetime.datetime) -> int:
    """
    Function that calculate days number in a time_lvl interval that contains PERIOD_DT
    
    Parameters
    ----------
    time_lvl : {'DAY', 'WEEK', 'MONTH'}
        vf_time_lvl or ml_time_lvl from config
    period_dt : datetime.datetime
        The start of forecasting period of one object
        
    Returns
    -------
    int
       Days number in a time_lvl interval that contains PERIOD_DT 
    """
    if time_lvl == 'DAY':
        return 1
    if time_lvl == 'WEEK':
        return 7
    if time_lvl == 'MONTH':
        return monthrange(period_dt.year, period_dt.month)[1]


def interval_forecast_correction(df : pd.DataFrame, config : dict) -> pd.DataFrame:
    """
    Step 3.6
    Calculate forecast share and volume of VF_FORECAST_VALUE and ML_FORECAST_VALUE
    proportionaly to number of day in interval [PERIOD_DT, PERIOD_END_DT]
    
    Parameters
    ----------
    df : pd.DataFrame
        The forecasts table obtained in the previous steps of the algorithm
    config : dict
        Configuration parameters used within the step
    Returns
    -------
    pd.DataFrame
        The forecasts table with shared and volumed forecasts
    """
    number_days_vf = df['PERIOD_DT'].apply(lambda x : number_days(config['vf_time_lvl'], x))
    number_days_ml = df['PERIOD_DT'].apply(lambda x : number_days(config['ml_time_lvl'], x))
    period_len = (df['PERIOD_END_DT'] - df['PERIOD_DT']).dt.days + 1
    df['ML_FORECAST_VALUE'] *= (period_len / number_days_ml)
    df['VF_FORECAST_VALUE'] *= (period_len / number_days_vf)
    return df


def reconcile(df : pd.DataFrame, config : dict) -> pd.DataFrame:
    """
    Step 3.2
    Reconcile VF_FORECAST_VALUE to ML_FORECAST_VALUE
    
    Parameters
    ----------
    df : pd.DataFrame
        The forecasts table obtained in the previous steps of the algorithm
    config : dict
        Configuration parameters used within the step
    
    Returns
    -------
    pd.DataFrame
        The forecasts table with reconciled forecasts
    """
    keys = ["PERIOD_DT", f"product_lvl_id{config['ml_product_lvl']}",
       f"location_lvl_id{config['ml_location_lvl']}", f"customer_lvl_id{config['ml_customer_lvl']}",
       f"distr_channel_lvl_id{config['ml_distr_channel_lvl']}"]

    sums = df.groupby(keys)[['ML_FORECAST_VALUE', 'VF_FORECAST_VALUE']].sum()
    ratio = pd.DataFrame(sums['VF_FORECAST_VALUE'] / sums['ML_FORECAST_VALUE'], columns=['ratio']).reset_index()
    df = pd.merge(df, ratio, on=keys)
    df['VF_FORECAST_VALUE_REC'] = df['ML_FORECAST_VALUE'] * df['ratio']
    df = df.drop(['VF_FORECAST_VALUE', 'ratio'], axis=1)
    return df


def add_segment_name(df : pd.DataFrame, VF_TS_SEGMENTS : pd.DataFrame) -> pd.DataFrame:
    """
    Step 3.3
    Adding segment names
    
    Parameters
    ----------
    df : pd.DataFrame
        Reconciled forecast
    VF_TS_SEGMENTS : pd.DataFrame
        VF_TS_SEGMENTS containg information about segment names
        
    Returns
    -------
    pd.DataFrame
        Reconciling forecast containing segment names
    """
    keys = VF_TS_SEGMENTS.columns.drop('SEGMENT_NAME').tolist()
    df = pd.merge(df, VF_TS_SEGMENTS, on=keys, how='left')
    return df


def reconcilation_algorithm(VF_FORECAST : pd.DataFrame, ML_FORECAST : pd.DataFrame, VF_TS_SEGMENTS : pd.DataFrame,
                            config : dict, IB_HIST_END_DT : datetime.datetime, IB_FC_HORIZ : int,
                            delays_config_length : int, hierarchies : dict) -> (pd.DataFrame, pd.DataFrame):
    """
    Pipeline for reconcilation algorithm.
    
    Forecast Reconciliation step is aimed at bringing ML and Statistical
        forecasts to the same granularity level (e.g. at product/location/day level).
    The result of this step is used to generate hybrid forecast value. Usually,
        the forecast is split to more granular level at the Reconciliation step.
    This step may include some PL processing logic (e.g. phase-in and phase-out dates) as well as  
    This step is not needed if only one approach (ML or Stat) is used for forecasting.
    
    Parameters
    ----------
    VF_FORECAST : pd.DataFrame
        Input table containing the keys columns, PERIOD_DT column, FORECAST_VALUE column.
    ML_FORECAST : pd.DataFrame
        Input table containing the keys columns, PERIOD_DT column,
        FORECAST_VALUE column, columns with types of DEMAND and ASSORTMENT.
    VF_TS_SEGMENTS : pd.DataFrame
        VF_TS_SEGMENTS containg information about segment names
    config : dict
        Configuration parameters used within the step
    IB_HIST_END_DT : datetime.datetime
        Last known date (i.e. sales and stock information is known)
    IB_FC_HORIZ : int
        Horizon of forecast
    hierarchies : dict
        Dictionary containg matches of key names with the relevant hierarchical tables
    delays_config_length : int
        Lenght of short-term forecasting period
        
    Returns
    -------
    pd.DataFrame
        Mid reconciled forecast
    pd.DataFrame
        Reconciled forecast
    
    """
    
    VF_FORECAST, mid_reconciled_forecast = periods_splitting(VF_FORECAST, IB_HIST_END_DT, delays_config_length)
    VF_FORECAST, ML_FORECAST = add_period_ends(VF_FORECAST, ML_FORECAST, config)
    VF_FORECAST, ML_FORECAST = match_keys(VF_FORECAST, ML_FORECAST, config, hierarchies)
    df = match_forecasts(VF_FORECAST, ML_FORECAST, IB_HIST_END_DT)
    df = interval_forecast_correction(df, config)
    df = reconcile(df, config)
    df = add_segment_name(df, VF_TS_SEGMENTS)
    return mid_reconciled_forecast, df
    