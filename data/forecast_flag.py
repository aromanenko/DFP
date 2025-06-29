import numpy as np, pandas as pd
from datetime import date
import datetime

FORECAST_FLAG = pd.NA

IB_UPDATE_HISTORY_DEPTH = pd.NA
IB_MAX_DT = pd.NA
IB_HIST_END_DT = pd.NA

def generate_input_unfolding(size):
    """
    Function to generate input data for unfolding algorithm
    """
    mplnTable = pd.concat([pd.Series(np.arange(10000, 10000 + size)) for _ in range(4)], axis=1)
    mplnTable.columns = ["product_lvl_id", "location_lvl_id", "customer_lvl_id", "distr_channel_lvl_id"]

    mpInProduct = pd.concat([pd.Series(np.arange(10000, 10000 + size)) for _ in range(26)], axis=1)
    mpInProduct = pd.concat([mpInProduct, pd.Series(np.random.choice([0, 1], size))], axis=1)
    mpInProduct.columns = ['product_lvl_id', 'PRODUCT_LVL_NM1', 'PRODUCT_LVL_DESC1', 'PRODUCT_LVL_ID2', 'PRODUCT_LVL_NM2', 'PRODUCT_LVL_DESC2',
                           'PRODUCT_LVL_ID3', 'PRODUCT_LVL_NM3', 'PRODUCT_LVL_DESC3', 'PRODUCT_LVL_ID4', 'PRODUCT_LVL_NM4', 'PRODUCT_LVL_DESC4',
                           'PRODUCT_LVL_ID5', 'PRODUCT_LVL_NM5', 'PRODUCT_LVL_DESC5', 'PRODUCT_LVL_ID6', 'PRODUCT_LVL_NM6', 'PRODUCT_LVL_DESC6',
                           'PRODUCT_LVL_ID7', 'PRODUCT_LVL_NM7', 'PRODUCT_LVL_DESC7', 'parent_product_id', 'product_id', 'PRODUCT_NM', 'PRODUCT_DESC',
                           'MODIFIED_DTTM', 'DELETE_FLG']
    
    mpInLocation = pd.concat([pd.Series(np.arange(10000, 10000 + size)) for _ in range(21)], axis=1)
    mpInLocation = pd.concat([mpInLocation, pd.Series(np.random.choice([0, 1], size))], axis=1)
    mpInLocation.columns = ['location_lvl_id', 'location_lvl_nm1', 'location_lvl_desc1', 'location_lvl_id2', 'location_lvl_nm2', 'location_lvl_desc2',
                            'location_lvl_id3', 'location_lvl_nm3', 'location_lvl_desc3', 'location_lvl_id4', 'location_lvl_nm4', 'location_lvl_desc4',
                            'location_lvl_id5', 'location_lvl_nm5', 'location_lvl_desc5', 'location_id', 'location_nm', 'location_desc', 'open_dttm',
                            'close_dttm', 'modified_dttm', 'del_flag']

    mpInCustomer = pd.concat([pd.Series(np.arange(10000, 10000 + size)) for _ in range(8)], axis=1)
    mpInCustomer = pd.concat([mpInCustomer, pd.Series(np.random.choice([0, 1], size))], axis=1)
    mpInCustomer.columns = ['customer_lvl_id', 'customer_lvl_desc5', 'customer_id', 'customer_nm', 'customer_desc', 'open_dttm', 'close_dttm', 
                            'modified_dttm', 'del_flag']

    mpInDistrChannel = pd.concat([pd.Series(np.arange(10000, 10000 + size)) for _ in range(8)], axis=1)
    mpInDistrChannel = pd.concat([mpInDistrChannel, pd.Series(np.random.choice([0, 1], size))], axis=1)
    mpInDistrChannel.columns = ['distr_channel_lvl_id', 'distr_channel_lvl_desc5', 'distr_channel_id', 'distr_channel_nm', 'distr_channel_desc', 'open_dttm',
                                'close_dttm', 'modified_dttm', 'del_flag']

    return mplnTable, mpInProduct, mpInLocation, mpInCustomer, mpInDistrChannel



def unfold_aggregated_data(mpInTable, mpInProduct, mpInLocation, mpInCustomer, mpInDistrChannel, mpInQuadruple=pd.NA):
    """
    Utility to unfold aggregated data to lower level of organizational hierarchy
    """
    table = mpInTable.merge(mpInProduct, on=["product_lvl_id"], how="left")
    table = table.merge(mpInLocation, on=["location_lvl_id"], how="left")
    table = table.merge(mpInCustomer, on=["customer_lvl_id"], how="left")
    table = table.merge(mpInDistrChannel, on=["distr_channel_lvl_id"], how="left")

    if not pd.isna(mpInQuadruple):
        table = table.merge(mpInQuadruple, on=['product_id', 'location_id', 'customer_id', 'distr_channel_id'], how='inner')

    return table



def generate_input_ilp(start_date, end_date):
    """
    Function to generate input data for incremental load preparation
    """
    IB_UPDATE_HISTORY_DEPTH = 3
    IB_MAX_DT = datetime.date.fromisoformat('2100-01-01')
    IB_HIST_END_DT = datetime.date.fromisoformat('2003-12-01')

    date_range = pd.date_range(start_date, end_date, freq='D')
    size = len(date_range)

    sales = pd.concat([pd.Series(np.arange(10000, 10000 + size)) for _ in range(5)], axis=1)
    sales = pd.concat([pd.Series(date_range), sales], axis=1)
    sales.columns = ['period_dt', 'product_id', 'location_id', 'customer_id', 'distr_channel_id', 'period_start_dt']

    stock = pd.concat([pd.Series(np.arange(10000, 10000 + size)) for _ in range(3)], axis=1)
    stock = pd.concat([stock, pd.Series(date_range)], axis=1)
    stock.columns = ['product_id', 'location_id', 'STOCK_QTY', 'period_start_dt']

    sell_in = pd.concat([pd.Series(np.arange(10000, 10000 + size)) for _ in range(16)], axis=1)
    sell_in = pd.concat([sell_in, pd.Series(date_range)], axis=1)
    sell_in = pd.concat([sell_in, pd.Series(np.random.choice([0, 1], size))], axis=1)
    sell_in.columns = ['product_id', 'location_id', 'customer_id', 'distr_channel_id', 'period_dt', 'ORDERS_QTY', 'ORDERS_AMOUNT',
                       'SHIPMENTS_QTY', 'SHIPMENTS_AMOUNT', 'INVOICES_QTY', 'INVOICES_AMOUNT', 'RETURNS_QTY', 'RETURNS_AMOUNT', 
                       'PROMO_FLG', 'PROMO_ID', 'COST', 'date', 'del_flag']

    sell_out = pd.concat([pd.Series(np.arange(10000, 10000 + size)) for _ in range(16)], axis=1)
    sell_out = pd.concat([sell_out, pd.Series(date_range)], axis=1)
    sell_out = pd.concat([sell_out, pd.Series(np.random.choice([0, 1], size))], axis=1)
    sell_out.columns = ['product_id', 'location_id', 'customer_id', 'distr_channel_id', 'period_dt', 'ORDERS_QTY', 'ORDERS_AMOUNT',
                        'SHIPMENTS_QTY', 'SHIPMENTS_AMOUNT', 'INVOICES_QTY', 'INVOICES_AMOUNT', 'RETURNS_QTY', 'RETURNS_AMOUNT', 
                        'PROMO_FLG', 'PROMO_ID', 'COST', 'date', 'del_flag']

    ASSORT_MATRIX = pd.concat([pd.Series(np.arange(10000, 10000 + size)) for _ in range(4)], axis=1)
    ASSORT_MATRIX = pd.concat([ASSORT_MATRIX, pd.Series(date_range)], axis=1)
    ASSORT_MATRIX = pd.concat([ASSORT_MATRIX, pd.Series(date_range + datetime.timedelta(days=15))], axis=1)
    ASSORT_MATRIX = pd.concat([ASSORT_MATRIX, pd.Series(np.random.choice([0, 1], size))], axis=1)
    ASSORT_MATRIX = pd.concat([ASSORT_MATRIX, pd.Series(["Active"] * size)], axis=1)
    ASSORT_MATRIX.columns = ["product_id", "location_id", "customer_id", "distr_channel_id", "start_dt", "end_dt", 'del_flag', 'status']

    LOCATION_LIFE = pd.concat([pd.Series(np.arange(10000, 10000 + size)) for _ in range(4)], axis=1)
    LOCATION_LIFE = pd.concat([LOCATION_LIFE, pd.Series(date_range)], axis=1)
    LOCATION_LIFE = pd.concat([LOCATION_LIFE, pd.Series(date_range + datetime.timedelta(days=15))], axis=1)
    LOCATION_LIFE.columns = ['product_lvl_id', 'location_id', 'customer_lvl_id', 'distr_channel_lvl_id', 'period_start_dt', 'period_end_dt']
    LOCATION_LIFE['location_successor_id'] = pd.Series(np.arange(10000 + np.random.choice([10, 20, 30, 40, 50]), 10000 + size, 100))
    LOCATION_LIFE['PERIOD_TYPE'] = pd.Series(np.random.choice(['reconstruction', 're-branding'], size))
    LOCATION_LIFE['del_flag'] = pd.Series(np.random.choice([0, 1], size))

    CUSTOMER_LIFE = pd.concat([pd.Series(np.arange(10000, 10000 + size)) for _ in range(4)], axis=1)
    CUSTOMER_LIFE = pd.concat([CUSTOMER_LIFE, pd.Series(date_range)], axis=1)
    CUSTOMER_LIFE = pd.concat([CUSTOMER_LIFE, pd.Series(date_range + datetime.timedelta(days=15))], axis=1)
    CUSTOMER_LIFE.columns = ['product_lvl_id', 'location_lvl_id', 'customer_id', 'distr_channel_lvl_id', 'period_start_dt', 'period_end_dt']
    CUSTOMER_LIFE['customer_successor_id'] = pd.Series(np.arange(10000 + np.random.choice([10, 20, 30, 40, 50]), 10000 + size, 100))
    CUSTOMER_LIFE['PERIOD_TYPE'] = pd.Series(np.random.choice(['active', 'blocked', 'end-of-life'], size))
    CUSTOMER_LIFE['del_flag'] = pd.Series(np.random.choice([0, 1], size))

    PRODUCT_LIFE = pd.concat([pd.Series(np.arange(10000, 10000 + size)) for _ in range(4)], axis=1)
    PRODUCT_LIFE = pd.concat([PRODUCT_LIFE, pd.Series(date_range)], axis=1)
    PRODUCT_LIFE = pd.concat([PRODUCT_LIFE, pd.Series(date_range + datetime.timedelta(days=15))], axis=1)
    PRODUCT_LIFE.columns = ['product_id', 'location_lvl_id', 'customer_lvl_id', 'distr_channel_lvl_id', 'period_start_dt', 'period_end_dt']
    PRODUCT_LIFE['product_successor_id'] = pd.Series(np.arange(10000 + np.random.choice([10, 20, 30, 40, 50]), 10000 + size, 100))
    PRODUCT_LIFE['PERIOD_TYPE'] = pd.Series(np.random.choice(['active', 'blocked', 'end-of-life'], size))
    PRODUCT_LIFE['del_flag'] = pd.Series(np.random.choice([0, 1], size))

    return sales, stock, sell_in, sell_out, ASSORT_MATRIX, LOCATION_LIFE, CUSTOMER_LIFE, PRODUCT_LIFE

def time_interval_utility(
    table: pd.DataFrame,
    mpTimeGranularity: str,
    mpDistanceTolerance: int,
    mpGroupBy: list
) -> pd.DataFrame:
    """
    Utility to union time-intervals within groups based on time granularity and tolerance.

    Parameters:
    ----------
    table : pd.DataFrame
        Input DataFrame containing time intervals. Must contain 'period_start_dt' and 'period_end_dt'.
    mpTimeGranularity : str
        'day', 'week', or 'month' – granularity of intervals.
    mpDistanceTolerance : int
        Number of granules to extend right interval bound.
    mpGroupBy : list
        List of column names to group by.

    Returns:
    ----------
    pd.DataFrame
        DataFrame with merged time intervals per group.
    """

    def round_dt(dt, granularity):
        if granularity == 'day':
            return dt.dt.normalize()
        elif granularity == 'week':
            return dt - pd.to_timedelta(dt.dt.dayofweek, unit='d')
        elif granularity == 'month':
            return dt.values.astype('datetime64[M]')
        else:
            raise ValueError(f"Unsupported granularity: {granularity}")

    # Validate required columns
    if 'period_start_dt' not in table.columns or 'period_end_dt' not in table.columns:
        raise ValueError("Input table must contain 'period_start_dt' and 'period_end_dt' columns.")

    # Ensure datetime format
    table['period_start_dt'] = pd.to_datetime(table['period_start_dt'], errors='coerce')
    table['period_end_dt'] = pd.to_datetime(table['period_end_dt'], errors='coerce')

    if table['period_start_dt'].isna().any() or table['period_end_dt'].isna().any():
        raise ValueError("Some values in 'period_start_dt' or 'period_end_dt' could not be converted to datetime.")

    # Round interval boundaries
    table['_start'] = round_dt(table['period_start_dt'], mpTimeGranularity)

    if mpTimeGranularity == 'day':
        offset = pd.to_timedelta(mpDistanceTolerance, unit='d')
    elif mpTimeGranularity == 'week':
        offset = pd.to_timedelta(7 * mpDistanceTolerance, unit='d')
    elif mpTimeGranularity == 'month':
        offset = pd.DateOffset(months=mpDistanceTolerance)

    table['_end'] = round_dt(table['period_end_dt'], mpTimeGranularity) + offset

    # Sort for merging
    table = table.sort_values(by=mpGroupBy + ['_start'])

    merged_intervals = []
    for _, group_df in table.groupby(mpGroupBy):
        group_df = group_df.reset_index(drop=True)
        current_start = group_df.loc[0, '_start']
        current_end = group_df.loc[0, '_end']
        key_values = group_df.loc[0, mpGroupBy].tolist()

        for i in range(1, len(group_df)):
            row = group_df.loc[i]
            if row['_start'] <= current_end:
                current_end = max(current_end, row['_end'])
            else:
                merged_intervals.append(key_values + [current_start, current_end])
                current_start = row['_start']
                current_end = row['_end']
                key_values = row[mpGroupBy].tolist()

        merged_intervals.append(key_values + [current_start, current_end])

    result_columns = mpGroupBy + ['period_start_dt', 'period_end_dt']
    result = pd.DataFrame(merged_intervals, columns=result_columns)

    print(f"[INFO] Merged {len(table)} intervals → {len(result)} rows")
    print(result.head(3))

    return result


def incremental_load_preparation(SALES, STOCK, SELL_IN, SELL_OUT, ASSORT_MATRIX, LOCATION_LIFE, start_date, end_date, PRODUCT_LIFE = None, CUSTOMER_LIFE = None):
  """
  Utility to unfold aggregated data to lower level of organizational hierarchy (3.1)

  Parameters
  ----------
  Sales : pd.DataFrame
    Sales Table

  Stock : pd.DataFrame
    Stocks Table

  SELL_IN : pd.DataFrame
    Sells in Table

  SELL_OUT : pd.DataFrame
    Sells out Table

  ASSORT_MATRIX : pd.DataFrame
    Assortment matrix

  Returns
  -------
  pd.DataFrame
    Qudrabples to delete

  pd.DataFrame
    Updated assortment matrix

  """

  if FORECAST_FLAG == None:
    IB_UPDATE_HISTORY_DEPTH = 0

  IB_UPDATE_HISTORY_DEPTH = 3
  IB_MAX_DT = date.fromisoformat('2030-12-12')
  IB_HIST_END_DT = date.fromisoformat('2003-12-01')

  table1 = SALES[['product_id', 'location_id', 'customer_id', 'distr_channel_id','period_dt']]
  table2 = ASSORT_MATRIX[['product_id', 'location_id', 'customer_id',\
                          'distr_channel_id']][ASSORT_MATRIX.del_flag == 1]
  table3 = LOCATION_LIFE[['product_lvl_id', 'location_id', 'customer_lvl_id',\
                          'distr_channel_lvl_id']][LOCATION_LIFE.del_flag == 1]

  QUADRUPLES_DELETE = pd.concat([table1,table2,table2],axis=0)

  SALES_UPDATE_FF = SALES.merge(QUADRUPLES_DELETE, on = ['product_id', 'location_id'],how='left')
  SALES_UPDATE_FF['IB_UPDATE_HISTORY_DEPTH'] = pd.Series(np.full(shape=len(SALES_UPDATE_FF), fill_value=IB_UPDATE_HISTORY_DEPTH))


  SALES_UPDATE_FF = SALES_UPDATE_FF.loc[(SALES_UPDATE_FF.IB_UPDATE_HISTORY_DEPTH<=0) | (pd.to_datetime(SALES_UPDATE_FF.period_dt_x).dt.date > (IB_HIST_END_DT - datetime.timedelta(days=IB_UPDATE_HISTORY_DEPTH))) | (SALES_UPDATE_FF.product_id is not np.nan)]
  
  ASSORT_MATRIX_UPDATE_FF = ASSORT_MATRIX.merge(QUADRUPLES_DELETE, on = ['product_id', 'location_id', 'customer_id', \
                          'distr_channel_id'],how='left')
  ASSORT_MATRIX_UPDATE_FF['IB_UPDATE_HISTORY_DEPTH'] = pd.Series(np.full(shape=len(ASSORT_MATRIX_UPDATE_FF), fill_value=IB_UPDATE_HISTORY_DEPTH))
  ASSORT_MATRIX_UPDATE_FF = ASSORT_MATRIX_UPDATE_FF.loc[(ASSORT_MATRIX_UPDATE_FF.IB_UPDATE_HISTORY_DEPTH<=0) | (pd.to_datetime(ASSORT_MATRIX_UPDATE_FF.period_dt).dt.date > (IB_HIST_END_DT - datetime.timedelta(days=IB_UPDATE_HISTORY_DEPTH))) | (ASSORT_MATRIX_UPDATE_FF.product_id is not np.nan)]

  ASSORT_MATRIX_UPDATE_FF['IB_UPDATE_HISTORY_DEPTH'] = pd.Series(np.full(shape=len(SALES_UPDATE_FF), fill_value=0))
  ASSORT_MATRIX_UPDATE_FF.period_dt = pd.Series(pd.date_range(start_date, end_date, freq='D'))
  ASSORT_MATRIX_UPDATE_FF['IB_UPDATE_HISTORY_DEPTH_date'] = pd.Series(pd.date_range(start_date, end_date, freq='D'))
  ASSORT_MATRIX_UPDATE_FF = ASSORT_MATRIX_UPDATE_FF.loc[(ASSORT_MATRIX_UPDATE_FF.IB_UPDATE_HISTORY_DEPTH<=0)]
  
  return SALES_UPDATE_FF, ASSORT_MATRIX_UPDATE_FF, QUADRUPLES_DELETE


def adding_fields(sales, stock, assort):
    """
    Function to add required fields into sales and stock (3.2)
    """
    table1 = sales[['location_id', 'product_id', 'period_dt_x']]
    table1.columns = ['location_id', 'product_id', 'period_dt']
    table2 = stock[['location_id', 'product_id', 'period_start_dt']]
    table2.columns = ['location_id', 'product_id', 'period_dt']

    print(table1)
    print(table2)

    res = pd.concat([table1, table2], axis=0)
    res = res.merge(assort[['product_id', 'location_id', 'customer_id', 'distr_channel_id']], 
                            on=['product_id', 'location_id'], 
                            how='left')

    print(res.columns)
    
    res['customer_id'] = res['customer_id'].bfill() 
    res['distr_channel_id'] = res['distr_channel_id'].bfill()  

    res['customer_id'] = res['customer_id'].fillna(res['customer_id'].min())
    res['distr_channel_id'] = res['distr_channel_id'].fillna(res['distr_channel_id'].min())
    
    return res

def calculate_fact_dates(T1, sell_in_data, sell_out_data):    

    T1['period_dt'] = pd.to_datetime(T1['period_dt'])
    sell_in_data['period_dt'] = pd.to_datetime(sell_in_data['period_dt'])
    sell_out_data['period_dt'] = pd.to_datetime(sell_out_data['period_dt'])
    fact_data = pd.merge(T1, sell_in_data[['product_id', 'location_id', 'customer_id', 'distr_channel_id', 'period_dt']], 
                         on=['product_id', 'location_id', 'customer_id', 'distr_channel_id', 'period_dt'], 
                         how='left')

    fact_data = pd.merge(fact_data, sell_out_data[['product_id', 'location_id', 'customer_id', 'distr_channel_id', 'period_dt']], 
                         on=['product_id', 'location_id', 'customer_id', 'distr_channel_id', 'period_dt'], 
                         how='left')

    fact_data = fact_data.drop_duplicates(subset=['product_id', 'location_id', 'customer_id', 'distr_channel_id', 'period_dt'])

    # 4. Вычисляем PERIOD_START_DT и PERIOD_END_DT для каждой комбинации
    fact_data['period_start_dt'] = fact_data.groupby(['product_id', 'location_id', 'customer_id', 'distr_channel_id'])['period_dt'].transform('min')
    fact_data['period_end_dt'] = fact_data.groupby(['product_id', 'location_id', 'customer_id', 'distr_channel_id'])['period_dt'].transform('max')

    # 5. Применяем utility для временных интервалов
    intervals = time_interval_utility(fact_data, mpTimeGranularity='day', mpDistanceTolerance=365, mpGroupBy=['product_id', 'location_id', 'customer_id',  'distr_channel_id'])

    return intervals

def assortment_matrix_calculation(assortment_matrix, IB_MAX_DT):
    """
    A function for processing the assortment table and calculating the end dates.
    """
    
    # Шаг 1: Заполнение пропусков в END_DT
    assortment_matrix['end_dt'] = assortment_matrix.groupby(['product_id', 'location_id', 'customer_id', 'distr_channel_id'])['end_dt'].fillna(method='bfill')
    
    # Если в конце данных нет следующей строки, заполняем значением IB_MAX_DT
    assortment_matrix['end_dt'] = assortment_matrix['end_dt'].fillna(IB_MAX_DT)
    
    # Шаг 2: Фильтрация только активных записей (STATUS = 'Active')
    active_assortment = assortment_matrix[assortment_matrix['status'] == 'Active']
    
    # Шаг 3: Заполняем пропуски в END_DT с использованием следующего периода или IB_MAX_DT
    active_assortment['end_dt'] = active_assortment.groupby(['product_id', 'location_id', 'customer_id', 'distr_channel_id'])['end_dt'].fillna(method='bfill')
    active_assortment['end_dt'] = active_assortment['end_dt'].fillna(IB_MAX_DT)
    
    # Шаг 4: Объединение всех интервалов по каждому сочетанию (PRODUCT_ID, LOCATION_ID, CUSTOMER_ID, DISTR_CHANNEL_ID)
    active_assortment['period_start_dt'] = active_assortment.groupby(['product_id', 'location_id', 'customer_id', 'distr_channel_id'])['start_dt'].transform('min')
    active_assortment['period_end_dt'] = active_assortment.groupby(['product_id', 'location_id', 'customer_id', 'distr_channel_id'])['end_dt'].transform('max')
    
    # Шаг 5: Возвращаем итоговую таблицу
    result = active_assortment[['product_id', 'location_id', 'customer_id', 'distr_channel_id', 'period_start_dt', 'period_end_dt']]
    
    return result

import pandas as pd

def transform_lifecycle_to_single_interval_format(life_table, name):
    """
    Функция для преобразования таблицы жизненных циклов из формата "преемник-предшественник"
    в формат с единичным интервалом.
    
    Parameters:
    ----------
    life_table : pd.DataFrame
        Таблица жизненного цикла (например, PRODUCT_LIFE, LOCATION_LIFE или CUSTOMER_LIFE)
        
    name : str
        Имя таблицы, которое будет использоваться в выводе (например, "product", "location" или "customer")
    
    Returns:
    ----------
    pd.DataFrame
        Объединенная таблица жизненных циклов с форматированными интервалами
    """
    if name == 'product':
        # a) Извлечение уникальных записей, где <name>_SUCCESSOR_ID пустой
        life_table_a = life_table[life_table[f'{name}_successor_id'].isna()]
        life_table_a = life_table_a[['period_start_dt', 'period_end_dt', f'{name}_id', f'{name}_successor_id', 'location_lvl_id', 'customer_lvl_id', 'distr_channel_lvl_id']]
    
        # b) Извлечение уникальных записей, где <name>_ID не равен <name>_SUCCESSOR_ID
        life_table_b = life_table[life_table[f'{name}_id'] != life_table[f'{name}_successor_id']]
        life_table_b = life_table_b[['period_start_dt', 'period_end_dt', f'{name}_successor_id', f'{name}_id', 'location_lvl_id', 'customer_lvl_id', 'distr_channel_lvl_id']]  # используем SUCCESSOR_ID вместо ID
    
        # c) Извлечение уникальных записей, где <name>_ID равен <name>_SUCCESSOR_ID
        life_table_c = life_table[life_table[f'{name}_id'] == life_table[f'{name}_successor_id']]
        life_table_c = life_table_c[['period_start_dt', 'period_end_dt', f'{name}_id', f'{name}_successor_id', 'location_lvl_id', 'customer_lvl_id', 'distr_channel_lvl_id']]
    
        # d) Объединение всех трех таблиц
        combined_life_table = pd.concat([life_table_a, life_table_b, life_table_c], axis=0, ignore_index=True)
    elif name == 'location':
        # a) Извлечение уникальных записей, где <name>_SUCCESSOR_ID пустой
        life_table_a = life_table[life_table[f'{name}_successor_id'].isna()]
        life_table_a = life_table_a[['period_start_dt', 'period_end_dt', f'{name}_id', f'{name}_successor_id', 'product_lvl_id', 'customer_lvl_id', 'distr_channel_lvl_id']]
    
        # b) Извлечение уникальных записей, где <name>_ID не равен <name>_SUCCESSOR_ID
        life_table_b = life_table[life_table[f'{name}_id'] != life_table[f'{name}_successor_id']]
        life_table_b = life_table_b[['period_start_dt', 'period_end_dt', f'{name}_successor_id', f'{name}_id', 'product_lvl_id', 'customer_lvl_id', 'distr_channel_lvl_id']]  # используем SUCCESSOR_ID вместо ID
    
        # c) Извлечение уникальных записей, где <name>_ID равен <name>_SUCCESSOR_ID
        life_table_c = life_table[life_table[f'{name}_id'] == life_table[f'{name}_successor_id']]
        life_table_c = life_table_c[['period_start_dt', 'period_end_dt', f'{name}_id', f'{name}_successor_id', 'product_lvl_id', 'customer_lvl_id', 'distr_channel_lvl_id']]
    
        # d) Объединение всех трех таблиц
        combined_life_table = pd.concat([life_table_a, life_table_b, life_table_c], axis=0, ignore_index=True)
    else:
        # a) Извлечение уникальных записей, где <name>_SUCCESSOR_ID пустой
        life_table_a = life_table[life_table[f'{name}_successor_id'].isna()]
        life_table_a = life_table_a[['period_start_dt', 'period_end_dt', f'{name}_id', f'{name}_successor_id', 'product_lvl_id', 'location_lvl_id', 'distr_channel_lvl_id']]
    
        # b) Извлечение уникальных записей, где <name>_ID не равен <name>_SUCCESSOR_ID
        life_table_b = life_table[life_table[f'{name}_id'] != life_table[f'{name}_successor_id']]
        life_table_b = life_table_b[['period_start_dt', 'period_end_dt', f'{name}_successor_id', f'{name}_id', 'product_lvl_id', 'location_lvl_id', 'distr_channel_lvl_id']]  # используем SUCCESSOR_ID вместо ID
    
        # c) Извлечение уникальных записей, где <name>_ID равен <name>_SUCCESSOR_ID
        life_table_c = life_table[life_table[f'{name}_id'] == life_table[f'{name}_successor_id']]
        life_table_c = life_table_c[['period_start_dt', 'period_end_dt', f'{name}_id', f'{name}_successor_id', 'product_lvl_id', 'location_lvl_id', 'distr_channel_lvl_id']]
    
        # d) Объединение всех трех таблиц
        combined_life_table = pd.concat([life_table_a, life_table_b, life_table_c], axis=0, ignore_index=True)


    # Возвращаем итоговую таблицу
    return combined_life_table


def life_cycle_information_merging(product_life, location_life, customer_life, IB_MAX_DT):
    """
    Функция для обработки таблиц жизненных циклов продуктов, локаций и клиентов,
    преобразует их в единичные интервалы, объединяет и вызывает утилиту time_interval_utility.

    Parameters:
    ----------
    product_life : pd.DataFrame
        Таблица жизненного цикла продуктов.
        
    location_life : pd.DataFrame
        Таблица жизненного цикла локаций.
        
    customer_life : pd.DataFrame
        Таблица жизненного цикла клиентов.
        
    IB_MAX_DT : datetime
        Максимальная дата для заполнения пропущенных значений.
    
    Returns:
    ----------
    pd.DataFrame
        Объединенная таблица жизненных циклов с нужной структурой.
    """
    
    # Преобразуем таблицы жизненных циклов для каждого типа (продукт, локация, клиент)
    product_life_transformed = transform_lifecycle_to_single_interval_format(product_life, "product")
    print(product_life_transformed)
    location_life_transformed = transform_lifecycle_to_single_interval_format(location_life, "location")
    print(location_life_transformed)
    customer_life_transformed = transform_lifecycle_to_single_interval_format(customer_life, "customer")
    print(customer_life_transformed)

    # Применяем утилиту time_interval_utility для каждой таблицы жизненных циклов с нужной группировкой
    # Для таблицы продуктов
    i = 0
    while i < 3:
        product_life_processed = time_interval_utility(
            product_life_transformed, 
            mpTimeGranularity='day', 
            mpDistanceTolerance=365, 
            mpGroupBy=['product_id', 'location_lvl_id', 'customer_lvl_id', 'distr_channel_lvl_id']
        )
        
        # Для таблицы локаций
        location_life_processed = time_interval_utility(
            location_life_transformed, 
            mpTimeGranularity='day', 
            mpDistanceTolerance=365, 
            mpGroupBy=['location_id', 'customer_lvl_id', 'product_lvl_id', 'distr_channel_lvl_id']
        )
        
        # Для таблицы клиентов
        customer_life_processed = time_interval_utility(
            customer_life_transformed, 
            mpTimeGranularity='day', 
            mpDistanceTolerance=365, 
            mpGroupBy=['customer_id', 'location_lvl_id', 'product_lvl_id', 'distr_channel_lvl_id']
        )
        i += 1

    # Объединяем все обработанные данные в единую таблицу
    all_processed_data = pd.concat([
        product_life_processed[['product_id', 'location_lvl_id', 'customer_lvl_id', 'distr_channel_lvl_id', 'period_start_dt', 'period_end_dt']],
        location_life_processed[['product_lvl_id', 'location_id', 'customer_lvl_id', 'distr_channel_lvl_id', 'period_start_dt', 'period_end_dt']],
        customer_life_processed[['product_lvl_id', 'location_lvl_id', 'customer_id', 'distr_channel_lvl_id', 'period_start_dt', 'period_end_dt']]
    ], axis=0, ignore_index=True)

    print(all_processed_data)

    # Возвращаем итоговую таблицу с нужной структурой
    return all_processed_data[['product_id', 'customer_id', 'location_id', 'period_start_dt', 'period_end_dt']]

def forecast_flag_calculation(ff_fact_dates, ff_assort_dates, ff_life_dates, quadruples_delete, forecast_flag):
    """
    Функция для выполнения расчета прогноза флага на основе временных таблиц и выполнения алгоритма из шага 3.6.
    
    Parameters:
    ----------
    ff_fact_dates : pd.DataFrame
        Временная таблица FF_FACT_DATES (T2)
        
    ff_assort_dates : pd.DataFrame
        Временная таблица FF_ASSORT_DATES
    
    ff_life_dates : pd.DataFrame
        Временная таблица FF_LIFE_DATES
    
    forecast_flag : pd.DataFrame
        Таблица FORECAST_FLAG, в которую будет обновлен результат
    
    quadruples_delete : pd.DataFrame
        Таблица с записями для удаления
    
    Returns:
    ----------
    pd.DataFrame
        Обновленная таблица FORECAST_FLAG с результатами флага прогноза
    """
    
    # 1. Объединяем входные таблицы в WORK.FF_DATES
    ff_dates = pd.concat([ff_fact_dates, ff_assort_dates, ff_life_dates], axis=0, ignore_index=True)
    
    # 2. Вызов утилиты для обработки временных интервалов
    ff_dates_transformed = time_interval_utility(
        ff_dates,
        mpTimeGranularity='day',
        mpDistanceTolerance=1,
        mpGroupBy=['product_id', 'location_id', 'customer_id', 'distr_channel_id']
    )
    
    # 3. Добавляем новое поле STATUS с постоянным значением 'active'
    ff_dates_transformed['STATUS'] = 'active'
    
    # 4. Обновление таблицы FORECAST_FLAG:
    # a. Удаляем все строки из FORECAST_FLAG, которые содержат записи из QUADRUPLES_DELETE
    if not forecast_flag.empty:
        forecast_flag_cleaned = forecast_flag[~forecast_flag.isin(quadruples_delete)].dropna()
    else:
        forecast_flag_cleaned = forecast_flag
    
    # b. Добавляем строки из шага 3 (ff_dates_transformed)
    forecast_flag_updated = pd.concat([forecast_flag_cleaned, ff_dates_transformed], axis=0, ignore_index=True)
    
    # 5. Вызов утилиты для объединения пересекающихся интервалов
    forecast_flag_final = time_interval_utility(
        forecast_flag_updated,
        mpTimeGranularity='day',
        mpDistanceTolerance=1,
        mpGroupBy=['product_id', 'location_id', 'customer_id', 'distr_channel_id']
    )
    
    # Возвращаем итоговую таблицу
    return forecast_flag_final

def filter_dictionaries(forecast_flag, product_df, location_df, customer_df, distr_channel_df):
    """
    Фильтрует справочники (PRODUCT, LOCATION, CUSTOMER, DISTR_CHANNEL), оставляя только те элементы,
    которые присутствуют в таблице FORECAST_FLAG.

    Parameters:
    ----------
    forecast_flag : pd.DataFrame
        Таблица с результатами прогноза (FORECAST_FLAG)

    product_df : pd.DataFrame
        Справочник продуктов

    location_df : pd.DataFrame
        Справочник локаций

    customer_df : pd.DataFrame
        Справочник клиентов

    distr_channel_df : pd.DataFrame
        Справочник каналов распределения

    Returns:
    ----------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Отфильтрованные справочники:
        PRODUCT_FILTERED, LOCATION_FILTERED, CUSTOMER_FILTERED, DISTR_CHANNEL_FILTERED
    """
    product_filtered = product_df[product_df['product_id'].isin(forecast_flag['product_id'].unique())].copy()
    location_filtered = location_df[location_df['location_id'].isin(forecast_flag['location_id'].unique())].copy()
    customer_filtered = customer_df[customer_df['customer_id'].isin(forecast_flag['customer_id'].unique())].copy()
    distr_channel_filtered = distr_channel_df[distr_channel_df['distr_channel_id'].isin(forecast_flag['distr_channel_id'].unique())].copy()

    return product_filtered, location_filtered, customer_filtered, distr_channel_filtered