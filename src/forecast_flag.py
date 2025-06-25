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
                           'PRODUCT_LVL_ID7', 'PRODUCT_LVL_NM7', 'PRODUCT_LVL_DESC7', 'parent_product_id', 'PRODUCT_ID', 'PRODUCT_NM', 'PRODUCT_DESC',
                           'MODIFIED_DTTM', 'DELETE_FLG']
    
    mpInLocation = pd.concat([pd.Series(np.arange(10000, 10000 + size)) for _ in range(21)], axis=1)
    mpInLocation = pd.concat([mpInLocation, pd.Series(np.random.choice([0, 1], size))], axis=1)
    mpInLocation.columns = ['location_lvl_id', 'location_lvl_nm1', 'location_lvl_desc1', 'location_lvl_id2', 'location_lvl_nm2', 'location_lvl_desc2',
                            'location_lvl_id3', 'location_lvl_nm3', 'location_lvl_desc3', 'location_lvl_id4', 'location_lvl_nm4', 'location_lvl_desc4',
                            'location_lvl_id5', 'location_lvl_nm5', 'location_lvl_desc5', 'location_id', 'location_nm', 'location_desc', 'open_dttm',
                            'close_dttm', 'modified_dttm', 'del_flag']

    mpInCustomer = pd.concat([pd.Series(np.arange(10000, 10000 + size)) for _ in range(8)], axis=1)
    mpInCustomer = pd.concat([mpInCustomer, pd.Series(np.random.choice([0, 1], size))], axis=1)
    mpInCustomer.columns = ['customer_lvl_id', 'location_lvl_desc5', 'location_id', 'location_nm', 'location_desc', 'open_dttm', 'close_dttm', 
                            'modified_dttm', 'del_flag']

    mpInDistrChannel = pd.concat([pd.Series(np.arange(10000, 10000 + size)) for _ in range(8)], axis=1)
    mpInDistrChannel = pd.concat([mpInDistrChannel, pd.Series(np.random.choice([0, 1], size))], axis=1)
    mpInDistrChannel.columns = ['distr_channel_lvl_id', 'location_lvl_desc5', 'location_id', 'location_nm', 'location_desc', 'open_dttm',
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
    IB_MAX_DT = datetime.date.fromisoformat('2030-12-12')
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

    LOCATION_LIFE = pd.concat([pd.Series(np.arange(10000, 10000 + size)) for _ in range(6)], axis=1)
    LOCATION_LIFE = pd.concat([LOCATION_LIFE, pd.Series(date_range)], axis=1)
    LOCATION_LIFE.columns = ['product_id', 'location_id', 'customer_id', 'distr_channel_id', 'PERIOD_DT', 'ORDERS_QTY', 'date']
    LOCATION_LIFE['PERIOD_TYPE'] = pd.Series(np.random.choice(['reconstruction', 're-branding'], size))
    LOCATION_LIFE['del_flag'] = pd.Series(np.random.choice([0, 1], size))

    return sales, stock, sell_in, sell_out, ASSORT_MATRIX, LOCATION_LIFE

def time_interval_utility(table, granularity, distance_tolerance, groupby):
    """
    Technical utility for time-intervals union

    Parameters:
    ----------
    table : pd.DataFrame
        The table to be transformed 

    granularity : str 
        Granularity for the dates, e.g., 'Day', 'Week', 'Month'

    groupby : str 
        The columns to group by

    Returns:
    ----------
    list
        List of transformed date intervals
    """
  
    dates = []

    # Group by the relevant columns
    grouped = table.groupby(groupby)

    for key, group in grouped:
        start_dt = group['period_dt'].min()
        end_dt = group['period_dt'].max()

        # Extend the end date based on the distance tolerance (365 days)
        new_end_dt = end_dt + pd.Timedelta(days=distance_tolerance)

        # Add the interval to the results
        dates.append([key[0], key[1], key[2], key[3], start_dt, new_end_dt])

    # Create a DataFrame from the intervals
    intervals_df = pd.DataFrame(dates, columns=['product_id', 'location_id', 'customer_id', 'distr_channel_id', 'period_start_dt', 'period_end_dt'])

    return intervals_df


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

  table1 = SALES[['product_id', 'location_id', 'customer_id', 'distr_channel_id','period_dt']].drop_duplicates()
  table2 = ASSORT_MATRIX[['product_id', 'location_id', 'customer_id',\
                          'distr_channel_id']][ASSORT_MATRIX.del_flag == 1].drop_duplicates()
  table3 = LOCATION_LIFE[['product_id', 'location_id', 'customer_id',\
                          'distr_channel_id']][LOCATION_LIFE.del_flag == 1].drop_duplicates()

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
  
  return SALES_UPDATE_FF, ASSORT_MATRIX_UPDATE_FF,QUADRUPLES_DELETE


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
    intervals = time_interval_utility(fact_data, granularity='Day', distance_tolerance=365, groupby=['product_id', 'location_id', 'customer_id',  'distr_channel_id'])

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

def life_cycle_information_merging(product_life, location_life, customer_life, IB_MAX_DT):
    """
    Функция для объединения данных о жизненных циклах продуктов, локаций и клиентов
    в единую таблицу.
    
    Parameters:
    ----------
    product_life : pd.DataFrame
        Таблица с данными о жизненном цикле продуктов, содержащая PRODUCT_ID, START_DT, END_DT, и другие.
        
    location_life : pd.DataFrame
        Таблица с данными о жизненном цикле локаций, содержащая LOCATION_ID, START_DT, END_DT, и другие.
        
    customer_life : pd.DataFrame
        Таблица с данными о жизненном цикле клиентов, содержащая CUSTOMER_ID, START_DT, END_DT, и другие.
        
    IB_MAX_DT : datetime
        Максимальная дата (например, 01/01/2100), используемая для заполнения недостающих END_DT.
        
    Returns:
    ----------
    pd.DataFrame
        Объединенная таблица с данными о жизненных циклах, которая будет содержать:
        - PRODUCT_ID, LOCATION_ID, CUSTOMER_ID, DISTR_CHANNEL_ID, PERIOD_START_DT, PERIOD_END_DT
    """
    
    # Преобразуем столбцы даты в datetime
    product_life['START_DT'] = pd.to_datetime(product_life['START_DT'])
    product_life['END_DT'] = pd.to_datetime(product_life['END_DT'])
    location_life['START_DT'] = pd.to_datetime(location_life['START_DT'])
    location_life['END_DT'] = pd.to_datetime(location_life['END_DT'])
    customer_life['START_DT'] = pd.to_datetime(customer_life['START_DT'])
    customer_life['END_DT'] = pd.to_datetime(customer_life['END_DT'])

    # Объединяем данные о жизненных циклах продуктов, локаций и клиентов
    combined_life = pd.concat([product_life[['PRODUCT_ID', 'START_DT', 'END_DT']],
                               location_life[['LOCATION_ID', 'START_DT', 'END_DT']],
                               customer_life[['CUSTOMER_ID', 'START_DT', 'END_DT']]], axis=0)

    # Заполняем пропуски в END_DT значением IB_MAX_DT
    combined_life['END_DT'] = combined_life['END_DT'].fillna(IB_MAX_DT)
    
    # Объединяем данные о жизненных циклах по ключам (PRODUCT_ID, LOCATION_ID, CUSTOMER_ID)
    combined_life['PERIOD_START_DT'] = combined_life.groupby(['PRODUCT_ID', 'LOCATION_ID', 'CUSTOMER_ID'])['START_DT'].transform('min')
    combined_life['PERIOD_END_DT'] = combined_life.groupby(['PRODUCT_ID', 'LOCATION_ID', 'CUSTOMER_ID'])['END_DT'].transform('max')
    
    # Убираем лишние столбцы и возвращаем итоговую таблицу
    result = combined_life[['PRODUCT_ID', 'LOCATION_ID', 'CUSTOMER_ID', 'PERIOD_START_DT', 'PERIOD_END_DT']]
    
    return result