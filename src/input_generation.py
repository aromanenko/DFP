import numpy as np
import pandas as pd
from datetime import date
import datetime

def generate_input_unfolding(size):
  """
  Function to generate input data for unfolding algorithm

  Parameters
  ----------
  size : int
    Number of rows in generated tables

  Returns
  -------
    pd.DataFrame
  Table to be unfolded later

    pd.DataFrame
  Product Hierarchy
  
    pd.DataFrame
  Location Hierarchy

    pd.DataFrame
  Customer Hierarchy

    pd.DataFrame
  Distribution Channel Hierarchy

  """
  mplnTable = pd.concat([pd.Series(np.arange(10000,10000+size)) for _ in range(4)],axis=1)
  mplnTable.columns = ["product_lvl_id", "location_lvl_id", "customer_lvl_id", "distr_channel_lvl_id"]

  mpInProduct = pd.concat([pd.Series(np.arange(10000,10000+size)) for _ in range(26)],axis=1)
  mpInProduct = pd.concat([mpInProduct, pd.Series(np.random.choice([0,1],size))],axis=1)
  mpInProduct.columns = ['product_lvl_id', 'PRODUCT_LVL_NM1', 'PRODUCT_LVL_DESC1',
       'PRODUCT_LVL_ID2', 'PRODUCT_LVL_NM2', 'PRODUCT_LVL_DESC2',
       'PRODUCT_LVL_ID3', 'PRODUCT_LVL_NM3', 'PRODUCT_LVL_DESC3',
       'PRODUCT_LVL_ID4', 'PRODUCT_LVL_NM4', 'PRODUCT_LVL_DESC4',
       'PRODUCT_LVL_ID5', 'PRODUCT_LVL_NM5', 'PRODUCT_LVL_DESC5',
       'PRODUCT_LVL_ID6', 'PRODUCT_LVL_NM6', 'PRODUCT_LVL_DESC6',
       'PRODUCT_LVL_ID7', 'PRODUCT_LVL_NM7', 'PRODUCT_LVL_DESC7',
       'parent_product_id', 'PRODUCT_ID', 'PRODUCT_NM', 'PRODUCT_DESC',
       'MODIFIED_DTTM', 'DELETE_FLG']
       
  mpInLocation = pd.concat([pd.Series(np.arange(10000,10000+size)) for _ in range(21)],axis=1)
  mpInLocation = pd.concat([mpInLocation, pd.Series(np.random.choice([0,1],size))],axis=1)
  mpInLocation.columns = ['location_lvl_id', 'location_lvl_nm1', 'location_lvl_desc1',
       'location_lvl_id2', 'location_lvl_nm2', 'location_lvl_desc2',
       'location_lvl_id3', 'location_lvl_nm3', 'location_lvl_desc3',
       'location_lvl_id4', 'location_lvl_nm4', 'location_lvl_desc4',
       'location_lvl_id5', 'location_lvl_nm5', 'location_lvl_desc5',
       'location_id', 'location_nm', 'location_desc', 'open_dttm',
       'close_dttm', 'modified_dttm','del_flag']

  mpInCustomer = pd.concat([pd.Series(np.arange(10000,10000+size)) for _ in range(8)],axis=1)
  mpInCustomer = pd.concat([mpInCustomer, pd.Series(np.random.choice([0,1],size))],axis=1)
  mpInCustomer.columns = ['customer_lvl_id','location_lvl_desc5',
       'location_id', 'location_nm', 'location_desc', 'open_dttm',
       'close_dttm', 'modified_dttm','del_flag']

  mpInDistrChannel = pd.concat([pd.Series(np.arange(10000,10000+size)) for _ in range(8)],axis=1)
  mpInDistrChannel = pd.concat([mpInDistrChannel, pd.Series(np.random.choice([0,1],size))],axis=1)
  mpInDistrChannel.columns = ['distr_channel_lvl_id','location_lvl_desc5',
       'location_id', 'location_nm', 'location_desc', 'open_dttm',
       'close_dttm', 'modified_dttm','del_flag']
  

  return mplnTable,mpInProduct,mpInLocation,mpInCustomer,mpInDistrChannel


def generate_input_ilp(start_date, end_date, seed=42):
  """
  Function to generate input data for incremental load preparation with realistic scenarios
  to demonstrate Forecast Flag impact

  Parameters
  ----------
  start_date : str
    First date of sales (ISO format)
  end_date : str
  	Last date of sales (ISO format)
  seed : int
    Random seed for reproducibility

  Returns
  -------
    pd.DataFrame
  Sales Table

    pd.DataFrame
  Stocks Hierarchy
  
    pd.DataFrame
  Sell in  Hierarchy

    pd.DataFrame
  Sell out Hierarchy

    pd.DataFrame
  Assortment Matrix 

  	pd.DataFrame
  Location File Hierarchy

  """
  np.random.seed(seed)

  IB_UPDATE_HISTORY_DEPTH = 3
  IB_MAX_DT = datetime.date.fromisoformat('2025-12-31')
  IB_HIST_END_DT = datetime.date.fromisoformat('2000-01-01')

  date = pd.date_range(start_date, end_date, freq='D')
  size = len(date)

  # Create base IDs for all combinations
  base_ids = np.arange(10000, 10000 + size)
  
  # Generate regular data with all valid product-location-customer-distribution channel combinations
  # Some combinations will be marked with del_flag == 1 in source tables to be filtered by Forecast Flag
  
  # Sales data - all combinations
  sales = pd.DataFrame({
    'period_dt': date,
    'product_id': base_ids,
    'location_id': base_ids,
    'customer_id': base_ids,
    'distr_channel_id': base_ids,
    'period_start_dt': date
  })

  # Stock - all combinations
  stock = pd.DataFrame({
    'PRODUCT_ID': base_ids,
    'LOCATION_ID': base_ids,
    'STOCK_QTY': np.random.randint(10, 100, size),
    'PERIOD_START_DT': date
  })

  # Sell in - all combinations
  sell_in = pd.DataFrame({
    'PRODUCT_ID': base_ids,
    'LOCATION_ID': base_ids,
    'CUSTOMER_ID': base_ids,
    'DISTR_CHANNEL_ID': base_ids,
    'PERIOD_DT': date,
    'ORDERS_QTY': np.random.randint(1, 50, size),
    'ORDERS_AMOUNT': np.random.randint(100, 1000, size),
    'SHIPMENTS_QTY': np.random.randint(1, 50, size),
    'SHIPMENTS_AMOUNT': np.random.randint(100, 1000, size),
    'INVOICES_QTY': np.random.randint(1, 50, size),
    'INVOICES_AMOUNT': np.random.randint(100, 1000, size),
    'RETURNS_QTY': np.random.randint(0, 5, size),
    'RETUNRS_AMOUNT': np.random.randint(0, 50, size),
    'PROMO_FLG': np.random.choice([0, 1], size),
    'PROMO_ID': np.random.randint(1, 10, size),
    'COST': np.random.randint(50, 500, size),
    'date': date,
    'del_flag': np.zeros(size)
  })

  sell_out = sell_in.copy()

  # ASSORT_MATRIX - all combinations, but some marked with del_flag == 1
  # 20% will be marked for deletion (del_flag == 1) - these will be filtered by Forecast Flag
  del_flag_percentage = 0.2
  num_to_delete = int(size * del_flag_percentage)
  
  # Create continuous availability for all
  assort_continuous = pd.DataFrame({
    'product_id': base_ids,
    'location_id': base_ids,
    'customer_id': base_ids,
    'distr_channel_id': base_ids,
    'stard_td': [date[0]] * size,
    'end_td': [date[-1]] * size,
    'del_flag': np.zeros(size)
  })
  
  # Mark some combinations with del_flag == 1 (to be filtered by Forecast Flag)
  # Randomly select combinations to mark for deletion
  np.random.seed(seed)
  delete_indices = np.random.choice(size, size=num_to_delete, replace=False)
  assort_continuous.loc[delete_indices, 'del_flag'] = 1
  
  ASSORT_MATRIX = assort_continuous.copy()

  # LOCATION_LIFE - all combinations, some marked with del_flag == 1
  LOCATION_LIFE = pd.DataFrame({
    'product_id': base_ids,
    'location_id': base_ids,
    'customer_id': base_ids,
    'distr_channel_id': base_ids,
    'PERIOD_DT': date,
    'ORDERS_QTY': np.random.randint(1, 20, size),
    'date': date,
    'PERIOD_TYPE': np.random.choice(['reconstruction','re-branding'], size),
    'del_flag': np.zeros(size)
  })
  
  # Mark some combinations for deletion in LOCATION_LIFE (different random selection)
  np.random.seed(seed + 1)  # Different seed for different selection
  delete_indices_life = np.random.choice(size, size=num_to_delete, replace=False)
  LOCATION_LIFE.loc[delete_indices_life, 'del_flag'] = 1

  return sales, stock, sell_in, sell_out, ASSORT_MATRIX, LOCATION_LIFE
 