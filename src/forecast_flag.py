import numpy as np
import pandas as pd
from datetime import date
import datetime

FORECAST_FLAG = None

IB_UPDATE_HISTORY_DEPTH = None
IB_MAX_DT = None
IB_HIST_END_DT = None

def unfold_aggregated_data(mpInTable, mpInProduct, mpInLocation, mpInCustomer,mpInDistrChannel, mpInQuadruple = None):
  """
  Utility to unfold aggregated data to lower level of organizational hierarchy (3)

  Parameters
  ----------
  mpInTable : pd.DataFrame
    Input Table that must be unfold

  mpInProduct : pd.DataFrame
    Product Hierarchy

  mpInLocation : pd.DataFrame
    Location Hierarchy

  mpInCustomer : pd.DataFrame
    Customer Hierarchy

  mpInDistrChannel : pd.DataFrame
    Distribution channel Hierarchy

  Returns
  -------
  pd.DataFrame
    Returns unfolded table
  """

  table = mpInTable.merge(mpInProduct, on=["product_lvl_id"], how="left")
  table = table.merge(mpInLocation, on=["location_lvl_id"], how="left")
  table = table.merge(mpInCustomer, on=["customer_lvl_id"], how="left")
  table = table.merge(mpInDistrChannel, on=["distr_channel_lvl_id"], how="left")
  if mpInQuadruple is not None:  
    table = table.merge(mpInQuadruple, on=['product_id', 'location_id', 'customer_id', 'distr_channel_id'], how = 'inner')

  return table


def _merge_intervals(df, group_cols, start_col, end_col, distance_tolerance=0):
  """
  Utility to merge intersecting or close intervals inside each group.

  Parameters
  ----------
  df : pd.DataFrame
    Input table with date intervals.
  group_cols : list
    Columns to group by.
  start_col : str
    Column name of interval start.
  end_col : str
    Column name of interval end.
  distance_tolerance : int
    Max gap (in days) between intervals to consider them continuous.

  Returns
  -------
    pd.DataFrame
    Table with merged intervals per group.
  """
  if df.empty:
    return df.copy()

  work = df.copy()
  work[start_col] = pd.to_datetime(work[start_col])
  work[end_col] = pd.to_datetime(work[end_col])

  merged_rows = []
  for _, g in work.sort_values(group_cols + [start_col, end_col]).groupby(group_cols):
    current_start = None
    current_end = None
    current_keys = None

    for _, row in g.iterrows():
      start = row[start_col]
      end = row[end_col]
      keys = tuple(row[col] for col in group_cols)

      if current_start is None:
        current_start, current_end, current_keys = start, end, keys
        continue

      # If intervals overlap or are within tolerance, extend the current interval
      if start <= current_end + pd.Timedelta(days=distance_tolerance):
        current_end = max(current_end, end)
      else: 
        merged_rows.append((*current_keys, current_start, current_end))
        current_start, current_end, current_keys = start, end, keys

    if current_start is not None:
      merged_rows.append((*current_keys, current_start, current_end))

  result_cols = group_cols + [start_col, end_col]
  return pd.DataFrame(merged_rows, columns=result_cols)


def incremental_load_preparation(SALES, STOCK, SELL_IN, SELL_OUT, ASSORT_MATRIX, LOCATION_LIFE, start_date, end_date, PRODUCT_LIFE = None, CUSTOMER_LIFE = None):
  """
  Utility to unfold aggregated data to lower level of organizational hierarchy (3.1)

  Parameters
  ----------
  SALES : pd.DataFrame
    Sales Table

  STOCK : pd.DataFrame
    Stocks Table

  SELL_IN : pd.DataFrame
    Sells in Table

  SELL_OUT : pd.DataFrame
    Sells out Table

  ASSORT_MATRIX : pd.DataFrame
    Assortment matrix

  LOCATION_LIFE : pd.DataFrame
    Location Life table

  start_date : datetime
    Start date for processing

  end_date : datetime
    End date for processing

  PRODUCT_LIFE : pd.DataFrame, optional
    Product Life table

  CUSTOMER_LIFE : pd.DataFrame, optional
    Customer Life table

  Returns
  -------
  pd.DataFrame
    Updated Sales table (SALES_UPDATE_FF)

  pd.DataFrame
    Updated Stock table (STOCK_UPDATE_FF)

  pd.DataFrame
    Updated Sell In table (SELL_IN_UPDATE_FF)

  pd.DataFrame
    Updated Sell Out table (SELL_OUT_UPDATE_FF)

  pd.DataFrame
    Updated Assortment matrix (ASSORT_MATRIX_UPDATE_FF)

  pd.DataFrame
    Quadruples to delete (QUADRUPLES_DELETE)

  """

  global FORECAST_FLAG, IB_UPDATE_HISTORY_DEPTH, IB_MAX_DT, IB_HIST_END_DT

  # Step 1: Check whether previous version of FORECAST_FLAG table is available
  if FORECAST_FLAG is None:
    IB_UPDATE_HISTORY_DEPTH = 0
  else:
    IB_UPDATE_HISTORY_DEPTH = 3

  IB_MAX_DT = date.fromisoformat('2025-12-31')
  IB_HIST_END_DT = date.fromisoformat('2000-01-01')

  # Step 2: Select list of quadruples that have DELETE_FLG = 1
  # Step 2a: Quadruples from SALES with del_flag==1 if present; otherwise none
  if 'del_flag' in SALES.columns:
    table1 = SALES[SALES['del_flag'] == 1][['product_id', 'location_id', 'customer_id', 'distr_channel_id']].drop_duplicates()
  else:
    table1 = SALES[['product_id', 'location_id', 'customer_id', 'distr_channel_id']].iloc[0:0]

  # Step 2b: Quadruples from ASSORT_MATRIX where DELETE_FLG = 1
  table2 = ASSORT_MATRIX[['product_id', 'location_id', 'customer_id', 'distr_channel_id']][
    ASSORT_MATRIX.del_flag == 1].drop_duplicates()

  # Step 2c: Quadruples from LOCATION_LIFE where DELETE_FLG = 1
  table3 = LOCATION_LIFE[['product_id', 'location_id', 'customer_id', 'distr_channel_id']][
    LOCATION_LIFE.del_flag == 1].drop_duplicates()

  # Step 2d: Union all quadruples
  tables_to_union = [table1, table2, table3]

  # Add PRODUCT_LIFE if provided
  if PRODUCT_LIFE is not None:
    table4 = PRODUCT_LIFE[['product_id', 'location_id', 'customer_id', 'distr_channel_id']][
      PRODUCT_LIFE.del_flag == 1].drop_duplicates()
    tables_to_union.append(table4)

  # Add CUSTOMER_LIFE if provided
  if CUSTOMER_LIFE is not None:
    table5 = CUSTOMER_LIFE[['product_id', 'location_id', 'customer_id', 'distr_channel_id']][
      CUSTOMER_LIFE.del_flag == 1].drop_duplicates()
    tables_to_union.append(table5)

  # Union all tables to create QUADRUPLES_DELETE
  QUADRUPLES_DELETE = pd.concat(tables_to_union, axis=0).drop_duplicates()

  # Step 4: Create updated tables with filtering
  # Calculate date threshold
  date_threshold = IB_HIST_END_DT - datetime.timedelta(days=IB_UPDATE_HISTORY_DEPTH)
  
  # Step 4a: Create SALES_UPDATE_FF
  # Merge only on product_id and location_id as per specification
  SALES_UPDATE_FF = SALES.merge(
    QUADRUPLES_DELETE[['product_id', 'location_id']].drop_duplicates(), 
    on=['product_id', 'location_id'], 
    how='left', 
    suffixes=('', '_delete')
  )
  
  # Create indicator column to check if merge was successful
  SALES_UPDATE_FF['_matched_quadruple'] = SALES_UPDATE_FF['product_id_delete'].notna() if 'product_id_delete' in SALES_UPDATE_FF.columns else pd.Series([False] * len(SALES_UPDATE_FF))
  
  # Filter SALES_UPDATE_FF according to specification
  SALES_UPDATE_FF = SALES_UPDATE_FF.loc[
    (IB_UPDATE_HISTORY_DEPTH <= 0) | 
    (pd.to_datetime(SALES_UPDATE_FF.period_dt).dt.date > date_threshold) | 
    (SALES_UPDATE_FF['_matched_quadruple'])
  ]
  
  # Drop the merge columns from QUADRUPLES_DELETE
  SALES_UPDATE_FF = SALES_UPDATE_FF.drop(columns=[col for col in SALES_UPDATE_FF.columns if col.endswith('_delete') or col == '_matched_quadruple'])

  # Step 4b: Create ASSORT_MATRIX_UPDATE_FF
  ASSORT_MATRIX_UPDATE_FF = ASSORT_MATRIX.merge(
    QUADRUPLES_DELETE, 
    on=['product_id', 'location_id', 'customer_id', 'distr_channel_id'], 
    how='left', 
    suffixes=('', '_delete')
  )
  
  # Create indicator column to check if merge was successful
  ASSORT_MATRIX_UPDATE_FF['_matched_quadruple'] = ASSORT_MATRIX_UPDATE_FF['product_id_delete'].notna() if 'product_id_delete' in ASSORT_MATRIX_UPDATE_FF.columns else pd.Series([False] * len(ASSORT_MATRIX_UPDATE_FF))
  
  # Find date columns in ASSORT_MATRIX (handle typo "stard_td" -> "start_dt")
  assort_cols_list = ASSORT_MATRIX_UPDATE_FF.columns.tolist()
  period_start_col = next((col for col in assort_cols_list if 'start' in col.lower() or 'stard' in col.lower()), None)
  period_end_col = next((col for col in assort_cols_list if 'end' in col.lower() and 'td' in col.lower()), None)
  
  # Filter ASSORT_MATRIX_UPDATE_FF
  if period_start_col and period_end_col:
    ASSORT_MATRIX_UPDATE_FF = ASSORT_MATRIX_UPDATE_FF.loc[
      (IB_UPDATE_HISTORY_DEPTH <= 0) | 
      (pd.to_datetime(ASSORT_MATRIX_UPDATE_FF[period_start_col]).dt.date > date_threshold) |
      (pd.to_datetime(ASSORT_MATRIX_UPDATE_FF[period_end_col]).dt.date > date_threshold) |
      (ASSORT_MATRIX_UPDATE_FF['_matched_quadruple'])
    ]
  else:
    # If date columns not found, filter only on matched_quadruple
    ASSORT_MATRIX_UPDATE_FF = ASSORT_MATRIX_UPDATE_FF.loc[
      (IB_UPDATE_HISTORY_DEPTH <= 0) | 
      (ASSORT_MATRIX_UPDATE_FF['_matched_quadruple'])
    ]
  # Drop the merge columns from QUADRUPLES_DELETE
  ASSORT_MATRIX_UPDATE_FF = ASSORT_MATRIX_UPDATE_FF.drop(columns=[col for col in ASSORT_MATRIX_UPDATE_FF.columns if col.endswith('_delete') or col == '_matched_quadruple'])

  # Create updated versions for STOCK, SELL_IN, SELL_OUT
  # STOCK_UPDATE_FF - merge on product_id and location_id
  # Create a temporary dataframe with uppercase column names for merging
  quadruples_stock = QUADRUPLES_DELETE[['product_id', 'location_id']].drop_duplicates().copy()
  quadruples_stock.columns = ['PRODUCT_ID', 'LOCATION_ID']
  
  STOCK_UPDATE_FF = STOCK.merge(
    quadruples_stock,
    on=['PRODUCT_ID', 'LOCATION_ID'],
    how='left',
    suffixes=('', '_delete')
  )
  
  # Create indicator column to check if merge was successful
  STOCK_UPDATE_FF['_matched_quadruple'] = STOCK_UPDATE_FF['PRODUCT_ID_delete'].notna() if 'PRODUCT_ID_delete' in STOCK_UPDATE_FF.columns else pd.Series([False] * len(STOCK_UPDATE_FF))
  
  STOCK_UPDATE_FF = STOCK_UPDATE_FF.loc[
    (IB_UPDATE_HISTORY_DEPTH <= 0) | 
    (pd.to_datetime(STOCK_UPDATE_FF.PERIOD_START_DT).dt.date > date_threshold) |
    (STOCK_UPDATE_FF['_matched_quadruple'])
  ]
  
  # Drop merge indicator columns
  STOCK_UPDATE_FF = STOCK_UPDATE_FF.drop(columns=[col for col in STOCK_UPDATE_FF.columns if col.endswith('_delete') or col == '_matched_quadruple'])

  # SELL_IN_UPDATE_FF - merge on all four keys
  # Create a temporary dataframe with uppercase column names for merging
  quadruples_sell = QUADRUPLES_DELETE.copy()
  quadruples_sell.columns = ['PRODUCT_ID', 'LOCATION_ID', 'CUSTOMER_ID', 'DISTR_CHANNEL_ID']
  
  SELL_IN_UPDATE_FF = SELL_IN.merge(
    quadruples_sell,
    on=['PRODUCT_ID', 'LOCATION_ID', 'CUSTOMER_ID', 'DISTR_CHANNEL_ID'],
    how='left',
    suffixes=('', '_delete')
  )
  
  # Create indicator column to check if merge was successful
  SELL_IN_UPDATE_FF['_matched_quadruple'] = SELL_IN_UPDATE_FF['PRODUCT_ID_delete'].notna() if 'PRODUCT_ID_delete' in SELL_IN_UPDATE_FF.columns else pd.Series([False] * len(SELL_IN_UPDATE_FF))
  
  SELL_IN_UPDATE_FF = SELL_IN_UPDATE_FF.loc[
    (IB_UPDATE_HISTORY_DEPTH <= 0) | 
    (pd.to_datetime(SELL_IN_UPDATE_FF.PERIOD_DT).dt.date > date_threshold) |
    (SELL_IN_UPDATE_FF['_matched_quadruple'])
  ]
  
  # Drop merge indicator columns
  SELL_IN_UPDATE_FF = SELL_IN_UPDATE_FF.drop(columns=[col for col in SELL_IN_UPDATE_FF.columns if col.endswith('_delete') or col == '_matched_quadruple'])

  # SELL_OUT_UPDATE_FF - merge on all four keys
  SELL_OUT_UPDATE_FF = SELL_OUT.merge(
    quadruples_sell,
    on=['PRODUCT_ID', 'LOCATION_ID', 'CUSTOMER_ID', 'DISTR_CHANNEL_ID'],
    how='left',
    suffixes=('', '_delete')
  )
  
  # Create indicator column to check if merge was successful
  SELL_OUT_UPDATE_FF['_matched_quadruple'] = SELL_OUT_UPDATE_FF['PRODUCT_ID_delete'].notna() if 'PRODUCT_ID_delete' in SELL_OUT_UPDATE_FF.columns else pd.Series([False] * len(SELL_OUT_UPDATE_FF))
  
  SELL_OUT_UPDATE_FF = SELL_OUT_UPDATE_FF.loc[
    (IB_UPDATE_HISTORY_DEPTH <= 0) | 
    (pd.to_datetime(SELL_OUT_UPDATE_FF.PERIOD_DT).dt.date > date_threshold) |
    (SELL_OUT_UPDATE_FF['_matched_quadruple'])
  ]
  
  # Drop merge indicator columns
  SELL_OUT_UPDATE_FF = SELL_OUT_UPDATE_FF.drop(columns=[col for col in SELL_OUT_UPDATE_FF.columns if col.endswith('_delete') or col == '_matched_quadruple'])
  
  return SALES_UPDATE_FF, STOCK_UPDATE_FF, SELL_IN_UPDATE_FF, SELL_OUT_UPDATE_FF, ASSORT_MATRIX_UPDATE_FF, QUADRUPLES_DELETE


def adding_fields(SALES_UPDATE_FF, STOCK_UPDATE_FF, ASSORT_MATRIX_UPDATE_FF):
  """
  Function to add required fields into sales and stock (3.2)

  Parameters
  ----------
  Sales : pd.DataFrame
    Sales Table

  Stock : pd.DataFrame
    Stocks Table

  Assort : pd.DataFrame
    Assortment matrix

  Returns
  -------
  pd.DataFrame
    Updated Sales table with CUSTOMER_ID and DISTR_CHANNEL_ID

  pd.DataFrame
    Updated Stock table with CUSTOMER_ID and DISTR_CHANNEL_ID

  """
  # Check if CUSTOMER_ID and DISTR_CHANNEL_ID already exist in SALES and STOCK
  sales_cols = [col.upper() for col in SALES_UPDATE_FF.columns]
  stock_cols = [col.upper() for col in STOCK_UPDATE_FF.columns]
  
  sales_has_customer = 'CUSTOMER_ID' in sales_cols
  sales_has_distr = 'DISTR_CHANNEL_ID' in sales_cols
  stock_has_customer = 'CUSTOMER_ID' in stock_cols
  stock_has_distr = 'DISTR_CHANNEL_ID' in stock_cols
  
  # Only proceed if columns are missing
  if (sales_has_customer and sales_has_distr) and (stock_has_customer and stock_has_distr):
    # Both tables already have the required columns, return as-is
    return SALES_UPDATE_FF.copy(), STOCK_UPDATE_FF.copy()
  
  # Step 1: Select LOCATION_ID, PRODUCT_ID, and PERIOD_DT from SALES and STOCK tables and union both results
  # Handle different column name cases (SALES uses lowercase, STOCK uses uppercase)
  sales_cols_list = SALES_UPDATE_FF.columns.tolist()
  stock_cols_list = STOCK_UPDATE_FF.columns.tolist()
  
  # Find the correct column names (case-insensitive)
  sales_location_col = next((col for col in sales_cols_list if col.upper() == 'LOCATION_ID'), None)
  sales_product_col = next((col for col in sales_cols_list if col.upper() == 'PRODUCT_ID'), None)
  sales_period_col = next((col for col in sales_cols_list if col.upper() in ['PERIOD_DT', 'PERIOD_START_DT']), None)
  
  stock_location_col = next((col for col in stock_cols_list if col.upper() == 'LOCATION_ID'), None)
  stock_product_col = next((col for col in stock_cols_list if col.upper() == 'PRODUCT_ID'), None)
  stock_period_col = next((col for col in stock_cols_list if col.upper() in ['PERIOD_DT', 'PERIOD_START_DT']), None)
  
  if None in [sales_location_col, sales_product_col, sales_period_col, stock_location_col, stock_product_col, stock_period_col]:
    raise ValueError("Required columns (LOCATION_ID, PRODUCT_ID, PERIOD_DT) not found in SALES or STOCK tables")
  
  # Extract and standardize column names
  table1 = SALES_UPDATE_FF[[sales_location_col, sales_product_col, sales_period_col]].copy()
  table1.columns = ['LOCATION_ID', 'PRODUCT_ID', 'PERIOD_DT']
  
  table2 = STOCK_UPDATE_FF[[stock_location_col, stock_product_col, stock_period_col]].copy()
  table2.columns = ['LOCATION_ID', 'PRODUCT_ID', 'PERIOD_DT']
  
  # Union both results
  union_table = pd.concat([table1, table2], axis=0).drop_duplicates()
  
  # Step 2: Left join ASSORT_MATRIX on PRODUCT_ID, LOCATION_ID, and PERIOD_DT BETWEEN START_DT and END_DT
  # Find date columns in ASSORT_MATRIX (handle typo "stard_td" -> "start_dt")
  assort_cols_list = ASSORT_MATRIX_UPDATE_FF.columns.tolist()
  assort_start_col = next((col for col in assort_cols_list if 'start' in col.lower() or 'stard' in col.lower()), None)
  assort_end_col = next((col for col in assort_cols_list if 'end' in col.lower() and 'td' in col.lower()), None)
  
  if assort_start_col is None or assort_end_col is None:
    raise ValueError("ASSORT_MATRIX_UPDATE_FF must contain START_DT and END_DT columns")
  
  # Find customer and distr_channel columns in ASSORT_MATRIX
  assort_customer_col = next((col for col in assort_cols_list if col.upper() == 'CUSTOMER_ID'), None)
  assort_distr_col = next((col for col in assort_cols_list if col.upper() == 'DISTR_CHANNEL_ID'), None)
  assort_product_col = next((col for col in assort_cols_list if col.upper() == 'PRODUCT_ID'), None)
  assort_location_col = next((col for col in assort_cols_list if col.upper() == 'LOCATION_ID'), None)
  
  if None in [assort_customer_col, assort_distr_col, assort_product_col, assort_location_col]:
    raise ValueError("ASSORT_MATRIX_UPDATE_FF must contain PRODUCT_ID, LOCATION_ID, CUSTOMER_ID, and DISTR_CHANNEL_ID columns")
  
  # Prepare ASSORT_MATRIX for merge
  assort_for_merge = ASSORT_MATRIX_UPDATE_FF[[
    assort_product_col, assort_location_col, assort_customer_col, assort_distr_col, 
    assort_start_col, assort_end_col
  ]].copy()
  
  # Standardize column names for merge
  assort_for_merge.columns = ['PRODUCT_ID', 'LOCATION_ID', 'CUSTOMER_ID', 'DISTR_CHANNEL_ID', 'START_DT', 'END_DT']
  
  # Convert dates to datetime if needed
  union_table['PERIOD_DT'] = pd.to_datetime(union_table['PERIOD_DT'])
  assort_for_merge['START_DT'] = pd.to_datetime(assort_for_merge['START_DT'])
  assort_for_merge['END_DT'] = pd.to_datetime(assort_for_merge['END_DT'])
  
  # Perform merge with date range condition
  # First merge on PRODUCT_ID and LOCATION_ID (this may create multiple rows if multiple date ranges match)
  merged = union_table.merge(
    assort_for_merge[['PRODUCT_ID', 'LOCATION_ID', 'CUSTOMER_ID', 'DISTR_CHANNEL_ID', 'START_DT', 'END_DT']],
    on=['PRODUCT_ID', 'LOCATION_ID'],
    how='left'
  )
  
  # Filter to keep only rows where PERIOD_DT is between START_DT and END_DT
  # For rows with multiple matches, keep all (as per specification)
  date_match = (merged['PERIOD_DT'] >= merged['START_DT']) & (merged['PERIOD_DT'] <= merged['END_DT'])
  merged.loc[~date_match, ['CUSTOMER_ID', 'DISTR_CHANNEL_ID']] = None
  
  # If there are multiple matches for the same (PRODUCT_ID, LOCATION_ID, PERIOD_DT), keep all
  # But for the merge back, we'll take the first non-null value per group
  # Drop the temporary date columns
  merged = merged.drop(columns=['START_DT', 'END_DT'])
  
  # Step 3: Fill missing values in CUSTOMER_ID and DISTR_CHANNEL_ID
  # Sort by PRODUCT_ID, LOCATION_ID, PERIOD_DT for proper forward/backward fill
  merged = merged.sort_values(['PRODUCT_ID', 'LOCATION_ID', 'PERIOD_DT'])
  
  # Group by PRODUCT_ID and LOCATION_ID for filling within each pair
  grouped = merged.groupby(['PRODUCT_ID', 'LOCATION_ID'])
  
  # 3a. Fill missing values with previous non-missing value (forward fill)
  merged['CUSTOMER_ID'] = grouped['CUSTOMER_ID'].ffill()
  merged['DISTR_CHANNEL_ID'] = grouped['DISTR_CHANNEL_ID'].ffill()
  
  # 3b. Fill missing values with next non-missing value (backward fill)
  merged['CUSTOMER_ID'] = grouped['CUSTOMER_ID'].bfill()
  merged['DISTR_CHANNEL_ID'] = grouped['DISTR_CHANNEL_ID'].bfill()
  
  # 3c. Fill remaining missing values with minimal CUSTOMER_ID (DISTR_CHANNEL_ID)
  if merged['CUSTOMER_ID'].isna().any():
    min_customer_id = merged['CUSTOMER_ID'].min()
    if pd.notna(min_customer_id):
      merged['CUSTOMER_ID'] = merged['CUSTOMER_ID'].fillna(min_customer_id)
  
  if merged['DISTR_CHANNEL_ID'].isna().any():
    min_distr_channel_id = merged['DISTR_CHANNEL_ID'].min()
    if pd.notna(min_distr_channel_id):
      merged['DISTR_CHANNEL_ID'] = merged['DISTR_CHANNEL_ID'].fillna(min_distr_channel_id)
  
  # For duplicate (PRODUCT_ID, LOCATION_ID, PERIOD_DT) combinations, take the first non-null value
  # This handles the case where multiple ASSORT_MATRIX rows match
  customer_distr = merged.groupby(['LOCATION_ID', 'PRODUCT_ID', 'PERIOD_DT']).agg({
    'CUSTOMER_ID': 'first',
    'DISTR_CHANNEL_ID': 'first'
  }).reset_index()
  
  # Now merge back to original SALES and STOCK tables
  # Prepare merge keys for SALES
  sales_merge_key = SALES_UPDATE_FF[[sales_location_col, sales_product_col, sales_period_col]].copy()
  sales_merge_key.columns = ['LOCATION_ID', 'PRODUCT_ID', 'PERIOD_DT']
  sales_merge_key['PERIOD_DT'] = pd.to_datetime(sales_merge_key['PERIOD_DT'])
  
  # Prepare merge keys for STOCK
  stock_merge_key = STOCK_UPDATE_FF[[stock_location_col, stock_product_col, stock_period_col]].copy()
  stock_merge_key.columns = ['LOCATION_ID', 'PRODUCT_ID', 'PERIOD_DT']
  stock_merge_key['PERIOD_DT'] = pd.to_datetime(stock_merge_key['PERIOD_DT'])
  
  # Merge CUSTOMER_ID and DISTR_CHANNEL_ID back to SALES
  sales_with_fields = SALES_UPDATE_FF.copy()
  sales_merged = sales_merge_key.merge(
    customer_distr,
    on=['LOCATION_ID', 'PRODUCT_ID', 'PERIOD_DT'],
    how='left'
  )
  
  # Add CUSTOMER_ID and DISTR_CHANNEL_ID to SALES if they don't exist
  if not sales_has_customer:
    # Use the original column name case if it exists, otherwise use CUSTOMER_ID
    sales_customer_col = 'customer_id' if 'customer_id' in SALES_UPDATE_FF.columns else 'CUSTOMER_ID'
    sales_with_fields[sales_customer_col] = sales_merged['CUSTOMER_ID'].values
  else:
    # Update existing column
    existing_col = next((col for col in SALES_UPDATE_FF.columns if col.upper() == 'CUSTOMER_ID'), None)
    if existing_col:
      sales_with_fields[existing_col] = sales_merged['CUSTOMER_ID'].values
  
  if not sales_has_distr:
    sales_distr_col = 'distr_channel_id' if 'distr_channel_id' in SALES_UPDATE_FF.columns else 'DISTR_CHANNEL_ID'
    sales_with_fields[sales_distr_col] = sales_merged['DISTR_CHANNEL_ID'].values
  else:
    existing_col = next((col for col in SALES_UPDATE_FF.columns if col.upper() == 'DISTR_CHANNEL_ID'), None)
    if existing_col:
      sales_with_fields[existing_col] = sales_merged['DISTR_CHANNEL_ID'].values
  
  # Merge CUSTOMER_ID and DISTR_CHANNEL_ID back to STOCK
  stock_with_fields = STOCK_UPDATE_FF.copy()
  stock_merged = stock_merge_key.merge(
    customer_distr,
    on=['LOCATION_ID', 'PRODUCT_ID', 'PERIOD_DT'],
    how='left'
  )
  
  # Add CUSTOMER_ID and DISTR_CHANNEL_ID to STOCK if they don't exist
  if not stock_has_customer:
    stock_with_fields['CUSTOMER_ID'] = stock_merged['CUSTOMER_ID'].values
  else:
    stock_with_fields['CUSTOMER_ID'] = stock_merged['CUSTOMER_ID'].values
  
  if not stock_has_distr:
    stock_with_fields['DISTR_CHANNEL_ID'] = stock_merged['DISTR_CHANNEL_ID'].values
  else:
    stock_with_fields['DISTR_CHANNEL_ID'] = stock_merged['DISTR_CHANNEL_ID'].values
  
  return sales_with_fields, stock_with_fields


def calculate_fact_dates(T1, SELL_IN_UPDATE_FF, SELL_OUT_UPDATE_FF):
  """
  3.3 Fact dates calculation

  Returns
  -------
  pd.DataFrame
    FF_FACT_DATES with columns:
    PRODUCT_ID, LOCATION_ID, CUSTOMER_ID, DISTR_CHANNEL_ID,
    PERIOD_START_DT, PERIOD_END_DT
  """
  def _pick_cols(df, col_names):
    picked = {}
    for target in col_names:
      match = next((c for c in df.columns if c.upper() == target), None)
      picked[target] = match
    return picked

  # Identify column names for each table
  required = ['PRODUCT_ID', 'LOCATION_ID', 'CUSTOMER_ID', 'DISTR_CHANNEL_ID', 'PERIOD_DT']

  def _standardize_cols(df):
    cols = _pick_cols(df, required)
    missing = [k for k, v in cols.items() if v is None]
    if missing:
      raise ValueError(f"Missing columns {missing} in input table")
    res = df[[cols['PRODUCT_ID'], cols['LOCATION_ID'], cols['CUSTOMER_ID'], cols['DISTR_CHANNEL_ID'], cols['PERIOD_DT']]].copy()
    res.columns = ['PRODUCT_ID', 'LOCATION_ID', 'CUSTOMER_ID', 'DISTR_CHANNEL_ID', 'PERIOD_DT']
    res['PERIOD_DT'] = pd.to_datetime(res['PERIOD_DT'])
    return res

  t1_std = _standardize_cols(T1)
  sell_in_std = _standardize_cols(SELL_IN_UPDATE_FF)
  sell_out_std = _standardize_cols(SELL_OUT_UPDATE_FF)

  union_df = pd.concat([t1_std, sell_in_std, sell_out_std], axis=0).drop_duplicates()

  union_df['PERIOD_START_DT'] = union_df['PERIOD_DT']
  union_df['PERIOD_END_DT'] = union_df['PERIOD_DT']

  ff_fact_dates = _merge_intervals(
    union_df[['PRODUCT_ID', 'LOCATION_ID', 'CUSTOMER_ID', 'DISTR_CHANNEL_ID', 'PERIOD_START_DT', 'PERIOD_END_DT']],
    ['PRODUCT_ID', 'LOCATION_ID', 'CUSTOMER_ID', 'DISTR_CHANNEL_ID'],
    'PERIOD_START_DT',
    'PERIOD_END_DT',
    distance_tolerance=365
  )

  return ff_fact_dates


def calculate_assortment_dates(ASSORT_MATRIX_UPDATE_FF, IB_MAX_DT=date.fromisoformat('2030-12-12')):
  """
  3.4 Assortment Matrix Calculation

  Returns
  -------
  pd.DataFrame
    FF_ASSORT_DATES with columns:
    PRODUCT_ID, LOCATION_ID, CUSTOMER_ID, DISTR_CHANNEL_ID,
    START_DT, END_DT
  """
  if ASSORT_MATRIX_UPDATE_FF.empty:
    return pd.DataFrame(columns=['PRODUCT_ID', 'LOCATION_ID', 'CUSTOMER_ID', 'DISTR_CHANNEL_ID', 'START_DT', 'END_DT'])

  df = ASSORT_MATRIX_UPDATE_FF.copy()
  # Identify columns
  def _find(col_name):
    return next((c for c in df.columns if c.upper() == col_name), None)

  prod_col = _find('PRODUCT_ID')
  loc_col = _find('LOCATION_ID')
  cust_col = _find('CUSTOMER_ID')
  distr_col = _find('DISTR_CHANNEL_ID')
  start_col = next((c for c in df.columns if 'start' in c.lower() or 'stard' in c.lower()), None)
  end_col = next((c for c in df.columns if 'end' in c.lower() and 'td' in c.lower()), None)

  for col, name in [(prod_col, 'PRODUCT_ID'), (loc_col, 'LOCATION_ID'), (cust_col, 'CUSTOMER_ID'),
                    (distr_col, 'DISTR_CHANNEL_ID'), (start_col, 'START_DT'), (end_col, 'END_DT')]:
    if col is None:
      raise ValueError(f"ASSORT_MATRIX_UPDATE_FF must contain {name}")

# ПЕРЕНАЗВАТЬ ПЕРЕМЕННУЮ!
  work = df[[prod_col, loc_col, cust_col, distr_col, start_col, end_col]].copy()
  work.columns = ['PRODUCT_ID', 'LOCATION_ID', 'CUSTOMER_ID', 'DISTR_CHANNEL_ID', 'START_DT', 'END_DT']
  work['START_DT'] = pd.to_datetime(work['START_DT'])
  work['END_DT'] = pd.to_datetime(work['END_DT'])

  # Fill missing END_DT with next START_DT within group or IB_MAX_DT
  work = work.sort_values(['PRODUCT_ID', 'LOCATION_ID', 'CUSTOMER_ID', 'DISTR_CHANNEL_ID', 'START_DT'])
  grouped = work.groupby(['PRODUCT_ID', 'LOCATION_ID', 'CUSTOMER_ID', 'DISTR_CHANNEL_ID'])

  def _fill_end(group):
    group = group.copy().sort_values('START_DT')
    group['END_DT'] = group['END_DT'].ffill()
    starts = group['START_DT'].tolist()
    ends = group['END_DT'].tolist()
    for i in range(len(group)):
      if pd.isna(ends[i]):
        if i + 1 < len(group):
          ends[i] = starts[i + 1]
        else:
          ends[i] = IB_MAX_DT
    group['END_DT'] = pd.to_datetime(ends)
    return group

  work = grouped.apply(_fill_end).reset_index(drop=True)

  # Optional filter Status = 'Active' if such a column exists
  status_col = next((c for c in df.columns if c.lower() == 'status'), None)
  if status_col:
    active_mask = df[status_col].str.lower() == 'active'
    work = work.loc[active_mask.values]

  # Distinct combinations
  ff_assort_dates = work[['PRODUCT_ID', 'LOCATION_ID', 'CUSTOMER_ID', 'DISTR_CHANNEL_ID', 'START_DT', 'END_DT']].drop_duplicates()
  return ff_assort_dates


def _combine_life_table(life_df, id_col, successor_col, start_col, end_col, extra_keys):
  """
  Implements step 3.5 (part 1) for a single lifecycle table.
  """
  if life_df is None or life_df.empty:
    return pd.DataFrame(columns=[id_col] + extra_keys + [start_col, end_col])

  df = life_df.copy()
  df[start_col] = pd.to_datetime(df[start_col])
  df[end_col] = pd.to_datetime(df[end_col])

  # a) successor missing
  a_df = df[df[successor_col].isna()][[id_col] + extra_keys + [start_col, end_col]]

  # b) successor differs -> use successor id as key
  b_df = df[df[successor_col].notna() & (df[id_col] != df[successor_col])][[successor_col] + extra_keys + [start_col, end_col]].copy()
  b_df.columns = [id_col] + extra_keys + [start_col, end_col]

  # c) successor equals id
  c_df = df[df[successor_col].notna() & (df[id_col] == df[successor_col])][[id_col] + extra_keys + [start_col, end_col]]

  combined = pd.concat([a_df, b_df, c_df], axis=0, ignore_index=True).drop_duplicates()
  return combined


def process_life_cycles(IN_CUSTOMER_LIFE_UPDATE_FF=None, IN_LOCATION_LIFE_UPDATE_FF=None, IN_PRODUCT_LIFE_UPDATE_FF=None):
  """
  3.5 Life-cycle information merging

  Returns
  -------
  pd.DataFrame
    FF_LIFE_DATES with columns:
    PRODUCT_ID, LOCATION_ID, CUSTOMER_ID, DISTR_CHANNEL_ID,
    PERIOD_START_DT, PERIOD_END_DT
  """
  combined_tables = []

  def _detect_cols(df, id_label):
    id_col = next((c for c in df.columns if c.upper() == f"{id_label}_ID"), None)
    successor_col = next((c for c in df.columns if c.upper() == f"{id_label}_SUCCESSOR_ID"), None)
    start_col = next((c for c in df.columns if 'PERIOD_START_DT' in c.upper()), None)
    end_col = next((c for c in df.columns if 'PERIOD_END_DT' in c.upper()), None)
    return id_col, successor_col, start_col, end_col

  def _process_single(df, id_label):
    if df is None:
      return None
    id_col, succ_col, start_col, end_col = _detect_cols(df, id_label)
    if None in [id_col, succ_col, start_col, end_col]:
      raise ValueError(f"Lifecycle table for {id_label} is missing required columns")

    # extra keys: keep any of the standard dimension cols that exist
    possible_keys = ['product_id', 'location_id', 'customer_id', 'distr_channel_id',
                     'product_lvl_id', 'location_lvl_id', 'customer_lvl_id', 'distr_channel_lvl_id']
    extra_keys = [c for c in df.columns if c in possible_keys and c.upper() != f"{id_label}_ID" and c.upper() != f"{id_label}_SUCCESSOR_ID"]

    combined = _combine_life_table(df, id_col, succ_col, start_col, end_col, extra_keys)
    combined.columns = [id_col] + extra_keys + ['PERIOD_START_DT', 'PERIOD_END_DT']
    return combined
  
# ДОБАВИТЬ КОММЕНТОВ
  customer_combined = _process_single(IN_CUSTOMER_LIFE_UPDATE_FF, 'CUSTOMER')
  location_combined = _process_single(IN_LOCATION_LIFE_UPDATE_FF, 'LOCATION')
  product_combined = _process_single(IN_PRODUCT_LIFE_UPDATE_FF, 'PRODUCT')

  for tbl in [customer_combined, location_combined, product_combined]:
    if tbl is not None and not tbl.empty:
      combined_tables.append(tbl)

  if not combined_tables:
    return pd.DataFrame(columns=['PRODUCT_ID', 'LOCATION_ID', 'CUSTOMER_ID', 'DISTR_CHANNEL_ID', 'PERIOD_START_DT', 'PERIOD_END_DT'])

  ff_life_dates = pd.concat(combined_tables, axis=0, ignore_index=True)

  # Attempt to unfold to lowest level; if hierarchy levels exist, leave as-is
  # Standardize column names where available
  rename_map = {c: c.upper() for c in ff_life_dates.columns}
  ff_life_dates = ff_life_dates.rename(columns=rename_map)

  # Ensure all key columns exist
  for col in ['PRODUCT_ID', 'LOCATION_ID', 'CUSTOMER_ID', 'DISTR_CHANNEL_ID']:
    if col not in ff_life_dates.columns:
      ff_life_dates[col] = np.nan

  ff_life_dates = ff_life_dates[['PRODUCT_ID', 'LOCATION_ID', 'CUSTOMER_ID', 'DISTR_CHANNEL_ID', 'PERIOD_START_DT', 'PERIOD_END_DT']]
  ff_life_dates = ff_life_dates.drop_duplicates()

  # Merge overlapping intervals with 1 day tolerance
  ff_life_dates = _merge_intervals(ff_life_dates, ['PRODUCT_ID', 'LOCATION_ID', 'CUSTOMER_ID', 'DISTR_CHANNEL_ID'], 'PERIOD_START_DT', 'PERIOD_END_DT', distance_tolerance=1)

  return ff_life_dates


def calculate_forecast_flag(FF_FACT_DATES, FF_ASSORT_DATES, FF_LIFE_DATES, FORECAST_FLAG=None, QUADRUPLES_DELETE=None):
  """
  3.6 Forecast Flag Calculation
  
  **FORECASTING GUIDANCE**: This function creates the Forecast Flag table that tells the forecasting
  algorithm WHICH combinations and time periods should be forecasted. The output table acts as a 
  filter/guide for forecasting:
  - Specifies valid PRODUCT_ID, LOCATION_ID, CUSTOMER_ID, DISTR_CHANNEL_ID combinations
  - Defines valid time periods (PERIOD_START_DT to PERIOD_END_DT) for each combination
  - Marks STATUS='active' to indicate these should be forecasted
  
  The forecasting algorithm should:
  1. Only forecast combinations present in this table
  2. Only forecast for dates within PERIOD_START_DT to PERIOD_END_DT for each combination
  3. Skip combinations/dates not in this table (they are discontinued, invalid, or not available)

  Returns
  -------
  pd.DataFrame
    Updated FORECAST_FLAG table with columns:
    - PRODUCT_ID, LOCATION_ID, CUSTOMER_ID, DISTR_CHANNEL_ID: Which combinations to forecast
    - PERIOD_START_DT, PERIOD_END_DT: When to forecast (valid time periods)
    - STATUS: 'active' indicates these should be forecasted
  """
  tables = []
  for tbl in [FF_FACT_DATES, FF_ASSORT_DATES, FF_LIFE_DATES]:
    if tbl is not None and not tbl.empty:
      tables.append(tbl.copy())

  if not tables:
    return FORECAST_FLAG.copy() if FORECAST_FLAG is not None else pd.DataFrame()

  def _std_dates(df):
    df = df.copy()
    # normalize column names
    rename = {c: c.upper() for c in df.columns}
    df = df.rename(columns=rename)
    # map possible date names
    if 'START_DT' not in df.columns and 'PERIOD_START_DT' in df.columns:
      df['START_DT'] = df['PERIOD_START_DT']
    if 'END_DT' not in df.columns and 'PERIOD_END_DT' in df.columns:
      df['END_DT'] = df['PERIOD_END_DT']
    for col in ['START_DT', 'END_DT']:
      if col not in df.columns:
        raise ValueError("Input tables must contain start/end dates")
      df[col] = pd.to_datetime(df[col])
    # ensure keys exist
    for col in ['PRODUCT_ID', 'LOCATION_ID', 'CUSTOMER_ID', 'DISTR_CHANNEL_ID']:
      if col not in df.columns:
        df[col] = np.nan
    return df[['PRODUCT_ID', 'LOCATION_ID', 'CUSTOMER_ID', 'DISTR_CHANNEL_ID', 'START_DT', 'END_DT']]

  # *** FORECASTING GUIDANCE PART 1: Combine all valid date sources ***
  # Combines dates from: fact data, assortment matrix, and lifecycle information
  # Each source provides valid time periods when combinations should be forecasted
  standardized = [_std_dates(t) for t in tables]
  ff_dates = pd.concat(standardized, axis=0, ignore_index=True)

  # *** FORECASTING GUIDANCE PART 2: Merge overlapping time intervals ***
  # Creates continuous valid periods for each combination
  # This tells the forecast algorithm: "For this combination, forecast from START to END date"
  ff_dates_merged = _merge_intervals(ff_dates, ['PRODUCT_ID', 'LOCATION_ID', 'CUSTOMER_ID', 'DISTR_CHANNEL_ID'], 'START_DT', 'END_DT', distance_tolerance=1)
  ff_dates_merged = ff_dates_merged.rename(columns={'START_DT': 'PERIOD_START_DT', 'END_DT': 'PERIOD_END_DT'})
  ff_dates_merged['STATUS'] = 'active'  # Mark as active = should be forecasted

  # *** HARD DELETE: drop any combinations present in QUADRUPLES_DELETE (del_flag==1) ***
  # This ensures anything flagged for deletion in source tables is NOT carried forward
  if QUADRUPLES_DELETE is not None and not QUADRUPLES_DELETE.empty:
    del_keys = QUADRUPLES_DELETE.rename(columns={c: c.upper() for c in QUADRUPLES_DELETE.columns})[['PRODUCT_ID', 'LOCATION_ID', 'CUSTOMER_ID', 'DISTR_CHANNEL_ID']].drop_duplicates()
    key_cols = ['PRODUCT_ID', 'LOCATION_ID', 'CUSTOMER_ID', 'DISTR_CHANNEL_ID']
    ff_dates_merged = ff_dates_merged.merge(del_keys.assign(_del=1), on=key_cols, how='left')
    ff_dates_merged = ff_dates_merged[ff_dates_merged['_del'].isna()].drop(columns=['_del'])

  if FORECAST_FLAG is not None and not FORECAST_FLAG.empty:
    existing = FORECAST_FLAG.copy()
    existing = existing.rename(columns={c: c.upper() for c in existing.columns})
  else:
    existing = pd.DataFrame(columns=['PRODUCT_ID', 'LOCATION_ID', 'CUSTOMER_ID', 'DISTR_CHANNEL_ID', 'PERIOD_START_DT', 'PERIOD_END_DT', 'STATUS'])

  # *** FORECASTING GUIDANCE PART 3: Remove deleted combinations ***
  # Tells forecasting: "DO NOT forecast these combinations - they are deleted/discontinued"
  if QUADRUPLES_DELETE is not None and not QUADRUPLES_DELETE.empty:
    del_keys = QUADRUPLES_DELETE.rename(columns={c: c.upper() for c in QUADRUPLES_DELETE.columns})[['PRODUCT_ID', 'LOCATION_ID', 'CUSTOMER_ID', 'DISTR_CHANNEL_ID']].drop_duplicates()
    key_cols = ['PRODUCT_ID', 'LOCATION_ID', 'CUSTOMER_ID', 'DISTR_CHANNEL_ID']
    existing = existing.merge(del_keys.assign(_del=1), on=key_cols, how='left')
    existing = existing[existing['_del'].isna()].drop(columns=['_del'])

  # *** FORECASTING GUIDANCE PART 4: Final merge of existing and new valid periods ***
  # Creates final table with all valid combinations and their time periods
  combined = pd.concat([existing, ff_dates_merged], axis=0, ignore_index=True)
  combined = _merge_intervals(combined, ['PRODUCT_ID', 'LOCATION_ID', 'CUSTOMER_ID', 'DISTR_CHANNEL_ID'], 'PERIOD_START_DT', 'PERIOD_END_DT', distance_tolerance=1)
  combined['STATUS'] = 'active'  # Final status: all entries are active = should be forecasted

  # *** OUTPUT: This table is used by forecasting algorithm to know:
  #   - WHAT to forecast: combinations in this table
  #   - WHEN to forecast: dates between PERIOD_START_DT and PERIOD_END_DT
  #   - What NOT to forecast: anything not in this table
  return combined


def filter_dictionaries(FORECAST_FLAG, PRODUCT, LOCATION, CUSTOMER, DISTR_CHANNEL):
  """
  3.7 Filter out excessive and obsolete elements of dictionaries.
  
  **FORECASTING GUIDANCE**: This function filters dimension tables (dictionaries) to only include
  elements that appear in FORECAST_FLAG. This tells the forecasting algorithm:
  - Only use these PRODUCTs in forecasting (remove discontinued products)
  - Only use these LOCATIONs in forecasting (remove closed locations)
  - Only use these CUSTOMERs in forecasting (remove inactive customers)
  - Only use these DISTR_CHANNELs in forecasting (remove inactive channels)
  
  This prevents the forecasting algorithm from trying to forecast combinations that don't exist
  or are no longer active, saving computational resources and preventing invalid forecasts.

  Returns
  -------
  tuple(pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame)
    PRODUCT_FILTERED, LOCATION_FILTERED, CUSTOMER_FILTERED, DISTR_CHANNEL_FILTERED
    These filtered dictionaries only contain elements that should be used in forecasting.
  """
  ff = FORECAST_FLAG.rename(columns={c: c.upper() for c in FORECAST_FLAG.columns})

  # *** FORECASTING GUIDANCE: Filter each dimension to only include IDs present in Forecast Flag ***
  # This ensures forecasting only works with valid, active elements
  def _filter(dim_df, key_col):
    if dim_df is None or dim_df.empty:
      return pd.DataFrame()
    dim_df = dim_df.copy()
    match_col = next((c for c in dim_df.columns if c.upper() == key_col), None)
    if match_col is None:
      raise ValueError(f"Dictionary missing required column {key_col}")
    # Only keep dimension elements that are in Forecast Flag (i.e., should be forecasted)
    filtered = dim_df[dim_df[match_col].isin(ff[key_col].dropna().unique())]
    return filtered

  prod_f = _filter(PRODUCT, 'PRODUCT_ID')
  loc_f = _filter(LOCATION, 'LOCATION_ID')
  cust_f = _filter(CUSTOMER, 'CUSTOMER_ID')
  distr_f = _filter(DISTR_CHANNEL, 'DISTR_CHANNEL_ID')

  return prod_f, loc_f, cust_f, distr_f


