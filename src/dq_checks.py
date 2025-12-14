import os
import glob
import pandas as pd

# Required input tables for DQ checks
REQ = [
    'DPS_ASSORT_MATRIX',
    'DPS_CUSTOMER',
    'DPS_CUSTOMER_ATTR',
    'DPS_DISTR_CHANNEL',
    'DPS_DISTR_CHANNEL_ATTR',
    'DPS_LOCATION',
    'DPS_LOCATION_ATTR',
    'DPS_LOCATION_LIFE',
    'DPS_PRICE',
    'DPS_PRODUCT',
    'DPS_PRODUCT_ATTR',
    'DPS_PRODUCT_LIFE',
    'DPS_PROMO',
    'DPS_PROMO_ATTR',
    'DPS_PROMO_TYPE',
    'DPS_SELL_IN',
    'DPS_SELL_OUT',
    'DPS_STOCK',
    'DPS_SALES',
]

# Tables to use for cross-key checks
CROSS = [
    'DPS_SALES',
    'DPS_STOCK',
    'DPS_PRICE',
    'DPS_PRODUCT',
    'DPS_LOCATION',
    'DPS_CUSTOMER',
    'DPS_DISTR_CHANNEL',
]

# Pairs of tables for time-based cross-checks
TPAIRS = [
    ('DPS_SALES', 'DPS_STOCK'),
    ('DPS_STOCK', 'DPS_SALES'),
    ('DPS_STOCK', 'DPS_ASSORT_MATRIX'),
    ('DPS_SALES', 'DPS_ASSORT_MATRIX'),
    ('DPS_STOCK', 'DPS_PRICE'),
    ('DPS_SALES', 'DPS_PRICE'),
]

# All potential key columns
KEYS_ALL = ['PRODUCT_ID', 'LOCATION_ID', 'CUSTOMER_ID', 'DISTR_CHANNEL_ID']


def ib_dfp_get_table_name(fn):
    """
    Extract a logical table name from a CSV filename.

    :param fn: Path to a CSV file.
    :return: Uppercase table name without extension.
    """
    base = os.path.basename(fn)
    return os.path.splitext(base)[0].upper()


def ib_dfp_read_dir(path):
    """
    Read input CSV files from a folder into a dict of DataFrames.

    The function:
      * reads only DPS_*.csv and optional *DQ_PARAMETERS*.csv,
      * skips data_quality_output.csv,
      * safely handles empty files,
      * converts column names to uppercase for consistency.

    :param path: Path to folder with CSV files.
    :return: Dict {table_name: DataFrame}.
    """
    path = str(path)

    # Collect all candidate files
    files = []
    files += glob.glob(os.path.join(path, 'DPS_*.csv'))
    files += glob.glob(os.path.join(path, '*DQ_PARAMETERS*.csv'))

    tables = {}

    for f in files:
        table_name = ib_dfp_get_table_name(f)

        # Skip our own DQ output, if present in the same folder
        if os.path.basename(f).lower() == 'data_quality_output.csv':
            continue

        # Handle truly empty files
        if os.path.getsize(f) == 0:
            df = pd.DataFrame()
        else:
            try:
                df = pd.read_csv(f)
            except pd.errors.EmptyDataError:
                df = pd.DataFrame()

        # Normalize column names
        if not df.empty:
            df.columns = [c.upper() for c in df.columns]

        tables[table_name] = df

    return tables


def ib_dfp_empty_df():
    """
    Create an empty DataFrame.

    :return: Empty pandas DataFrame.
    """
    return pd.DataFrame()


def ib_dfp_ensure_required(tables):
    """
    Ensure that all required tables exist in the dictionary.

    Missing tables are added as empty DataFrames.

    :param tables: Dict of DataFrames.
    :return: Updated dict with all required keys present.
    """
    for t in REQ:
        if t not in tables:
            tables[t] = ib_dfp_empty_df()
    return tables


def ib_dfp_output_columns():
    """
    Provide the schema of the data_quality_output.

    :return: List of column names in required order.
    """
    return [
        'PRODUCT_LVL_ID',
        'LOCATION_LVL_ID',
        'CUSTOMER_LVL_ID',
        'DISTR_CHANNEL_LVL_ID',
        'PRODUCT_LVL',
        'LOCATION_LVL',
        'CUSTOMER_LVL',
        'DISTR_CHANNEL_LVL',
        'PERIOD_DT',
        'INPUT_COLUMN',
        'INPUT_TABLE',
        'INPUT_VALUE',
        'WARNING_TYPE',
        'WARNING',
    ]


def ib_dfp_pick_time_column(df):
    """
    Detect a time column in a DataFrame.

    The function looks for known candidates in priority order.

    :param df: Input DataFrame.
    :return: Time column name or None.
    """
    for c in ['PERIOD_DT', 'PERIOD_START_DT', 'START_DT']:
        if c in df.columns:
            return c
    return None


def ib_dfp_build_level_ids(df):
    """
    Build level/id fields based on leaf keys.

    This simplified mapping uses *_ID columns directly
    as *_LVL_ID in the output.

    :param df: Source DataFrame.
    :return: DataFrame with level/id columns for output.
    """
    res = pd.DataFrame(index=df.index)
    res['PRODUCT_LVL_ID'] = df['PRODUCT_ID'] if 'PRODUCT_ID' in df.columns else None
    res['LOCATION_LVL_ID'] = df['LOCATION_ID'] if 'LOCATION_ID' in df.columns else None
    res['CUSTOMER_LVL_ID'] = df['CUSTOMER_ID'] if 'CUSTOMER_ID' in df.columns else None
    res['DISTR_CHANNEL_LVL_ID'] = (
        df['DISTR_CHANNEL_ID'] if 'DISTR_CHANNEL_ID' in df.columns else None
    )
    # In this simplified version hierarchy levels are not resolved
    res['PRODUCT_LVL'] = None
    res['LOCATION_LVL'] = None
    res['CUSTOMER_LVL'] = None
    res['DISTR_CHANNEL_LVL'] = None
    return res


def ib_dfp_make_output_block(df, table_name, column_name, warning_type, warning_text, value_column):
    """
    Convert problem rows into data_quality_output format.

    :param df: DataFrame with problematic rows.
    :param table_name: Source table name.
    :param column_name: Source column name (or list joined as string).
    :param warning_type: Warning type identifier.
    :param warning_text: Human-readable warning message.
    :param value_column: Column name to copy as INPUT_VALUE (can be None).
    :return: Formatted DataFrame with output schema.
    """
    base = ib_dfp_build_level_ids(df)
    tc = ib_dfp_pick_time_column(df)

    # Copy time information if available
    if tc is not None:
        base['PERIOD_DT'] = df[tc].values
    else:
        base['PERIOD_DT'] = None

    base['INPUT_COLUMN'] = column_name
    base['INPUT_TABLE'] = table_name

    # Copy value for INPUT_VALUE when requested
    if value_column is not None and value_column in df.columns:
        base['INPUT_VALUE'] = df[value_column].values
    else:
        base['INPUT_VALUE'] = None

    base['WARNING_TYPE'] = warning_type
    base['WARNING'] = warning_text
    base = base[ib_dfp_output_columns()]
    return base


def ib_dfp_append_output(dq, part):
    """
    Append a partial output block to the main output.

    :param dq: Main output DataFrame.
    :param part: Partial output DataFrame.
    :return: Combined DataFrame.
    """
    if part is None or part.empty:
        return dq
    return pd.concat([dq, part], ignore_index=True)


def ib_dfp_negative_columns_sell(df):
    """
    Detect quantity/amount-like columns for SELL_IN/SELL_OUT checks.

    :param df: Source DataFrame.
    :return: List of candidate numeric columns.
    """
    cols = []
    for c in df.columns:
        if c.endswith('_QTY') or c.endswith('_AMOUNT') or c == 'COST':
            cols.append(c)
    return cols


def ib_dfp_check_negative_values(tables):
    """
    Run negative value checks for key numeric fields.

    Covers PRICE, PROMO_PRICE, SALES_QTY, STOCK_QTY
    and generic *_QTY/*_AMOUNT fields in SELL_IN/SELL_OUT.

    :param tables: Dict of input DataFrames.
    :return: Output DataFrame with NEGATIVE_VALUE warnings.
    """
    dq = pd.DataFrame(columns=ib_dfp_output_columns())

    # PRICE < 0
    if 'DPS_PRICE' in tables and not tables['DPS_PRICE'].empty and 'PRICE' in tables['DPS_PRICE'].columns:
        df = tables['DPS_PRICE']
        x = pd.to_numeric(df['PRICE'], errors='coerce')
        bad = df[x < 0]
        dq = ib_dfp_append_output(
            dq,
            ib_dfp_make_output_block(
                bad,
                'DPS_PRICE',
                'PRICE',
                'NEGATIVE_VALUE',
                'Negative price in DPS_PRICE',
                'PRICE',
            ),
        )

    # PROMO_PRICE < 0
    if 'DPS_PROMO' in tables and not tables['DPS_PROMO'].empty and 'PROMO_PRICE' in tables['DPS_PROMO'].columns:
        df = tables['DPS_PROMO']
        x = pd.to_numeric(df['PROMO_PRICE'], errors='coerce')
        bad = df[x < 0]
        dq = ib_dfp_append_output(
            dq,
            ib_dfp_make_output_block(
                bad,
                'DPS_PROMO',
                'PROMO_PRICE',
                'NEGATIVE_VALUE',
                'Negative promo price in DPS_PROMO',
                'PROMO_PRICE',
            ),
        )

    # SALES_QTY < 0
    if 'DPS_SALES' in tables and not tables['DPS_SALES'].empty and 'SALES_QTY' in tables['DPS_SALES'].columns:
        df = tables['DPS_SALES']
        x = pd.to_numeric(df['SALES_QTY'], errors='coerce')
        bad = df[x < 0]
        dq = ib_dfp_append_output(
            dq,
            ib_dfp_make_output_block(
                bad,
                'DPS_SALES',
                'SALES_QTY',
                'NEGATIVE_VALUE',
                'Negative sales in DPS_SALES',
                'SALES_QTY',
            ),
        )

    # SELL_OUT numeric negatives
    if 'DPS_SELL_OUT' in tables and not tables['DPS_SELL_OUT'].empty:
        df = tables['DPS_SELL_OUT']
        for c in ib_dfp_negative_columns_sell(df):
            x = pd.to_numeric(df[c], errors='coerce')
            bad = df[x < 0]
            dq = ib_dfp_append_output(
                dq,
                ib_dfp_make_output_block(
                    bad,
                    'DPS_SELL_OUT',
                    c,
                    'NEGATIVE_VALUE',
                    'Negative value in DPS_SELL_OUT',
                    c,
                ),
            )

    # SELL_IN numeric negatives
    if 'DPS_SELL_IN' in tables and not tables['DPS_SELL_IN'].empty:
        df = tables['DPS_SELL_IN']
        for c in ib_dfp_negative_columns_sell(df):
            x = pd.to_numeric(df[c], errors='coerce')
            bad = df[x < 0]
            dq = ib_dfp_append_output(
                dq,
                ib_dfp_make_output_block(
                    bad,
                    'DPS_SELL_IN',
                    c,
                    'NEGATIVE_VALUE',
                    'Negative value in DPS_SELL_IN',
                    c,
                ),
            )

    # STOCK_QTY < 0
    if 'DPS_STOCK' in tables and not tables['DPS_STOCK'].empty and 'STOCK_QTY' in tables['DPS_STOCK'].columns:
        df = tables['DPS_STOCK']
        x = pd.to_numeric(df['STOCK_QTY'], errors='coerce')
        bad = df[x < 0]
        dq = ib_dfp_append_output(
            dq,
            ib_dfp_make_output_block(
                bad,
                'DPS_STOCK',
                'STOCK_QTY',
                'NEGATIVE_VALUE',
                'Negative stock in DPS_STOCK',
                'STOCK_QTY',
            ),
        )

    return dq


def ib_dfp_key_to_str(df, keys):
    """
    Build a string representation of a composite key.

    :param df: DataFrame containing key columns.
    :param keys: List of key column names.
    :return: Series of joined key values.
    """
    if df.empty:
        return pd.Series([], dtype='object')
    return df[keys].astype(str).agg('|'.join, axis=1)


def ib_dfp_check_cross_keys(tables):
    """
    Run cross-table key consistency checks.

    For each ordered pair (A, B) in CROSS,
    finds keys present in A but missing in B.

    :param tables: Dict of input DataFrames.
    :return: Output DataFrame with MISSING_KEY warnings.
    """
    dq = pd.DataFrame(columns=ib_dfp_output_columns())

    for a in CROSS:
        for b in CROSS:
            if a == b:
                continue
            if a not in tables or b not in tables:
                continue

            df1 = tables[a]
            df2 = tables[b]

            if df1.empty or df2.empty:
                continue

            # Find keys that exist in both tables
            keys = [k for k in KEYS_ALL if k in df1.columns and k in df2.columns]
            if not keys:
                continue

            # Distinct keys in both tables
            k1 = df1[keys].drop_duplicates()
            k2 = df2[keys].drop_duplicates()

            # Left join to find missing keys in B
            joined = k1.merge(k2, on=keys, how='left', indicator=True)
            miss = joined[joined['_merge'] == 'left_only'].drop(columns=['_merge'])
            if miss.empty:
                continue

            coln = ','.join(keys)
            tmp = miss.copy()
            tmp['INPUT_VALUE'] = ib_dfp_key_to_str(tmp, keys).values

            part = ib_dfp_make_output_block(
                tmp,
                a,
                coln,
                'MISSING_KEY',
                'Key from %s not found in %s' % (a, b),
                None,
            )
            part['INPUT_VALUE'] = tmp['INPUT_VALUE'].values
            dq = ib_dfp_append_output(dq, part)

    return dq


def ib_dfp_get_parameters(tables):
    """
    Extract DQ parameters DataFrame if present.

    Any CSV with 'DQ_PARAMETERS' in the filename
    is treated as parameters storage.

    :param tables: Dict of input DataFrames.
    :return: Parameters DataFrame or empty DataFrame.
    """
    for k in tables.keys():
        if 'DQ_PARAMETERS' in k:
            return tables[k]
    return ib_dfp_empty_df()


def ib_dfp_get_threshold(par):
    """
    Determine threshold for time cross checks.

    If no valid parameters found, returns default 0.1.

    :param par: Parameters DataFrame.
    :return: Float threshold.
    """
    if par.empty:
        return 0.1
    if 'VAR_NAME' not in par.columns or 'VAR_VALUE' not in par.columns:
        return 0.1

    p = par.copy()
    p['VAR_NAME'] = p['VAR_NAME'].astype(str).str.upper()
    s = p[p['VAR_NAME'].str.contains('THRESHOLD', na=False)]

    if s.empty:
        return 0.1

    v = s['VAR_VALUE'].iloc[0]
    try:
        return float(v)
    except Exception:
        return 0.1


def ib_dfp_find_time_column_pair(df1, df2):
    """
    Find a shared time column for a pair of DataFrames.

    :param df1: First DataFrame.
    :param df2: Second DataFrame.
    :return: Shared time column name or None.
    """
    for c in ['PERIOD_DT', 'PERIOD_START_DT']:
        if c in df1.columns and c in df2.columns:
            return c
    return None


def ib_dfp_check_time_consistency(tables):
    """
    Run time-based cross-consistency checks.

    For each pair (A, B) in TPAIRS,
    checks missing (PRODUCT_ID, LOCATION_ID, PERIOD) combos
    and reports only if missing share > threshold.

    :param tables: Dict of input DataFrames.
    :return: Output DataFrame with TIME_CROSS_MISSING warnings.
    """
    dq = pd.DataFrame(columns=ib_dfp_output_columns())
    par = ib_dfp_get_parameters(tables)
    thr = ib_dfp_get_threshold(par)

    for a, b in TPAIRS:
        if a not in tables or b not in tables:
            continue

        df1 = tables[a]
        df2 = tables[b]

        if df1.empty or df2.empty:
            continue

        # Keys that must be shared
        keys = [k for k in ['PRODUCT_ID', 'LOCATION_ID'] if k in df1.columns and k in df2.columns]
        if not keys:
            continue

        tc = ib_dfp_find_time_column_pair(df1, df2)
        if tc is None:
            continue

        # All unique key+time combos in both tables
        a_keys = df1[keys + [tc]].drop_duplicates()
        b_keys = df2[keys + [tc]].drop_duplicates()

        joined = a_keys.merge(b_keys, on=keys + [tc], how='left', indicator=True)
        miss = joined[joined['_merge'] == 'left_only'].drop(columns=['_merge'])
        if miss.empty:
            continue

        # Share of missing combinations
        share = float(len(miss)) / float(len(a_keys)) if len(a_keys) else 0.0
        if share <= thr:
            continue

        coln = ','.join(keys + [tc])
        tmp = miss.copy()
        tmp['INPUT_VALUE'] = ib_dfp_key_to_str(tmp, keys + [tc]).values

        part = ib_dfp_make_output_block(
            tmp,
            a,
            coln,
            'TIME_CROSS_MISSING',
            'Time cross missing in %s vs %s' % (a, b),
            None,
        )
        part['PERIOD_DT'] = tmp[tc].values
        part['INPUT_VALUE'] = tmp['INPUT_VALUE'].values
        dq = ib_dfp_append_output(dq, part)

    return dq


def run(path, out_csv='data_quality_output.csv'):
    """
    Run all DQ checks for a folder with CSV inputs.

    The function:
      * reads files,
      * ensures required tables exist,
      * runs negative value checks,
      * runs cross-table key checks,
      * runs time-based consistency checks,
      * writes the output CSV into the same folder.

    :param path: Path to folder with input CSV files.
    :param out_csv: Output filename.
    :return: Final data_quality_output DataFrame.
    """
    tables = ib_dfp_read_dir(path)
    tables = ib_dfp_ensure_required(tables)

    dq1 = ib_dfp_check_negative_values(tables)
    dq2 = ib_dfp_check_cross_keys(tables)
    dq3 = ib_dfp_check_time_consistency(tables)

    dq = pd.concat([dq1, dq2, dq3], ignore_index=True)
    out_path = os.path.join(path, out_csv)
    dq.to_csv(out_path, index=False)
    return dq
