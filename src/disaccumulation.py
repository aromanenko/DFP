import os
import glob
import pandas as pd


def ib_dfp_read_csv(path):
    """
    Read a CSV file and normalize column names to upper case.

    :param path: Path to the CSV file.
    :return: Pandas DataFrame with upper-case column names.
    """
    df = pd.read_csv(path)
    df.columns = [c.upper() for c in df.columns]
    return df


def ib_dfp_find_forecast_file(data_dir):
    """
    Locate the input forecast file in the data directory.

    The function searches for files matching the patterns
    'AGG_HYB_FCST*.csv' and 'DPS_AGG_HYB_FCST*.csv' and returns
    the first match in lexicographical order.

    :param data_dir: Path to the data folder.
    :return: Path to the forecast CSV file.
    :raises FileNotFoundError: If no matching files are found.
    """
    patterns = [
        os.path.join(data_dir, 'AGG_HYB_FCST*.csv'),
        os.path.join(data_dir, 'DPS_AGG_HYB_FCST*.csv'),
    ]

    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))

    files = sorted(set(files))
    if not files:
        raise FileNotFoundError('AGG_HYB_FCST*.csv not found in data folder')

    return files[0]


def ib_dfp_week_start(ts):
    """
    Compute the Monday (week start) for a given timestamp.

    This function uses the standard pandas Timestamp.dayofweek
    attribute (0 = Monday, ..., 6 = Sunday) and returns the
    normalized timestamp of the corresponding Monday.

    :param ts: A pandas-compatible datetime value.
    :return: Timestamp representing the Monday of the same week.
    """
    ts = pd.Timestamp(ts)
    return (ts - pd.to_timedelta(ts.dayofweek, unit='D')).normalize()


def ib_dfp_bucket_date(ts, out_time_lvl):
    """
    Map a date to a time bucket according to the target granularity.

    * 'DAY'    -> each calendar date forms its own bucket
    * 'WEEK.2' -> dates are mapped to the Monday of the corresponding week

    :param ts: A pandas-compatible datetime value.
    :param out_time_lvl: Target time level ('DAY' or 'WEEK.2').
    :return: Timestamp representing the bucket identifier.
    :raises ValueError: If an unsupported time level is provided.
    """
    ts = pd.Timestamp(ts)

    if out_time_lvl == 'DAY':
        return ts.normalize()

    if out_time_lvl == 'WEEK.2':
        return ib_dfp_week_start(ts)

    raise ValueError('Unsupported out_time_lvl: %s' % out_time_lvl)


def ib_dfp_final_granularity_delivered(df, out_time_lvl):
    """
    Check whether the target time granularity is already delivered.

    The idea is the following: for each forecast row we look at
    the interval [PERIOD_DT, PERIOD_END_DT]. If both endpoints of
    the interval fall into the *same* time bucket under
    `out_time_lvl`, then no further disaccumulation is needed.

    :param df: Forecast DataFrame.
    :param out_time_lvl: Target time level ('DAY' or 'WEEK.2').
    :return: True if no disaccumulation is required, False otherwise.
    """
    if df.empty:
        return True

    x = df[['PERIOD_DT', 'PERIOD_END_DT']].dropna()
    if x.empty:
        return True

    s = x['PERIOD_DT'].apply(lambda z: ib_dfp_bucket_date(z, out_time_lvl))
    e = x['PERIOD_END_DT'].apply(lambda z: ib_dfp_bucket_date(z, out_time_lvl))

    bad = (s != e)
    return not bad.any()


def ib_dfp_value_columns(df):
    """
    Detect forecast value columns present in the DataFrame.

    :param df: Forecast DataFrame.
    :return: List of column names that store forecast values.
    """
    candidates = ['VF_FORECAST_VALUE', 'ML_FORECAST_VALUE', 'HYBRID_FORECAST_VALUE']
    return [c for c in candidates if c in df.columns]


def ib_dfp_disaccumulate(df, out_time_lvl='DAY'):
    """
    Apply the forecast disaccumulation algorithm in a vectorized way.

    Vectorized approach:
      1. Convert PERIOD_DT and PERIOD_END_DT to datetime.
      2. Check if final granularity is already delivered.
      3. For each row build a daily date range [PERIOD_DT, PERIOD_END_DT].
      4. Explode all ranges into one big table of days.
      5. Map each day to a bucket (DAY or WEEK.2) using vectorized ops.
      6. Group by (ROW_ID, BUCKET) to:
         - get PERIOD_DT (min day in bucket),
         - get PERIOD_END_DT (max day in bucket),
         - get number of days in bucket.
      7. Compute share = days_in_bucket / total_days_in_original_interval.
      8. Multiply each forecast value column by this share.

    :param df: Input forecast DataFrame (AGG_HYB_FCST).
    :param out_time_lvl: Target time level ('DAY' or 'WEEK.2').
    :return: DataFrame with disaccumulated forecast.
    """
    if df.empty:
        return df

    df = df.copy()
    df['PERIOD_DT'] = pd.to_datetime(df['PERIOD_DT'], errors='coerce')
    df['PERIOD_END_DT'] = pd.to_datetime(df['PERIOD_END_DT'], errors='coerce')

    if ib_dfp_final_granularity_delivered(df, out_time_lvl):
        return df

    value_cols = ib_dfp_value_columns(df)
    if not value_cols:
        return df

    # 1) assign row id
    df = df.reset_index(drop=True)
    df['ROW_ID'] = df.index

    # 2) build daily date ranges per row
    day_ranges = []
    for s, e in zip(df['PERIOD_DT'], df['PERIOD_END_DT']):
        if pd.isna(s) or pd.isna(e):
            day_ranges.append(pd.DatetimeIndex([]))
            continue
        s = pd.Timestamp(s).normalize()
        e = pd.Timestamp(e).normalize()
        if e < s:
            day_ranges.append(pd.DatetimeIndex([]))
            continue
        day_ranges.append(pd.date_range(start=s, end=e, freq='D'))

    df['DAY_LIST'] = day_ranges

    # 3) explode into one row per day
    days_df = df.explode('DAY_LIST')
    days_df = days_df.rename(columns={'DAY_LIST': 'DAY'})
    days_df = days_df[days_df['DAY'].notna()]

    if days_df.empty:
        return df

    # 4) vectorized bucket mapping
    if out_time_lvl == 'DAY':
        days_df['BUCKET'] = days_df['DAY'].dt.normalize()
    elif out_time_lvl == 'WEEK.2':
        dow = days_df['DAY'].dt.dayofweek
        days_df['BUCKET'] = (
            days_df['DAY'] - pd.to_timedelta(dow, unit='D')
        ).dt.normalize()
    else:
        raise ValueError('Unsupported out_time_lvl: %s' % out_time_lvl)

    # 5) aggregate by (ROW_ID, BUCKET): get start, end, number of days
    grp = (
        days_df
        .groupby(['ROW_ID', 'BUCKET'])['DAY']
        .agg(PERIOD_DT='min',
             PERIOD_END_DT='max',
             DAYS_IN_BUCKET='count')
        .reset_index()
    )

    # 6) total days per original row and share
    total_days = grp.groupby('ROW_ID')['DAYS_IN_BUCKET'].transform('sum')
    grp['SHARE'] = grp['DAYS_IN_BUCKET'] / total_days

    # 7) bring back all other columns (dimensions + value columns)
    drop_cols = ['PERIOD_DT', 'PERIOD_END_DT', 'DAY_LIST']
    base_cols = [c for c in df.columns if c not in drop_cols]
    base = df[base_cols]

    res = grp.merge(base, on='ROW_ID', how='left')

    # 8) scale value columns by share
    for c in value_cols:
        res[c] = pd.to_numeric(res[c], errors='coerce') * res['SHARE']

    # 9) cleanup
    res = res.drop(columns=['DAYS_IN_BUCKET', 'SHARE'])
    # we no longer need ROW_ID in final result
    if 'ROW_ID' in res.columns:
        res = res.drop(columns=['ROW_ID'])

    # put PERIOD_DT/PERIOD_END_DT first for readability
    front = ['PERIOD_DT', 'PERIOD_END_DT']
    other = [c for c in res.columns if c not in front]
    res = res[front + other]

    return res


def run(
    data_dir,
    out_time_lvl='DAY',
    fcst_file=None,
    out_file='ACC_AGG_HYBRID_FORECAST.csv',
):
    """
    High-level entry point for running forecast disaccumulation on CSV files.

    The function encapsulates the full pipeline:
      * locating the input forecast file (if not provided explicitly);
      * reading it via `ib_dfp_read_csv`;
      * running the core disaccumulation algorithm;
      * writing the result to a CSV file in the same directory.

    :param data_dir: Path to the data directory containing the forecast file.
    :param out_time_lvl: Target time level ('DAY' or 'WEEK.2').
    :param fcst_file: Optional explicit path to the forecast CSV file.
    :param out_file: Name of the output CSV file.
    :return: Tuple (input_df, output_df, output_path).
    """
    if fcst_file is None:
        fcst_file = ib_dfp_find_forecast_file(data_dir)

    df = ib_dfp_read_csv(fcst_file)
    inp = df.copy()

    out = ib_dfp_disaccumulate(df, out_time_lvl=out_time_lvl)

    out_path = os.path.join(data_dir, out_file)
    out.to_csv(out_path, index=False)

    return inp, out, out_path
