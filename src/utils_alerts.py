"""
In this file I implement utilities for alerts, specifically:

- column normalisation
- table.column name processing
- config construction
- various aggregation methods
- datetime manipulation
- timegrid construction based on forecastingflag
"""

from __future__ import annotations

import re
from pathlib import Path
import pandas as pd
import numpy as np


# ##############################
# Basic utility functions
# ##############################

def normalize_columns(df):
    """
    Returns an UPPERCASE copy of df
    """
    if df is None:
        return None
    x = df.copy()
    x.columns = [str(c).upper() for c in x.columns]
    return x


def pick_first_existing_column(df, candidates):
    """
    Out of a list of candidate columns, locates the first one which exists in the df
    """
    if df is None:
        return None
    cols = set(str(c).upper() for c in df.columns)
    for c in candidates:
        cu = str(c).upper()
        if cu in cols:
            return cu
    return None


def parse_table_column_ref(x):
    """
    Parses a string of the form 'TABLE_POSTFIX.COLUMN'
    
    Output: table, column, kpi_name, tgt_type
    (tgt_type is imputed from the table's suffix: *_POS / *_SELLOUT / *_SELLIN)
    """
    x = str(x)
    if "." not in x:
        return x.strip().upper(), None, None, None

    tbl, col = x.split(".", 1)
    tbl = tbl.strip().upper()
    col = col.strip().upper()

    tgt_type = None
    m = re.search(r"_([A-Z]+)$", tbl)
    if m:
        tgt_type = m.group(1).lower()

    kpi_nm = f"{tgt_type}.{col}" if (tgt_type and col) else col
    return tbl, col, kpi_nm, tgt_type


def build_config(config_parameters=None, initial_global=None):
    """
    Joins CONFIG.CONFIG_PARAMETERS and initial_global into one dictionary object
    """
    cfg = {}
    if initial_global:
        cfg.update(initial_global)

    if config_parameters is None:
        return cfg

    cp = normalize_columns(config_parameters)
    name_col = pick_first_existing_column(cp, ["PARAMETER_NAME", "NAME"])
    val_col = pick_first_existing_column(cp, ["PARAMETER_VALUE", "VALUE"])
    if not name_col or not val_col:
        return cfg

    for _, r in cp[[name_col, val_col]].iterrows():
        k = str(r[name_col]).strip()
        if not k:
            continue
        cfg[k] = r[val_col]
    return cfg


def prepare_alert_params(alert_parameters):
    """
    Converts CONFIG.ALERT_PARAMETERS into an appropriate form for further operations

    with fields:
    - tgt_type
    - alert_id
    - al_product_lvl, al_location_lvl, al_customer_lvl, al_distr_channel_lvl, al_time_lvl
    - alert_threshold_val
    - Input_table_table
    - Input_column
    - also target_table_column, as it's referenced in the documentation multiple times

    if input_table_table has no suffix, this adds it via tgt_type
    """
    ap = normalize_columns(alert_parameters).copy()
    if ap is None:
        return pd.DataFrame()

    if "TARGET_TABLE_COLUMN" not in ap.columns:
        if "INPUT_TABLE_TABLE" in ap.columns and "INPUT_COLUMN" in ap.columns:
            tbl = ap["INPUT_TABLE_TABLE"].astype(str).str.upper().str.strip()
            col = ap["INPUT_COLUMN"].astype(str).str.upper().str.strip()

            if "TGT_TYPE" in ap.columns:
                tgt = ap["TGT_TYPE"].astype(str).str.upper().str.strip()
                known = ("_POS", "_SELLOUT", "_SELLIN")
                need_suffix = ~tbl.str.endswith(known)
                tbl = tbl.where(~need_suffix, tbl + "_" + tgt)

            ap["TARGET_TABLE_COLUMN"] = tbl + "." + col

    return ap


# ##############################
# Handling hierarchy
# ##############################

def col_for_hierarchy_level(prefix, lvl):
    """
    Returns the column name w/ respect to the hierarchy level
    """
    prefix = str(prefix).upper().strip()
    lvl = int(lvl)

    if prefix == "PRODUCT":
        if lvl >= 8:
            return "PRODUCT_ID"
        return f"PRODUCT_LVL_ID{lvl}"
    if prefix == "LOCATION":
        if lvl >= 6:
            return "LOCATION_ID"
        return f"LOCATION_LVL_ID{lvl}"
    if prefix == "CUSTOMER":
        if lvl >= 6:
            return "CUSTOMER_ID"
        return f"CUSTOMER_LVL_ID{lvl}"
    if prefix == "DISTR_CHANNEL":
        if lvl >= 2:
            return "DISTR_CHANNEL_ID"
        return f"DISTR_CHANNEL_LVL_ID{lvl}"
    return None


def add_level_columns(df, prod, loc, cust, dc, al_p, al_l, al_c, al_d):
    """
    adds hierarchy level columns to the df if they happen to not be there
    """
    x = normalize_columns(df)
    prod = normalize_columns(prod)
    loc = normalize_columns(loc)
    cust = normalize_columns(cust)
    dc = normalize_columns(dc)

    p_col = col_for_hierarchy_level("PRODUCT", al_p)
    l_col = col_for_hierarchy_level("LOCATION", al_l)
    c_col = col_for_hierarchy_level("CUSTOMER", al_c)
    d_col = col_for_hierarchy_level("DISTR_CHANNEL", al_d)

    out = x.copy()

    # join by keys 
    if prod is not None and p_col not in out.columns and "PRODUCT_ID" in out.columns and "PRODUCT_ID" in prod.columns:
        out = out.merge(prod, on="PRODUCT_ID", how="left", suffixes=("", "_P"))
    if loc is not None and l_col not in out.columns and "LOCATION_ID" in out.columns and "LOCATION_ID" in loc.columns:
        out = out.merge(loc, on="LOCATION_ID", how="left", suffixes=("", "_L"))
    if cust is not None and c_col not in out.columns and "CUSTOMER_ID" in out.columns and "CUSTOMER_ID" in cust.columns:
        out = out.merge(cust, on="CUSTOMER_ID", how="left", suffixes=("", "_C"))
    if dc is not None and d_col not in out.columns and "DISTR_CHANNEL_ID" in out.columns and "DISTR_CHANNEL_ID" in dc.columns:
        out = out.merge(dc, on="DISTR_CHANNEL_ID", how="left", suffixes=("", "_D"))

    return out, p_col, l_col, c_col, d_col


# ##############################
# Datetime conversion
# ##############################

def bucket_time(dt_ser, al_time_lvl):
    """
    Converts dt to granularity al_time_lvl: day / week.2 (неделя с понедельника) / week / month
    """
    al_time_lvl = str(al_time_lvl).lower().strip()
    dt = pd.to_datetime(dt_ser, errors="coerce")

    if al_time_lvl.startswith("day"):
        return dt.dt.normalize()

    if al_time_lvl.startswith("week.2"):
        # weeks starting from monday of the current week rather than from the current day
        return dt.dt.to_period("W-MON").dt.start_time.dt.normalize()

    if al_time_lvl.startswith("week"):
        return dt.dt.to_period("W").dt.start_time.dt.normalize()

    if al_time_lvl.startswith("month"):
        return dt.dt.to_period("M").dt.start_time.dt.normalize()

    return dt.dt.normalize()


# ##############################
# Aggregation
# ##############################

def agg_forecast(df, value_col, prod, loc, cust, dc, al_p, al_l, al_c, al_d, al_t):
    """
    Aggregates forecast
     
    Returns mean(value_col) by given levels + PERIOD_DT
    """
    if df is None:
        return pd.DataFrame()

    x = normalize_columns(df)
    value_col = str(value_col).upper()
    if value_col not in x.columns:
        return pd.DataFrame()

    x, p_col, l_col, c_col, d_col = add_level_columns(x, prod, loc, cust, dc, al_p, al_l, al_c, al_d)
    if "PERIOD_DT" not in x.columns:
        return pd.DataFrame()

    x["PERIOD_DT"] = bucket_time(x["PERIOD_DT"], al_t)

    gcols = [p_col, l_col, c_col, d_col, "PERIOD_DT"]
    gcols = [c for c in gcols if c in x.columns]

    # aggregation
    res = x.groupby(gcols, dropna=False)[value_col].mean().reset_index()
    res = res.rename(columns={value_col: "STAT_NOM_VAL"})
    return res


def agg_demand(df, prod, loc, cust, dc, al_p, al_l, al_c, al_d, al_t):
    """
    Aggregates demand 
    
    Returns mean(SALES_QTY_R) by levels + PERIOD_DT
    """
    if df is None:
        return pd.DataFrame()

    x = normalize_columns(df)
    use_col = pick_first_existing_column(x, ["SALES_QTY_R", "SALESTGT_QTY_R"]) # whichever of the two can be found first
    if use_col is None:
        return pd.DataFrame()

    x, p_col, l_col, c_col, d_col = add_level_columns(x, prod, loc, cust, dc, al_p, al_l, al_c, al_d)
    if "PERIOD_DT" not in x.columns:
        return pd.DataFrame()

    x["PERIOD_DT"] = bucket_time(x["PERIOD_DT"], al_t)

    gcols = [p_col, l_col, c_col, d_col, "PERIOD_DT"]
    gcols = [c for c in gcols if c in x.columns]

    # aggregation
    res = x.groupby(gcols, dropna=False)[use_col].mean().reset_index()
    res = res.rename(columns={use_col: "STAT_DEN_VAL"})
    return res


def agg_flags(df, prod, loc, cust, dc, al_p, al_l, al_c, al_d, active_status_list):
    """
    Aggregates FORECAST_FLAG on given alert levels
    from MIN_START_DT = min(PERIOD_START_DT)
    until MAX_END_DT   = max(PERIOD_END_DT)
    by (product/location/customer/distr_channel)
    """
    if df is None:
        return pd.DataFrame()

    x = normalize_columns(df)
    if "PERIOD_START_DT" not in x.columns or "PERIOD_END_DT" not in x.columns:
        return pd.DataFrame()
    if "STATUS" in x.columns and active_status_list:
        act = [str(s).lower() for s in active_status_list]
        x = x[x["STATUS"].astype(str).str.lower().isin(act)].copy()

    x, p_col, l_col, c_col, d_col = add_level_columns(x, prod, loc, cust, dc, al_p, al_l, al_c, al_d)

    gcols = [p_col, l_col, c_col, d_col]
    gcols = [c for c in gcols if c in x.columns]

    x["PERIOD_START_DT"] = pd.to_datetime(x["PERIOD_START_DT"], errors="coerce").dt.normalize()
    x["PERIOD_END_DT"] = pd.to_datetime(x["PERIOD_END_DT"], errors="coerce").dt.normalize()

    # aggregation
    res = x.groupby(gcols, dropna=False).agg(
        MIN_START_DT=("PERIOD_START_DT", "min"),
        MAX_END_DT=("PERIOD_END_DT", "max"),
    ).reset_index()
    return res


def shift_last_year(df, al_t):
    """
    Shifts the date by a year. If the chosen parameter is week, the shift is done
    by 52 weeks (364 days) - i.e., the total number of complete weeks in a year
    """
    if df is None or df.empty or "PERIOD_DT" not in df.columns:
        return df

    al_t = str(al_t).lower().strip()
    out = df.copy()
    dt = pd.to_datetime(out["PERIOD_DT"], errors="coerce")

    if al_t.startswith("month"):
        out["PERIOD_DT"] = dt + pd.DateOffset(months=12)
    elif al_t.startswith("week"):
        out["PERIOD_DT"] = dt + pd.to_timedelta(52 * 7, unit="D")
    else:
        out["PERIOD_DT"] = dt + pd.to_timedelta(365, unit="D")

    out["PERIOD_DT"] = pd.to_datetime(out["PERIOD_DT"], errors="coerce").dt.normalize()
    return out


def expand_period_grid(flags_agg, al_t, hist_end_dt, fc_horiz_weeks):
    """
    Constructs a time period grid (PERIOD_DI) based on aggregated flags
    For each, we obtain the following period:
    [max(MIN_START_DT, hist_end+1), min(MAX_END_DT, hist_end + fc_horiz_weeks*7)].
    """
    if flags_agg is None or flags_agg.empty:
        return pd.DataFrame()

    al_t = str(al_t).lower().strip()
    hist_end = pd.to_datetime(hist_end_dt, errors="coerce")
    if pd.isna(hist_end):
        return pd.DataFrame()
    hist_end = hist_end.normalize()

    fc_weeks = int(fc_horiz_weeks or 0)
    fc_end = hist_end + pd.to_timedelta(fc_weeks * 7, unit="D") # weeks to days conversion

    rows = []
    fa = flags_agg.copy()

    for _, r in fa.iterrows():
        # grid construction
        ms = pd.to_datetime(r.get("MIN_START_DT"), errors="coerce")
        me = pd.to_datetime(r.get("MAX_END_DT"), errors="coerce")

        ms = (hist_end + pd.to_timedelta(1, unit="D")) if pd.isna(ms) else max(ms, hist_end + pd.to_timedelta(1, unit="D"))
        me = fc_end if pd.isna(me) else min(me, fc_end)

        if pd.isna(ms) or pd.isna(me) or ms > me:
            continue

        if al_t.startswith("day"):
            rng = pd.date_range(ms, me, freq="D")
        elif al_t.startswith("week"):
            rng = pd.date_range(ms, me, freq="W-MON")
        elif al_t.startswith("month"):
            rng = pd.date_range(ms, me, freq="MS")
        else:
            rng = pd.date_range(ms, me, freq="D")

        base = r.to_dict()
        for d in rng:
            base2 = dict(base)
            base2["PERIOD_DT"] = bucket_time(pd.Series([d]), al_t).iloc[0]
            rows.append(base2)

    if not rows:
        return pd.DataFrame()

    res = pd.DataFrame(rows)
    res = res.drop(columns=[c for c in ["MIN_START_DT", "MAX_END_DT"] if c in res.columns], errors="ignore")
    return res


# def fill_common_out(df, alert_type, kpi_nm, input_tbl, thr, stat_nom_nm, stat_den_nm):
#     """
#     Adds common alert output terms to the output table
#     """
#     out = df.copy()
#     out["ALERT_TYPE"] = alert_type
#     out["KPI_NM"] = kpi_nm
#     out["INPUT_TABLE"] = input_tbl
#     out["STAT_NOM_NM"] = stat_nom_nm
#     out["STAT_DEN_NM"] = stat_den_nm
#     out["ALERT_THRESHOLD"] = thr

#     if "STAT_DEN_VAL" in out.columns and "STAT_NOM_VAL" in out.columns:
#         out["ALERT_STAT_VAL"] = out["STAT_NOM_VAL"].astype(float) / out["STAT_DEN_VAL"].astype(float)

#     return out

def fill_common_out(df, alert_type, kpi_nm, input_tbl, thr, stat_nom_nm, stat_den_nm, input_col=None):
    """
    Adds common alert output terms to the output table

    Fixed version, which passes all the cells, including the ones required by the documentation.
    Currently don't have the time to switch the demo variable names to this, however I can do that 
    if needed.
    """
    out = df.copy()

    out["ALERT_TYPE"] = alert_type
    out["KPI_NM"] = kpi_nm

    # keep old field (table name only) for backwards compatibility
    out["INPUT_TABLE"] = input_tbl

    # explicit column + "table.column" combined field
    if input_col is not None:
        out["INPUT_COLUMN"] = input_col
        out["INPUT_TABLE_COLUMN"] = str(input_tbl) + "." + str(input_col)
    else:
        # try to recover if caller already has it
        if "INPUT_COLUMN" in out.columns:
            out["INPUT_TABLE_COLUMN"] = str(input_tbl) + "." + out["INPUT_COLUMN"].astype(str)

    out["STAT_NOM_NM"] = stat_nom_nm
    out["STAT_DEN_NM"] = stat_den_nm
    out["ALERT_THRESHOLD"] = thr

    # cods-friendly aliases
    if "STAT_NOM_VAL" in out.columns and "STAT_NOM" not in out.columns:
        out["STAT_NOM"] = out["STAT_NOM_VAL"]
    if "STAT_DEN_VAL" in out.columns and "STAT_DEN" not in out.columns:
        out["STAT_DEN"] = out["STAT_DEN_VAL"]

    # alert statistic
    if "STAT_DEN_VAL" in out.columns and "STAT_NOM_VAL" in out.columns:
        den = pd.to_numeric(out["STAT_DEN_VAL"], errors="coerce")
        nom = pd.to_numeric(out["STAT_NOM_VAL"], errors="coerce")
        out["ALERT_STAT_VAL"] = nom / den

    return out
