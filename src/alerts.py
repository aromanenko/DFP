"""
Here I implement all ten alerts 
(the functions which are not necessarily related to 
running the alerts themselves are implemented in a 
separate utils file)

P.S. The first steps in most alerts are quite similar. 
I did not separate them into a standalone function in 
order to more strictly follow the documentation, which (iirc) 
did not account for that. If there's any need to separate these
into another function though, I'd be happy to oblige!
"""

# ##############################
# Prep for alerts
# ##############################

from __future__ import annotations

import pandas as pd
import numpy as np

# importing all of the necessary utility files
from utils_alerts import (
    normalize_columns,
    prepare_alert_params,
    build_config,
    parse_table_column_ref,
    col_for_hierarchy_level,
    add_level_columns,
    bucket_time,
    agg_forecast,
    agg_demand,
    agg_flags,
    shift_last_year,
    expand_period_grid,
    fill_common_out,
)


def run_alerts(tables,
                      alert_parameters,
                      prod=None, loc=None, cust=None, dc=None,
                      config_parameters=None,
                      initial_global=None,
                      tgt_var_cfg=None):
    """
    A function which runs all of the alerts based on the CONFIG.ALERT_PARAMETERS data.

    Input:
    - Other than that data, the function also requires the tables parameter: dict[str, DataFrame]
      At the very least, it should include FORECAST_FLAG, RESTORED_DEMAND and the forecasting tables.
    - prod - products table, for aggregation by AL_PRODUCT_LVL
    - loc - locations table, for aggregation by AL_LOCATION_LVL
    - cust - customers table, for aggregation by AL_CUSTOMER_LVL
    - dc - distribution channels table, for aggregation by AL_DISTR_CHANNEL_LVL
    - misc. configs from the documentation (word file 6_Алерты_v1.0.docx)

    Output:
    - A complete table of all the alert results
    """
    # normalising columns of all of the tables necessary for aggregation
    prod = normalize_columns(prod)
    loc = normalize_columns(loc)
    cust = normalize_columns(cust)
    dc = normalize_columns(dc)

    # normalising all of the forecast and demand tables - i.e., the outputs of the previous steps of the pipeline
    t = {}
    for k, v in (tables or {}).items():
        t[str(k).upper()] = normalize_columns(v)

    # since the doc mentions both naming conventions (with the DataMart. prefix and w/o it), I decided to account for both:
    if "DM.FORECAST_FLAG" in t and "FORECAST_FLAG" not in t:
        t["FORECAST_FLAG"] = t["DM.FORECAST_FLAG"]
    if "DM.RESTORED_DEMAND" in t and "RESTORED_DEMAND" not in t:
        t["RESTORED_DEMAND"] = t["DM.RESTORED_DEMAND"]

    # convert alert_params to an appropriate schema + convert configparams and initial_global to an appropriate input format as well
    ap = prepare_alert_params(alert_parameters)
    cfg = build_config(config_parameters, initial_global)

    # finally run all of the alerts for which a config got passed (should be all of them)
    res = []
    for _, r in ap.iterrows():
        aid = int(r.get("ALERT_ID", 0))
        if aid == 1:
            out = alert1(r, t, prod, loc, cust, dc, cfg)
        elif aid == 2:
            out = alert2_3(r, t, prod, loc, cust, dc, cfg, mode=2)
        elif aid == 3:
            out = alert2_3(r, t, prod, loc, cust, dc, cfg, mode=3)
        elif aid == 4:
            out = alert4_5(r, t, prod, loc, cust, dc, cfg, mode=4)
        elif aid == 5:
            out = alert4_5(r, t, prod, loc, cust, dc, cfg, mode=5)
        elif aid == 6:
            out = alert6(r, t, prod, loc, cust, dc, cfg)
        elif aid == 7:
            out = alert7(r, t, prod, loc, cust, dc, cfg)
        elif aid == 8:
            out = alert8(r, t, prod, loc, cust, dc, cfg, tgt_var_cfg)
        elif aid == 98:
            out = alert9(r, t, prod, loc, cust, dc, cfg)
        elif aid == 810:
            out = alert10(r, t, prod, loc, cust, dc, cfg)
        else:
            out = pd.DataFrame()

        if out is not None and not out.empty:
            res.append(out)
    if not res:
        return pd.DataFrame() # empty df if nothing works properly

    return pd.concat(res, ignore_index=True, sort=False)


def alert_preambule(ap_row, tables, cfg, thr_flag=False, thr_def=0, min_val_flag=False, min_val_def=0, hist_end_flag=False, fc_h_flag=False, fc_h_def=0):
    """
    Sets up the majority of the core data that is used in all of the alerts. Practically a blackbox to an outsider.
    Could have theoretically been done a tad more elegantly, but this approach should probably suffice
    """
    # get the target table column
    tbl_col = ap_row.get("TARGET_TABLE_COLUMN") or ap_row.get("INPUT_COLUMN")
    if tbl_col is None:
        return pd.DataFrame()

    # parse the data into a few separate variables (tables, columns, kpi_nm, type of target)
    tbl, col, kpi_nm, tgt_type = parse_table_column_ref(tbl_col)
    df = tables.get(tbl)
    if df is None:
        return pd.DataFrame()

    # fetch all of the relevant levels for each aspect related to the products
    al_p = ap_row.get("AL_PRODUCT_LVL")
    al_l = ap_row.get("AL_LOCATION_LVL")
    al_c = ap_row.get("AL_CUSTOMER_LVL")
    al_d = ap_row.get("AL_DISTR_CHANNEL_LVL")
    al_t = ap_row.get("AL_TIME_LVL")

    # fetch the threshold, min alert value, final history date and the forecast horison
    if thr_flag:
        thr = float(ap_row.get("ALERT_THRESHOLD_VAL", ap_row.get("ALERT_THRESHOLD", thr_def)))
    else:
        thr = None
    if min_val_flag:
        min_val = float(cfg.get("IB_ALERT_MIN_VAL", cfg.get("IB_ALERT_MIN_VALUE", min_val_def)))
    else:
        min_val = None
    if hist_end_flag:
        hist_end = pd.to_datetime(cfg.get("IB_HIST_END_DT"), errors="coerce").normalize()
    else:
        hist_end = None
    if fc_h_flag:
        fc_h = int(cfg.get("IB_FC_HORIZ", cfg.get("IB_FC_HORIZ_WEEKS", fc_h_def)))
    else:
        fc_h = None

    return df, tbl, col, kpi_nm, tgt_type, al_p, al_l, al_c, al_d, al_t, thr, min_val, hist_end, fc_h


# ##############################
# Alert 1: NAREG / ZEROREG
# ##############################


def alert1(ap_row, tables, prod, loc, cust, dc, cfg):
    """
    Alert 1 (NAREG/ZEROREG): Forecast is missing or equals 0 for regular assortment.

    Input:
    - all of the parameters described in run_alerts
    Output:
    - the rows on which the alert got triggered
    """

    # fetch all of the non-unique basic info
    df, tbl, col, kpi_nm, _, al_p, al_l, al_c, al_d, al_t, thr, _, hist_end, fc_h = alert_preambule(ap_row, tables, cfg,
                                                                                                    thr_flag=True, thr_def=0,
                                                                                                    hist_end_flag=True,
                                                                                                    fc_h_flag=True, fc_h_def=12)

    # aggregate forecast for the alert (using the aforementioned (in alert_preambule) hierarchy levels)
    a = agg_forecast(df, col, prod, loc, cust, dc, al_p, al_l, al_c, al_d, al_t)

    # aggregate forecast flag
    ff = tables.get("FORECAST_FLAG")
    active_status = cfg.get("IB_FF_ACTIVE_STATUS_LIST", ["active"])
    ffa = agg_flags(ff, prod, loc, cust, dc, al_p, al_l, al_c, al_d, active_status)

    if a.empty or ffa.empty or pd.isna(hist_end):
        return pd.DataFrame()

    # define forecast timestamp grid
    b = expand_period_grid(ffa, al_t, hist_end, fc_h)
    if b.empty:
        return pd.DataFrame()

    # join predictions with actual forecasts
    id_cols = [c for c in b.columns if c != "PERIOD_DT"]
    jcols = id_cols + ["PERIOD_DT"]
    d = b.merge(a, on=jcols, how="left")

    # only leave those where the value is missing or lower than some given threshold
    d["STAT_NOM_VAL_RAW"] = d["STAT_NOM_VAL"]
    d["STAT_NOM_VAL"] = pd.to_numeric(d["STAT_NOM_VAL"], errors="coerce")

    # NA rows (missing forecast)
    na_mask = d["STAT_NOM_VAL_RAW"].isna()

    # ZERO/low rows (forecast present but <= threshold)
    # keep it strict when thr==0: exactly 0 (or <=0) but not NA
    zero_mask = (~na_mask) & (d["STAT_NOM_VAL"].fillna(0.0) <= float(thr))

    outs = []

    if na_mask.any():
        na = d.loc[na_mask, id_cols + ["PERIOD_DT"]].copy()
        na["STAT_NOM_VAL"] = np.nan
        na["STAT_DEN_VAL"] = 1.0
        na = fill_common_out(na, "NAREG", kpi_nm, tbl, thr, "Forecast value", "na")
        outs.append(na)

    if zero_mask.any():
        zz = d.loc[zero_mask, id_cols + ["PERIOD_DT"]].copy()
        zz["STAT_NOM_VAL"] = d.loc[zero_mask, "STAT_NOM_VAL"].values
        zz["STAT_DEN_VAL"] = 1.0
        zz = fill_common_out(zz, "ZEROREG", kpi_nm, tbl, thr, "Forecast value", "na")
        outs.append(zz)

    if not outs:
        return pd.DataFrame()
    
    # create relevant table, return it
    out = pd.concat(outs, ignore_index=True, sort=False)
    return out


# ##############################
# Alert 2/3: INCRREG / DECRREG
# ##############################


def alert2_3(ap_row, tables, prod, loc, cust, dc, cfg, mode):
    """
    Alert 2: Forecast too high vs last-year (INCRREG)
    Alert 3: Forecast too low  vs last-year (DECRREG)

    Input:
    - all of the parameters described in run_alerts
    Output:
    - the rows on which the alert got triggered
    """

    # fetch all of the non-unique basic info
    df, tbl, col, kpi_nm, tgt_type, al_p, al_l, al_c, al_d, al_t, thr, min_val, hist_end, _ = alert_preambule(ap_row, tables, cfg,
                                                                                                    thr_flag=True, thr_def=2,
                                                                                                    min_val_flag=True, min_val_def=0.1,
                                                                                                    hist_end_flag=True)

    # aggregate forecast for the alert (using the aforementioned levels)
    a = agg_forecast(df, col, prod, loc, cust, dc, al_p, al_l, al_c, al_d, al_t)

    # aggregate restored demand
    rd = tables.get("RESTORED_DEMAND")
    dem = agg_demand(rd, prod, loc, cust, dc, al_p, al_l, al_c, al_d, al_t)

    # if there's no data, empty output
    if a.empty or dem.empty or pd.isna(hist_end):
        return pd.DataFrame()

    # 1-year shift to obtain last year's demand
    den = shift_last_year(dem, al_t)

    # join previous and curr data
    id_cols = [c for c in a.columns if c.startswith(("PRODUCT", "LOCATION", "CUSTOMER", "DISTR_CHANNEL"))]
    id_cols = [c for c in id_cols if ("LVL" in c) or c.endswith("_ID")]
    id_cols = sorted(list(set(id_cols)))
    jcols = id_cols + ["PERIOD_DT"]
    d = a.merge(den, on=jcols, how="left")

    # only take non-null values larger than the min possible
    d = d[d["STAT_DEN_VAL"].notna() & (d["STAT_DEN_VAL"] > min_val)].copy()
    if d.empty:
        return pd.DataFrame()

    # get their difference in scale
    stat = d["STAT_NOM_VAL"].astype(float) / d["STAT_DEN_VAL"].astype(float)

    # trigger either 2 (stat is higher than the threshold) or 3 (lower)
    if mode == 2:
        d = d[stat > thr].copy()
        alert_type = "INCRREG"
        stat_nom_nm = "Forecast value"
        stat_den_nm = "Last year demand value"
    else:
        d = d[stat < 1.0 / thr].copy()
        alert_type = "DECRREG"
        stat_nom_nm = "Forecast value"
        stat_den_nm = "Last year demand value"

    if d.empty:
        return pd.DataFrame()

    # create relevant table, return it
    d = fill_common_out(d, alert_type, kpi_nm, tbl, thr, stat_nom_nm, stat_den_nm, input_col=col)
    return d


# ##############################
# # Alert 4/5: HIGHREG / LOWREG
# ##############################


def alert4_5(ap_row, tables, prod, loc, cust, dc, cfg, mode):
    """
    Alert 4: Forecast too high vs average demand in last 3 months (HIGHREG)
    Alert 5: Forecast too low  vs average demand in last 3 months (LOWREG)

    Input:
    - all of the parameters described in run_alerts
    Output:
    - the rows on which the alert got triggered
    """

    # fetch all of the non-unique basic info
    df, tbl, col, kpi_nm, _, al_p, al_l, al_c, al_d, al_t, thr, min_val, hist_end, _ = alert_preambule(ap_row, tables, cfg,
                                                                                                        thr_flag=True, thr_def=5,
                                                                                                        min_val_flag=True, min_val_def=0.1,
                                                                                                        hist_end_flag=True)

    # aggregate forecast for the alert (using the aforementioned levels)
    a = agg_forecast(df, col, prod, loc, cust, dc, al_p, al_l, al_c, al_d, al_t)

    # aggregate restored demand
    rd = tables.get("RESTORED_DEMAND")
    dem = agg_demand(rd, prod, loc, cust, dc, al_p, al_l, al_c, al_d, al_t)

    # if there's no data, empty output
    if a.empty or dem.empty or pd.isna(hist_end):
        return pd.DataFrame()

    # normalise dt (get rid of the time)
    dem["PERIOD_DT"] = pd.to_datetime(dem["PERIOD_DT"], errors="coerce").dt.normalize()

    # initialise a 12-week bin
    al_t_l = str(al_t).lower().strip()
    if al_t_l.startswith("day"):
        win_start = hist_end - pd.to_timedelta(84, unit="D")
    elif al_t_l.startswith("week"):
        win_start = hist_end - pd.to_timedelta(12 * 7, unit="D")
    elif al_t_l.startswith("month"):
        win_start = hist_end - pd.DateOffset(months=3)
    else:
        win_start = hist_end - pd.to_timedelta(84, unit="D")

    # get average demand for the bin
    id_cols = [c for c in dem.columns if c.startswith(("PRODUCT", "LOCATION", "CUSTOMER", "DISTR_CHANNEL"))]
    id_cols = [c for c in id_cols if ("LVL" in c) or c.endswith("_ID")]
    id_cols = sorted(list(set(id_cols)))

    dem_win = dem[(dem["PERIOD_DT"] <= hist_end) & (dem["PERIOD_DT"] > win_start)].copy()
    den = dem_win.groupby(id_cols, dropna=False)["STAT_DEN_VAL"].mean().reset_index()

    # join with predicted data
    d = a.merge(den, on=id_cols, how="left")
    d = d[d["STAT_DEN_VAL"].notna() & (d["STAT_DEN_VAL"] > min_val)].copy()
    if d.empty:
        return pd.DataFrame()

    # get their difference in scale
    stat = d["STAT_NOM_VAL"].astype(float) / d["STAT_DEN_VAL"].astype(float)

    # trigger either 4 (stat is higher than the threshold) or 5 (lower)
    if mode == 4:
        d = d[stat > thr].copy()
        alert_type = "HIGHREG"
    else:
        d = d[stat < 1.0 / thr].copy()
        alert_type = "LOWREG"

    if d.empty:
        return pd.DataFrame()

    # create relevant table, return it
    d = fill_common_out(d, alert_type, kpi_nm, tbl, thr, "Forecast value", "Average demand value within last 3 months",
                        input_col=col)
    return d


# ##############################
# # Alert 6: DEVWK
# ##############################


def alert6(ap_row, tables, prod, loc, cust, dc, cfg):
    """
    Alert 6: Abnormally maximum deviation with respect to last 3 month's data (DEVWK)

    Input:
    - all of the parameters described in run_alerts
    Output:
    - the rows on which the alert got triggered
    """
    # fetch all of the non-unique basic info
    df, tbl, col, kpi_nm, tgt_type, al_p, al_l, al_c, al_d, al_t, thr, min_val, hist_end, _ = alert_preambule(ap_row, tables, cfg,
                                                                                                    thr_flag=True, thr_def=2,
                                                                                                    min_val_flag=True, min_val_def=0.1,
                                                                                                    hist_end_flag=True)

    # aggregate forecast
    a = agg_forecast(df, col, prod, loc, cust, dc, al_p, al_l, al_c, al_d, al_t)

    # aggregate restored demand
    rd = tables.get("RESTORED_DEMAND")
    c = agg_demand(rd, prod, loc, cust, dc, al_p, al_l, al_c, al_d, al_t)

    # if there's no data, empty output
    if a.empty or c.empty or pd.isna(hist_end):
        return pd.DataFrame()

    # get forecast and demand, then union
    a["_SRC"] = "F"
    c = c.rename(columns={"STAT_DEN_VAL": "STAT_NOM_VAL"})
    c["_SRC"] = "D"

    u = pd.concat([a, c], ignore_index=True, sort=False)
    u["PERIOD_DT"] = pd.to_datetime(u["PERIOD_DT"], errors="coerce").dt.normalize()

    u = u.rename(columns={"STAT_NOM_VAL": "_VALUE"})
    u["_VALUE"] = pd.to_numeric(u["_VALUE"], errors="coerce")

    id_cols = [x for x in u.columns if x.startswith(("PRODUCT", "LOCATION", "CUSTOMER", "DISTR_CHANNEL"))]
    id_cols = [x for x in id_cols if ("LVL" in x) or x.endswith("_ID")]
    id_cols = sorted(list(set(id_cols)))
    
    # get offset
    al_t_l = str(al_t).lower().strip()
    if al_t_l.startswith("day"):
        win_start = hist_end - pd.to_timedelta(84, unit="D")
    elif al_t_l.startswith("week"):
        win_start = hist_end - pd.to_timedelta(12 * 7, unit="D")
    elif al_t_l.startswith("month"):
        win_start = hist_end - pd.DateOffset(months=3)
    else:
        win_start = hist_end - pd.to_timedelta(84, unit="D")
    
    # calculate the "difference for each quad"
    u = u.sort_values(id_cols + ["PERIOD_DT"])
    u["PREV_VAL"] = u.groupby(id_cols, dropna=False)["_VALUE"].shift(1)
    u["_DIFVALUE"] = u["_VALUE"] - u["PREV_VAL"]

    # calculate statistic using the query from the documentation
    last_hist = (
        u[(u["_SRC"] == "D") & (u["PERIOD_DT"] <= hist_end)]
        .groupby(id_cols, dropna=False)
        .tail(1)[id_cols + ["_VALUE"]]
        .rename(columns={"_VALUE": "LAST_HIST_VAL"})
    )

    hist_win = u[(u["_SRC"] == "D") & (u["PERIOD_DT"] <= hist_end) & (u["PERIOD_DT"] > win_start)]
    den = (hist_win.groupby(id_cols, dropna=False)["_VALUE"]
                .agg(lambda s: s.max() - s.min())
                .reset_index()
                .rename(columns={"_VALUE": "STAT_DEN_VAL"}))

    fut = u[(u["_SRC"] == "F") & (u["PERIOD_DT"] > hist_end)].copy()

    # join, only leave rows where stat exists
    fut = fut.merge(last_hist, on=id_cols, how="left").merge(den, on=id_cols, how="left")
    fut = fut[fut["STAT_DEN_VAL"].notna() & (fut["STAT_DEN_VAL"] > min_val)].copy()
    fut["STAT_NOM_VAL"] = (fut["_VALUE"] - fut["LAST_HIST_VAL"]).abs()
    fut["ALERT_STAT_VAL"] = fut["STAT_NOM_VAL"] / fut["STAT_DEN_VAL"]
    d = fut[fut["ALERT_STAT_VAL"] > thr].copy()


    if d.empty:
        return pd.DataFrame()

    # create complete table, return it
    d = fill_common_out(
        d, "DEVWK", kpi_nm, tbl, thr,
        "Forecast Deviation", "Demand Deviation within last 3 months",
        input_col=col
    )
    keep = id_cols + ["PERIOD_DT", "STAT_NOM_VAL", "STAT_DEN_VAL",
                      "ALERT_TYPE", "KPI_NM", "INPUT_TABLE",
                      "STAT_NOM_NM", "STAT_DEN_NM", "ALERT_THRESHOLD", "ALERT_STAT_VAL", col]
    keep = [c for c in keep if c in d.columns]
    return d[keep]


# ##############################
# Alert 7: ZEROFLG
# ##############################


def alert7(ap_row, tables, prod, loc, cust, dc, cfg):
    """
    Alert 7: Non-zero forecast for assortments with missing forecast flags (ZEROFLG).

    Input:
    - all of the parameters described in run_alerts
    Output:
    - the rows on which the alert got triggered
    """

    # fetch all of the non-unique basic info
    df, tbl, col, kpi_nm, _, al_p, al_l, al_c, al_d, al_t, _, min_val, hist_end, fc_h = alert_preambule(
        ap_row, tables, cfg,
        hist_end_flag=True,
        fc_h_flag=True, fc_h_def=12,
        min_val_flag=True, min_val_def=float(cfg.get("IB_ALERT_MIN_VALUE", cfg.get("IB_ALERT_MIN_VAL", 0.0))),
        thr_flag=False,
    )

    # aggregate forecast (step a)
    a = agg_forecast(df, col, prod, loc, cust, dc, al_p, al_l, al_c, al_d, al_t)
    if a.empty or pd.isna(hist_end):
        return pd.DataFrame()

    # keep only future horizon (otherwise history would always look like "missing flags")
    a["PERIOD_DT"] = pd.to_datetime(a["PERIOD_DT"], errors="coerce").dt.normalize()
    a = a[a["PERIOD_DT"].notna() & (a["PERIOD_DT"] > hist_end)].copy()
    if a.empty:
        return pd.DataFrame()

    # aggregate forecast flags (step b/c, same as Alert 1)
    ff = tables.get("FORECAST_FLAG")
    active_status = cfg.get("IB_FF_ACTIVE_STATUS_LIST", ["active"])
    ffa = agg_flags(ff, prod, loc, cust, dc, al_p, al_l, al_c, al_d, active_status)
    if ffa.empty:
        return pd.DataFrame()

    # expand flags to the forecasting grid (same utility as Alert 1)
    b = expand_period_grid(ffa, al_t, hist_end, fc_h)
    if b.empty:
        return pd.DataFrame()

    # join predictions with the flag grid and keep only rows where the flag-period is missing
    id_cols = [c for c in b.columns if c != "PERIOD_DT"]
    jcols = id_cols + ["PERIOD_DT"]
    d = a.merge(b[jcols].drop_duplicates(), on=jcols, how="left", indicator=True)
    d = d[d["_merge"] == "left_only"].drop(columns=["_merge"])
    if d.empty:
        return pd.DataFrame()

    # only leave rows where forecast exists (non-zero / > min threshold)
    d["STAT_NOM_VAL"] = pd.to_numeric(d["STAT_NOM_VAL"], errors="coerce")
    d = d[d["STAT_NOM_VAL"].fillna(0.0) > float(min_val)].copy()
    if d.empty:
        return pd.DataFrame()

    # complete the output table (spec: threshold is missing, denom=1, stat=forecast value)
    d["STAT_DEN_VAL"] = 1.0
    d = fill_common_out(
        d, "ZEROFLG", kpi_nm, tbl, np.nan,
        "Forecast value", "na",
        input_col=col
    )
    return d



# ##############################
# Alert 8: SHRNEW
# ##############################

def alert8(ap_row, tables, prod, loc, cust, dc, cfg, tgt_var_cfg):
    """
    Alert 8: Abnormal share of new assortment forecast (SHRNEW)

    Input:
    - all of the parameters described in run_alerts
    Output:
    - the rows on which the alert got triggered
    """
    # fetch all of the non-unique basic info
    df, tbl, col, kpi_nm, tgt_type, al_p, al_l, al_c, al_d, al_t, thr, min_val, hist_end, _ = alert_preambule(
        ap_row, tables, cfg,
        hist_end_flag=True,
        min_val_flag=True, min_val_def=0.0,
        thr_flag=True, thr_def=20
    )

    # params
    min_obs = int(cfg.get("IB_ALERT_MIN_OBS", 10))
    max_np = int(cfg.get("IB_MAX_NP_HISTORY", 30))
    active_status = cfg.get("IB_FF_ACTIVE_STATUS_LIST", ["active"])

    ff = tables.get("FORECAST_FLAG")
    if ff is None or pd.isna(hist_end):
        return pd.DataFrame()

    # product-based aggregation level for baseline
    p_agg = None
    if tgt_var_cfg and tgt_type and tgt_type in tgt_var_cfg:
        p_agg = int(tgt_var_cfg[tgt_type].get("AL_PRODUCT_AGG_LVL", al_p - 1))
    if p_agg is None:
        p_agg = max(al_p - 1, 1)

    # aggregate forecast flags => split to new / regular
    ffa = agg_flags(ff, prod, loc, cust, dc, al_p, al_l, al_c, al_d, active_status)
    if ffa.empty:
        return pd.DataFrame()

    ffa["MIN_START_DT"] = pd.to_datetime(ffa["MIN_START_DT"], errors="coerce").dt.normalize()
    ffa["NP_DAYS"] = (hist_end - ffa["MIN_START_DT"]).dt.days

    new_keys = ffa[ffa["NP_DAYS"] <= max_np].copy()
    reg_keys = ffa[ffa["NP_DAYS"] > max_np].copy()

    # prepare forecast (nom) on al_* + PERIOD_DT
    x = normalize_columns(df)
    col = str(col).upper()
    if col not in x.columns or "PERIOD_DT" not in x.columns:
        return pd.DataFrame()

    x, p_col, l_col, c_col, d_col = add_level_columns(x, prod, loc, cust, dc, al_p, al_l, al_c, al_d)
    x["PERIOD_DT"] = bucket_time(x["PERIOD_DT"], al_t)
    x["PERIOD_DT"] = pd.to_datetime(x["PERIOD_DT"], errors="coerce").dt.normalize()

    # use only forecast horizon (future) to match shifted denominator
    x = x[x["PERIOD_DT"].notna() & (x["PERIOD_DT"] > hist_end)].copy()
    if x.empty:
        return pd.DataFrame()

    # add product agg column
    p_agg_col = col_for_hierarchy_level("PRODUCT", p_agg)
    if p_agg_col not in x.columns and "PRODUCT_ID" in x.columns and prod is not None:
        prod_u = normalize_columns(prod)
        if "PRODUCT_ID" in prod_u.columns and p_agg_col in prod_u.columns:
            x = x.merge(prod_u[["PRODUCT_ID", p_agg_col]], on="PRODUCT_ID", how="left")

    id_cols = [p_col, l_col, c_col, d_col]
    id_cols = [c for c in id_cols if c in x.columns and c is not None]
    if not id_cols or p_col not in id_cols:
        return pd.DataFrame()

    # select only new keys for numerator
    join_ids = [c for c in id_cols if c in new_keys.columns]
    if not join_ids:
        return pd.DataFrame()

    new = x.merge(new_keys[join_ids], on=join_ids, how="inner")
    if new.empty:
        return pd.DataFrame()

    # numerator: average forecast for new assortment (per key+period)
    new_g = [p_col] + [c for c in id_cols if c != p_col] + ["PERIOD_DT"]
    new_stat = new.groupby(new_g, dropna=False)[col].mean().reset_index()
    new_stat = new_stat.rename(columns={col: "STAT_NOM_VAL"})

    # add p_agg to numerator for join
    if p_agg_col not in new_stat.columns:
        if p_agg_col not in x.columns:
            return pd.DataFrame()
        tmp = x[[p_col, p_agg_col]].dropna().drop_duplicates()
        new_stat = new_stat.merge(tmp, on=p_col, how="left")

    # denominator from restored demand: regular assortment only, grouped by p_agg + loc/cust/dc + period
    rd = None
    if tgt_type:
        for nm in [f"RESTORED_DEMAND_{tgt_type}", f"RESTORED_DEMAND_{str(tgt_type).upper()}"]:
            if nm in tables:
                rd = tables.get(nm)
                break
    if rd is None:
        rd = tables.get("RESTORED_DEMAND")

    if rd is None or rd.empty:
        return pd.DataFrame()

    rd = normalize_columns(rd)

    # demand column candidates (include SALES_QTY_R!)
    den_col = None
    for cand in ["SALES_QTY_R", "SALESTGT_QTY_R", "TGTSALES_QTY_R",
                 "DEMAND", "RESTORED_DEMAND", "SALES_QTY", "QTY", "VALUE"]:
        if cand in rd.columns:
            den_col = cand
            break
    if den_col is None:
        return pd.DataFrame()

    # ensure PERIOD_DT
    if "PERIOD_DT" not in rd.columns:
        if "PERIOD_START_DT" in rd.columns:
            rd["PERIOD_DT"] = rd["PERIOD_START_DT"]
        else:
            return pd.DataFrame()

    rd["PERIOD_DT"] = bucket_time(rd["PERIOD_DT"], al_t)
    rd["PERIOD_DT"] = pd.to_datetime(rd["PERIOD_DT"], errors="coerce").dt.normalize()

    # historical only (will be shifted forward)
    rd = rd[rd["PERIOD_DT"].notna() & (rd["PERIOD_DT"] <= hist_end)].copy()
    if rd.empty:
        return pd.DataFrame()

    # add level columns to rd
    rd, rp_col, rl_col, rc_col, rd_col = add_level_columns(rd, prod, loc, cust, dc, al_p, al_l, al_c, al_d)

    # ensure p_agg in rd
    if p_agg_col not in rd.columns and "PRODUCT_ID" in rd.columns and prod is not None:
        prod_u = normalize_columns(prod)
        if "PRODUCT_ID" in prod_u.columns and p_agg_col in prod_u.columns:
            rd = rd.merge(prod_u[["PRODUCT_ID", p_agg_col]], on="PRODUCT_ID", how="left")

    if p_agg_col not in rd.columns or rp_col is None:
        return pd.DataFrame()

    # regular-only join via flags
    reg_join_ids = [c for c in [rp_col, rl_col, rc_col, rd_col]
                    if c is not None and c in rd.columns and c in reg_keys.columns]
    if not reg_join_ids:
        return pd.DataFrame()

    reg_rd = rd.merge(reg_keys[reg_join_ids].drop_duplicates(), on=reg_join_ids, how="inner")
    if reg_rd.empty:
        return pd.DataFrame()

    reg_g = [p_agg_col] + [c for c in [rl_col, rc_col, rd_col] if c is not None and c in reg_rd.columns] + ["PERIOD_DT"]
    reg_stat = (
        reg_rd.groupby(reg_g, dropna=False)
              .agg(
                  STAT_DEN_VAL=(den_col, "mean"),
                  STAT_COUNT=(rp_col, pd.Series.nunique),
              )
              .reset_index()
    )

    # shift denominator one year forward (step f)
    reg_stat = shift_last_year(reg_stat, al_t)

    # join numerator with denominator for the same future period
    join_cols = [p_agg_col] + [c for c in id_cols if c != p_col] + ["PERIOD_DT"]
    d = new_stat.merge(reg_stat, on=join_cols, how="left")

    # leave only rows with enough statistics and valid denominator
    d = d[d["STAT_COUNT"].fillna(0) >= min_obs].copy()
    d = d[d["STAT_DEN_VAL"].notna() & (d["STAT_DEN_VAL"] > min_val)].copy()
    if d.empty:
        return pd.DataFrame()

    stat = d["STAT_NOM_VAL"].astype(float) / d["STAT_DEN_VAL"].astype(float)
    d = d[(stat > thr) | (stat < 1.0 / thr)].copy()
    if d.empty:
        return pd.DataFrame()

    d = fill_common_out(
        d, "SHRNEW", kpi_nm, tbl, thr,
        "Forecast for New Assortment",
        "Average Demand for Regular Assortment",
        input_col=col
    )
    return d



# ##############################
# Alert 9: LOWNEW
# ##############################


def alert9(ap_row, tables, prod, loc, cust, dc, cfg):
    """
    Alert 9: Abnormally low value for new assortment (LOWNEW)

    Input:
    - all of the parameters described in run_alerts
    Output:
    - the rows on which the alert got triggered
    """
    # fetch all of the non-unique basic info
    df, tbl, col, kpi_nm, _, al_p, al_l, al_c, al_d, al_t, thr, _, hist_end, _ = alert_preambule(
        ap_row, tables, cfg,
        hist_end_flag=True,
        thr_flag=True, thr_def=1
    )

    # params
    min_obs = int(cfg.get("IB_ALERT_MIN_OBS", 10))
    max_np = int(cfg.get("IB_MAX_NP_HISTORY", 30))
    active_status = cfg.get("IB_FF_ACTIVE_STATUS_LIST", ["active"])

    # get forecast flag
    ff = tables.get("FORECAST_FLAG")
    if ff is None or pd.isna(hist_end):
        return pd.DataFrame()

    # aggregate forecast flags => get new flags
    ffa = agg_flags(ff, prod, loc, cust, dc, al_p, al_l, al_c, al_d, active_status)
    if ffa.empty:
        return pd.DataFrame()

    ffa["MIN_START_DT"] = pd.to_datetime(ffa["MIN_START_DT"], errors="coerce").dt.normalize()
    ffa["NP_DAYS"] = (hist_end - ffa["MIN_START_DT"]).dt.days
    new_keys = ffa[ffa["NP_DAYS"] <= max_np].copy()

    x = normalize_columns(df)
    col = str(col).upper()
    if col not in x.columns or "PERIOD_DT" not in x.columns:
        return pd.DataFrame()

    # add level-based columns
    x, p_col, l_col, c_col, d_col = add_level_columns(x, prod, loc, cust, dc, al_p, al_l, al_c, al_d)

    x["PERIOD_DT"] = bucket_time(x["PERIOD_DT"], al_t)
    x["PERIOD_DT"] = pd.to_datetime(x["PERIOD_DT"], errors="coerce").dt.normalize()

    # sum only over forecasting horizon (future)
    x = x[x["PERIOD_DT"].notna() & (x["PERIOD_DT"] > hist_end)].copy()
    if x.empty:
        return pd.DataFrame()

    gcols = [p_col, l_col, c_col, d_col]
    gcols = [c for c in gcols if c in x.columns]

    # calculate stat
    a = x.groupby(gcols, dropna=False)[col].agg(
        STAT_NOM_VAL="sum",
        STAT_COUNT="count",
    ).reset_index()

    id_cols = [p_col, l_col, c_col, d_col]
    id_cols = [c for c in id_cols if c in a.columns and c in new_keys.columns]
    if not id_cols:
        return pd.DataFrame()

    # remove values over the threshold
    a = a.merge(new_keys[id_cols], on=id_cols, how="inner")
    a = a[a["STAT_COUNT"] >= min_obs].copy()
    a["STAT_DEN_VAL"] = 1.0
    a = a[a["STAT_NOM_VAL"] < thr].copy()

    if a.empty:
        return pd.DataFrame()

    # construct table, return it
    a["PERIOD_DT"] = pd.NaT
    a = fill_common_out(
        a, "LOWNEW", kpi_nm, tbl, thr,
        "Total forecast value on the whole forecasting period", "na",
        input_col=col
    )
    return a


# ##############################
# Alert 10: NONSEAS
# ##############################


def alert10(ap_row, tables, prod, loc, cust, dc, cfg):
    """
    Alert 10: NONSEAS — non-seasonal segment is used for long time series

    Input:
    - all of the parameters described in run_alerts
    Output:
    - the rows on which the alert got triggered
    """
    df, tbl, col, kpi_nm, _, al_p, al_l, al_c, al_d, _, _, _, hist_end, _ = alert_preambule(ap_row, tables, cfg,
                                                                                                    hist_end_flag=True)

    # get VF forecast segments
    seg = tables.get("VF_SEGMENTS")
    if seg is None or seg.empty:
        return pd.DataFrame()

    # get forecast flag
    ff = tables.get("FORECAST_FLAG")
    active_status = cfg.get("IB_FF_ACTIVE_STATUS_LIST", ["active"])
    hist_end = pd.to_datetime(cfg.get("IB_HIST_END_DT"), errors="coerce").normalize()
    if ff is None or pd.isna(hist_end):
        return pd.DataFrame()

    # get long history keys
    min_hist_days = int(cfg.get("IB_TS_LONG_HISTORY_DAYS", 2 * 365))

    ffa = agg_flags(ff, prod, loc, cust, dc, al_p, al_l, al_c, al_d, active_status)
    if ffa.empty:
        return pd.DataFrame()

    # count days passed
    ffa["MIN_START_DT"] = pd.to_datetime(ffa["MIN_START_DT"], errors="coerce").dt.normalize()
    ffa["HIST_DAYS"] = (hist_end - ffa["MIN_START_DT"]).dt.days
    long_keys = ffa[ffa["HIST_DAYS"] >= min_hist_days].copy()
    if long_keys.empty:
        return pd.DataFrame()

    # normalize + add level columns to seg
    x = normalize_columns(seg)
    x, p_col, l_col, c_col, d_col = add_level_columns(x, prod, loc, cust, dc, al_p, al_l, al_c, al_d)

    id_cols = [c for c in [p_col, l_col, c_col, d_col] if c in x.columns and c in long_keys.columns]
    if not id_cols:
        return pd.DataFrame()

    x = x.merge(long_keys[id_cols], on=id_cols, how="inner")

    # segment column selection 
    seg_col = None
    for cand in ["SEGMENT_NAME", "VF_SEGMENT_NAME", "SEGMENT", "SEGMENT_NM"]:
        if cand in x.columns:
            seg_col = cand
            break
    if seg_col is None:
        return pd.DataFrame()

    sn = x[seg_col].astype(str).str.upper()

    # flag explicitly NONSEAS / NONSEASON*
    bad = (~sn.str.contains(r"_SEASON", regex=True)) & (~sn.str.startswith("SEASON"))
    x = x[bad].copy()
    if x.empty:
        return pd.DataFrame()

    x["STAT_NOM_VAL"] = np.nan
    x["STAT_DEN_VAL"] = 1.0
    x["PERIOD_DT"] = pd.NaT

    x = fill_common_out(
        x, "NONSEAS", kpi_nm, tbl,
        ap_row.get("ALERT_THRESHOLD_VAL", np.nan),
        "Time Series Segment’", "na",
        input_col=col
    )

    keep = id_cols + ["PERIOD_DT", "STAT_NOM_VAL", "STAT_DEN_VAL",
                      "ALERT_TYPE", "KPI_NM", "INPUT_TABLE",
                      "STAT_NOM_NM", "STAT_DEN_NM", "ALERT_THRESHOLD", seg_col, col]
    keep = [c for c in keep if c in x.columns]
    return x[keep]
