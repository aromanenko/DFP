"""
This file can probably be removed later. 
As of now, its purpose is pregenerating input for the demo version of the Alerts pipeline stage that would trigger all of the existing alerts
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_dimensions(n_products=400, n_locations=60, n_customers=20, n_channels=2, seed=7):
    """
    Generates hierarchies - PRODUCT / LOCATION / CUSTOMER / DISTR_CHANNEL
    with level columns for alerts
    """
    rng = np.random.default_rng(seed)

    # simulating various groups - i.e., ID7 contains all the individual products, whereas ID6 groups each 20 of them, etc.
    prod = pd.DataFrame({
        "PRODUCT_ID": np.arange(1, n_products + 1, dtype=int),
    })
    prod["PRODUCT_LVL_ID7"] = prod["PRODUCT_ID"]
    prod["PRODUCT_LVL_ID6"] = (prod["PRODUCT_ID"] - 1) // 20 + 1
    prod["PRODUCT_LVL_ID5"] = (prod["PRODUCT_ID"] - 1) // 80 + 1

    loc = pd.DataFrame({
        "LOCATION_ID": np.arange(1, n_locations + 1, dtype=int),
    })
    loc["LOCATION_LVL_ID5"] = (loc["LOCATION_ID"] - 1) // 10 + 1
    loc["LOCATION_LVL_ID4"] = (loc["LOCATION_ID"] - 1) // 25 + 1

    cust = pd.DataFrame({
        "CUSTOMER_ID": np.arange(1, n_customers + 1, dtype=int),
    })
    cust["CUSTOMER_LVL_ID3"] = (cust["CUSTOMER_ID"] - 1) // 5 + 1
    cust["CUSTOMER_LVL_ID2"] = (cust["CUSTOMER_ID"] - 1) // 10 + 1

    dc = pd.DataFrame({
        "DISTR_CHANNEL_ID": np.arange(1, n_channels + 1, dtype=int),
    })
    dc["DISTR_CHANNEL_LVL_ID1"] = dc["DISTR_CHANNEL_ID"]

    return prod, loc, cust, dc



def generate_keys(prod, loc, cust, dc, n_series=2500, seed=7):
    """
    Generates a set of keys (PRODUCT_ID, LOCATION_ID, CUSTOMER_ID, DISTR_CHANNEL_ID)

    Random sampling for all alerts, specific subsampling for alert 8 to have enough observations to function 
    (as the param min_obs exists)
    """
    rng = np.random.default_rng(seed)

    p = rng.choice(prod["PRODUCT_ID"].values, size=n_series, replace=True)
    l = rng.choice(loc["LOCATION_ID"].values, size=n_series, replace=True)
    c = rng.choice(cust["CUSTOMER_ID"].values, size=n_series, replace=True)
    d = rng.choice(dc["DISTR_CHANNEL_ID"].values, size=n_series, replace=True)

    keys = pd.DataFrame({
        "PRODUCT_ID": p.astype(int),
        "LOCATION_ID": l.astype(int),
        "CUSTOMER_ID": c.astype(int),
        "DISTR_CHANNEL_ID": d.astype(int),
    })

    # for alert 8 
    dense_loc = int(loc["LOCATION_ID"].iloc[0])
    dense_cust = int(cust["CUSTOMER_ID"].iloc[0])
    dense_dc = int(dc["DISTR_CHANNEL_ID"].iloc[0])

    prod_u = prod.copy()
    # choose the lvl 6 group which has the most products to add to alert 8 
    if "PRODUCT_LVL_ID6" in prod_u.columns:
        g = int(prod_u["PRODUCT_LVL_ID6"].value_counts().idxmax())
        dense_products = prod_u[prod_u["PRODUCT_LVL_ID6"] == g]["PRODUCT_ID"].head(40).astype(int).to_numpy()
        dense_block = pd.DataFrame({
            "PRODUCT_ID": dense_products,
            "LOCATION_ID": dense_loc,
            "CUSTOMER_ID": dense_cust,
            "DISTR_CHANNEL_ID": dense_dc,
        })
        keys = pd.concat([keys, dense_block], ignore_index=True)

    keys = keys.drop_duplicates(ignore_index=True)
    return keys

def compute_hist_end_dt():
    """
    Returns the date of the last monday as IB_HIST_END_DT
    """
    today = pd.Timestamp.today().normalize()
    shift = (today.weekday() - 0) % 7
    return (today - pd.Timedelta(days=shift)).normalize()


def generate_restored_demand(keys, hist_end_dt, hist_years=3, base=100.0, seed=7):
    """
    Generates RESTORED_DEMAND with periodicity W-MON.
    """
    rng = np.random.default_rng(seed)

    n_weeks = int(hist_years * 52)
    hist_dates = pd.date_range(end=hist_end_dt, periods=n_weeks, freq="W-MON")

    # base demand depends on PRODUCT_LVL_ID7
    key_df = keys.copy()
    key_df["P_GRP"] = (key_df["PRODUCT_ID"] - 1) // 20 + 1
    grp_scale = 0.8 + 0.6 * (key_df["P_GRP"] / key_df["P_GRP"].max())

    rep_k = np.repeat(np.arange(len(key_df)), len(hist_dates))
    rep_d = np.tile(hist_dates.values, len(key_df))

    out = key_df.iloc[rep_k][["PRODUCT_ID", "LOCATION_ID", "CUSTOMER_ID", "DISTR_CHANNEL_ID", "P_GRP"]].reset_index(drop=True)
    out["PERIOD_DT"] = pd.to_datetime(rep_d).normalize()

    # adding seasonality and noise
    week_idx = (out["PERIOD_DT"].dt.isocalendar().week.astype(int)).to_numpy()
    seasonal = 1.0 + 0.15 * np.sin(2 * np.pi * week_idx / 52.0)

    noise = rng.normal(0.0, 8.0, size=len(out))
    lvl = base * grp_scale.iloc[rep_k].to_numpy()
    val = np.maximum(0.0, lvl * seasonal + noise)

    out["SALES_QTY_R"] = val.astype(float)
    out = out.drop(columns=["P_GRP"])
    return out


def generate_forecast_flag(keys, hist_end_dt, fc_weeks=12, max_np_days=30, seed=7):
    """
    Generates FORECAST_FLAG:
    - a fraction of the keys are regular and start a long time ago
    - a fraction of the keys are new, started in the last max_np_days
    - some keys simply son't have a forecast flag (alert 7)
    """
    rng = np.random.default_rng(seed)

    ff = keys.copy()

    # status set to active
    ff["STATUS"] = "active"

    # end_dt is the end of the forecast horizon
    fc_end = hist_end_dt + pd.Timedelta(days=fc_weeks * 7)

    # regular: 2-4 years ago
    reg_start = hist_end_dt - pd.to_timedelta(rng.integers(365 * 2, 365 * 4, size=len(ff)), unit="D")
    # new: 0-max_np_days days ago
    new_start = hist_end_dt - pd.to_timedelta(rng.integers(0, max_np_days + 1, size=len(ff)), unit="D")

    # 10% - new assortment
    is_new = rng.random(len(ff)) < 0.10
    ff["PERIOD_START_DT"] = pd.to_datetime(np.where(is_new, new_start.values, reg_start.values)).normalize()
    ff["PERIOD_END_DT"] = pd.to_datetime(fc_end).normalize()

    # 3% - should not be forecasted
    drop_mask = rng.random(len(ff)) < 0.03
    ff = ff[~drop_mask].reset_index(drop=True)

    return ff


def generate_forecasts(keys, demand_hist, hist_end_dt, fc_weeks=12, seed=7):
    """
    Generates forecasts of the types:
    - ACC_AGG_HYBRID_FORECAST_{POS/SELLOUT/SELLIN}
    - VF_FORECAST_POS (required for alert 10)
    """
    rng = np.random.default_rng(seed)

    fut_dates = pd.date_range(start=hist_end_dt + pd.Timedelta(days=7), periods=fc_weeks, freq="W-MON")

    # last-year demand for the future
    dem = demand_hist.copy()
    dem["PERIOD_DT"] = pd.to_datetime(dem["PERIOD_DT"]).dt.normalize()
    dem_ly = dem.copy()
    dem_ly["PERIOD_DT"] = dem_ly["PERIOD_DT"] + pd.Timedelta(days=52 * 7)  

    dem_ly = dem_ly[dem_ly["PERIOD_DT"].isin(fut_dates)].copy()
    if dem_ly.empty:
        last = dem.groupby(["PRODUCT_ID","LOCATION_ID","CUSTOMER_ID","DISTR_CHANNEL_ID"]).tail(52).copy()
        last["PERIOD_DT"] = last["PERIOD_DT"] + pd.Timedelta(days=52*7)
        dem_ly = last[last["PERIOD_DT"].isin(fut_dates)].copy()

    # basic forecast = last-year demand + noise
    dem_ly["HYBRID_FORECAST_VALUE"] = np.maximum(
        0.0,
        dem_ly["SALES_QTY_R"].to_numpy() * (1.0 + rng.normal(0.0, 0.05, size=len(dem_ly)))
    )

    base_fc = dem_ly[["PRODUCT_ID","LOCATION_ID","CUSTOMER_ID","DISTR_CHANNEL_ID","PERIOD_DT","HYBRID_FORECAST_VALUE"]].copy()

    # create 3 forecasting tables (for different tgt_types)
    tabs = {}
    for postfix in ["POS", "SELLOUT", "SELLIN"]:
        df = base_fc.copy()
        # slight differences
        mult = {"POS": 1.0, "SELLOUT": 0.9, "SELLIN": 1.1}[postfix]
        df["HYBRID_FORECAST_VALUE"] = df["HYBRID_FORECAST_VALUE"] * mult
        tabs[f"ACC_AGG_HYBRID_FORECAST_{postfix}"] = df

    # VF_FORECAST_POS: we just copy it from the table we just made, as alert 10 only requires the existence of said table
    tabs["VF_FORECAST_POS"] = base_fc.rename(columns={"HYBRID_FORECAST_VALUE": "VF_FORECAST_VALUE"}).copy()

    return tabs


def generate_vf_segments(keys, seed=7):
    """
    Generates VF_SEGMENTS: SEGMENT_NAME for each TS.
    """
    rng = np.random.default_rng(seed)
    seg = keys.copy()
    # 20% are non-seasonal
    seg["SEGMENT_NAME"] = np.where(rng.random(len(seg)) < 0.20, "NONSEAS", "SEASONAL")
    return seg


def inject_alert_anomalies(tables, ff, demand_hist, hist_end_dt, fc_weeks=12, seed=7):
    """
    Injects anomalies for some of the other alerts to work
    Is applied to ACC_AGG_HYBRID_FORECAST_* tables
    """
    rng = np.random.default_rng(seed)

    fut_dates = pd.date_range(start=hist_end_dt + pd.Timedelta(days=7), periods=fc_weeks, freq="W-MON")
    demand_hist = demand_hist.copy()
    demand_hist["PERIOD_DT"] = pd.to_datetime(demand_hist["PERIOD_DT"]).dt.normalize()

    # select some keys which will be mallied
    all_keys = demand_hist[["PRODUCT_ID","LOCATION_ID","CUSTOMER_ID","DISTR_CHANNEL_ID"]].drop_duplicates()
    if len(all_keys) < 50:
        return tables
    sample_keys = all_keys.sample(n=11, random_state=seed).reset_index(drop=True)

    # Alert 1 (ZEROREG/NAREG): regular key -> forecast = 0 or None for a few periods
    k1 = sample_keys.iloc[0].to_dict()
    for tnm in [k for k in tables if k.startswith("ACC_AGG_HYBRID_FORECAST_")]:
        df = tables[tnm]
        m = (df["PRODUCT_ID"] == k1["PRODUCT_ID"]) & (df["LOCATION_ID"] == k1["LOCATION_ID"]) & (df["CUSTOMER_ID"] == k1["CUSTOMER_ID"]) & (df["DISTR_CHANNEL_ID"] == k1["DISTR_CHANNEL_ID"])
        pick_dates = fut_dates[:2]
        m = m & (df["PERIOD_DT"].isin(pick_dates))
        df.loc[m, "HYBRID_FORECAST_VALUE"] = 0.0
        tables[tnm] = df
    
    k1_2 = sample_keys.iloc[10].to_dict()
    for tnm in [k for k in tables if k.startswith("ACC_AGG_HYBRID_FORECAST_")]:
        df = tables[tnm]
        m = (df["PRODUCT_ID"] == k1_2["PRODUCT_ID"]) & (df["LOCATION_ID"] == k1_2["LOCATION_ID"]) & (df["CUSTOMER_ID"] == k1_2["CUSTOMER_ID"]) & (df["DISTR_CHANNEL_ID"] == k1["DISTR_CHANNEL_ID"])
        pick_dates = fut_dates[:2]
        m = m & (df["PERIOD_DT"].isin(pick_dates))
        df.loc[m, "HYBRID_FORECAST_VALUE"] = None
        tables[tnm] = df

    # Alert 2 (INCRREG): forecast is a lot higher than last year
    k2 = sample_keys.iloc[1].to_dict()
    for tnm in [k for k in tables if k.startswith("ACC_AGG_HYBRID_FORECAST_")]:
        df = tables[tnm]
        m = (df["PRODUCT_ID"] == k2["PRODUCT_ID"]) & (df["LOCATION_ID"] == k2["LOCATION_ID"]) & (df["CUSTOMER_ID"] == k2["CUSTOMER_ID"]) & (df["DISTR_CHANNEL_ID"] == k2["DISTR_CHANNEL_ID"])
        df.loc[m, "HYBRID_FORECAST_VALUE"] = df.loc[m, "HYBRID_FORECAST_VALUE"] * 4.0
        tables[tnm] = df

    # Alert 3 (DECRREG): forecast is a lot lower than last year
    k3 = sample_keys.iloc[2].to_dict()
    for tnm in [k for k in tables if k.startswith("ACC_AGG_HYBRID_FORECAST_")]:
        df = tables[tnm]
        m = (df["PRODUCT_ID"] == k3["PRODUCT_ID"]) & (df["LOCATION_ID"] == k3["LOCATION_ID"]) & (df["CUSTOMER_ID"] == k3["CUSTOMER_ID"]) & (df["DISTR_CHANNEL_ID"] == k3["DISTR_CHANNEL_ID"])
        df.loc[m, "HYBRID_FORECAST_VALUE"] = df.loc[m, "HYBRID_FORECAST_VALUE"] * 0.15
        tables[tnm] = df

    # Alert 4/5: same as before, but for the last 12 weeks
    # set the forecast as a lot higher/lower than the average for the last two weeks
    last12 = demand_hist[demand_hist["PERIOD_DT"] > hist_end_dt - pd.Timedelta(days=12*7)]
    avg12 = last12.groupby(["PRODUCT_ID","LOCATION_ID","CUSTOMER_ID","DISTR_CHANNEL_ID"])["SALES_QTY_R"].mean().reset_index()

    k4 = sample_keys.iloc[3].to_dict()
    v4 = avg12.merge(pd.DataFrame([k4]), on=["PRODUCT_ID","LOCATION_ID","CUSTOMER_ID","DISTR_CHANNEL_ID"], how="inner")
    v4 = float(v4["SALES_QTY_R"].iloc[0]) if len(v4) else 100.0

    k5 = sample_keys.iloc[4].to_dict()
    v5 = avg12.merge(pd.DataFrame([k5]), on=["PRODUCT_ID","LOCATION_ID","CUSTOMER_ID","DISTR_CHANNEL_ID"], how="inner")
    v5 = float(v5["SALES_QTY_R"].iloc[0]) if len(v5) else 100.0

    for tnm in [k for k in tables if k.startswith("ACC_AGG_HYBRID_FORECAST_")]:
        df = tables[tnm]
        m4 = (df["PRODUCT_ID"] == k4["PRODUCT_ID"]) & (df["LOCATION_ID"] == k4["LOCATION_ID"]) & (df["CUSTOMER_ID"] == k4["CUSTOMER_ID"]) & (df["DISTR_CHANNEL_ID"] == k4["DISTR_CHANNEL_ID"])
        df.loc[m4, "HYBRID_FORECAST_VALUE"] = v4 * 7.0  # > 5x

        m5 = (df["PRODUCT_ID"] == k5["PRODUCT_ID"]) & (df["LOCATION_ID"] == k5["LOCATION_ID"]) & (df["CUSTOMER_ID"] == k5["CUSTOMER_ID"]) & (df["DISTR_CHANNEL_ID"] == k5["DISTR_CHANNEL_ID"])
        df.loc[m5, "HYBRID_FORECAST_VALUE"] = max(0.01, v5 / 7.0)  # < 1/5
        tables[tnm] = df

    # Alert 6: a future date's forecast is significantly higher
    k6 = sample_keys.iloc[5].to_dict()
    for tnm in [k for k in tables if k.startswith("ACC_AGG_HYBRID_FORECAST_")]:
        df = tables[tnm]
        m = (df["PRODUCT_ID"] == k6["PRODUCT_ID"]) & (df["LOCATION_ID"] == k6["LOCATION_ID"]) & (df["CUSTOMER_ID"] == k6["CUSTOMER_ID"]) & (df["DISTR_CHANNEL_ID"] == k6["DISTR_CHANNEL_ID"])
        # set one of the values to be extremely high
        if m.any():
            d0 = fut_dates[0]
            df.loc[m & (df["PERIOD_DT"] == d0), "HYBRID_FORECAST_VALUE"] = df.loc[m, "HYBRID_FORECAST_VALUE"].median() + 2000.0
        tables[tnm] = df

    # Alert 8: the forecast for the new product group is significantly higher than the average
    # Choose one group from lvl 6 with a lot of regulars, and add a new product
    keys2 = ff.copy()
    keys2["P_GRP"] = (keys2["PRODUCT_ID"] - 1) // 20 + 1
    grp_counts = keys2.groupby("P_GRP")["PRODUCT_ID"].nunique().sort_values(ascending=False)
    if len(grp_counts):
        grp = int(grp_counts.index[0])
        grp_keys = keys2[keys2["P_GRP"] == grp].copy()
        # choose one combo of loc, cust, dc which has enough products (as there is a min boundary for alert 8)
        grp_keys["LC"] = grp_keys["LOCATION_ID"].astype(str)+"-"+grp_keys["CUSTOMER_ID"].astype(str)+"-"+grp_keys["DISTR_CHANNEL_ID"].astype(str)
        lc = grp_keys["LC"].value_counts().idxmax()
        subset = grp_keys[grp_keys["LC"] == lc].copy()
        # list products in this subset
        prod_list = subset["PRODUCT_ID"].unique()
        if len(prod_list) >= 15:
            new_prod = int(prod_list[0])
            # make one of the products new by shifting it close to the end of the forecasting
            ff.loc[ff["PRODUCT_ID"] == new_prod, "PERIOD_START_DT"] = hist_end_dt - pd.Timedelta(days=10)

            # make its forecast way higher than it should be
            for tnm in [k for k in tables if k.startswith("ACC_AGG_HYBRID_FORECAST_")]:
                df = tables[tnm]
                m = (df["PRODUCT_ID"] == new_prod) & (df["LOCATION_ID"] == int(subset["LOCATION_ID"].iloc[0])) & (df["CUSTOMER_ID"] == int(subset["CUSTOMER_ID"].iloc[0])) & (df["DISTR_CHANNEL_ID"] == int(subset["DISTR_CHANNEL_ID"].iloc[0]))
                df.loc[m, "HYBRID_FORECAST_VALUE"] = df.loc[m, "HYBRID_FORECAST_VALUE"] * 30.0  # > 20x
                tables[tnm] = df
    
    # Alert 9: new product with an extremely low sum of forecasts
    k9 = sample_keys.iloc[6].to_dict()
    # make a product "new"
    ff.loc[
        (ff["PRODUCT_ID"] == k9["PRODUCT_ID"]) &
        (ff["LOCATION_ID"] == k9["LOCATION_ID"]) &
        (ff["CUSTOMER_ID"] == k9["CUSTOMER_ID"]) &
        (ff["DISTR_CHANNEL_ID"] == k9["DISTR_CHANNEL_ID"]),
        "PERIOD_START_DT"
    ] = hist_end_dt - pd.Timedelta(days=5)

    for tnm in [k for k in tables if k.startswith("ACC_AGG_HYBRID_FORECAST_")]:
        df = tables[tnm]
        m = (df["PRODUCT_ID"] == k9["PRODUCT_ID"]) & (df["LOCATION_ID"] == k9["LOCATION_ID"]) & (df["CUSTOMER_ID"] == k9["CUSTOMER_ID"]) & (df["DISTR_CHANNEL_ID"] == k9["DISTR_CHANNEL_ID"])
        df.loc[m, "HYBRID_FORECAST_VALUE"] = 0.05  # 12*0.05 -> 0.6 < 1, small enough
        tables[tnm] = df

    return tables, ff


def build_demo_alert_params():
    """
    Creates CONFIG.ALERT_PARAMETERS to run all of the alerts
    One df row per each alert_id & tgt_type
    """
    rows = []
    for tgt in ["POS", "SELLOUT", "SELLIN"]:
        for aid in [1,2,3,4,5,6,7,8,98,810]:
            rows.append({
                "TGT_TYPE": tgt,
                "ALERT_ID": aid,
                "AL_PRODUCT_LVL": 7,
                "AL_LOCATION_LVL": 5,
                "AL_CUSTOMER_LVL": 3,
                "AL_DISTR_CHANNEL_LVL": 1,
                "AL_TIME_LVL": "week.2",
                "ALERT_THRESHOLD_VAL": {
                    1: 0,
                    2: 2,
                    3: 2,
                    4: 5,
                    5: 5,
                    6: 2,
                    7: 0,
                    8: 20,
                    98: 1,
                    810: np.nan,
                }[aid],
                "INPUT_TABLE_TABLE": "ACC_AGG_HYBRID_FORECAST" if aid != 10 else "VF_FORECAST",
                "INPUT_COLUMN": "HYBRID_FORECAST_VALUE" if aid != 10 else "VF_FORECAST_VALUE",
            })
    return pd.DataFrame(rows)


def build_demo_config_parameters():
    """
    Creates CONFIG.CONFIG_PARAMETERS
    """
    rows = [
        ("IB_FC_HORIZ", 12),
        ("IB_ALERT_MIN_VAL", 0.1),
        ("IB_ALERT_MIN_OBS", 10),
        ("IB_MAX_NP_HISTORY", 30),
        ("IB_FF_ACTIVE_STATUS_LIST", ["active"]),
    ]
    return pd.DataFrame({"PARAMETER_NAME": [r[0] for r in rows],
                         "PARAMETER_VALUE": [r[1] for r in rows]})


def build_demo_tgt_var_cfg():
    """
    Creates TGT_VAR_CONFIG: (product aggregation levels based on tgt_type)
    """
    return {
        "pos": {"AL_PRODUCT_AGG_LVL": 6},
        "sellout": {"AL_PRODUCT_AGG_LVL": 6},
        "sellin": {"AL_PRODUCT_AGG_LVL": 6},
    }


def generate_demo_inputs(n_products=400, n_locations=60, n_customers=20, n_channels=2,
                                n_series=2500, hist_years=3, fc_weeks=12, seed=7):
    """
    Combines all of the functions to generate a set of starters for the demo
    """
    prod, loc, cust, dc = generate_dimensions(
        n_products=n_products,
        n_locations=n_locations,
        n_customers=n_customers,
        n_channels=n_channels,
        seed=seed,
    )
    keys = generate_keys(prod, loc, cust, dc, n_series=n_series, seed=seed)

    hist_end_dt = compute_hist_end_dt()

    demand_hist = generate_restored_demand(keys, hist_end_dt, hist_years=hist_years, seed=seed)

    ff = generate_forecast_flag(keys, hist_end_dt, fc_weeks=fc_weeks, seed=seed)

    tables = generate_forecasts(keys, demand_hist, hist_end_dt, fc_weeks=fc_weeks, seed=seed)
    tables["RESTORED_DEMAND"] = demand_hist
    tables["FORECAST_FLAG"] = ff
    tables["VF_SEGMENTS"] = generate_vf_segments(keys, seed=seed)

    tables, ff2 = inject_alert_anomalies(tables, ff, demand_hist, hist_end_dt, fc_weeks=fc_weeks, seed=seed)
    tables["FORECAST_FLAG"] = ff2

    alert_params_df = build_demo_alert_params()
    config_params_df = build_demo_config_parameters()
    initial_global = {"IB_HIST_END_DT": hist_end_dt}
    tgt_var_cfg = build_demo_tgt_var_cfg()

    return tables, prod, loc, cust, dc, alert_params_df, config_params_df, initial_global, tgt_var_cfg