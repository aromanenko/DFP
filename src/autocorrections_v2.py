import pandas as pd
import numpy as np
import os
import sys

project_path = os.path.abspath(os.path.join('..'))

if project_path not in sys.path:
    sys.path.append(project_path)


def autocorrections_type1(I1,  # DISACC_DISAGG_HYBRID_FORECAST_
                          I2  # FORECAST_FLAG_
                          ):
    T1 = pd.merge(I1, I2, how='outer')

    # set 1 if 'status' is not 'active'
    T1['flg_apply_corr1'] = (T1['status'] != 'active').astype(int)

    # set 1 if 'hybrid_forecast_value' is not numeric
    T1['flg_apply_corr2'] = pd.to_numeric(T1['hybrid_forecast_value'], errors='coerce').isna().astype(int)

    return T1


def autocorrections_type2(T1,
                          PRE_ABT_,
                          DEMAND_RESTORED_,
                          CONFIG_PARAMETERS,
                          CONFIG_FILE,
                          INITIAL_GLOBAL_FILE):
    # when flg_apply_corr2 is encountered, forecast value is replaced with either:
    # 1: average of past values for the unit (given above min number of values in min number of days)
    # 2: average of past values across similar units

    def own_average(unit, T1, CONFIG_PARAMETERS):
        # 1: Replaces missing hybrid forecast values with the units average regulated by CONFIG_PARAMS

        same = T1[(T1['product_id'] == unit['product_id']) & (T1['location_id'] == unit['location_id']) & (
                T1['customer_id'] == unit['customer_id']) & (T1['distr_channel_id'] == unit['distr_channel_id']) & (
                          T1['demand_type'] == unit['demand_type']) & (T1['period_dt'] < unit['period_dt'])]
        # choose max close dates
        close_dates = pd.date_range(unit['period_dt'] - pd.Timedelta(days=CONFIG_PARAMETERS['ib_npf_max_hist_depth']),
                                    unit['period_dt'] - pd.Timedelta(days=1))
        same = pd.merge(same, pd.DataFrame(close_dates, columns=['period_dt']), how='right')

        # choose 'ib_adj2_min_observ_num' (7) nearest dates
        same = same.dropna().tail(CONFIG_PARAMETERS['ib_adj2_min_observ_num'])

        if same.shape[0] == CONFIG_PARAMETERS['ib_adj2_min_observ_num']:
            return same['hybrid_forecast_value'].mean()

    def group_average(unit, T1, CONFIG_PARAMETERS):
        # 2: When own data is scarce, replaces missing hybrid forecast values with average over similar units

        close_dates = pd.date_range(
            unit['period_dt'] - pd.Timedelta(days=CONFIG_PARAMETERS['ib_npf_max_hist_depth'] - 1), unit['period_dt'])
        # fist omit 'distr_channel_id'
        similar = T1[(T1['product_id'] == unit['product_id']) & (T1['location_id'] == unit['location_id']) & (
                T1['customer_id'] == unit['customer_id']) & (T1['demand_type'] == unit['demand_type']) & (
                         T1['period_dt'].isin(close_dates))].sort_values(by='period_dt', ascending=False)
        # next omit 'customer_id'
        similar = pd.concat([similar,
                             T1[(T1['product_id'] == unit['product_id']) & (
                                     T1['location_id'] == unit['location_id']) & (
                                        T1['demand_type'] == unit['demand_type']) & (
                                    T1['period_dt'].isin(close_dates))].sort_values(by='period_dt', ascending=False)],
                            ignore_index=True)
        # next omit 'location_id'
        similar = pd.concat([similar,
                             T1[(T1['product_id'] == unit['product_id']) & (
                                     T1['demand_type'] == unit['demand_type']) & (
                                    T1['period_dt'].isin(close_dates))].sort_values(by='period_dt', ascending=False)],
                            ignore_index=True)
        similar = similar.drop_duplicates()
        # choose 'ib_adj2_min_observ_num' (7) nearest dates
        similar = similar.dropna().head(CONFIG_PARAMETERS['ib_adj2_min_observ_num'])

        if similar.shape[0] == CONFIG_PARAMETERS['ib_adj2_min_observ_num']:
            return similar['hybrid_forecast_value'].mean()

    # first try own_average, then go group_average
    # treat only observations in the forecast period
    part_1 = T1[((T1['flg_apply_corr2'] == 1)) & (T1['period_dt'] > INITIAL_GLOBAL_FILE['IB_HIST_END_DT'])]
    if part_1.shape[0] > 0: part_1['hybrid_forecast_value_aft2'] = part_1.apply(own_average, axis=1,
                                                                                args=(T1, CONFIG_PARAMETERS))

    # now try to use group_average for units that do not have enough past values
    part_2 = part_1[part_1['hybrid_forecast_value_aft2'].isna() == True]
    if part_2.shape[0] > 0: part_2['hybrid_forecast_value_aft2'] = part_2.apply(group_average, axis=1,
                                                                                args=(T1, CONFIG_PARAMETERS))

    # combine datasets
    T2 = pd.concat([part_1, part_2], ignore_index=True)
    T2 = T2.sort_values(by=['period_dt', 'product_id', 'location_id', 'customer_id', 'distr_channel_id'])
    T2 = pd.merge(T1, T2, how='left')
    T2['hybrid_forecast_value_aft2'] = T2['hybrid_forecast_value_aft2'].combine_first(T2['hybrid_forecast_value'])
    T2.loc[T2['flg_apply_corr1'] == 1, 'hybrid_forecast_value_aft2'] = np.nan

    # return T1 dataframe with 'hybrid_forecast_value_aft2' column containing Type-2-adjusted forecasts
    return T2


def autocorrections_type3(T2,
                          PRE_ABT_,
                          DEMAND_RESTORED_,
                          CONFIG_PARAMETERS,
                          CONFIG_FILE,
                          INITIAL_GLOBAL_FILE, ARLEY_CRIT):
    # correction is done only based on hist values

    # for each unit find last hist values (n_obs of them)
    # if n_obs >= ib_adj3_min_observ_num', calculate Arley bounds
    # if forecast outside Arley bounds, replace forecast value with mean or bound

    # function to get 5% crit value for Arley criterion
    def crit_lookup(n_obs, ARLEY_CRIT):
        crit_intervals = pd.IntervalIndex.from_arrays(
            ARLEY_CRIT['cnt_observations_lbound'],
            ARLEY_CRIT['cnt_observations_ubound'],
            closed='both')
        if n_obs >= crit_intervals[-1].right:
            return ARLEY_CRIT.iloc[-1, -1]
        else:
            return ARLEY_CRIT.iloc[crit_intervals.get_loc(n_obs)]['k_arley_005']

    def own_average_hist(unit, T, CONFIG_PARAMETERS, INITIAL_GLOBAL_FILE):

        if pd.isna(unit['hybrid_forecast_value_aft2']) == True:
            return unit['hybrid_forecast_value_aft2']

        # find last historic records for the unit (max 'ib_npf_max_hist_depth' of them)
        close_hist_dates = pd.date_range(
            INITIAL_GLOBAL_FILE['IB_HIST_END_DT'] - pd.Timedelta(days=CONFIG_PARAMETERS['ib_npf_max_hist_depth']),
            INITIAL_GLOBAL_FILE['IB_HIST_END_DT'])
        same = T[(T['product_id'] == unit['product_id']) & (T['location_id'] == unit['location_id']) & (
                T['customer_id'] == unit['customer_id']) & (T['distr_channel_id'] == unit['distr_channel_id']) & (
                         T['demand_type'] == unit['demand_type']) & (T['period_dt'].isin(close_hist_dates))]
        same = same.dropna()
        n_obs = same['hybrid_forecast_value_aft2'].count()

        # if enough values are present, apply Arley criterion
        if n_obs >= CONFIG_PARAMETERS['ib_adj3_min_observ_num']:
            same_mean = same['hybrid_forecast_value_aft2'].mean()
            same_std = same['hybrid_forecast_value_aft2'].std()
            # find corresponding Arley crit values, calculate bounds
            upper = same_mean + np.sqrt((n_obs - 1) / n_obs) * same_std * crit_lookup(n_obs, ARLEY_CRIT)
            lower = same_mean - np.sqrt((n_obs - 1) / n_obs) * same_std * crit_lookup(n_obs, ARLEY_CRIT)

            # apply outlier treatment method
            if CONFIG_PARAMETERS['ib_adj3_correction_method'] == 'mean':

                if (unit['hybrid_forecast_value_aft2'] < lower) or (unit['hybrid_forecast_value_aft2'] > upper):
                    return same_mean
                else:
                    return unit['hybrid_forecast_value_aft2']

            if CONFIG_PARAMETERS['ib_adj3_correction_method'] == 'bound':

                if (unit['hybrid_forecast_value_aft2'] < lower):
                    return lower
                elif (unit['hybrid_forecast_value_aft2'] > upper):
                    return upper
                else:
                    return unit['hybrid_forecast_value_aft2']

                    # apply own_average_hist() on forecast period on non-promo units only with autocorrection flags 0, 1 == 0

    T3 = T2[(T2['period_dt'] > INITIAL_GLOBAL_FILE['IB_HIST_END_DT']) & (T2['demand_type'] == 'regular') & (
            T2['flg_apply_corr1'] == 0) & (T2['flg_apply_corr2'] == 0)]
    T3['hybrid_forecast_value_aft3'] = T3['hybrid_forecast_value_aft2']
    T3['hybrid_forecast_value_aft3'] = T3.apply(own_average_hist, axis=1,
                                                args=(T2, CONFIG_PARAMETERS, INITIAL_GLOBAL_FILE))
    T3 = pd.merge(T2, T3, how='left')
    T3['hybrid_forecast_value_aft3'] = T3['hybrid_forecast_value_aft3'].combine_first(T2['hybrid_forecast_value_aft2'])
    T3['flg_apply_corr3'] = (~(T3['hybrid_forecast_value_aft3'].isna()) & (
            T3['hybrid_forecast_value_aft3'] != T3['hybrid_forecast_value_aft2'])).astype(int)
    return T3
