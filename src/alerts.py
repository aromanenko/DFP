# alerts.py

import pandas as pd
from datetime import timedelta

# Вспомогательные функции

def week_start_date(date):
    """Вернуть дату начала недели (понедельник) для заданной даты."""
    # Если date = среда, вычитаем её номер дня (Mon=0,...,Sun=6), чтобы получить понедельник
    return date - timedelta(days=date.weekday())

def parse_config(config_params):
    """Извлечь необходимые параметры конфигурации в словарь."""
    # Поддержка config_params в виде DataFrame (колонки 'Column Name' и 'Value') или dict
    if isinstance(config_params, pd.DataFrame):
        conf = {row['Column Name']: row['Value'] for _, row in config_params.iterrows()}
    else:
        conf = dict(config_params)
    conf_values = {}
    # Преобразуем дату конца исторического периода в Timestamp
    if 'IB_HIST_END_DT' in conf:
        conf_values['IB_HIST_END_DT'] = pd.to_datetime(conf['IB_HIST_END_DT'])
    else:
        conf_values['IB_HIST_END_DT'] = None
    conf_values['IB_FC_HORIZ'] = conf.get('IB_FC_HORIZ')               # горизонт прогнозирования (в неделях)
    conf_values['IB_FF_ACTIVE_STATUS_LIST'] = conf.get('IB_FF_ACTIVE_STATUS_LIST', ['active'])
    conf_values['IB_ALERT_MIN_VAL'] = conf.get('IB_ALERT_MIN_VAL', 0.1) # минимальный значимый объём прогноза
    conf_values['IB_ALERT_MIN_OBS'] = conf.get('IB_ALERT_MIN_OBS', 10)  # минимальное число наблюдений/SKU для расчётов
    conf_values['IB_MAX_NP_HISTORY'] = conf.get('IB_MAX_NP_HISTORY', 30) # дни для определения нового товара (новый, если <= 30)
    # Уровни иерархии для VF (статистического прогноза), если заданы
    conf_values['VF_PRODUCT_LVL'] = conf.get('IB_VF_PRODUCT_LVL')
    conf_values['VF_LOCATION_LVL'] = conf.get('IB_VF_LOCATION_LVL')
    conf_values['VF_CUSTOMER_LVL'] = conf.get('IB_VF_CUSTOMER_LVL')
    conf_values['VF_DISTR_CHANNEL_LVL'] = conf.get('IB_VF_DISTR_LVL') or conf.get('IB_VF_DISTR_CHANNEL_LVL')
    return conf_values

def generate_timeline(ff_active_df, group_keys, conf, time_lvl):
    """Сгенерировать полный таймлайн периодов (например, недельных) от MIN_START_DT до MAX_END_DT для каждой комбинации."""
    # Вычисляем минимальный и максимальный период для каждой комбинации по активным флагам
    range_df = ff_active_df.groupby(group_keys).agg(
        MIN_START_DT=pd.NamedAgg(column='PERIOD_START_DT', aggfunc='min'),
        MAX_END_DT=pd.NamedAgg(column='PERIOD_END_DT', aggfunc='max')
    ).reset_index()
    # Корректируем границы периода с учётом исторического отрезка и горизонта прогноза
    hist_end = conf.get('IB_HIST_END_DT')
    horizon_weeks = conf.get('IB_FC_HORIZ')
    if hist_end is not None and horizon_weeks is not None:
        horiz_end_date = hist_end + pd.Timedelta(weeks=int(horizon_weeks))
    else:
        horiz_end_date = None
    range_df['ADJ_START_DT'] = range_df['MIN_START_DT']
    range_df['ADJ_END_DT'] = range_df['MAX_END_DT']
    if hist_end is not None:
        # Начало прогноза не может быть раньше дня, следующего за концом истории
        range_df['ADJ_START_DT'] = range_df['ADJ_START_DT'].apply(lambda d: max(d, hist_end + pd.Timedelta(days=1)))
    if horiz_end_date is not None:
        # Конец периода не может превышать конец горизонта прогнозирования
        range_df['ADJ_END_DT'] = range_df['ADJ_END_DT'].apply(lambda d: min(d, horiz_end_date))
    # Генерируем список записей (combo + период) для каждого комбо с шагом, соответствующим granularity времени
    timeline_records = []
    for _, row in range_df.iterrows():
        start = pd.to_datetime(row['ADJ_START_DT']); end = pd.to_datetime(row['ADJ_END_DT'])
        if pd.isna(start) or pd.isna(end):
            continue
        # Определяем шаг времени: неделя, месяц или день
        if isinstance(time_lvl, str) and 'week' in time_lvl:
            current = week_start_date(start)
            end_period = week_start_date(end)
            step = pd.Timedelta(weeks=1)
        elif isinstance(time_lvl, str) and 'month' in time_lvl:
            current = start.replace(day=1)
            end_period = end.replace(day=1)
            step = pd.DateOffset(months=1)
        else:
            current = start
            end_period = end
            step = pd.Timedelta(days=1)
        combo_keys = {key: row[key] for key in group_keys}  # фиксированные значения комбинированных ключей
        while current <= end_period:
            rec = combo_keys.copy()
            rec['PERIOD_START_DT'] = current
            timeline_records.append(rec)
            current += step
    return pd.DataFrame(timeline_records)

# Общий список колонок для результирующих таблиц алертов (T1)
OUTPUT_COLUMNS = [
    'PRODUCT_ID','STORE_LOCATION_ID','PERIOD_START_DT','SEGMENT_NAME',
    'VF_FORECAST_VALUE','DEMAND_TYPE','ASSORTMENT_TYPE','ML_FORECAST_VALUE','HYBRID_FORECAST_VALUE',
    'ALERT_TYPE','product_lvl_id','product_lvl','location_lvl_id','location_lvl',
    'customer_lvl_id','customer_lvl','distr_channel_lvl_id','distr_channel_lvl',
    'KPI_NM','INPUT_TABLE','STAT_NOM_NM','STAT_DEN_NM','STAT_NOM','STAT_DEN','ALERT_THRESHOLD','ALERT_STAT_VAL'
]

# Функции вычисления алертов 1–10
def alert1(hybrid_forecast_df, forecast_flag_df, alert_parameters_df, config):
    """
    Алерт 1: Отсутствие или нулевое значение прогноза для регулярного SKU (тип NAREG/ZEROREG).
    """
    conf = parse_config(config)
    params = alert_parameters_df[alert_parameters_df['alert_id'] == 1].iloc[0]
    prod_lvl = int(params['al_product_lvl'])
    loc_lvl = int(params['al_location_lvl'])
    cust_lvl = params.get('al_customer_lvl')
    dist_lvl = params.get('al_distr_channel_lvl')
    time_lvl = params['al_tim_lvl']
    tgt_table = params['Input_table_table']
    tgt_col = params['Input_column']
    target_type = params.get('tgt_type', '')

    # 1) Фильтрация по старому ассортименту и регулярному спросу
    df = hybrid_forecast_df.copy()
    df = df[(df['ASSORTMENT_TYPE'] == 'old') & (df['DEMAND_TYPE'] == 'regular')]

    # 2) Агрегация прогноза по неделям
    df['PERIOD_START_DT'] = df['PERIOD_DT'].apply(week_start_date)
    group_keys = [f'PRODUCT_LVL_ID{prod_lvl}', f'LOCATION_LVL_ID{loc_lvl}', 'PERIOD_START_DT']
    if cust_lvl and not pd.isna(cust_lvl):
        group_keys.insert(2, f'CUSTOMER_LVL_ID{int(cust_lvl)}')
    if dist_lvl and not pd.isna(dist_lvl):
        group_keys.insert(len(group_keys)-1, f'DISTR_CHANNEL_LVL_ID{int(dist_lvl)}')

    agg_df = (
        df
        .groupby(group_keys, dropna=False)[tgt_col]
        .mean()
        .reset_index()
        .rename(columns={tgt_col: 'FORECAST_VALUE'})
    )

    # 3) Генерация полного таймлайна из FORECAST_FLAG
    ff_active = forecast_flag_df[forecast_flag_df['STATUS'].isin(conf['IB_FF_ACTIVE_STATUS_LIST'])]
    ff_group_keys = [k for k in group_keys if k != 'PERIOD_START_DT']
    timeline_df = generate_timeline(ff_active, ff_group_keys, conf, time_lvl)

    # 4) Левый джойн прогноза на таймлайн
    merged = timeline_df.merge(agg_df, on=group_keys, how='left')

    # 5) Отбор строк, где прогноз отсутствует или <= min_val
    min_val = conf['IB_ALERT_MIN_VAL']
    alert_mask = merged['FORECAST_VALUE'].isna() | (merged['FORECAST_VALUE'] <= min_val)
    alert_rows = merged[alert_mask].copy()
    if alert_rows.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    alert_rows['ALERT_TYPE'] = alert_rows['FORECAST_VALUE'].apply(
        lambda x: 'NAREG' if pd.isna(x) else 'ZEROREG'
    )

    # ——— ИСПРАВЛЕНИЕ: правильный merge контекста ———
    # Берём контекст (SEGMENT_NAME, VF_FORECAST_VALUE и т.д.) из исходного df
    context_keys = [k for k in group_keys if k != 'PERIOD_START_DT']
    context_cols = [
        'SEGMENT_NAME','VF_FORECAST_VALUE','DEMAND_TYPE',
        'ASSORTMENT_TYPE','ML_FORECAST_VALUE','HYBRID_FORECAST_VALUE'
    ]
    context_df = df.drop_duplicates(subset=context_keys)[context_keys + context_cols]

    # Объединяем alert_rows с context_df по тем же ключам (до переименования)
    out = alert_rows.merge(context_df, on=context_keys, how='left')

    # 6) Переименовываем уровневые колонки в PRODUCT_ID, STORE_LOCATION_ID и т.д.
    rename_map = {
        f'PRODUCT_LVL_ID{prod_lvl}': 'PRODUCT_ID',
        f'LOCATION_LVL_ID{loc_lvl}': 'STORE_LOCATION_ID'
    }
    if cust_lvl and not pd.isna(cust_lvl):
        rename_map[f'CUSTOMER_LVL_ID{int(cust_lvl)}'] = 'CUSTOMER_ID'
    if dist_lvl and not pd.isna(dist_lvl):
        rename_map[f'DISTR_CHANNEL_LVL_ID{int(dist_lvl)}'] = 'DISTR_CHANNEL_ID'

    out = out.rename(columns=rename_map)

    # 7) Заполняем все остальные обязательные поля
    out['product_lvl_id'] = out['PRODUCT_ID']
    out['product_lvl'] = prod_lvl
    out['location_lvl_id'] = out['STORE_LOCATION_ID']
    out['location_lvl'] = loc_lvl
    out['customer_lvl_id'] = out.get('CUSTOMER_ID', pd.NA)
    out['customer_lvl'] = int(cust_lvl) if cust_lvl and not pd.isna(cust_lvl) else pd.NA
    out['distr_channel_lvl_id'] = out.get('DISTR_CHANNEL_ID', pd.NA)
    out['distr_channel_lvl'] = int(dist_lvl) if dist_lvl and not pd.isna(dist_lvl) else pd.NA

    kpi_nm = f"{target_type}.{tgt_col}" if target_type else tgt_col
    out['KPI_NM'] = kpi_nm

    input_table_name = tgt_table
    if target_type and input_table_name.endswith(f"_{target_type}"):
        input_table_name = input_table_name[:-(len(target_type) + 1)]
    out['INPUT_TABLE'] = input_table_name

    out['STAT_NOM_NM'] = 'Forecast value'
    out['STAT_DEN_NM'] = 'na'
    out['STAT_NOM'] = out['FORECAST_VALUE'].fillna(pd.NA)
    out['STAT_DEN'] = 1
    out['ALERT_THRESHOLD'] = pd.NA
    out['ALERT_STAT_VAL'] = out['FORECAST_VALUE'].fillna(pd.NA)

    # 8) Оставляем только нужные колонки в нужном порядке
    return out[OUTPUT_COLUMNS]


def alert2(hybrid_forecast_df, restored_demand_df, alert_parameters_df, config):
    """
    Алерт 2: Аномально высокое значение прогноза для регулярного ассортимента (INCRREG).
    """
    conf = parse_config(config)
    params = alert_parameters_df[alert_parameters_df['alert_id'] == 2].iloc[0]
    prod_lvl = int(params['al_product_lvl'])
    loc_lvl = int(params['al_location_lvl'])
    cust_lvl = params.get('al_customer_lvl')
    dist_lvl = params.get('al_distr_channel_lvl')
    threshold_val = float(params['alert_threshold_val'])
    tgt_table = params['Input_table_table']
    tgt_col = params['Input_column']
    target_type = params.get('tgt_type', '')

    # --- Шаг 1: фильтрация исходного прогноза ---
    df = hybrid_forecast_df.copy()
    df = df[(df['ASSORTMENT_TYPE'] == 'old') & (df['DEMAND_TYPE'] == 'regular')]
    df['PERIOD_START_DT'] = df['PERIOD_DT'].apply(week_start_date)

    # --- Шаг 2: агрегируем прогноз по неделям ---
    agg_keys = [f'PRODUCT_LVL_ID{prod_lvl}', f'LOCATION_LVL_ID{loc_lvl}']
    if cust_lvl and not pd.isna(cust_lvl):
        agg_keys.append(f'CUSTOMER_LVL_ID{int(cust_lvl)}')
    if dist_lvl and not pd.isna(dist_lvl):
        agg_keys.append(f'DISTR_CHANNEL_LVL_ID{int(dist_lvl)}')
    agg_keys.append('PERIOD_START_DT')

    forecast_agg = (
        df
        .groupby(agg_keys)[tgt_col]
        .mean()
        .reset_index()
        .rename(columns={tgt_col: 'STAT_NOM_VAL'})
    )

    # --- Шаг 3: агрегируем восстановленный спрос прошлого года ---
    demand = restored_demand_df.copy()
    demand['PERIOD_START_DT'] = demand['PERIOD_DT'].apply(week_start_date)
    if target_type and 'TGT_TYPE' in demand.columns:
        demand = demand[demand['TGT_TYPE'] == target_type]
    demand_agg = (
        demand
        .groupby(agg_keys)['SALESTGT_QTY_R']
        .mean()
        .reset_index()
        .rename(columns={'SALESTGT_QTY_R': 'STAT_DEN_VAL'})
    )
    # сдвигаем на 52 недели вперёд
    demand_agg['PERIOD_START_DT'] = demand_agg['PERIOD_START_DT'] + pd.DateOffset(weeks=52)

    # --- Шаг 4: соединяем прогноз и спрос ---
    combined = forecast_agg.merge(
        demand_agg[agg_keys + ['STAT_DEN_VAL']],
        on=agg_keys,
        how='left'
    )
    valid = combined[
        combined['STAT_DEN_VAL'].notna() &
        (combined['STAT_DEN_VAL'] > conf['IB_ALERT_MIN_VAL'])
    ].copy()
    if valid.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    valid['ALERT_STAT_VAL'] = valid['STAT_NOM_VAL'] / valid['STAT_DEN_VAL']
    alert_rows = valid[valid['ALERT_STAT_VAL'] > threshold_val].copy()
    if alert_rows.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    alert_rows['ALERT_TYPE'] = 'INCRREG'

    # --- Шаг 5: формируем контекст (SEGMENT_NAME, VF_FORECAST_VALUE и пр.) ---
    # Берём из оригинального df по тем же agg_keys (кроме даты)
    context_keys = [k for k in agg_keys if k != 'PERIOD_START_DT']
    context_cols = [
        'SEGMENT_NAME',
        'VF_FORECAST_VALUE',
        'DEMAND_TYPE',
        'ASSORTMENT_TYPE',
        'ML_FORECAST_VALUE',
        'HYBRID_FORECAST_VALUE'
    ]
    context_df = df.drop_duplicates(subset=context_keys)[context_keys + context_cols]

    # Соединяем alert_rows с context_df на оригинальных ключах
    out = alert_rows.merge(context_df, on=context_keys, how='left')

    # --- Шаг 6: переименовываем уровневые колонки в PRODUCT_ID, STORE_LOCATION_ID и т.д. ---
    rename_map = {
        f'PRODUCT_LVL_ID{prod_lvl}': 'PRODUCT_ID',
        f'LOCATION_LVL_ID{loc_lvl}': 'STORE_LOCATION_ID'
    }
    if cust_lvl and not pd.isna(cust_lvl):
        rename_map[f'CUSTOMER_LVL_ID{int(cust_lvl)}'] = 'CUSTOMER_ID'
    if dist_lvl and not pd.isna(dist_lvl):
        rename_map[f'DISTR_CHANNEL_LVL_ID{int(dist_lvl)}'] = 'DISTR_CHANNEL_ID'

    out = out.rename(columns=rename_map)

    # --- Шаг 7: добавляем все остальные поля формата T1 ---
    out['product_lvl_id'] = out['PRODUCT_ID']
    out['product_lvl'] = prod_lvl
    out['location_lvl_id'] = out['STORE_LOCATION_ID']
    out['location_lvl'] = loc_lvl
    out['customer_lvl_id'] = out.get('CUSTOMER_ID', pd.NA)
    out['customer_lvl'] = int(cust_lvl) if cust_lvl and not pd.isna(cust_lvl) else pd.NA
    out['distr_channel_lvl_id'] = out.get('DISTR_CHANNEL_ID', pd.NA)
    out['distr_channel_lvl'] = int(dist_lvl) if dist_lvl and not pd.isna(dist_lvl) else pd.NA

    # KPI_NM и INPUT_TABLE
    out['KPI_NM'] = f"{target_type}.{tgt_col}" if target_type else tgt_col
    input_table_name = tgt_table
    if target_type and input_table_name.endswith(f"_{target_type}"):
        input_table_name = input_table_name[:-(len(target_type) + 1)]
    out['INPUT_TABLE'] = input_table_name

    out['STAT_NOM_NM'] = 'Average Forecast Value'
    out['STAT_DEN_NM'] = 'Average Demand Value a year ago'
    out['STAT_NOM'] = out['STAT_NOM_VAL']
    out['STAT_DEN'] = out['STAT_DEN_VAL']
    out['ALERT_THRESHOLD'] = threshold_val
    # ALERT_STAT_VAL уже рассчитано

    # --- Финальная выборка колонок ---
    return out[OUTPUT_COLUMNS]


def alert3(hybrid_forecast_df, restored_demand_df, alert_parameters_df, config):
    """
    Алерт 3: Аномально низкое значение прогноза для регулярного ассортимента (DECRREG).
    """
    conf = parse_config(config)
    params = alert_parameters_df[alert_parameters_df['alert_id'] == 3].iloc[0]
    prod_lvl = int(params['al_product_lvl'])
    loc_lvl = int(params['al_location_lvl'])
    cust_lvl = params.get('al_customer_lvl')
    dist_lvl = params.get('al_distr_channel_lvl')
    threshold_val = float(params['alert_threshold_val'])
    tgt_table = params['Input_table_table']
    tgt_col = params['Input_column']
    target_type = params.get('tgt_type', '')

    # 1) Фильтрация исходного прогноза
    df = hybrid_forecast_df.copy()
    df = df[(df['ASSORTMENT_TYPE'] == 'old') & (df['DEMAND_TYPE'] == 'regular')]
    df['PERIOD_START_DT'] = df['PERIOD_DT'].apply(week_start_date)

    # 2) Агрегация прогноза по неделям
    agg_keys = [f'PRODUCT_LVL_ID{prod_lvl}', f'LOCATION_LVL_ID{loc_lvl}']
    if cust_lvl and not pd.isna(cust_lvl):
        agg_keys.append(f'CUSTOMER_LVL_ID{int(cust_lvl)}')
    if dist_lvl and not pd.isna(dist_lvl):
        agg_keys.append(f'DISTR_CHANNEL_LVL_ID{int(dist_lvl)}')
    agg_keys.append('PERIOD_START_DT')

    forecast_agg = (
        df
        .groupby(agg_keys)[tgt_col]
        .mean()
        .reset_index()
        .rename(columns={tgt_col: 'STAT_NOM_VAL'})
    )
    forecast_agg['PERIOD_LAST_YEAR'] = forecast_agg['PERIOD_START_DT'] - pd.DateOffset(weeks=52)

    # 3) Агрегация спроса год назад
    demand = restored_demand_df.copy()
    demand['PERIOD_START_DT'] = demand['PERIOD_DT'].apply(week_start_date)
    if target_type and 'TGT_TYPE' in demand.columns:
        demand = demand[demand['TGT_TYPE'] == target_type]

    demand_agg = (
        demand
        .groupby(agg_keys)['SALESTGT_QTY_R']
        .mean()
        .reset_index()
        .rename(columns={'SALESTGT_QTY_R': 'STAT_DEN_VAL'})
    )
    demand_agg['PERIOD_START_DT'] = demand_agg['PERIOD_START_DT'] + pd.DateOffset(weeks=52)

    # 4) Соединяем и отбираем по порогу
    combined = forecast_agg.merge(
        demand_agg[agg_keys + ['STAT_DEN_VAL']],
        on=agg_keys,
        how='left'
    )
    valid = combined[
        (combined['STAT_DEN_VAL'].notna()) &
        (combined['STAT_DEN_VAL'] > conf['IB_ALERT_MIN_VAL'])
    ].copy()
    if valid.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    valid['ALERT_STAT_VAL'] = valid['STAT_NOM_VAL'] / valid['STAT_DEN_VAL']
    alert_rows = valid[valid['ALERT_STAT_VAL'] < 1 / threshold_val].copy()
    if alert_rows.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    alert_rows['ALERT_TYPE'] = 'DECRREG'

    # 5) Контекст из исходного df
    context_keys = [k for k in agg_keys if k != 'PERIOD_START_DT']
    context_cols = [
        'SEGMENT_NAME',
        'VF_FORECAST_VALUE',
        'DEMAND_TYPE',
        'ASSORTMENT_TYPE',
        'ML_FORECAST_VALUE',
        'HYBRID_FORECAST_VALUE'
    ]
    context_df = df.drop_duplicates(subset=context_keys)[context_keys + context_cols]

    out = alert_rows.merge(context_df, on=context_keys, how='left')

    # 6) Переименование уровневых полей
    rename_map = {
        f'PRODUCT_LVL_ID{prod_lvl}': 'PRODUCT_ID',
        f'LOCATION_LVL_ID{loc_lvl}': 'STORE_LOCATION_ID'
    }
    if cust_lvl and not pd.isna(cust_lvl):
        rename_map[f'CUSTOMER_LVL_ID{int(cust_lvl)}'] = 'CUSTOMER_ID'
    if dist_lvl and not pd.isna(dist_lvl):
        rename_map[f'DISTR_CHANNEL_LVL_ID{int(dist_lvl)}'] = 'DISTR_CHANNEL_ID'
    out = out.rename(columns=rename_map)

    # 7) Заполняем остальные поля из спецификации T1
    out['product_lvl_id'] = out['PRODUCT_ID']
    out['product_lvl'] = prod_lvl
    out['location_lvl_id'] = out['STORE_LOCATION_ID']
    out['location_lvl'] = loc_lvl
    out['customer_lvl_id'] = out.get('CUSTOMER_ID', pd.NA)
    out['customer_lvl'] = int(cust_lvl) if cust_lvl and not pd.isna(cust_lvl) else pd.NA
    out['distr_channel_lvl_id'] = out.get('DISTR_CHANNEL_ID', pd.NA)
    out['distr_channel_lvl'] = int(dist_lvl) if dist_lvl and not pd.isna(dist_lvl) else pd.NA

    out['KPI_NM'] = f"{target_type}.{tgt_col}" if target_type else tgt_col
    input_table_name = params['Input_table_table']
    if target_type and input_table_name.endswith(f"_{target_type}"):
        input_table_name = input_table_name[:-(len(target_type) + 1)]
    out['INPUT_TABLE'] = input_table_name

    out['STAT_NOM_NM'] = 'Average Forecast Value'
    out['STAT_DEN_NM'] = 'Average Demand Value a year ago'
    out['STAT_NOM'] = out['STAT_NOM_VAL']
    out['STAT_DEN'] = out['STAT_DEN_VAL']
    out['ALERT_THRESHOLD'] = threshold_val
    # ALERT_STAT_VAL уже есть

    return out[OUTPUT_COLUMNS]


def alert4(hybrid_forecast_df, restored_demand_df, alert_parameters_df, config):
    """
    Алерт 4: Аномально высокое значение прогноза относительно последних данных (HIGHREG).
    Если средний недельный прогноз превышает средний недельный фактический спрос за последние L=12 недель более чем в alert_threshold раз (например, 5×), 
    то формируется алерт HIGHREG:contentReference[oaicite:10]{index=10}:contentReference[oaicite:11]{index=11}. 
    """
    conf = parse_config(config)
    params = alert_parameters_df[alert_parameters_df['alert_id'] == 4].iloc[0]
    prod_lvl = int(params['al_product_lvl']); loc_lvl = int(params['al_location_lvl'])
    threshold_val = float(params['alert_threshold_val'])
    # 1. Средний прогноз по неделям (за каждый будущий период)
    df = hybrid_forecast_df.copy()
    df = df[(df['ASSORTMENT_TYPE'] == 'old') & (df['DEMAND_TYPE'] == 'regular')]
    df['PERIOD_START_DT'] = df['PERIOD_DT'].apply(week_start_date)
    forecast_weekly = df.groupby([f'PRODUCT_LVL_ID{prod_lvl}', f'LOCATION_LVL_ID{loc_lvl}', 'PERIOD_START_DT'])['HYBRID_FORECAST_VALUE'] \
                        .mean().reset_index().rename(columns={'HYBRID_FORECAST_VALUE': 'STAT_NOM_VAL'})
    # 2. Средний фактический спрос по неделям за последние 12 недель истории
    demand = restored_demand_df.copy()
    demand['PERIOD_START_DT'] = demand['PERIOD_DT'].apply(week_start_date)
    demand_weekly = demand.groupby([f'PRODUCT_LVL_ID{prod_lvl}', f'LOCATION_LVL_ID{loc_lvl}', 'PERIOD_START_DT'])['SALESTGT_QTY_R'] \
                          .mean().reset_index().rename(columns={'SALESTGT_QTY_R': 'DEMAND'})
    # Ограничиваемся периодом последних L недель истории (L=12 недель):contentReference[oaicite:12]{index=12}
    hist_end = conf.get('IB_HIST_END_DT')
    if hist_end is not None:
        cutoff_date = hist_end - pd.Timedelta(weeks=12)
        demand_weekly = demand_weekly[demand_weekly['PERIOD_START_DT'] > cutoff_date]
    # Вычисляем средний спрос за последние L недель (один показатель на комбинацию)
    demand_L_mean = demand_weekly.groupby([f'PRODUCT_LVL_ID{prod_lvl}', f'LOCATION_LVL_ID{loc_lvl}'])['DEMAND'] \
                                  .mean().reset_index().rename(columns={'DEMAND': 'STAT_DEN_VAL'})
    # 3. Объединяем прогноз с рассчитанным средним спросом
    combined = forecast_weekly.merge(demand_L_mean, on=[f'PRODUCT_LVL_ID{prod_lvl}', f'LOCATION_LVL_ID{loc_lvl}'], how='left')
    valid = combined[(combined['STAT_DEN_VAL'].notna()) & (combined['STAT_DEN_VAL'] > conf['IB_ALERT_MIN_VAL'])].copy()
    if valid.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    # 4. Рассчитываем отношение прогноза к среднему спросу, отбираем большие отклонения
    valid['ALERT_STAT_VAL'] = valid['STAT_NOM_VAL'] / valid['STAT_DEN_VAL']
    alert_rows = valid[valid['ALERT_STAT_VAL'] > threshold_val].copy()
    if alert_rows.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    alert_rows['ALERT_TYPE'] = 'HIGHREG'
    # 5. Формируем итоговые поля вывода
    out = alert_rows.copy()
    out.rename(columns={
        f'PRODUCT_LVL_ID{prod_lvl}': 'PRODUCT_ID',
        f'LOCATION_LVL_ID{loc_lvl}': 'STORE_LOCATION_ID'
    }, inplace=True)
    out['product_lvl_id'] = out['PRODUCT_ID']; out['product_lvl'] = prod_lvl
    out['location_lvl_id'] = out['STORE_LOCATION_ID']; out['location_lvl'] = loc_lvl
    # KPI_NM и INPUT_TABLE (для простоты берем без разделения по типам, т.к. target_type не задан явно)
    out['KPI_NM'] = 'HYBRID_FORECAST_VALUE'
    out['INPUT_TABLE'] = 'ACC_AGG_HYBRID_FORECAST'
    out['STAT_NOM_NM'] = 'Average Forecast Value'
    out['STAT_DEN_NM'] = 'Average Demand Value within last 3 months'
    out['STAT_NOM'] = out['STAT_NOM_VAL']; out['STAT_DEN'] = out['STAT_DEN_VAL']
    out['ALERT_THRESHOLD'] = threshold_val
    # Добавляем недостающие колонки как <NA> для соответствия OUTPUT_COLUMNS
    for col in OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[OUTPUT_COLUMNS]
    return out

def alert5(hybrid_forecast_df, restored_demand_df, alert_parameters_df, config):
    """
    Алерт 5: Аномально низкое значение прогноза относительно последних данных (LOWREG).
    Если средний прогноз **ниже** среднего фактического спроса за последние 12 недель более чем в alert_threshold раз 
    (например, <1/5 от среднего спроса), формируется алерт LOWREG:contentReference[oaicite:13]{index=13}:contentReference[oaicite:14]{index=14}.
    """
    conf = parse_config(config)
    params = alert_parameters_df[alert_parameters_df['alert_id'] == 5].iloc[0]
    prod_lvl = int(params['al_product_lvl']); loc_lvl = int(params['al_location_lvl'])
    threshold_val = float(params['alert_threshold_val'])
    # Логика аналогична alert4, но фильтр обратный (прогноз << исторического спроса)
    df = hybrid_forecast_df.copy()
    df = df[(df['ASSORTMENT_TYPE'] == 'old') & (df['DEMAND_TYPE'] == 'regular')]
    df['PERIOD_START_DT'] = df['PERIOD_DT'].apply(week_start_date)
    forecast_weekly = df.groupby([f'PRODUCT_LVL_ID{prod_lvl}', f'LOCATION_LVL_ID{loc_lvl}', 'PERIOD_START_DT'])['HYBRID_FORECAST_VALUE'] \
                        .mean().reset_index().rename(columns={'HYBRID_FORECAST_VALUE': 'STAT_NOM_VAL'})
    demand = restored_demand_df.copy()
    demand['PERIOD_START_DT'] = demand['PERIOD_DT'].apply(week_start_date)
    demand_weekly = demand.groupby([f'PRODUCT_LVL_ID{prod_lvl}', f'LOCATION_LVL_ID{loc_lvl}', 'PERIOD_START_DT'])['SALESTGT_QTY_R'] \
                          .mean().reset_index().rename(columns={'SALESTGT_QTY_R': 'DEMAND'})
    hist_end = conf.get('IB_HIST_END_DT')
    if hist_end is not None:
        cutoff_date = hist_end - pd.Timedelta(weeks=12)
        demand_weekly = demand_weekly[demand_weekly['PERIOD_START_DT'] > cutoff_date]
    demand_L_mean = demand_weekly.groupby([f'PRODUCT_LVL_ID{prod_lvl}', f'LOCATION_LVL_ID{loc_lvl}'])['DEMAND'] \
                                  .mean().reset_index().rename(columns={'DEMAND': 'STAT_DEN_VAL'})
    combined = forecast_weekly.merge(demand_L_mean, on=[f'PRODUCT_LVL_ID{prod_lvl}', f'LOCATION_LVL_ID{loc_lvl}'], how='left')
    valid = combined[(combined['STAT_DEN_VAL'].notna()) & (combined['STAT_DEN_VAL'] > conf['IB_ALERT_MIN_VAL'])].copy()
    if valid.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    valid['ALERT_STAT_VAL'] = valid['STAT_NOM_VAL'] / valid['STAT_DEN_VAL']
    alert_rows = valid[valid['ALERT_STAT_VAL'] < 1 / threshold_val].copy()
    if alert_rows.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    alert_rows['ALERT_TYPE'] = 'LOWREG'
    out = alert_rows.copy()
    out.rename(columns={
        f'PRODUCT_LVL_ID{prod_lvl}': 'PRODUCT_ID',
        f'LOCATION_LVL_ID{loc_lvl}': 'STORE_LOCATION_ID'
    }, inplace=True)
    out['product_lvl_id'] = out['PRODUCT_ID']; out['product_lvl'] = prod_lvl
    out['location_lvl_id'] = out['STORE_LOCATION_ID']; out['location_lvl'] = loc_lvl
    out['KPI_NM'] = 'HYBRID_FORECAST_VALUE'
    out['INPUT_TABLE'] = 'ACC_AGG_HYBRID_FORECAST'
    out['STAT_NOM_NM'] = 'Average Forecast Value'
    out['STAT_DEN_NM'] = 'Average Demand Value within last 3 months'
    out['STAT_NOM'] = out['STAT_NOM_VAL']; out['STAT_DEN'] = out['STAT_DEN_VAL']
    out['ALERT_THRESHOLD'] = threshold_val
    for col in OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[OUTPUT_COLUMNS]
    return out

def alert6(hybrid_forecast_df, restored_demand_df, forecast_flag_df, alert_parameters_df, config):
    """
    Алерт 6: Аномальное максимальное отклонение прогноза (DEVWK).
    Сравнивает максимальный скачок прогноза (между последней фактической неделей и первой прогнозной, либо внутри горизонта) 
    с типичным разбросом спроса за последние 3 месяца:contentReference[oaicite:15]{index=15}:contentReference[oaicite:16]{index=16}. 
    Если скачок прогноза более чем в threshold раз превышает среднее недельное изменение спроса, формируется алерт DEVWK.
    """
    conf = parse_config(config)
    params = alert_parameters_df[alert_parameters_df['alert_id'] == 6].iloc[0]
    prod_lvl = int(params['al_product_lvl']); loc_lvl = int(params['al_location_lvl'])
    threshold_val = float(params['alert_threshold_val'])
    # 1. Получаем недельный ряд прогноза и фактического спроса
    df_forecast = hybrid_forecast_df.copy()
    df_forecast = df_forecast[(df_forecast['ASSORTMENT_TYPE'] == 'old') & (df_forecast['DEMAND_TYPE'] == 'regular')]
    df_forecast['PERIOD_START_DT'] = df_forecast['PERIOD_DT'].apply(week_start_date)
    forecast_weekly = df_forecast.groupby([f'PRODUCT_LVL_ID{prod_lvl}', f'LOCATION_LVL_ID{loc_lvl}', 'PERIOD_START_DT'])['HYBRID_FORECAST_VALUE'] \
                                  .mean().reset_index().rename(columns={'HYBRID_FORECAST_VALUE': 'VALUE'})
    df_demand = restored_demand_df.copy()
    df_demand['PERIOD_START_DT'] = df_demand['PERIOD_DT'].apply(week_start_date)
    demand_weekly = df_demand.groupby([f'PRODUCT_LVL_ID{prod_lvl}', f'LOCATION_LVL_ID{loc_lvl}', 'PERIOD_START_DT'])['SALESTGT_QTY_R'] \
                              .mean().reset_index().rename(columns={'SALESTGT_QTY_R': 'VALUE'})
    # Объединяем исторические данные и прогноз (соединение не по ключам, а последовательное объединение)
    combined = pd.concat([demand_weekly, forecast_weekly], ignore_index=True)
    if combined.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    # 2. Рассчитываем разницу между текущей и предыдущей неделей для каждой комбинации (first difference):contentReference[oaicite:17]{index=17}:contentReference[oaicite:18]{index=18} 
    combined.sort_values('PERIOD_START_DT', inplace=True)
    combined['DIFF'] = combined.groupby([f'PRODUCT_LVL_ID{prod_lvl}', f'LOCATION_LVL_ID{loc_lvl}'])['VALUE'].diff()
    diff_df = combined.dropna(subset=['DIFF']).copy()
    if diff_df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    # 3. Рассчитываем среднее абсолютное изменение спроса за последние 12 недель истории (STAT_DEN_VAL):contentReference[oaicite:19]{index=19}
    hist_end = conf.get('IB_HIST_END_DT')
    if hist_end is not None:
        # Берём различия, у которых период <= конец истории и период >= (конец истории - 12 недель)
        hist_diff = diff_df[(diff_df['PERIOD_START_DT'] <= hist_end) & 
                             (diff_df['PERIOD_START_DT'] > (hist_end - pd.Timedelta(weeks=12)))]
    else:
        hist_diff = diff_df
    avg_diff = hist_diff.groupby([f'PRODUCT_LVL_ID{prod_lvl}', f'LOCATION_LVL_ID{loc_lvl}'])['DIFF'] \
                        .agg(lambda x: x.abs().mean()).reset_index().rename(columns={'DIFF': 'STAT_DEN_VAL'})
    # 4. Выбираем различия прогноза (после истории) и сравниваем с историческим средним
    future_diff = diff_df[diff_df['PERIOD_START_DT'] > hist_end] if hist_end is not None else diff_df.iloc[0:0]
    if future_diff.empty or avg_diff.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    alerts = future_diff.merge(avg_diff, on=[f'PRODUCT_LVL_ID{prod_lvl}', f'LOCATION_LVL_ID{loc_lvl}'], how='left')
    alerts = alerts[(alerts['STAT_DEN_VAL'].notna()) & (alerts['STAT_DEN_VAL'] > conf['IB_ALERT_MIN_VAL'])]
    if alerts.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    # Вычисляем отношение модуля отклонения прогноза к среднему модулю отклонения спроса
    alerts['ALERT_STAT_VAL'] = alerts['DIFF'].abs() / alerts['STAT_DEN_VAL']
    alerts = alerts[alerts['ALERT_STAT_VAL'] > threshold_val].copy()
    if alerts.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    alerts['ALERT_TYPE'] = 'DEVWK'
    # 5. Формируем выходные поля
    alerts.rename(columns={
        f'PRODUCT_LVL_ID{prod_lvl}': 'PRODUCT_ID',
        f'LOCATION_LVL_ID{loc_lvl}': 'STORE_LOCATION_ID'
    }, inplace=True)
    alerts['product_lvl_id'] = alerts['PRODUCT_ID']; alerts['product_lvl'] = prod_lvl
    alerts['location_lvl_id'] = alerts['STORE_LOCATION_ID']; alerts['location_lvl'] = loc_lvl
    alerts['KPI_NM'] = 'HYBRID_FORECAST_VALUE'
    alerts['INPUT_TABLE'] = 'ACC_AGG_HYBRID_FORECAST'
    alerts['STAT_NOM_NM'] = 'Forecast Deviation'                          # Отклонение прогноза
    alerts['STAT_DEN_NM'] = 'Demand Deviation within last 3 months'       # Отклонение спроса (среднее)
    alerts['STAT_NOM'] = alerts['DIFF'].abs(); alerts['STAT_DEN'] = alerts['STAT_DEN_VAL']
    alerts['ALERT_THRESHOLD'] = threshold_val
    for col in OUTPUT_COLUMNS:
        if col not in alerts.columns:
            alerts[col] = pd.NA
    out = alerts[OUTPUT_COLUMNS]
    return out

def alert7(hybrid_forecast_df, forecast_flag_df, alert_parameters_df, config):
    """
    Алерт 7: Ненулевой прогноз для ассортимента без флага прогнозирования (ZEROFLG).
    Указывает на наличие прогноза там, где согласно флагам FORECAST_FLAG его быть не должно (прогноз вне периода активности):contentReference[oaicite:20]{index=20}:contentReference[oaicite:21]{index=21}.
    """
    conf = parse_config(config)
    params = alert_parameters_df[alert_parameters_df['alert_id'] == 7].iloc[0]
    prod_lvl = int(params['al_product_lvl']); loc_lvl = int(params['al_location_lvl'])
    cust_lvl = params.get('al_customer_lvl'); dist_lvl = params.get('al_distr_channel_lvl')
    tgt_table = params['Input_table_table']; tgt_col = params['Input_column']
    target_type = params.get('tgt_type', '')
    # 1. Агрегируем прогноз как в alert1 (по заданным уровням и неделям)
    df = hybrid_forecast_df.copy()
    df = df[(df['ASSORTMENT_TYPE'] == 'old') & (df['DEMAND_TYPE'] == 'regular')]
    df['PERIOD_START_DT'] = df['PERIOD_DT'].apply(week_start_date)
    group_keys = [f'PRODUCT_LVL_ID{prod_lvl}', f'LOCATION_LVL_ID{loc_lvl}', 'PERIOD_START_DT']
    agg_df = df.groupby(group_keys)[tgt_col].mean().reset_index().rename(columns={tgt_col: 'FORECAST_VALUE'})
    # 2. Генерируем таймлайн активных периодов по FORECAST_FLAG (аналогично alert1)
    ff_active = forecast_flag_df[forecast_flag_df['STATUS'].isin(conf['IB_FF_ACTIVE_STATUS_LIST'])].copy()
    ff_group_keys = [f'PRODUCT_LVL_ID{prod_lvl}', f'LOCATION_LVL_ID{loc_lvl}']
    timeline_df = generate_timeline(ff_active, ff_group_keys, conf, params['al_tim_lvl'])
    # 3. Выполняем **левое** соединение прогноза (a) с флагами (c) и находим случаи, где флага нет (период отсутствует в c):contentReference[oaicite:22]{index=22} 
    merged = agg_df.merge(timeline_df, on=[f'PRODUCT_LVL_ID{prod_lvl}', f'LOCATION_LVL_ID{loc_lvl}', 'PERIOD_START_DT'], how='left', indicator=True)
    alert_rows = merged[merged['_merge'] == 'left_only'].copy()  # строки, присутствующие только в прогнозе (нет соответствующего флага)
    # Оставляем только те, где прогнозное значение существенно > 0 (чтобы отсечь тривиальные нули)
    alert_rows = alert_rows[alert_rows['FORECAST_VALUE'] > conf['IB_ALERT_MIN_VAL']]
    if alert_rows.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    alert_rows['ALERT_TYPE'] = 'ZEROFLG'
    # 4. Формируем итоговую таблицу (аналогично alert1)
    out = alert_rows.copy()
    out.rename(columns={
        f'PRODUCT_LVL_ID{prod_lvl}': 'PRODUCT_ID',
        f'LOCATION_LVL_ID{loc_lvl}': 'STORE_LOCATION_ID'
    }, inplace=True)
    out['product_lvl_id'] = out['PRODUCT_ID']; out['product_lvl'] = prod_lvl
    out['location_lvl_id'] = out['STORE_LOCATION_ID']; out['location_lvl'] = loc_lvl
    # KPI_NM и INPUT_TABLE
    kpi_nm = f"{target_type}.{tgt_col}" if target_type else tgt_col
    out['KPI_NM'] = kpi_nm
    input_table_name = str(tgt_table)
    if target_type and input_table_name.endswith(f"_{target_type}"):
        input_table_name = input_table_name[:-(len(target_type) + 1)]
    out['INPUT_TABLE'] = input_table_name
    out['STAT_NOM_NM'] = 'Forecast value'; out['STAT_DEN_NM'] = 'na'
    out['STAT_NOM'] = out['FORECAST_VALUE']; out['STAT_DEN'] = 1
    out['ALERT_THRESHOLD'] = pd.NA
    out['ALERT_STAT_VAL'] = out['FORECAST_VALUE']
    for col in OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[OUTPUT_COLUMNS]
    return out

def alert8(hybrid_forecast_df, restored_demand_df, forecast_flag_df, product_df, alert_parameters_df, config):
    """
    Алерт 8: Аномальная доля прогноза нового ассортимента (SHRNEW).
    Если доля прогноза нового продукта значительно отличается от среднего по группе.
    """
    conf = parse_config(config)
    params = alert_parameters_df[alert_parameters_df['alert_id'] == 8].iloc[0]
    prod_lvl = int(params['al_product_lvl'])
    loc_lvl = int(params['al_location_lvl'])
    agg_lvl = int(params.get('al_product_agg_lvl', prod_lvl - 1))
    threshold_val = float(params['alert_threshold_val'])
    target_type = params.get('tgt_type', '')

    # Определяем имена колонок уровней
    prod_col = f'PRODUCT_LVL_ID{prod_lvl}'
    parent_col = f'PRODUCT_LVL_ID{agg_lvl}'
    loc_col = f'LOCATION_LVL_ID{loc_lvl}'
    date_col = 'PERIOD_START_DT'

    # --- подготовка мэппинга продукт→родитель ---
    try:
        mapping = product_df[[prod_col, parent_col]].drop_duplicates()
    except KeyError:
        # fallback: если нет product_df или нет нужных колонок, 
        # считаем, что родитель = сам продукт
        unique_ids = hybrid_forecast_df[prod_col].unique()
        mapping = pd.DataFrame({
            prod_col: unique_ids,
            parent_col: unique_ids
        })

    # 1) получаем только новый ассортимент
    df_new = hybrid_forecast_df.copy()
    df_new = df_new[df_new['ASSORTMENT_TYPE'] == 'new']
    df_new['PERIOD_START_DT'] = df_new['PERIOD_DT'].apply(week_start_date)
    # мержим, чтобы получить parent_col
    df_new = df_new.merge(mapping, on=prod_col, how='left')

    # 2) агрегируем прогноз по продукту и его родительской группе
    keys_new = [prod_col, parent_col, loc_col, 'PERIOD_START_DT']
    forecast_new = (
        df_new
        .groupby(keys_new)['HYBRID_FORECAST_VALUE']
        .mean()
        .reset_index()
        .rename(columns={'HYBRID_FORECAST_VALUE': 'STAT_NOM_VAL'})
    )

    # 3) агрегируем восстановленный спрос по группе (исключая новые)
    demand = restored_demand_df.copy()
    demand['PERIOD_START_DT'] = demand['PERIOD_DT'].apply(week_start_date)
    # присоединяем mapping и флаг новых
    demand = demand.merge(mapping, on=prod_col, how='left')
    ff_active = forecast_flag_df[forecast_flag_df['STATUS'].isin(conf['IB_FF_ACTIVE_STATUS_LIST'])]
    ff_min = (
        ff_active
        .groupby([prod_col, loc_col])
        .agg(MIN_START_DT=('PERIOD_START_DT','min'))
        .reset_index()
    )
    hist_end = conf.get('IB_HIST_END_DT')
    max_new = conf.get('IB_MAX_NP_HISTORY', 0)
    ff_min['IS_NEW'] = ff_min['MIN_START_DT'].apply(
        lambda d: (hist_end - pd.to_datetime(d)).days <= max_new if hist_end and pd.notna(d) else False
    )
    demand = demand.merge(ff_min[[prod_col,'IS_NEW']], on=prod_col, how='left')
    demand_reg = demand[demand['IS_NEW'] == False]
    # усредняем спрос по группе
    keys_group = [parent_col, loc_col, 'PERIOD_START_DT']
    demand_reg_agg = (
        demand_reg
        .groupby(keys_group)['SALESTGT_QTY_R']
        .agg(['mean','count'])
        .reset_index()
        .rename(columns={'mean':'STAT_DEN_VAL','count':'STAT_COUNT'})
    )

    # 4) соединяем прогноз нового продукта с усреднённым спросом группы
    alerts = forecast_new.merge(demand_reg_agg, on=[parent_col, loc_col, 'PERIOD_START_DT'], how='left')
    alerts = alerts[
        alerts['STAT_DEN_VAL'].notna() &
        (alerts['STAT_DEN_VAL'] > conf['IB_ALERT_MIN_VAL']) &
        (alerts['STAT_COUNT'] >= conf['IB_ALERT_MIN_OBS'])
    ].copy()
    if alerts.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    # 5) считаем отношение и отбираем отклонения
    alerts['ALERT_STAT_VAL'] = alerts['STAT_NOM_VAL'] / alerts['STAT_DEN_VAL']
    alerts = alerts[
        (alerts['ALERT_STAT_VAL'] > threshold_val) | 
        (alerts['ALERT_STAT_VAL'] < 1/threshold_val)
    ].copy()
    if alerts.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    alerts['ALERT_TYPE'] = 'SHRNEW'

    # 6) формируем окончательный результат — переименование и доп. поля
    rename_map = {
        prod_col: 'PRODUCT_ID',
        loc_col: 'STORE_LOCATION_ID'
    }
    alerts = alerts.rename(columns=rename_map)
    alerts['product_lvl_id'] = alerts['PRODUCT_ID']
    alerts['product_lvl'] = prod_lvl
    alerts['location_lvl_id'] = alerts['STORE_LOCATION_ID']
    alerts['location_lvl'] = loc_lvl

    alerts['KPI_NM'] = f"{target_type}.HYBRID_FORECAST_VALUE" if target_type else 'HYBRID_FORECAST_VALUE'
    alerts['INPUT_TABLE'] = 'ACC_AGG_HYBRID_FORECAST'
    alerts['STAT_NOM_NM'] = 'Forecast for New Assortment'
    alerts['STAT_DEN_NM'] = 'Average Forecast for Regular Assortment'
    alerts['STAT_NOM'] = alerts['STAT_NOM_VAL']
    alerts['STAT_DEN'] = alerts['STAT_DEN_VAL']
    alerts['ALERT_THRESHOLD'] = threshold_val

    # ensure OUTPUT_COLUMNS order
    for col in OUTPUT_COLUMNS:
        if col not in alerts.columns:
            alerts[col] = pd.NA
    return alerts[OUTPUT_COLUMNS]


def alert9(hybrid_forecast_df, forecast_flag_df, alert_parameters_df, config):
    """
    Алерт 9: Аномально низкий суммарный прогноз для нового ассортимента (LOWNEW).
    Если суммарный прогноз по новому продукту на весь горизонт прогнозирования < порогового значения (например, <1 штуки в год), 
    формируется алерт LOWNEW:contentReference[oaicite:31]{index=31}:contentReference[oaicite:32]{index=32}.
    """
    conf = parse_config(config)
    params = alert_parameters_df[alert_parameters_df['alert_id'] == 9].iloc[0]
    prod_lvl = int(params['al_product_lvl']); loc_lvl = int(params['al_location_lvl'])
    threshold_val = float(params['alert_threshold_val'])
    # 1. Суммарный прогноз (по всему горизонту) для каждого нового товара
    df_new = hybrid_forecast_df.copy()
    df_new = df_new[df_new['ASSORTMENT_TYPE'] == 'new']
    total_forecast = df_new.groupby([f'PRODUCT_LVL_ID{prod_lvl}', f'LOCATION_LVL_ID{loc_lvl}'])['HYBRID_FORECAST_VALUE'] \
                           .sum().reset_index().rename(columns={'HYBRID_FORECAST_VALUE': 'STAT_NOM_VAL'})
    total_forecast['STAT_COUNT'] = df_new.groupby([f'PRODUCT_LVL_ID{prod_lvl}', f'LOCATION_LVL_ID{loc_lvl}'])['HYBRID_FORECAST_VALUE'].count().values
    # 2. Определяем, какие из этих продуктов действительно "новые" (по FORECAST_FLAG) и достаточность данных
    ff_active = forecast_flag_df[forecast_flag_df['STATUS'].isin(conf['IB_FF_ACTIVE_STATUS_LIST'])].copy()
    ff_min = ff_active.groupby([f'PRODUCT_LVL_ID{prod_lvl}', f'LOCATION_LVL_ID{loc_lvl}']) \
                     .agg(MIN_START_DT=('PERIOD_START_DT', 'min')).reset_index()
    hist_end = conf.get('IB_HIST_END_DT'); max_new_days = conf.get('IB_MAX_NP_HISTORY')
    ff_min['IS_NEW'] = ff_min['MIN_START_DT'].apply(lambda d: (hist_end - pd.to_datetime(d)).days <= max_new_days if hist_end and pd.notna(d) else False)
    alerts = total_forecast.merge(ff_min, on=[f'PRODUCT_LVL_ID{prod_lvl}', f'LOCATION_LVL_ID{loc_lvl}'], how='left')
    alerts = alerts[alerts['IS_NEW'] == True].copy()
    alerts = alerts[alerts['STAT_COUNT'] >= conf['IB_ALERT_MIN_OBS']]  # минимально необходимое кол-во периодов прогноза
    alerts = alerts[alerts['STAT_NOM_VAL'] < threshold_val].copy()
    if alerts.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    alerts['ALERT_TYPE'] = 'LOWNEW'
    # 3. Оформляем итоговую таблицу
    alerts.rename(columns={
        f'PRODUCT_LVL_ID{prod_lvl}': 'PRODUCT_ID',
        f'LOCATION_LVL_ID{loc_lvl}': 'STORE_LOCATION_ID'
    }, inplace=True)
    alerts['product_lvl_id'] = alerts['PRODUCT_ID']; alerts['product_lvl'] = prod_lvl
    alerts['location_lvl_id'] = alerts['STORE_LOCATION_ID']; alerts['location_lvl'] = loc_lvl
    alerts['KPI_NM'] = 'HYBRID_FORECAST_VALUE'
    alerts['INPUT_TABLE'] = 'ACC_AGG_HYBRID_FORECAST'
    alerts['STAT_NOM_NM'] = 'Total forecast value on the whole forecasting period'  # Суммарный объём прогноза на всём горизонте
    alerts['STAT_DEN_NM'] = 'na'
    alerts['STAT_NOM'] = alerts['STAT_NOM_VAL']; alerts['STAT_DEN'] = 1
    alerts['ALERT_THRESHOLD'] = threshold_val
    alerts['ALERT_STAT_VAL'] = alerts['STAT_NOM_VAL']  # Стат. значение алерта = суммарный прогноз (тк знаменатель = 1)
    for col in OUTPUT_COLUMNS:
        if col not in alerts.columns:
            alerts[col] = pd.NA
    out = alerts[OUTPUT_COLUMNS]
    return out

def alert10(vf_forecast_df, vf_segments_df, forecast_flag_df, alert_parameters_df, config):
    """
    Алерт 10: Неприменение сезонной модели при длинном ряде данных (NONSEAS).
    Выявляет случаи, когда для временного ряда с историей >2 лет была выбрана несезонная модель (напр. ETS/ARIMA без сезонности) 
    во время статистического прогнозирования VF:contentReference[oaicite:33]{index=33}:contentReference[oaicite:34]{index=34}. Applicable only for VF forecast.
    """
    conf = parse_config(config)
    params = alert_parameters_df[alert_parameters_df['alert_id'] == 10].iloc[0]
    vf_prod_lvl = int(params['al_product_lvl']); vf_loc_lvl = int(params['al_location_lvl'])
    threshold_years = float(params['alert_threshold_val'])  # например, 2 (года)
    # 1. Объединяем данные статистического прогноза VF с сегментами VF_SEGMENTS для получения названия сегмента модели
    vf = vf_forecast_df.copy()
    vf = vf.merge(vf_segments_df, on=[f'PRODUCT_LVL_ID{vf_prod_lvl}', f'LOCATION_LVL_ID{vf_loc_lvl}'], how='left')
    # 2. Ограничиваем FORECAST_FLAG комбинациями на уровне VF (например, уровень продукта = IB_VF_PRODUCT_LVL), исключая более низкие уровни
    ff_active = forecast_flag_df[forecast_flag_df['STATUS'].isin(conf['IB_FF_ACTIVE_STATUS_LIST'])].copy()
    ff_vf = ff_active.copy()
    next_level_col = f'PRODUCT_LVL_ID{vf_prod_lvl + 1}'
    if next_level_col in ff_vf.columns:
        # Если есть колонка для уровня ниже, оставляем только записи без значения на уровне ниже (т.е. агрегированный уровень)
        ff_vf = ff_vf[ff_vf[next_level_col].isna()]
    # Вычисляем дату начала прогноза (MIN_START_DT) для каждой комбинации VF
    ff_min = ff_vf.groupby([f'PRODUCT_LVL_ID{vf_prod_lvl}', f'LOCATION_LVL_ID{vf_loc_lvl}']) \
                  .agg(MIN_START_DT=('PERIOD_START_DT', 'min')).reset_index()
    # 3. Отбираем комбинации с историей более 2 лет: IB_HIST_END_DT - MIN_START_DT >= 2 * 365 дней:contentReference[oaicite:35]{index=35}
    hist_end = conf.get('IB_HIST_END_DT')
    ff_min['LONG_HISTORY'] = ff_min['MIN_START_DT'].apply(lambda d: (hist_end - pd.to_datetime(d)).days >= threshold_years * 365 if hist_end and pd.notna(d) else False)
    # 4. Объединяем с сегментами и отбираем случаи, где история >2 лет и сегмент несезонный (имя сегмента не содержит "SEASON"):contentReference[oaicite:36]{index=36}:contentReference[oaicite:37]{index=37}
    alerts = vf.merge(ff_min, on=[f'PRODUCT_LVL_ID{vf_prod_lvl}', f'LOCATION_LVL_ID{vf_loc_lvl}'], how='left')
    alerts = alerts[(alerts['LONG_HISTORY'] == True)]
    if alerts.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    alerts = alerts[alerts['SEGMENT_NAME'].str.contains('SEASON', case=False, na=False) == False].copy()
    if alerts.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    alerts['ALERT_TYPE'] = 'NONSEAS'
    # 5. Заполняем требуемые выходные поля
    alerts.rename(columns={
        f'PRODUCT_LVL_ID{vf_prod_lvl}': 'PRODUCT_ID',
        f'LOCATION_LVL_ID{vf_loc_lvl}': 'STORE_LOCATION_ID'
    }, inplace=True)
    alerts['product_lvl_id'] = alerts['PRODUCT_ID']; alerts['product_lvl'] = vf_prod_lvl
    alerts['location_lvl_id'] = alerts['STORE_LOCATION_ID']; alerts['location_lvl'] = vf_loc_lvl
    alerts['KPI_NM'] = params['Input_column']               # например, 'FORECAST_VALUE'
    alerts['INPUT_TABLE'] = str(params['Input_table_table']) # например, 'ACC_VF_FORECAST'
    alerts['STAT_NOM_NM'] = 'Time Series Segment'; alerts['STAT_DEN_NM'] = 'na'
    alerts['STAT_NOM'] = pd.NA; alerts['STAT_DEN'] = 1
    alerts['ALERT_THRESHOLD'] = threshold_years
    alerts['ALERT_STAT_VAL'] = pd.NA  # нет числового значения для данного алерта, используем флаговый признак сегмента
    for col in OUTPUT_COLUMNS:
        if col not in alerts.columns:
            alerts[col] = pd.NA
    out = alerts[OUTPUT_COLUMNS]
    return out
