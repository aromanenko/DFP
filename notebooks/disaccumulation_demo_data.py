# disaccumulation_demo_data.py
# Генерация синтетического прогноза для демонстрации дизаккумуляции.

import os
import numpy as np
import pandas as pd


def generate_demo_forecast(data_dir,
                           n_products=30,
                           n_locations=10,
                           start='2024-01-01',
                           months=12,
                           seed=42,
                           filename='AGG_HYB_FCST_DEMO.csv'):
    """
    Generate synthetic monthly forecast for several products and locations.

    :param data_dir: Path to data folder (string or Path).
    :param n_products: Number of products.
    :param n_locations: Number of locations.
    :param start: Start date for the first month (YYYY-MM-DD).
    :param months: Number of monthly periods to generate.
    :param seed: Random seed for reproducibility.
    :param filename: Name of forecast CSV file to create.
    :return: Generated DataFrame.
    """
    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    dates = pd.date_range(start=start, periods=months, freq='MS')

    rows = []
    for p in range(1, n_products + 1):
        for loc in range(1, n_locations + 1):
            base = rng.uniform(50, 500)
            for i, d in enumerate(dates):
                period_start = d.normalize()
                # last day of month
                period_end = (d + pd.offsets.MonthEnd(0)).normalize()

                # simple seasonality
                season_factor = 1.0 + 0.5 * np.sin(2.0 * np.pi * i / 12.0)

                vf = base * season_factor
                ml = vf * (1.0 + rng.normal(0.0, 0.05))
                hyb = 0.5 * vf + 0.5 * ml

                rows.append({
                    'PRODUCT_ID': p,
                    'LOCATION_ID': loc,
                    'PERIOD_DT': period_start,
                    'PERIOD_END_DT': period_end,
                    'VF_FORECAST_VALUE': vf,
                    'ML_FORECAST_VALUE': ml,
                    'HYBRID_FORECAST_VALUE': hyb,
                })

    df = pd.DataFrame(rows)
    out_path = os.path.join(data_dir, filename)
    df.to_csv(out_path, index=False)
    return df


if __name__ == '__main__':
    # Simple manual run for quick check
    generate_demo_forecast('../data')
