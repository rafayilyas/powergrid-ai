import pandas as pd
import numpy as np
from pathlib import Path


def generate(path: str = 'data/raw/household_power_consumption.txt', periods: int = 24*30):
    Path('data/raw').mkdir(parents=True, exist_ok=True)
    # Create hourly timestamps for one month
    rng = pd.date_range(start='2020-01-01', periods=periods, freq='H')
    hour = rng.hour
    dayofweek = rng.dayofweek
    # Generate synthetic demand with daily and weekly seasonality
    base = 1 + 0.5 * np.sin(2 * np.pi * hour / 24)
    weekly = 0.2 * (dayofweek >= 5)
    noise = 0.1 * np.random.randn(periods)
    demand = 2 + base + weekly + noise
    voltage = 230 + 5 * np.sin(2 * np.pi * hour / 24)
    temperature = 20 + 7 * np.sin(2 * np.pi * (hour-14) / 24)
    df = pd.DataFrame({
        'Date': rng.strftime('%d/%m/%Y'),
        'Time': rng.strftime('%H:%M:%S'),
        'Global_active_power': demand,
        'Global_reactive_power': 0.1 + 0.01 * np.random.randn(periods),
        'Voltage': voltage,
        'Global_intensity': demand / 230 * 100,
        'Sub_metering_1': np.random.randint(0, 2, size=periods),
        'Sub_metering_2': np.random.randint(0, 2, size=periods),
        'Sub_metering_3': np.random.randint(0, 2, size=periods),
    })
    # Introduce some missing values as '?'
    mask = np.random.rand(len(df)) < 0.02
    df.loc[mask, 'Global_active_power'] = np.nan
    # Write with semicolon separator and NA represented as '?'
    df.to_csv(path, sep=';', index=False, na_rep='?')
    return path

if __name__ == '__main__':
    print('Generating sample data...')
    print(generate())
