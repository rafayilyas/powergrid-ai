import pandas as pd
from pathlib import Path


def preprocess(raw_path: str = 'data/raw/household_power_consumption.txt', out_path: str = 'data/processed/processed.csv'):
    p = Path(raw_path)
    if not p.exists():
        raise FileNotFoundError(f"Raw data not found at {raw_path}")
    # The UCI 'Individual household electric power consumption' dataset uses ';' and '?' as NaN markers
    if p.suffix.lower() in ['.txt', '.csv']:
        df = pd.read_csv(raw_path, sep=';', na_values=['?', ''], low_memory=False)
    else:
        df = pd.read_csv(raw_path)
    # Try to parse datetime
    if 'Date' in df.columns and 'Time' in df.columns:
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True, errors='coerce')
        # keep original Date and Time if needed, drop afterwards
        df = df.drop(columns=['Date', 'Time'])
    elif 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    else:
        # Expect 'timestamp' or 'datetime'
        df['datetime'] = pd.to_datetime(df.iloc[:, 0], errors='coerce')

    df = df.dropna(subset=['datetime'])
    df = df.sort_values('datetime')
    # Aggregate to hourly mean if multiple rows per hour
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    # The UCI dataset has 'Global_active_power' as the active power consumed in kilowatts
    if 'Global_active_power' not in df.columns:
        # Use a fallback column if missing
        df['Global_active_power'] = df.iloc[:, 1]
    # Clean the Global_active_power column and convert to numeric
    df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
    # Drop rows where the active power is missing, since it is our target
    df = df.dropna(subset=['Global_active_power'])
    # Create a temperature column if missing (synthetic)
    if 'temperature' not in df.columns:
        df['temperature'] = 25 + 5 * (df['dayofweek'] >= 5) + 3 * ((df['hour'] >= 14) & (df['hour'] <= 18))
    # Voltage column in UCI is 'Voltage'
    if 'Voltage' in df.columns and 'voltage' not in df.columns:
        df['voltage'] = pd.to_numeric(df['Voltage'], errors='coerce')
    if 'voltage' not in df.columns:
        df['voltage'] = 230 + 10 * (df['hour'] >= 17)

    # Build hour-level time series; aggregate by hour through datetime
    grouped = df.groupby(['datetime', 'hour', 'dayofweek'], as_index=False).agg(
        demand=('Global_active_power', 'mean'),
        temperature=('temperature', 'mean'),
        voltage=('voltage', 'mean')
    )
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(out_path, index=False)
    return out_path


if __name__ == '__main__':
    print('Preprocessing...')
    print(preprocess())
