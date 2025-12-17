import pandas as pd
from pathlib import Path

def preprocess(raw_path: str = 'data/raw/household_power_consumption.txt', out_path: str = 'data/processed/processed.csv'):
    p = Path(raw_path)
    if not p.exists():
        raise FileNotFoundError(f"Raw data not found at {raw_path}")

    print("Loading and parsing data (this may take a moment)...")
    
    # OPTIMIZATION: Combine Date and Time *during* the read operation.
    # parse_dates={'datetime': ['Date', 'Time']} creates a 'datetime' column automatically.
    # index_col='datetime' sets it as index immediately (optional, but saves memory).
    if p.suffix.lower() in ['.txt', '.csv']:
        df = pd.read_csv(
            raw_path, 
            sep=';', 
            na_values=['?', ''], 
            low_memory=False,
            parse_dates={'datetime': ['Date', 'Time']}, # Combines columns automatically
            dayfirst=True, # vital for dd/mm/yyyy format
            infer_datetime_format=True # Speeds up parsing
        )
    else:
        df = pd.read_csv(raw_path)

    # If datetime parsing failed (NaT), drop those rows
    df = df.dropna(subset=['datetime'])
    
    # Sort just in case
    df = df.sort_values('datetime')

    # Clean Target Column ('Global_active_power')
    if 'Global_active_power' not in df.columns:
        # Fallback if names are different
        df['Global_active_power'] = df.iloc[:, 0] # 0 because Date/Time are now the index/datetime col
    
    df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
    df = df.dropna(subset=['Global_active_power'])

    # Feature Engineering
    # We access the .dt accessor on the datetime column
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek

    # Synthetic Temperature
    if 'temperature' not in df.columns:
        df['temperature'] = 25 + 5 * (df['dayofweek'] >= 5) + 3 * ((df['hour'] >= 14) & (df['hour'] <= 18))

    # Clean/Synthetic Voltage
    if 'Voltage' in df.columns:
        df['voltage'] = pd.to_numeric(df['Voltage'], errors='coerce')
    elif 'voltage' not in df.columns:
        df['voltage'] = 230 + 10 * (df['hour'] >= 17)

    # Aggregate to Hourly Data
    print("Aggregating to hourly data...")
    df['ts_hour'] = df['datetime'].dt.floor('h')

    grouped = df.groupby(['ts_hour'], as_index=False).agg(
        demand=('Global_active_power', 'mean'),
        temperature=('temperature', 'mean'),
        voltage=('voltage', 'mean')
    )

    grouped = grouped.rename(columns={'ts_hour': 'datetime'})

    # Save
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(out_path, index=False)
    
    print(f"Done! Processed data saved to {out_path}")
    return out_path

if __name__ == '__main__':
    try:
        preprocess()
    except Exception as e:
        print(f"Error: {e}")