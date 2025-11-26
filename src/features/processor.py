import pandas as pd
import numpy as np
from pathlib import Path

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

def encode_cyclical_time(df, col, max_val):
    """
    Encodes a time column into sin/cos coordinates.
    Example: Hour 23 becomes close to Hour 0 in vector space.
    """
    df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / max_val)
    df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / max_val)
    return df

def generate_features(meter_type='electricity'):
    input_path = PROCESSED_DATA_DIR / f"{meter_type}_merged.parquet"
    if not input_path.exists():
        print(f"Error: {input_path} not found. Run ETL first.")
        return

    print(f"Loading {meter_type} data...")
    df = pd.read_parquet(input_path)

    # 1. Basic Time Features
    print("Generating Time Features...")
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # 2. Cyclical Encoding (The "Pro" Move)
    print("Encoding Cyclical Time...")
    df = encode_cyclical_time(df, 'hour', 24)
    df = encode_cyclical_time(df, 'month', 12)
    df = encode_cyclical_time(df, 'day_of_week', 7)

    # 3. Lag Features (History)
    # The model needs context: "What was usage 24 hours ago?"
    # NOTE: We must group by building_id so we don't shift data from Building A into Building B
    print("Generating Lag Features (Past History)...")
    
    # Sort to ensure shifts are correct
    df = df.sort_values(['building_id', 'timestamp'])
    
    # Lag 1: What was usage 1 hour ago?
    df['lag_1h'] = df.groupby('building_id')['meter_reading'].shift(1)
    
    # Lag 24: What was usage exactly 1 day ago? (Strongest predictor)
    df['lag_24h'] = df.groupby('building_id')['meter_reading'].shift(24)

    # 4. Rolling Statistics
    # "Average usage over the last 6 hours"
    print("Generating Rolling Statistics...")
    df['rolling_mean_6h'] = df.groupby('building_id')['meter_reading'].transform(
        lambda x: x.rolling(window=6).mean()
    )

    # 5. Clean NaNs created by lagging
    # The first 24 hours of data for every building will now be NaN. We drop them.
    df = df.dropna()

    # Save
    output_path = PROCESSED_DATA_DIR / f"{meter_type}_features.parquet"
    print(f"Saving feature-rich dataset to {output_path}...")
    df.to_parquet(output_path)
    print("Success! Feature Engineering Complete.")

if __name__ == "__main__":
    generate_features('electricity')