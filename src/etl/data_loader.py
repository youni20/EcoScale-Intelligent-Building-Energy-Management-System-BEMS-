import pandas as pd
import numpy as np
import os
from pathlib import Path

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

def load_metadata() -> pd.DataFrame:
    """Loads and cleans the building metadata with defensive coding."""
    print("Loading Metadata...")
    meta_path = RAW_DATA_DIR / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found at {meta_path}")
        
    meta = pd.read_csv(meta_path)
    
    # --- SENIOR DEV FIX: Normalize & Deduplicate ---
    # 1. Normalize names
    column_mapping = {
        'primaryspaceusage': 'primary_use',
        'sqm': 'square_feet',
        'sqft': 'square_feet',
        'yearbuilt': 'year_built'
    }
    meta = meta.rename(columns=column_mapping)
    
    # 2. DROP DUPLICATES (The Fix): If both 'sqm' and 'sqft' existed, 
    # we now have two 'square_feet' columns. We keep the first one.
    meta = meta.loc[:, ~meta.columns.duplicated()]
    
    # 3. Filter for desired columns only
    desired_cols = ['building_id', 'site_id', 'primary_use', 'square_feet', 'year_built']
    existing_cols = [c for c in desired_cols if c in meta.columns]
    
    return meta[existing_cols]

def load_weather() -> pd.DataFrame:
    """Loads weather data and fixes timestamps."""
    print("Loading Weather Data...")
    weather_path = RAW_DATA_DIR / "weather.csv"
    if not weather_path.exists():
        raise FileNotFoundError(f"Weather data not found at {weather_path}")

    weather = pd.read_csv(weather_path)
    weather['timestamp'] = pd.to_datetime(weather['timestamp'])
    
    # --- SENIOR DEV FIX: Clean CamelCase Columns ---
    # The error log showed these specific column names exist and should be dropped to save space
    cols_to_drop = [
        'cloudCoverage', 'precipDepth1HR', 'precipDepth6HR', 
        'seaLvlPressure', 'windDirection', 'windSpeed'
    ]
    weather = weather.drop(columns=cols_to_drop, errors='ignore')
    
    # Interpolate temperature gaps (Linear fill)
    if 'airTemperature' in weather.columns:
        weather['airTemperature'] = weather.groupby('site_id')['airTemperature'].transform(
            lambda x: x.interpolate(method='linear').ffill().bfill()
        )
    return weather

def process_meter_data(meter_type: str = 'electricity'):
    """
    ETL Pipeline:
    1. Load wide-format meter data.
    2. Melt to long-format.
    3. Merge with Metadata.
    4. Merge with Weather.
    5. Save as Parquet.
    """
    print(f"\n--- Processing {meter_type} Data ---")
    
    # 1. Load Meter Data
    file_path = RAW_DATA_DIR / f"{meter_type}_cleaned.csv"
    if not file_path.exists():
        print(f"Error: {file_path} not found. Skipping.")
        return

    print(f"Reading {file_path.name}...")
    df = pd.read_csv(file_path)
    
    # 2. Reshape (Wide to Long)
    print("Reshaping data (Melt)...")
    df_long = df.melt(id_vars=['timestamp'], var_name='building_id', value_name='meter_reading')
    
    # Clean up memory
    df_long = df_long.dropna(subset=['meter_reading'])
    df_long['timestamp'] = pd.to_datetime(df_long['timestamp'])

    # 3. Merge Metadata
    meta = load_metadata()
    print("Merging Metadata...")
    merged_df = df_long.merge(meta, on='building_id', how='left')

    # 4. Merge Weather
    weather = load_weather()
    print("Merging Weather...")
    # Merge on site_id AND timestamp
    final_df = merged_df.merge(weather, on=['site_id', 'timestamp'], how='left')

    # 5. Save to Parquet
    output_path = PROCESSED_DATA_DIR / f"{meter_type}_merged.parquet"
    print(f"Saving to {output_path}...")
    final_df.to_parquet(output_path, engine='pyarrow', index=False)
    
    print(f"Success! Saved {len(final_df)} rows to {output_path}")
    return final_df

if __name__ == "__main__":
    process_meter_data('electricity')