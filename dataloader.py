import pandas as pd
import numpy as np

def load_and_prepare_data(energy_type='electricity'):
    # 1. Load Metadata to get site_id for each building
    meta = pd.read_csv('datasets/metadata.csv')
    
    # 2. Load Energy Data
    energy_df = pd.read_csv(f'datasets/{energy_type}_cleaned.csv')
    
    # 3. Load Weather
    weather = pd.read_csv('datasets/weather.csv')
    weather['timestamp'] = pd.to_datetime(weather['timestamp'])
    
    # 4. Reshape Energy Data (It's currently wide format, we need long format)
    # The BDG2 dataset usually has buildings as columns. We need to "melt" it.
    energy_melted = energy_df.melt(id_vars='timestamp', var_name='building_id', value_name='meter_reading')
    energy_melted['timestamp'] = pd.to_datetime(energy_melted['timestamp'])

    # 5. Merge Strategy
    # Merge metadata to get site_id
    merged = energy_melted.merge(meta[['building_id', 'site_id', 'primary_use', 'square_feet']], on='building_id', how='left')
    
    # Merge weather based on site_id and timestamp
    final_df = merged.merge(weather, on=['site_id', 'timestamp'], how='left')
    
    # 6. Optimization: Save as Parquet
    final_df.to_parquet(f'datasets/processed_{energy_type}.parquet')
    return final_df