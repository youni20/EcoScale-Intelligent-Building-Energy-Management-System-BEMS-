import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "src" / "models"
OUTPUT_DIR = BASE_DIR / "data" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Average commercial electricity rate ($/kWh)
COST_PER_KWH = 0.14 

def detect_anomalies(meter_type='electricity'):
    print(f"--- Running Anomaly Detection for {meter_type} ---")
    
    # 1. Load Data and Model
    data_path = DATA_DIR / f"{meter_type}_features.parquet"
    model_path = MODELS_DIR / f"lgbm_{meter_type}.joblib"
    
    if not data_path.exists() or not model_path.exists():
        print("Error: Missing data or model.")
        return

    print("Loading data...")
    df = pd.read_parquet(data_path)
    
    print("Loading model...")
    model = joblib.load(model_path)

    # 2. Prepare Features (Convert categories again just to be safe)
    # The model expects the exact same columns as training
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')

    # Ensure we only use the features the model was trained on
    # (LightGBM is picky about column order, so we filter by the model's feature names)
    model_features = model.feature_name_
    X = df[model_features]

    # 3. Predict 'Expected' Usage
    print("Generating predictions (Virtual Meter)...")
    df['expected_reading'] = model.predict(X)
    
    # 4. Calculate Anomalies
    # Anomaly = Actual Reading is significantly higher than Expected
    # We use a dynamic threshold: 2 Standard Deviations + 10% buffer
    
    df['deviation'] = df['meter_reading'] - df['expected_reading']
    
    # Calculate a dynamic threshold per building to be fair to small vs big buildings
    # We calculate the standard deviation of the ERROR (residuals) per building
    print("Calculating dynamic thresholds...")
    df['std_error'] = df.groupby('building_id')['deviation'].transform('std')
    
    # Logic: If usage is > Expected + (2 * StdDev), it's an anomaly
    df['is_anomaly'] = df['deviation'] > (2 * df['std_error'])
    
    # Filter for only positive anomalies (Waste) - we don't care if they saved energy for now
    anomalies = df[df['is_anomaly'] & (df['deviation'] > 0)].copy()
    
    # 5. Calculate Financial Impact
    anomalies['wasted_kwh'] = anomalies['deviation']
    anomalies['wasted_cost'] = anomalies['wasted_kwh'] * COST_PER_KWH
    
    # 6. Save Report
    output_file = OUTPUT_DIR / f"{meter_type}_anomalies.csv"
    
    # Save a lightweight version for the dashboard (top 5000 anomalies to keep it fast)
    # Or save all of them if you want. Let's save the top most expensive ones.
    top_anomalies = anomalies.sort_values('wasted_cost', ascending=False).head(5000)
    
    cols_to_save = ['timestamp', 'building_id', 'meter_reading', 'expected_reading', 'wasted_kwh', 'wasted_cost']
    top_anomalies[cols_to_save].to_csv(output_file, index=False)
    
    print(f"\nAnalysis Complete!")
    print(f"Total Anomalies Detected: {len(anomalies)}")
    print(f"Total Estimated Waste Cost: ${anomalies['wasted_cost'].sum():,.2f}")
    print(f"Top 5,000 most expensive anomalies saved to: {output_file}")

if __name__ == "__main__":
    detect_anomalies('electricity')