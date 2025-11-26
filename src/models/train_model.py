import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "src" / "models"
MODELS_DIR.mkdir(exist_ok=True)

def train_model(meter_type='electricity'):
    input_path = PROCESSED_DATA_DIR / f"{meter_type}_features.parquet"
    if not input_path.exists():
        print("Error: Data not found. Run feature generation first.")
        return

    print(f"Loading {meter_type} data for training...")
    df = pd.read_parquet(input_path)

    # --- SENIOR DEV FIX: Handle Categorical Data ---
    # LightGBM crashes on strings ('object'). We must convert them to 'category'.
    for col in df.columns:
        if df[col].dtype == 'object':
            print(f"Converting column '{col}' to category...")
            df[col] = df[col].astype('category')

    # --- 1. Define Features & Target ---
    target = 'meter_reading'
    # Exclude non-feature columns
    drop_cols = [target, 'timestamp', 'building_id', 'site_id']
    features = [c for c in df.columns if c not in drop_cols]
    
    print(f"Training with {len(features)} features.")

    # --- 2. TIME-BASED SPLIT ---
    # Sort by time to ensure we respect causality
    df = df.sort_values('timestamp')
    
    # 80% Train, 20% Test
    split_idx = int(len(df) * 0.8)
    
    X_train = df[features].iloc[:split_idx]
    y_train = df[target].iloc[:split_idx]
    X_test = df[features].iloc[split_idx:]
    y_test = df[target].iloc[split_idx:]
    
    print(f"Training Set: {len(X_train)} rows")
    print(f"Test Set: {len(X_test)} rows")

    # --- 3. Train LightGBM ---
    print("Training LightGBM Model (this may take 1-2 minutes)...")
    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1
    )
    
    # We use 'early_stopping' to stop if the model stops improving
    callbacks = [
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='rmse',
        callbacks=callbacks
    )

    # --- 4. Evaluate ---
    print("\n--- Model Evaluation ---")
    predictions = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE:  {mae:.2f}")
    print(f"Interpretation: On average, the model error is {mae:.2f} kWh.")

    # --- 5. Save the Model ---
    model_path = MODELS_DIR / f"lgbm_{meter_type}.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # --- 6. Feature Importance ---
    lgb.plot_importance(model, max_num_features=10, figsize=(10, 6))
    plt.title("Top 10 Drivers of Energy Consumption")
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "feature_importance.png")
    print("Feature importance plot saved to src/models/")

if __name__ == "__main__":
    train_model('electricity')