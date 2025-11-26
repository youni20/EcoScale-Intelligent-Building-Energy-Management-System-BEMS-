import subprocess
import sys
import time
from pathlib import Path

def run_step(script_path, step_name):
    """Runs a python script and handles errors/logging."""
    print(f"\n{'='*60}")
    print(f"🚀 STARTING STEP: {step_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    

    try:
        subprocess.run([sys.executable, str(script_path)], check=True)
        elapsed = time.time() - start_time
        print(f" COMPLETED: {step_name} in {elapsed:.2f} seconds.")
    except subprocess.CalledProcessError:
        print(f" FAILED: {step_name}. Stopping pipeline.")
        sys.exit(1)

def main():
    base_dir = Path(__file__).parent
    
    print("\n EcoScale Pipeline Initialized ")
    print("-----------------------------------")
    
    # 1. ETL: Load and Merge Data
    run_step(base_dir / "src" / "etl" / "data_loader.py", "Data Ingestion & Merging")
    
    # 2. Features: Cyclical Time & Lags
    run_step(base_dir / "src" / "features" / "processor.py", "Feature Engineering")
    
    # 3. Model: Train LightGBM
    run_step(base_dir / "src" / "models" / "train_model.py", "Model Training (LightGBM)")
    
    # 4. Inference: Detect Anomalies
    run_step(base_dir / "src" / "models" / "detect_anomalies.py", "Anomaly Detection Engine")

    print(f"\n{'='*60}")
    print("🎉 PIPELINE SUCCESSFUL!")
    print(f"{'='*60}")
    print("Next Step: Run the dashboard to view results.")
    print("Command: streamlit run dashboard/app.py")

if __name__ == "__main__":
    main()