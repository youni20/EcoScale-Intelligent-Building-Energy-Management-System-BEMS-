# ⚡ EcoScale: Intelligent Building Energy Management System

![Python](https://img.shields.io/badge/Python-3.11-blue) ![LightGBM](https://img.shields.io/badge/Model-LightGBM-green) ![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red) ![Status](https://img.shields.io/badge/Status-MVP_Complete-success)

**EcoScale** is an end-to-end Machine Learning pipeline designed to optimize energy consumption across commercial real estate portfolios. By creating "Virtual Meters" using Gradient Boosting (LightGBM), the system identifies operational inefficiencies and "phantom loads" that traditional rule-based systems miss.

![Dashboard Preview](assets/image.png)

---

## 🚀 Project Overview

Commercial buildings account for a significant portion of global energy waste. This project leverages historical meter data to predict *expected* energy usage based on weather and operational patterns. Deviations from this baseline are flagged as anomalies, allowing Facility Managers to target maintenance and reduce costs.

**Dataset:** [Building Data Genome Project 2 (Kaggle)](https://www.kaggle.com/datasets/claytonmiller/buildingdatagenomeproject2)
* **Scale:** 100GB+ of hourly meter readings (Electricity, Chilled Water, Steam, etc.).
* **Scope:** 1,600+ buildings across 19 diverse sites.

---

## 📊 Key Results & Impact

| Metric | Performance | Business Impact |
| :--- | :--- | :--- |
| **Model Accuracy (MAE)** | **6.34 kWh** | High-precision forecasting allows for detection of subtle HVAC faults. |
| **Forecast Horizon** | **24 Hours** | Enables Next-Day operational planning. |
| **Total Waste Identified** | **$1.46 Million** | Annualized potential savings detected across the test portfolio. |
| **Anomalies Flagged** | **320,000+** | Granular events identified for facility manager review. |

---

## 🏗️ Technical Architecture

The system is architected as a modular pipeline, separating data engineering, modeling, and serving layers.

### 1. Data Engineering (ETL)
* **Ingestion:** Merges disparate meter readings with site-specific weather telemetry using Pandas.
* **Optimization:** Converts raw CSVs (wide-format) into optimized **Parquet** files (long-format), reducing I/O time by ~95% and storage footprint significantly.
* **Robustness:** Handles missing sensor data via linear interpolation and forward-filling strategies.

### 2. Feature Engineering
* **Cyclical Encoding:** Transforms timestamp data (Hour, Month) into Sine/Cosine vectors ($x_{sin} = \sin(2\pi \cdot t/T)$) to preserve temporal continuity for the model.
* **Lag Features:** Generates 1h and 24h lag windows to capture serial autocorrelation (yesterday's usage is the strongest predictor of today's).
* **Rolling Statistics:** 6-hour rolling means to smooth out sensor noise.

### 3. Machine Learning (The "Virtual Meter")
* **Model:** **LightGBM Regressor** (Gradient Boosting Decision Trees).
* **Strategy:** Time-series split (Train: Jan-Oct, Test: Nov-Dec) to strictly prevent data leakage.
* **Inference:** The model predicts *expected* usage. If `Actual > Expected + Threshold`, an anomaly is flagged.
* **Thresholding:** Dynamic thresholding based on rolling standard deviation to account for building-specific volatility.

### 4. Visualization & Deployment
* **Dashboard:** A **Streamlit** application providing interactive exploration of energy profiles.
* **Reporting:** Automates the calculation of financial waste based on average commercial energy rates ($0.14/kWh).

---

## 🚀 Quick Start

### Prerequisites
* Python 3.9+
* pip

### Installation

1. **Clone the repository**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/EcoScale.git](https://github.com/YOUR_USERNAME/EcoScale.git)
   cd EcoScale
````

2.  **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Full Pipeline (ETL + Training + Inference)**
    This orchestrator script automates the entire workflow from raw data to anomaly report.

    ```bash
    python run_pipeline.py
    ```

4.  **Launch the Dashboard**

    ```bash
    streamlit run dashboard/app.py
    ```

-----

## 📂 Project Structure

EcoScale/
├── assets/                  # Images and static assets for README
├── data/
│   ├── raw/                 # Original BDG2 Datasets (CSV) - GitIgnored
│   ├── processed/           # Optimized Parquet files - GitIgnored
│   └── outputs/             # Anomaly reports and CSV exports
├── src/
│   ├── etl/                 # Data cleaning and merging scripts
│   ├── features/            # Cyclical encoding and lag generation
│   ├── models/              # LightGBM training and inference logic
│   └── utils/               # Shared utility functions
├── dashboard/               # Streamlit frontend application
├── run_pipeline.py          # Main pipeline orchestrator
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

-----

## 🔮 Future Improvements

  * **Containerization:** Dockerize the application for cloud deployment (AWS ECS/Azure Container Apps).
  * **Granularity:** Incorporate 15-minute interval data for sharper peak-load detection.
  * **Weather Integration:** Add solar irradiance features to better predict HVAC cooling loads.

-----

*Created by Younus Mashoor*

````