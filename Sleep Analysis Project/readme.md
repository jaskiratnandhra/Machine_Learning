# 😴 Sleep Health Predictor

An **interactive machine learning app** built with **Streamlit** that predicts sleep health outcomes (**classification**) or estimates continuous values (**regression**) — it automatically chooses the right type based on your target column.

- **For anyone:** simple 3 steps → *Choose target → Train → Predict (single or batch)*  
- **For engineers:** robust preprocessing, schema alignment, rare-class handling, and clear project structure.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue" />
  <img src="https://img.shields.io/badge/Streamlit-App-red" />
  <img src="https://img.shields.io/badge/ScikitLearn-ML-green" />
  <img src="https://img.shields.io/badge/Pandas-Data-orange" />
</p>

---

## ✨ Features

- **Layman-friendly interface**  
  Three steps only: *Choose target → Train → Predict (single / batch)*  

- **Automatic task detection**  
  - Categorical target → **RandomForestClassifier**  
  - Numeric/continuous target → **RandomForestRegressor**  

- **Robust training pipeline**  
  - Imputation (median/mode)  
  - Scaling (StandardScaler)  
  - One-hot encoding for categoricals  
  - Drops ultra-rare classes (< 2 rows) to prevent training errors  
  - Saves both model & schema (`sleep_model.joblib`)  

- **Safe schema alignment**  
  - Adds missing columns as `NA` so imputers handle them  
  - Drops extra/unexpected columns  
  - Prevents “missing column” errors  

- **Prediction options**  
  - **Single Entry**: Fill in a simple form → instant prediction  
  - **Batch Upload**: Upload CSV → download predictions with appended `prediction` column  

---

## ⚡ Quickstart

```bash
# 1) Clone
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2) (Optional) Create virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# 3) Install
pip install -r requirements.txt

# 4) Add your data
# Place your CSV at data/sleep.csv (default path used in app.py)

# 5) Run
streamlit run app.py
