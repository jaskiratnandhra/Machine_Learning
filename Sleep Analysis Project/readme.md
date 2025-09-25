# 😴 Sleep Health Predictor

An **interactive machine learning app** built with **Streamlit** that predicts sleep health outcomes (classification) or estimates continuous values (regression), depending on your dataset.  

Designed to be **easy for non-technical users** while also being **technically robust** under the hood. Just upload your data, click **Train**, and start making predictions.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue" />
  <img src="https://img.shields.io/badge/Streamlit-App-red" />
  <img src="https://img.shields.io/badge/ScikitLearn-ML-green" />
  <img src="https://img.shields.io/badge/Pandas-Data-orange" />
</p>

---

## ✨ Features

✅ **Layman-friendly interface**  
Simple 3-step workflow: *Choose target → Train → Predict (single or batch)*  

✅ **Automatic task detection**  
- If your target is categorical → trains a **Random Forest Classifier**  
- If your target is numeric → trains a **Random Forest Regressor**  

✅ **Robust training pipeline**  
- Handles missing values (numeric imputation, categorical mode)  
- Scales numerics & one-hot encodes categoricals  
- Drops ultra-rare classes (<2 rows) to prevent training errors  
- Aligns prediction input with training schema (no “missing column” issues)  

✅ **Prediction options**  
- **Single Entry:** Fill in a form → instant prediction  
- **Batch Mode:** Upload a CSV → download results with predictions appended  

✅ **Technical safeguards**  
- Schema alignment ensures consistency  
- Model + schema saved as `sleep_model.joblib`  
- Works even if dataset contains ID-like columns (`Person ID`, `User ID`)  

---

## 🏗️ Architecture

```mermaid
flowchart LR
    A[CSV Data] --> B[Preprocessing]
    B -->|Imputation / Scaling / Encoding| C[Model Training]
    C -->|Random Forest (clf/reg)| D[Trained Model Bundle]
    D --> E[Streamlit App]
    E -->|Form Inputs| F[Single Prediction]
    E -->|CSV Upload| G[Batch Prediction]
