# app.py — Simple Streamlit app (handles classification OR regression automatically)
from pathlib import Path
import os
import pandas as pd
import streamlit as st
import joblib

from src.train import train_and_save
from src.preprocess import detect_types

# -------------------- Page setup --------------------
st.set_page_config(page_title="Sleep Health Predictor", page_icon="😴", layout="centered")
st.title("😴 Sleep Health Predictor")
st.caption("Three steps: choose what to predict → train → get a simple result.")

# -------------------- Paths --------------------
BASE_DIR = Path(__file__).parent.resolve()
DATA_PATH = BASE_DIR / "data" / "sleep.csv"
MODEL_PATH = BASE_DIR / "sleep_model.joblib"
DEFAULT_TARGET = "Sleep Disorder"  # change if needed

# -------------------- Data --------------------
@st.cache_data
def load_data(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    return df

if not DATA_PATH.exists():
    st.error(f"Please place your CSV at **{DATA_PATH}** (named `sleep.csv`).")
    st.stop()

df = load_data(DATA_PATH)

# -------------------- Prediction alignment helpers --------------------
def _expected_columns_from_bundle(bundle, X_current: pd.DataFrame):
    expected = []
    if isinstance(bundle, dict):
        expected = (bundle.get("numeric") or []) + (bundle.get("categorical") or [])
    if not expected and isinstance(bundle, dict) and "model" in bundle:
        model = bundle["model"]
        if hasattr(model, "feature_names_in_"):
            expected = list(model.feature_names_in_)
    if not expected:
        expected = list(X_current.columns)
    return expected

def align_features(bundle, X: pd.DataFrame) -> pd.DataFrame:
    expected = _expected_columns_from_bundle(bundle, X)
    for col in expected:
        if col not in X.columns:
            X[col] = pd.NA
    X = X[expected]
    return X

# -------------------- Step 1: choose target --------------------
st.header("Step 1 — What are we predicting?")
if DEFAULT_TARGET in df.columns:
    target = st.selectbox("Pick your target column", options=[DEFAULT_TARGET] + [c for c in df.columns if c != DEFAULT_TARGET])
else:
    target = st.selectbox("Pick your target column", options=list(df.columns))

if target not in df.columns:
    st.error("Please pick a valid target column.")
    st.stop()

# -------------------- Step 2: train --------------------
st.header("Step 2 — Train the model")
if st.button("Train", type="primary"):
    try:
        _ = train_and_save(str(DATA_PATH), target, model_out=str(MODEL_PATH))
        st.success("Model is ready! ✅")
    except Exception as e:
        st.error(f"Training failed: {e}")

model_ready = MODEL_PATH.exists()

# -------------------- Step 3: predict --------------------
st.header("Step 3 — Try it out")
tabs = st.tabs(["🧍 Single entry", "📁 File upload"])

with tabs[0]:
    st.write("Fill in the details below and click **Get result**.")
    X = df.drop(columns=[target])
    numeric_cols, categorical_cols = detect_types(df, target)

    form = st.form("predict_form")
    user_vals = {}

    # Numeric → sliders
    for c in numeric_cols:
        ser = pd.to_numeric(X[c], errors="coerce")
        if ser.notna().sum() == 0:
            user_vals[c] = form.number_input(c, value=0.0)
            continue
        q05, q50, q95 = ser.quantile([0.05, 0.5, 0.95]).values
        lo = float(min(q05, q95))
        hi = float(max(q05, q95))
        default = float(q50) if lo <= q50 <= hi else float((lo + hi) / 2)
        user_vals[c] = form.slider(c, min_value=float(lo), max_value=float(hi), value=float(default))

    # Categorical → dropdowns
    for c in categorical_cols:
        options = sorted([str(v) for v in pd.Series(X[c].dropna().unique()).astype(str)]) or [""]
        user_vals[c] = form.selectbox(c, options=options, index=0)

    submitted = form.form_submit_button("Get result")

    if submitted:
        if not model_ready:
            st.warning("Please train the model first (Step 2).")
        else:
            try:
                bundle = joblib.load(MODEL_PATH)
                model = bundle["model"]
                task = bundle.get("task", "classification")
                inp = pd.DataFrame([user_vals])
                inp = align_features(bundle, inp)

                if task == "regression":
                    value = float(model.predict(inp)[0])
                    st.success(f"**Estimated value:** {round(value, 2)}")
                    st.caption("Simple estimate shown. No extra numbers.")
                else:
                    pred = model.predict(inp)[0]
                    st.success(f"**Our best guess:** {pred}")
                    st.caption("Simple result. No extra numbers.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

with tabs[1]:
    st.write("Upload a file with the same columns you see in the form (no target column).")
    up = st.file_uploader("Choose a CSV file", type=["csv"])
    if up is not None:
        if not model_ready:
            st.warning("Please train the model first (Step 2).")
        else:
            try:
                bundle = joblib.load(MODEL_PATH)
                model = bundle["model"]
                task = bundle.get("task", "classification")
                Xup = pd.read_csv(up)
                if target in Xup.columns:
                    Xup = Xup.drop(columns=[target])
                Xup = align_features(bundle, Xup)

                preds = model.predict(Xup)
                out = Xup.copy()
                out["prediction"] = preds

                csv_bytes = out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download results",
                    data=csv_bytes,
                    file_name="predictions.csv",
                    mime="text/csv",
                    type="primary",
                )
                st.success("Your results are ready to download.")
            except Exception as e:
                st.error(f"Batch prediction failed: {e}")

st.info("Tip: If you change the dataset columns, retrain the model so it learns the new pattern.")
    