# src/preprocess.py
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Common ID-like columns to ignore in features (case-insensitive)
IGNORE_COLS = {
    "id", "person id", "person_id", "user id", "userid",
    "employee id", "employee_id", "record id", "record_id",
    "name"  # include only if your dataset has a raw 'Name' column
}

def _feature_columns(df: pd.DataFrame, target_col: str):
    return [c for c in df.columns if c.lower() not in IGNORE_COLS and c != target_col]

def split_features_target(df: pd.DataFrame, target_col: str):
    cols = _feature_columns(df, target_col)
    X = df[cols]
    y = df[target_col]
    return X, y

def detect_types(df: pd.DataFrame, target_col: str):
    cols = _feature_columns(df, target_col)
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    return numeric, categorical

def build_preprocessor(numeric, categorical):
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, numeric),
        ("cat", cat_pipe, categorical)
    ])
    return pre
