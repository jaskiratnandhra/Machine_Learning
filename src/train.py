# src/train.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    mean_absolute_error, mean_squared_error
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib

from src.preprocess import (
    split_features_target,
    detect_types,
    build_preprocessor,
)

# ---------------- Heuristics to pick task ----------------
def _is_integer_series(y: pd.Series) -> bool:
    """True if all finite values are (close to) integers."""
    y_num = pd.to_numeric(y, errors="coerce").dropna().values
    if y_num.size == 0:
        return False
    return np.allclose(y_num, np.round(y_num))

def _choose_task(y: pd.Series) -> str:
    """
    Return 'classification' or 'regression' based on target.
    - Non-numeric -> classification
    - Numeric with small number of distinct values (<= 8) OR all integers -> classification
    - Otherwise -> regression
    """
    is_numeric = pd.api.types.is_numeric_dtype(y)
    nunique = y.nunique(dropna=True)

    if not is_numeric:
        return "classification"
    if nunique <= 8:
        return "classification"
    if _is_integer_series(y):
        return "classification"
    return "regression"

# -------------- Classification helper (rare classes) --------------
def _prepare_labels(X: pd.DataFrame, y: pd.Series, min_count: int = 2):
    """
    Ensure every class has at least `min_count` rows.
    Drops rows belonging to ultra-rare classes and returns filtered X, y, and notes.
    """
    vc = y.value_counts(dropna=False)
    rare_classes = vc[vc < min_count].index.tolist()
    notes = {}

    if rare_classes:
        mask = ~y.isin(rare_classes)
        dropped = int((~mask).sum())
        notes["dropped_rare_classes"] = {
            "classes": [str(c) for c in rare_classes],
            "dropped_rows": dropped,
        }
        X = X.loc[mask].reset_index(drop=True)
        y = y.loc[mask].reset_index(drop=True)

    if y.nunique() < 2:
        raise ValueError(
            f"After removing ultra-rare classes, only one class remains ({y.unique().tolist()}). "
            "Choose a different target, collect more data, or merge labels."
        )

    vc2 = y.value_counts()
    can_stratify = (vc2.min() >= 2)
    notes["stratify_used"] = bool(can_stratify)
    return X, y, notes

# --------------------- Main entry point ---------------------
def train_and_save(csv_path: str, target: str, model_out: str = "sleep_model.joblib"):
    """
    Train a model with preprocessing and save the fitted pipeline.
    - Auto-detects task (classification vs regression)
    - For classification: handles ultra-rare classes
    - Saves schema (numeric/categorical) + task for safe prediction
    """
    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not in CSV columns: {list(df.columns)}")

    df = df.dropna(subset=[target])
    X, y = split_features_target(df, target)

    task = _choose_task(y)
    numeric, categorical = detect_types(df, target)
    pre = build_preprocessor(numeric, categorical)

    if task == "classification":
        model = RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced"
        )
        pipe = Pipeline([("pre", pre), ("clf", model)])

        # Handle ultra-rare classes & choose stratify
        X_f, y_f, notes = _prepare_labels(X, y, min_count=2)
        stratify_arg = y_f if notes["stratify_used"] else None

        Xtr, Xte, ytr, yte = train_test_split(
            X_f, y_f, test_size=0.2, random_state=42, stratify=stratify_arg
        )
        pipe.fit(Xtr, ytr)

        preds = pipe.predict(Xte)
        metrics = {
            "task": task,
            "ConfusionMatrix": confusion_matrix(yte, preds).tolist(),
            "Report": classification_report(yte, preds, output_dict=True),
            "classes": sorted(pd.Series(y_f).unique().tolist()),
            "notes": notes,
            "n_train": int(len(Xtr)),
            "n_test": int(len(Xte)),
        }
        try:
            proba = pipe.predict_proba(Xte)
            if len(metrics["classes"]) > 2:
                auc = roc_auc_score(yte, proba, multi_class="ovr")
            else:
                auc = roc_auc_score(yte, proba[:, 1])
            metrics["AUC"] = float(auc)
        except Exception:
            pass

    else:  # regression
        model = RandomForestRegressor(
            n_estimators=300, random_state=42
        )
        pipe = Pipeline([("pre", pre), ("reg", model)])

        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        pipe.fit(Xtr, ytr)

        preds = pipe.predict(Xte)
        metrics = {
            "task": task,
            "MAE": float(mean_absolute_error(yte, preds)),
            "RMSE": float(np.sqrt(mean_squared_error(yte, preds))),
            "n_train": int(len(Xtr)),
            "n_test": int(len(Xte)),
        }

    # Save bundle with schema + task for prediction alignment
    joblib.dump(
        {
            "model": pipe,
            "task": task,
            "target": target,
            "numeric": numeric,
            "categorical": categorical,
        },
        model_out,
    )
    return metrics
