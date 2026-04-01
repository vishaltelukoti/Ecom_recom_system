"""
LightGBM vs XGBoost comparison report.

This module's only responsibility is comparison and reporting.
All XGBoost training logic lives in train_xgboost.py.

Usage (standalone):
    python -m src.models.evaluate

Called by train_pipeline.py as Step 5.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config.features import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COL
from src.models.train_xgboost import compute_scale_pos_weight, split_data
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def _build_lgbm_pipeline(scale_pos_weight: float) -> ImbPipeline:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", cat_pipe, CATEGORICAL_FEATURES),
    ], remainder="drop")

    return ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42, k_neighbors=5)),
        ("model", LGBMClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.8,
            min_child_weight=2,
            objective="binary",
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
            verbose=-1,
        )),
    ])


def compare_models(
    xgb_pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    spw = compute_scale_pos_weight(y_train)

    log.info("Training LightGBM for comparison …")
    lgbm_pipeline = _build_lgbm_pipeline(spw)
    t0 = time.perf_counter()
    lgbm_pipeline.fit(X_train, y_train)
    lgbm_time = time.perf_counter() - t0
    lgbm_proba = lgbm_pipeline.predict_proba(X_test)[:, 1]
    lgbm_metrics = {
        "roc_auc": float(roc_auc_score(y_test, lgbm_proba)),
        "pr_auc":  float(average_precision_score(y_test, lgbm_proba)),
        "train_time_s": round(lgbm_time, 4),
    }

    log.info("Evaluating XGBoost …")
    t0 = time.perf_counter()
    xgb_proba = xgb_pipeline.predict_proba(X_test)[:, 1]
    xgb_time = time.perf_counter() - t0
    xgb_metrics = {
        "roc_auc": float(roc_auc_score(y_test, xgb_proba)),
        "pr_auc":  float(average_precision_score(y_test, xgb_proba)),
        "inference_time_s": round(xgb_time, 4),
    }

    _write_report(xgb_metrics, lgbm_metrics)
    return {"xgboost": xgb_metrics, "lightgbm": lgbm_metrics}


def _write_report(xgb: dict, lgbm: dict) -> None:
    path = METRICS_DIR / "model_comparison.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("XGBoost vs LightGBM — Head-to-Head Comparison\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"{'Metric':<30} {'XGBoost':>10} {'LightGBM':>10}\n")
        f.write("-" * 55 + "\n")
        f.write(f"{'ROC-AUC':<30} {xgb['roc_auc']:>10.4f} {lgbm['roc_auc']:>10.4f}\n")
        f.write(f"{'PR-AUC':<30} {xgb['pr_auc']:>10.4f} {lgbm['pr_auc']:>10.4f}\n")
        f.write(f"{'XGBoost inference time (s)':<30} {xgb['inference_time_s']:>10.4f} {'N/A':>10}\n")
        f.write(f"{'LightGBM train time (s)':<30} {'N/A':>10} {lgbm['train_time_s']:>10.4f}\n")
        f.write("\nDecision: XGBoost selected for production.\n")
        f.write(
            "Rationale: SHAP-based regulatory explainability outweighs "
            "LightGBM's speed advantage at current dataset scale.\n"
        )
    log.info("Comparison report → %s", path)


def main() -> None:
    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    X_train, X_test, y_train, y_test = split_data(df)

    xgb_path = ARTIFACTS_DIR / "models" / "xgboost_pipeline.pkl"
    if not xgb_path.exists():
        raise FileNotFoundError(
            f"XGBoost pipeline not found at {xgb_path}. "
            "Run train_xgboost.py first."
        )
    xgb_pipeline = joblib.load(xgb_path)
    compare_models(xgb_pipeline, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()