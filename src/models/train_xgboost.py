"""
XGBoost purchase-propensity model — canonical production trainer.

Training strategy
-----------------
* ColumnTransformer handles mixed feature types (numeric + categorical).
* SMOTE oversamples the minority class (~2.8% positive rate) AFTER splitting
  so no resampled rows leak into the validation fold.
* RandomizedSearchCV tunes 6 XGBoost hyperparameters with stratified 5-fold CV.
* scale_pos_weight is passed as a fallback baseline when SMOTE is disabled.
* Best pipeline is serialised with joblib for use by the API and SHAP explainer.
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score,average_precision_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.config.features import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COL

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


PARAM_DIST = {
    "model__n_estimators":     [100, 150, 200, 300, 400],
    "model__max_depth":        [3, 4, 5, 6],
    "model__learning_rate":    [0.01, 0.05, 0.08, 0.1, 0.15],
    "model__subsample":        [0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "model__min_child_weight": [1, 2, 3, 5, 10],
}


def load_features() -> pd.DataFrame:
    path = PROCESSED_DIR / "features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}. Run build_features.py first.")
    return pd.read_parquet(path)


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def compute_scale_pos_weight(y: pd.Series) -> float:
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    return float(neg / pos) if pos > 0 else 1.0


def build_imbalanced_pipeline(scale_pos_weight: float) -> ImbPipeline:
    return ImbPipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("smote", SMOTE(random_state=42, k_neighbors=5)),
        ("model", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
        )),
    ])


def split_data(df: pd.DataFrame):
    X = df[ALL_FEATURES].copy()
    y = df[TARGET_COL].copy()
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def run_hyperparameter_search(
    pipeline: ImbPipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 20,
    cv_folds: int = 5,
) -> RandomizedSearchCV:
    log.info("Starting RandomizedSearchCV  n_iter=%d  cv=%d", n_iter, cv_folds)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=PARAM_DIST,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    log.info("Best CV AUC-ROC: %.4f", search.best_score_)
    log.info("Best params: %s", search.best_params_)
    return search


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate a trained pipeline on the test set.

    Metrics returned
    ----------------
    roc_auc  : Area under the ROC curve. Threshold-independent measure of
               ranking quality. Good for comparing models but can be
               over-optimistic on heavily imbalanced datasets.

    pr_auc   : Area under the Precision-Recall curve (average precision).
               More informative than ROC-AUC when the positive class is rare
               (~2.8% purchase rate). A high PR-AUC means the model ranks
               true purchasers near the top without too many false positives.

    confusion_matrix        : Raw counts of TP, FP, TN, FN at the 0.5 threshold.
    classification_report   : Per-class precision, recall, f1 at 0.5 threshold.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def save_artifacts(model, metrics: dict, best_params: dict) -> None:
    joblib.dump(model, MODELS_DIR / "xgboost_pipeline.pkl")
    joblib.dump(metrics, METRICS_DIR / "xgboost_metrics.pkl")
    with open(METRICS_DIR / "xgboost_metrics.txt", "w", encoding="utf-8") as f:
        f.write("XGBoost + SMOTE + RandomizedSearchCV  Metrics\n")
        f.write("=" * 50 + "\n")
        f.write(f"ROC-AUC  (test): {metrics['roc_auc']:.4f}\n")
        f.write(f"PR-AUC   (test): {metrics['pr_auc']:.4f}\n\n")
        f.write("Best hyperparameters:\n")
        for k, v in best_params.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(f"{metrics['confusion_matrix']}\n\n")
        f.write("Classification Report:\n")
        f.write(pd.DataFrame(metrics["classification_report"]).transpose().to_string())
    log.info("Artifacts saved → %s", MODELS_DIR / "xgboost_pipeline.pkl")


def train(n_iter: int = 20, cv_folds: int = 5) -> tuple:
    df = load_features()
    X_train, X_test, y_train, y_test = split_data(df)
    scale_pos_weight = compute_scale_pos_weight(y_train)

    log.info("Train rows: %d  |  Test rows: %d  |  scale_pos_weight: %.3f",
             len(X_train), len(X_test), scale_pos_weight)

    pipeline = build_imbalanced_pipeline(scale_pos_weight)
    search = run_hyperparameter_search(pipeline, X_train, y_train, n_iter=n_iter, cv_folds=cv_folds)

    best_pipeline = search.best_estimator_
    metrics = evaluate_model(best_pipeline, X_test, y_test)
    save_artifacts(best_pipeline, metrics, search.best_params_)

    log.info("Test ROC-AUC: %.4f", metrics["roc_auc"])
    return best_pipeline, metrics, search.best_params_


def main() -> None:
    train()


if __name__ == "__main__":
    main()