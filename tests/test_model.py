"""
Unit tests for src/models/train_xgboost.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

from src.models.train_xgboost import (
    ALL_FEATURES,
    TARGET_COL,
    build_preprocessor,
    compute_scale_pos_weight,
    evaluate_model,
    split_data,
)

import tempfile
from pathlib import Path

import joblib


# ---------------------------------------------------------------------------
# Additional model regression tests
# ---------------------------------------------------------------------------

class TestFeatureSchemaCompatibility:

    def test_pipeline_accepts_exact_all_features_schema(self, synthetic_df):
        X = synthetic_df[ALL_FEATURES].copy()
        y = synthetic_df[TARGET_COL].copy()

        pipeline = _build_test_pipeline(compute_scale_pos_weight(y))
        pipeline.fit(X, y)

        proba = pipeline.predict_proba(X.head(5))
        assert proba.shape == (5, 2)



class TestArtifactRoundTrip:

    def test_trained_pipeline_can_be_saved_and_loaded(self, synthetic_df):
        X = synthetic_df[ALL_FEATURES].copy()
        y = synthetic_df[TARGET_COL].copy()

        pipeline = _build_test_pipeline(compute_scale_pos_weight(y))
        pipeline.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_pipeline.pkl"
            joblib.dump(pipeline, model_path)

            assert model_path.exists()

            loaded = joblib.load(model_path)
            proba = loaded.predict_proba(X.head(4))
            assert proba.shape == (4, 2)

    def test_loaded_pipeline_predictions_match_original(self, synthetic_df):
        X = synthetic_df[ALL_FEATURES].copy()
        y = synthetic_df[TARGET_COL].copy()

        pipeline = _build_test_pipeline(compute_scale_pos_weight(y))
        pipeline.fit(X, y)
        original = pipeline.predict_proba(X.head(10))

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_pipeline.pkl"
            joblib.dump(pipeline, model_path)
            loaded = joblib.load(model_path)

        restored = loaded.predict_proba(X.head(10))
        assert np.allclose(original, restored)


class TestEvaluateModelOutputStructure:

    def test_confusion_matrix_is_2x2(self, synthetic_df):
        X_train, X_test, y_train, y_test = split_data(synthetic_df)
        pipeline = _build_test_pipeline(compute_scale_pos_weight(y_train))
        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(pipeline, X_test, y_test)
        cm = metrics["confusion_matrix"]

        assert isinstance(cm, list)
        assert len(cm) == 2
        assert all(len(row) == 2 for row in cm)


class TestInferenceRobustness:

    def test_pipeline_handles_small_batch_prediction(self, synthetic_df):
        X = synthetic_df[ALL_FEATURES].copy()
        y = synthetic_df[TARGET_COL].copy()

        pipeline = _build_test_pipeline(compute_scale_pos_weight(y))
        pipeline.fit(X, y)

        single = X.head(1)
        proba = pipeline.predict_proba(single)

        assert proba.shape == (1, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)



def _build_test_pipeline(scale_pos_weight: float) -> ImbPipeline:
    """
    Test-safe pipeline using k_neighbors=2 so SMOTE works even when
    the positive class has very few samples in the training split.
    Production uses k_neighbors=5 — do not change that value.
    """
    return ImbPipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("smote", SMOTE(random_state=42, k_neighbors=2)),
        ("model", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            n_estimators=50,
            scale_pos_weight=scale_pos_weight,
        )),
    ])


# ---------------------------------------------------------------------------
# Shared synthetic dataset
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame({
        "user_age":                     rng.integers(18, 60, n).astype(float),
        "price":                        rng.uniform(100, 15000, n),
        "price_log":                    np.log1p(rng.uniform(100, 15000, n)),
        "session_duration":             rng.uniform(0.1, 30, n),
        "browsing_time_last_7d_mins":   rng.uniform(0, 60, n),
        "browsing_time_last_7d_log":    rng.uniform(0, 4, n),
        "days_since_last_purchase":     rng.uniform(0, 200, n),
        "days_since_last_purchase_log": rng.uniform(0, 5.3, n),
        "avg_cart_value":               rng.uniform(200, 10000, n),
        "avg_cart_value_log":           rng.uniform(5, 9, n),
        "total_orders":                 rng.integers(0, 20, n).astype(float),
        "total_browsing_events":        rng.integers(1, 100, n).astype(float),
        "total_clicks":                 rng.integers(0, 50, n).astype(float),
        "click_through_proxy":          rng.uniform(0, 1, n),
        "category_affinity":            rng.integers(0, 2, n).astype(float),
        "is_affordable":                rng.integers(0, 2, n).astype(float),
        "price_to_avg_cart_ratio":      rng.uniform(0.1, 3, n),
        "event_hour":                   rng.integers(0, 24, n).astype(float),
        "event_dayofweek":              rng.integers(0, 7, n).astype(float),
        "user_city_tier":               rng.choice(["Tier-1", "Tier-2", "Tier-3"], n),
        "product_category":             rng.choice(["Electronics", "Fashion", "FMCG"], n),
        "product_brand":                rng.choice(["Sony", "Nike", "Samsung"], n),
        TARGET_COL:                     rng.choice([0, 1], n, p=[0.85, 0.15]),
    })


# ---------------------------------------------------------------------------
# compute_scale_pos_weight
# ---------------------------------------------------------------------------

class TestComputeScalePosWeight:

    def test_correct_ratio(self):
        y = pd.Series([0] * 90 + [1] * 10)
        assert np.isclose(compute_scale_pos_weight(y), 9.0)

# ---------------------------------------------------------------------------
# split_data
# ---------------------------------------------------------------------------

class TestSplitData:

    def test_sizes_add_up(self, synthetic_df):
        X_train, X_test, y_train, y_test = split_data(synthetic_df)
        assert len(X_train) + len(X_test) == len(synthetic_df)


# ---------------------------------------------------------------------------
# _build_test_pipeline  (uses k_neighbors=2, safe for small datasets)
# ---------------------------------------------------------------------------

class TestBuildPipeline:

    def test_pipeline_fits_without_error(self, synthetic_df):
        X = synthetic_df[ALL_FEATURES]
        y = synthetic_df[TARGET_COL]
        pipeline = _build_test_pipeline(compute_scale_pos_weight(y))
        pipeline.fit(X, y)


    def test_probabilities_in_valid_range(self, synthetic_df):
        X = synthetic_df[ALL_FEATURES]
        y = synthetic_df[TARGET_COL]
        pipeline = _build_test_pipeline(compute_scale_pos_weight(y))
        pipeline.fit(X, y)
        proba = pipeline.predict_proba(X)
        assert (proba >= 0).all() and (proba <= 1).all()


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------

class TestEvaluateModel:

    def test_required_keys_present(self, synthetic_df):
        X_train, X_test, y_train, y_test = split_data(synthetic_df)
        pipeline = _build_test_pipeline(compute_scale_pos_weight(y_train))
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)
        assert "roc_auc" in metrics
        assert "pr_auc" in metrics
        assert "confusion_matrix" in metrics
        assert "classification_report" in metrics

    def test_roc_auc_in_valid_range(self, synthetic_df):
        X_train, X_test, y_train, y_test = split_data(synthetic_df)
        pipeline = _build_test_pipeline(compute_scale_pos_weight(y_train))
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)
        assert 0.0 <= metrics["roc_auc"] <= 1.0
