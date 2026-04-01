from __future__ import annotations

import pandas as pd
import pytest

from src.config.features import ALL_FEATURES, TARGET_COL
from src.explainability.shap_explainer import _map_feature_name
from src.features.build_features import (
    add_model_ready_columns,
    select_final_columns,
)


# ---------------------------------------------------------------------------
# _map_feature_name tests
# ---------------------------------------------------------------------------

class TestMapFeatureName:

    def test_numeric_prefix_stripped(self):
        assert _map_feature_name("num__user_age") == "user_age"

    def test_price_vs_price_log_disambiguation(self):
        result = _map_feature_name("num__price_log")
        assert result == "price_log"
        assert result != "price"

    def test_categorical_ohe_feature(self):
        assert _map_feature_name("cat__product_category_Electronics") == "product_category"

    def test_no_prefix_returned_as_is(self):
        assert _map_feature_name("user_age") == "user_age"

    def test_unknown_feature_returns_stripped_name(self):
        result = _map_feature_name("num__some_unknown_feature_xyz")
        assert result == "some_unknown_feature_xyz"


# ---------------------------------------------------------------------------
# Feature engineering integrity tests
# ---------------------------------------------------------------------------

@pytest.fixture
def engineered_base_df() -> pd.DataFrame:
    return pd.DataFrame({
        "user_id": ["U_001", "U_002"],
        "product_id": ["P_001", "P_002"],
        "event_time": pd.to_datetime(["2025-09-10 10:00:00", "2025-09-11 12:00:00"]),
        "age": [28, 35],
        "city_tier": ["Tier-1", "Tier-2"],
        "category": ["Electronics", "Fashion"],
        "brand": ["Sony", "Nike"],
        "price": [5000.0, 3000.0],
        "session_duration": [5.0, 7.0],
        "browsing_time_last_7d_mins": [12.0, 8.0],
        "days_since_last_purchase": [10.0, 45.0],
        "avg_cart_value": [4000.0, 2500.0],
        "total_orders": [5.0, 2.0],
        "total_browsing_events": [30.0, 15.0],
        "total_clicks": [10.0, 4.0],
        "click_through_proxy": [0.33, 0.27],
        "category_affinity": [1.0, 0.0],
        "is_affordable": [0.0, 1.0],
        "price_to_avg_cart_ratio": [1.25, 1.2],
        "event_hour": [10.0, 12.0],
        "event_dayofweek": [2.0, 3.0],
        TARGET_COL: [1, 0],
    })


class TestAddModelReadyColumns:

    def test_creates_required_renamed_columns(self, engineered_base_df):
        out = add_model_ready_columns(engineered_base_df)
        assert "user_age" in out.columns
        assert "user_city_tier" in out.columns
        assert "product_category" in out.columns
        assert "product_brand" in out.columns




class TestSelectFinalColumns:

    def test_contains_all_inference_features(self, engineered_base_df):
        out = add_model_ready_columns(engineered_base_df)
        final_df = select_final_columns(out)
        missing = [col for col in ALL_FEATURES if col not in final_df.columns]
        assert missing == []



class TestFeatureSchemaConsistency:

    def test_all_features_are_unique(self):
        assert len(ALL_FEATURES) == len(set(ALL_FEATURES))

    def test_target_not_in_all_features(self):
        assert TARGET_COL not in ALL_FEATURES

    def test_expected_core_columns_present(self):
        expected = {
            "user_age",
            "price",
            "price_log",
            "session_duration",
            "browsing_time_last_7d_mins",
            "days_since_last_purchase",
            "avg_cart_value",
            "total_orders",
            "total_browsing_events",
            "total_clicks",
            "click_through_proxy",
            "category_affinity",
            "is_affordable",
            "price_to_avg_cart_ratio",
            "event_hour",
            "event_dayofweek",
            "user_city_tier",
            "product_category",
            "product_brand",
        }
        assert expected.issubset(set(ALL_FEATURES))