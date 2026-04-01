"""
Integration tests for the FastAPI /recommendations endpoint.
Uses mocked state so no real model files are needed.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Mock data — mirrors the real feature store shape
# ---------------------------------------------------------------------------

MOCK_FEATURE_DF = pd.DataFrame({
    "user_id":                      ["U_001", "U_001", "U_002"],
    "product_id":                   ["P_001", "P_002", "P_003"],
    "event_time":                   pd.to_datetime(["2025-09-10", "2025-09-09", "2025-09-08"]),
    "user_age":                     [28.0, 28.0, 35.0],
    "price":                        [5000.0, 3000.0, 1200.0],
    "price_log":                    [np.log1p(5000), np.log1p(3000), np.log1p(1200)],
    "session_duration":             [5.0, 3.0, 7.0],
    "browsing_time_last_7d_mins":   [12.0, 12.0, 5.0],
    "browsing_time_last_7d_log":    [np.log1p(12), np.log1p(12), np.log1p(5)],
    "days_since_last_purchase":     [10.0, 10.0, 45.0],
    "days_since_last_purchase_log": [np.log1p(10), np.log1p(10), np.log1p(45)],
    "avg_cart_value":               [4000.0, 4000.0, 1000.0],
    "avg_cart_value_log":           [np.log1p(4000), np.log1p(4000), np.log1p(1000)],
    "total_orders":                 [5.0, 5.0, 2.0],
    "total_browsing_events":        [30.0, 30.0, 10.0],
    "total_clicks":                 [10.0, 10.0, 3.0],
    "click_through_proxy":          [0.33, 0.33, 0.3],
    "category_affinity":            [1.0, 0.0, 1.0],
    "is_affordable":                [0.0, 1.0, 1.0],
    "price_to_avg_cart_ratio":      [1.25, 0.75, 1.2],
    "event_hour":                   [14.0, 10.0, 9.0],
    "event_dayofweek":              [1.0, 3.0, 5.0],
    "user_city_tier":               ["Tier-1", "Tier-1", "Tier-2"],
    "product_category":             ["Electronics", "Fashion", "FMCG"],
    "product_brand":                ["Sony", "Nike", "HUL"],
})


def _make_mock_pipeline():
    """
    Build a mock pipeline whose predict_proba returns the correct number
    of rows based on the input size — not a hardcoded 3-row array.
    A hardcoded array causes 'Length of values does not match length of index'
    whenever candidate_rows has fewer rows than the mock output.
    """
    mock_pipeline = MagicMock()

    def _predict_proba(X):
        n = len(X)
        # Return descending scores so rank ordering is deterministic
        scores = np.linspace(0.8, 0.2, n)
        return np.column_stack([1 - scores, scores])

    mock_pipeline.predict_proba.side_effect = _predict_proba
    return mock_pipeline


def _make_explain_predictions(n_rows: int | None = None):
    """
    Return a mock for explain_predictions that produces the correct number
    of reason lists based on input size.
    """
    def _explain(pipeline, explainer, X_input, top_n=3):
        n = len(X_input) if n_rows is None else n_rows
        return [["Reason A", "Reason B"] for _ in range(n)]
    return _explain


@pytest.fixture
def client():
    """
    TestClient with _state pre-populated so lifespan never runs
    and no real files are loaded.
    """
    import src.api.main as api_module

    @asynccontextmanager
    async def _noop_lifespan(app):
        yield

    # Prevent the real lifespan from loading real files
    api_module.app.router.lifespan_context = _noop_lifespan

    # Inject mocked runtime state
    api_module._state["pipeline"] = _make_mock_pipeline()
    api_module._state["feature_df"] = MOCK_FEATURE_DF.copy()
    api_module._state["shap_explainer"] = MagicMock()

    with patch(
        "src.api.main.explain_predictions",
        side_effect=_make_explain_predictions(),
    ):
        with TestClient(api_module.app, raise_server_exceptions=True) as c:
            yield c

    api_module._state.clear()


# ---------------------------------------------------------------------------
# GET endpoints
# ---------------------------------------------------------------------------


class TestHealthEndpoint:

    def test_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_status_is_ok(self, client):
        assert client.get("/health").json()["status"] == "ok"


class TestDebugEndpoint:

    def test_returns_200(self, client):
        assert client.get("/debug/sample-payload").status_code == 200

# ---------------------------------------------------------------------------
# POST /recommendations — 200 cases
# ---------------------------------------------------------------------------

class TestRecommendationsSuccess:

    def test_valid_request_returns_200(self, client):
        payload = {"user_id": "U_001", "product_ids": ["P_001", "P_002"]}
        assert client.post("/recommendations", json=payload).status_code == 200

    def test_response_has_required_fields(self, client):
        payload = {"user_id": "U_001", "product_ids": ["P_001"]}
        data = client.post("/recommendations", json=payload).json()
        assert "user_id" in data
        assert "total_recommendations" in data
        assert "summary" in data
        assert "recommendations" in data

    def test_ranks_are_sequential(self, client):
        payload = {"user_id": "U_001", "product_ids": ["P_001", "P_002"]}
        data = client.post("/recommendations", json=payload).json()
        ranks = [r["rank"] for r in data["recommendations"]]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_score_label_present(self, client):
        payload = {"user_id": "U_001", "product_ids": ["P_001"]}
        data = client.post("/recommendations", json=payload).json()
        for rec in data["recommendations"]:
            assert rec["recommendation_score"]["label"] in [
                "Very High", "High", "Medium", "Low"
            ]

    def test_reasons_present(self, client):
        payload = {"user_id": "U_001", "product_ids": ["P_001"]}
        data = client.post("/recommendations", json=payload).json()
        for rec in data["recommendations"]:
            assert len(rec["reasons"]) >= 1

    def test_duplicate_product_ids_deduped(self, client):
        payload = {"user_id": "U_001", "product_ids": ["P_001", "P_001", "P_002"]}
        data = client.post("/recommendations", json=payload).json()
        returned_ids = [r["product_id"] for r in data["recommendations"]]
        assert len(returned_ids) == len(set(returned_ids))

    def test_unknown_product_returns_empty_list(self, client):
        payload = {"user_id": "U_001", "product_ids": ["P_UNKNOWN"]}
        data = client.post("/recommendations", json=payload).json()
        assert data["total_recommendations"] == 0
        assert data["recommendations"] == []


# ---------------------------------------------------------------------------
# POST /recommendations — 404
# ---------------------------------------------------------------------------

class TestRecommendations404:

    def test_unknown_user_returns_404(self, client):
        payload = {"user_id": "U_UNKNOWN", "product_ids": ["P_001"]}
        assert client.post("/recommendations", json=payload).status_code == 404


# ---------------------------------------------------------------------------
# POST /recommendations — 422 validation errors
# ---------------------------------------------------------------------------

class TestRecommendations422:

    def test_empty_user_id(self, client):
        payload = {"user_id": "", "product_ids": ["P_001"]}
        assert client.post("/recommendations", json=payload).status_code == 422


    def test_empty_product_ids_list(self, client):
        payload = {"user_id": "U_001", "product_ids": []}
        assert client.post("/recommendations", json=payload).status_code == 422

    def test_blank_string_in_product_ids(self, client):
        payload = {"user_id": "U_001", "product_ids": ["P_001", ""]}
        assert client.post("/recommendations", json=payload).status_code == 422

    def test_extra_field_forbidden(self, client):
        payload = {"user_id": "U_001", "product_ids": ["P_001"], "top_k": 3}
        assert client.post("/recommendations", json=payload).status_code == 422

    def test_user_id_with_only_whitespace(self, client):
        payload = {"user_id": "   ", "product_ids": ["P_001"]}
        assert client.post("/recommendations", json=payload).status_code == 422


# ---------------------------------------------------------------------------
# Additional regression tests
# ---------------------------------------------------------------------------

class TestRecommendationsAdditionalCoverage:

    def test_recommendations_sorted_by_score_desc(self, client):
        payload = {"user_id": "U_001", "product_ids": ["P_001", "P_002"]}
        data = client.post("/recommendations", json=payload).json()
        scores = [r["recommendation_score"]["value"] for r in data["recommendations"]]
        assert scores == sorted(scores, reverse=True)


    def test_mixed_valid_and_invalid_products_returns_only_valid_matches(self, client):
        payload = {"user_id": "U_001", "product_ids": ["P_001", "P_UNKNOWN"]}
        data = client.post("/recommendations", json=payload).json()
        returned_ids = [r["product_id"] for r in data["recommendations"]]
        assert returned_ids == ["P_001"]
        assert data["total_recommendations"] == 1


    def test_short_shap_output_does_not_crash_endpoint(self, client):
        import src.api.main as api_module
        api_module._state["pipeline"] = _make_mock_pipeline()

        payload = {"user_id": "U_001", "product_ids": ["P_001", "P_002"]}
        with patch(
            "src.api.main.explain_predictions",
            return_value=[["Only one SHAP reason set"]],
        ):
            response = client.post("/recommendations", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert len(data["recommendations"]) >= 1
        for rec in data["recommendations"]:
            assert len(rec["reasons"]) >= 1
    
def test_debug_sample_payload_requires_initialized_state(monkeypatch):
    import src.api.main as main_module
    from fastapi.testclient import TestClient

    main_module._state.clear()

    app = main_module.app
    with TestClient(app) as client:
        response = client.get("/debug/sample-payload")
        assert response.status_code == 503