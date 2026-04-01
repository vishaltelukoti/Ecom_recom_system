"""
Unit tests for src/assistant/chain.py
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest
import numpy as np

import src.assistant.chain as chain_module
from src.assistant.schemas import ConversationSession, ShoppingQuery


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class DummyExtractor:
    def __init__(self, parsed_query):
        self.parsed_query = parsed_query
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload)
        return self.parsed_query


def _mock_ranked_df() -> pd.DataFrame:
    return pd.DataFrame({
        "product_id": ["P_001", "P_002", "P_003"],
        "title": ["Sony Headphones", "Nike Shoes", "Samsung Phone"],
        "category": ["Electronics", "Fashion", "Electronics"],
        "brand": ["Sony", "Nike", "Samsung"],
        "price": [5000.0, 3000.0, 12000.0],
        "user_age": [28.0, 28.0, 28.0],
        "price_log": [8.51, 8.01, 9.39],
        "session_duration": [5.0, 4.0, 6.0],
        "browsing_time_last_7d_mins": [12.0, 8.0, 10.0],
        "browsing_time_last_7d_log": [2.56, 2.20, 2.40],
        "days_since_last_purchase": [10.0, 20.0, 30.0],
        "days_since_last_purchase_log": [2.39, 3.04, 3.43],
        "avg_cart_value": [4000.0, 4000.0, 4000.0],
        "avg_cart_value_log": [8.29, 8.29, 8.29],
        "total_orders": [5.0, 5.0, 5.0],
        "total_browsing_events": [30.0, 30.0, 30.0],
        "total_clicks": [10.0, 10.0, 10.0],
        "click_through_proxy": [0.33, 0.33, 0.33],
        "category_affinity": [1.0, 0.0, 1.0],
        "is_affordable": [0.0, 1.0, 0.0],
        "price_to_avg_cart_ratio": [1.25, 0.75, 3.0],
        "event_hour": [14.0, 10.0, 18.0],
        "event_dayofweek": [1.0, 3.0, 5.0],
        "user_city_tier": ["Tier-1", "Tier-1", "Tier-1"],
        "product_category": ["Electronics", "Fashion", "Electronics"],
        "product_brand": ["Sony", "Nike", "Samsung"],
    })


@pytest.fixture
def parsed_query():
    return ShoppingQuery(category="Electronics", max_price=15000, brand=None)


@pytest.fixture
def extractor(parsed_query):
    return DummyExtractor(parsed_query)


@pytest.fixture
def mock_pipeline():
    pipeline = MagicMock()
    pipeline.predict_proba.return_value = np.array([
        [0.2, 0.8],
        [0.4, 0.6],
        [0.7, 0.3],
    ])
    return pipeline


@pytest.fixture
def ranked_df():
    return _mock_ranked_df()


@pytest.fixture
def patched_assistant(monkeypatch, extractor, mock_pipeline, ranked_df):
    monkeypatch.setattr(chain_module, "_get_query_extractor", lambda: extractor)
    monkeypatch.setattr(chain_module, "model_pipeline", mock_pipeline)
    monkeypatch.setattr(chain_module, "_shap_explainer", MagicMock())
    monkeypatch.setattr(chain_module, "_ensure_runtime_resources", lambda: None)

    monkeypatch.setattr(    
        chain_module,
        "get_candidate_products",
        lambda category=None, max_price=None, brand=None: pd.DataFrame({
            "product_id": ["P_001", "P_002", "P_003"],
            "title": ["Sony Headphones", "Nike Shoes", "Samsung Phone"],
            "category": ["Electronics", "Fashion", "Electronics"],
            "brand": ["Sony", "Nike", "Samsung"],
            "price": [5000.0, 3000.0, 12000.0],
        }),
    )
    monkeypatch.setattr(
        chain_module,
        "filter_products_by_query_text",
        lambda df, query_text: df,
    )
    monkeypatch.setattr(
        chain_module,
        "get_ranked_candidates_for_user",
        lambda user_id, candidate_products: ranked_df.copy(),
    )

    return {
        "extractor": extractor,
        "pipeline": mock_pipeline,
        "ranked_df": ranked_df,
    }

# ---------------------------------------------------------------------------
# Memory / multi-turn behavior
# ---------------------------------------------------------------------------

class TestRunAssistantMemory:

    def test_history_is_passed_to_extractor(self, patched_assistant, monkeypatch):
        monkeypatch.setattr(
            chain_module,
            "explain_predictions",
            lambda pipeline, explainer, X_input: [["Reason A"], ["Reason B"], ["Reason C"]],
        )

        _, session = chain_module.run_assistant(
            user_id="U_001",
            user_message="Show me electronics",
        )
        _, _ = chain_module.run_assistant(
            user_id="U_001",
            user_message="Make it cheaper",
            session=session,
        )

        extractor = patched_assistant["extractor"]
        assert len(extractor.calls) == 2
        assert "history" in extractor.calls[1]
        assert len(extractor.calls[1]["history"]) >= 1


# ---------------------------------------------------------------------------
# SHAP fallback behavior
# ---------------------------------------------------------------------------

class TestAssistantExplanationFallback:

    def test_uses_shap_reasons_when_available(self, patched_assistant, monkeypatch):
        monkeypatch.setattr(
            chain_module,
            "explain_predictions",
            lambda pipeline, explainer, X_input: [
                ["SHAP Reason 1"],
                ["SHAP Reason 2"],
                ["SHAP Reason 3"],
            ],
        )

        response, _ = chain_module.run_assistant(
            user_id="U_001",
            user_message="Recommend electronics",
        )

        assert response.recommendations[0].reasons == ["SHAP Reason 1"]

    def test_falls_back_to_assistant_reasons_when_shap_fails(self, patched_assistant, monkeypatch):
        monkeypatch.setattr(
            chain_module,
            "explain_predictions",
            lambda pipeline, explainer, X_input: (_ for _ in ()).throw(Exception("SHAP failed")),
        )

        response, _ = chain_module.run_assistant(
            user_id="U_001",
            user_message="Recommend electronics under 15000",
        )

        assert len(response.recommendations) >= 1
        assert len(response.recommendations[0].reasons) >= 1

# ---------------------------------------------------------------------------
# Query extractor / config behavior
# ---------------------------------------------------------------------------

class TestQueryExtractorCaching:

    def test_get_query_extractor_builds_once(self, monkeypatch):
        sentinel = object()
        calls = {"count": 0}

        def fake_builder():
            calls["count"] += 1
            return sentinel

        monkeypatch.setattr(chain_module, "_query_extractor", None)
        monkeypatch.setattr(chain_module, "build_query_extractor", fake_builder)

        extractor_1 = chain_module._get_query_extractor()
        extractor_2 = chain_module._get_query_extractor()

        assert extractor_1 is sentinel
        assert extractor_2 is sentinel
        assert calls["count"] == 1

        
def test_extractor_runtime_error_does_not_call_save_context(monkeypatch):
    import src.assistant.chain as chain_module

    monkeypatch.setattr(chain_module, "_ensure_runtime_resources", lambda: None)

    def boom():
        raise RuntimeError("missing api key")

    monkeypatch.setattr(chain_module, "_get_query_extractor", boom)

    response, session = chain_module.run_assistant(
        user_id="U_001",
        user_message="Recommend headphones",
    )

    assert response.assistant_message == "missing api key"
    assert isinstance(session.memory, list)
    assert len(session.memory) >= 2