"""
FastAPI recommendation service.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional, Union

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.explainability.shap_explainer import build_explainer, explain_predictions
from src.config.features import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COL
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "artifacts" / "models"

MODEL_PATH = MODELS_DIR / "xgboost_pipeline.pkl"
FEATURES_PATH = PROCESSED_DIR / "features.parquet"

_state: dict = {}

DEFAULT_REASON = "Recommended based on the user's browsing and purchase patterns."


def _require_state_keys(*keys: str) -> None:
    missing = [key for key in keys if key not in _state]
    if missing:
        raise HTTPException(
            status_code=503,
            detail=f"Service not initialized. Missing runtime resources: {missing}",
        )


def _validate_feature_columns(df: pd.DataFrame) -> None:
    missing = [col for col in ALL_FEATURES if col not in df.columns]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Feature store is missing required model columns: {missing}",
        )
    
def score_label(score: float) -> str:
    """Map raw model score to a business-friendly label."""
    if score >= 0.75:
        return "Very High"
    if score >= 0.60:
        return "High"
    if score >= 0.40:
        return "Medium"
    return "Low"


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading model from %s ...", MODEL_PATH)
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found: {MODEL_PATH}. Run the training pipeline first.")
    _state["pipeline"] = joblib.load(MODEL_PATH)

    log.info("Loading feature store from %s ...", FEATURES_PATH)
    if not FEATURES_PATH.exists():
        raise RuntimeError(f"Feature store not found: {FEATURES_PATH}. Run build_features.py first.")
    _state["feature_df"] = pd.read_parquet(FEATURES_PATH)

    log.info("Building SHAP TreeExplainer ...")
    bg_df = _state["feature_df"][ALL_FEATURES].sample(
        min(200, len(_state["feature_df"])),
        random_state=42,
    )
    _state["shap_explainer"] = build_explainer(_state["pipeline"], bg_df)

    log.info("Startup complete. Feature store rows: %d", len(_state["feature_df"]))
    yield

    _state.clear()
    log.info("Server shutdown - state cleared.")


app = FastAPI(
    title="E-Commerce Recommendation Engine",
    version="2.1.0",
    description="Purchase propensity model with SHAP-powered explanations.",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class RecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(..., min_length=1)
    product_ids: List[str] = Field(..., min_length=1)

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("user_id must be a non-empty string")
        return v

    @field_validator("product_ids")
    @classmethod
    def validate_product_ids(cls, values: List[str]) -> List[str]:
        cleaned: List[str] = []
        for item in values:
            if not isinstance(item, str):
                raise ValueError("each product_id must be a string")
            item = item.strip()
            if not item:
                raise ValueError("product_ids must not contain empty strings")
            cleaned.append(item)

        if not cleaned:
            raise ValueError("product_ids must contain at least one item")

        # Deduplicate while preserving order
        seen: set = set()
        deduped: List[str] = []
        for pid in cleaned:
            if pid not in seen:
                seen.add(pid)
                deduped.append(pid)

        return deduped


class MetricInfo(BaseModel):
    value: Union[float, int]
    label: Optional[str] = None
    explanation: str


class RecommendationItem(BaseModel):
    product_id: str
    rank: int
    recommendation_score: MetricInfo
    reasons: List[str]


class RecommendationResponse(BaseModel):
    user_id: str
    total_recommendations: int
    summary: str
    recommendations: List[RecommendationItem]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "E-Commerce Recommendation Engine API",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    feature_df = _state.get("feature_df")
    ready = all(key in _state for key in ("pipeline", "feature_df", "shap_explainer"))

    rows = int(len(feature_df)) if feature_df is not None else 0
    unique_users = int(feature_df["user_id"].nunique()) if feature_df is not None else 0
    unique_products = int(feature_df["product_id"].nunique()) if feature_df is not None else 0

    return {
        "status": "ok" if ready else "degraded",
        "model_loaded": "pipeline" in _state,
        "shap_explainer_ready": "shap_explainer" in _state,
        "rows_in_feature_store": {
            "value": rows,
            "explanation": "Total number of precomputed user-product rows available for scoring.",
        },
        "unique_users": {
            "value": unique_users,
            "explanation": "Number of distinct users in the feature store.",
        },
        "unique_products": {
            "value": unique_products,
            "explanation": "Number of distinct products in the feature store.",
        },
    }


@app.get("/debug/sample-payload")
def debug_sample_payload():
    _require_state_keys("feature_df")
    feature_df = _state["feature_df"]

    sample_user = feature_df["user_id"].iloc[0]
    sample_products = (
        feature_df[feature_df["user_id"] == sample_user]["product_id"]
        .drop_duplicates()
        .head(3)
        .tolist()
    )
    return {
        "user_id": sample_user,
        "product_ids": sample_products,
        "note": "Use this payload to test the /recommendations endpoint.",
    }


@app.post("/recommendations", response_model=RecommendationResponse)
def get_recommendations(request: RecommendationRequest):
    """
    POST /recommendations — ranked product list with SHAP explanations.

    Q17: Latency design — achieving <200ms per request
    ===================================================
    XGBoost inference on a small candidate set takes ~10ms (measured:
    0.0096s in artifacts/metrics/xgboost_metrics.txt). The remaining
    ~190ms budget is spent on network I/O and response serialization,
    which is comfortable at this payload size.

    Three design decisions ensure we stay within 200ms:

    1. Model loaded once at startup via lifespan(), not per request.
       joblib.load() on a 192KB pipeline takes ~80ms — doing this
       per request would blow the budget before inference even starts.

    2. SHAP TreeExplainer built once at startup against a background
       sample and cached in _state. Rebuilding it per request would
       add ~50-100ms of overhead on top of inference.

    3. Feature store loaded into RAM at startup as a DataFrame.
       Candidate rows are retrieved with a simple boolean mask
       (no database round-trip), keeping data access to <1ms.

    At runtime the critical path is:
      Feature lookup   ~1ms   (in-memory DataFrame mask)
      XGBoost scoring  ~10ms  (pipeline.predict_proba on candidates)
      SHAP explanation ~15ms  (TreeExplainer on scored rows)
      Serialization    ~5ms   (Pydantic response model)
      Total            ~31ms  — well within the 200ms SLA
    """
    _require_state_keys("feature_df", "pipeline", "shap_explainer")
    feature_df: pd.DataFrame = _state["feature_df"]
    model_pipeline = _state["pipeline"]
    shap_explainer = _state["shap_explainer"]

    # 1. Look up user
    user_rows = feature_df[feature_df["user_id"] == request.user_id].copy()
    if user_rows.empty:
        raise HTTPException(
            status_code=404,
            detail=f"user_id '{request.user_id}' not found in feature store.",
        )

    # 2. Filter to requested products — no fallback
    candidate_rows = user_rows[user_rows["product_id"].isin(request.product_ids)].copy()

    if candidate_rows.empty:
        return RecommendationResponse(
            user_id=request.user_id,
            total_recommendations=0,
            summary="No matching candidate products were found for this user.",
            recommendations=[],
        )

    # 3. Deduplicate and sort by recency before scoring
    candidate_rows = (
        candidate_rows
        .sort_values("event_time", ascending=False)
        .drop_duplicates(subset=["user_id", "product_id"])
        .reset_index(drop=True)
    )

    # 4. Score with XGBoost
    _validate_feature_columns(candidate_rows)
    X_input = candidate_rows[ALL_FEATURES].copy()
    proba = model_pipeline.predict_proba(X_input)

    if len(proba.shape) != 2 or proba.shape[1] < 2 or len(proba) != len(candidate_rows):
        raise HTTPException(
            status_code=500,
            detail="Model predict_proba output has an unexpected shape.",
        )

    candidate_rows["score"] = proba[:, 1]

    # 5. Sort by score descending and reset index so rank = index + 1
    candidate_rows = candidate_rows.sort_values("score", ascending=False).reset_index(drop=True)

    # 6. Rebuild feature frame from the now-sorted rows so SHAP rows
    #    are in exactly the same order as the ranked candidates.
    #    This prevents explanation/rank misalignment.
    X_ranked = candidate_rows[ALL_FEATURES].copy()

    # 7. SHAP reasons — primary explanation mode for the API.
    #    Always uses explain_predictions() — never heuristic rules.
    #    Returns list[list[str]], one inner list per row,
    #    in the same positional order as X_ranked.
    #    If SHAP fails, explain_predictions() returns default reasons
    #    internally so this endpoint never returns a 500 due to SHAP.
    reasons_list: list[list[str]] = explain_predictions(
        pipeline=model_pipeline,
        explainer=shap_explainer,
        X_input=X_ranked,
    )

    # 8. Build response items
    recommendations: List[RecommendationItem] = []
    for idx, row in candidate_rows.iterrows():
        score = round(float(row["score"]), 4)

        # reasons_list is positional — idx matches the reset_index above
        reasons = reasons_list[idx] if idx < len(reasons_list) else []
        if not reasons:
            reasons = ["Recommended based on the user's browsing and purchase patterns."]

        recommendations.append(
            RecommendationItem(
                product_id=row["product_id"],
                rank=idx + 1,
                recommendation_score=MetricInfo(
                    value=score,
                    label=score_label(score),
                    explanation=(
                        "Probability that this user will purchase this product "
                        "within 7 days, as predicted by the XGBoost model."
                    ),
                ),
                reasons=reasons[:3],
            )
        )

    total = len(recommendations)
    return RecommendationResponse(
        user_id=request.user_id,
        total_recommendations=total,
        summary=f"{total} product(s) scored and ranked for user '{request.user_id}'.",
        recommendations=recommendations,
    )
