from __future__ import annotations

import logging
import os
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv

from src.assistant.prompts import SYSTEM_PROMPT
from src.assistant.schemas import (
    AssistantResponse,
    ConversationSession,
    RecommendationResult,
    ShoppingQuery,
)
from src.assistant.tools import (
    filter_products_by_query_text,
    get_candidate_products,
    get_ranked_candidates_for_user,
    normalize_category,
)
from src.config.features import ALL_FEATURES
from src.explainability.shap_explainer import build_explainer, explain_predictions


load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "artifacts" / "models"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_PATH = MODELS_DIR / "xgboost_pipeline.pkl"
FEATURES_PATH = PROCESSED_DIR / "features.parquet"

model_pipeline = None
_shap_explainer = None
_features_df = None
_query_extractor = None


def build_query_extractor():
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_groq import ChatGroq

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your environment or .env file before running the assistant."
        )

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=groq_api_key,
    )
    structured_llm = llm.with_structured_output(ShoppingQuery)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{user_message}"),
    ])

    return prompt | structured_llm


def _get_query_extractor():
    global _query_extractor
    if _query_extractor is None:
        _query_extractor = build_query_extractor()
    return _query_extractor


def _ensure_runtime_resources():
    global model_pipeline, _shap_explainer, _features_df

    if model_pipeline is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(f"Model not found: {MODEL_PATH}. Run the training pipeline first.")
        model_pipeline = joblib.load(MODEL_PATH)

    if _features_df is None:
        if not FEATURES_PATH.exists():
            raise RuntimeError(f"Feature store not found: {FEATURES_PATH}. Run the training pipeline first.")
        _features_df = pd.read_parquet(FEATURES_PATH)

    if _shap_explainer is None:
        bg = _features_df[ALL_FEATURES].sample(
            min(200, len(_features_df)),
            random_state=42,
        )
        _shap_explainer = build_explainer(model_pipeline, bg)


def _append_turn(session: ConversationSession, user_message: str, assistant_message: str) -> None:
    try:
        from langchain_core.messages import AIMessage, HumanMessage
        session.memory.append(HumanMessage(content=user_message))
        session.memory.append(AIMessage(content=assistant_message))
    except Exception:
        session.memory.append({"role": "user", "content": user_message})
        session.memory.append({"role": "assistant", "content": assistant_message})


def _validate_ranked_features(ranked: pd.DataFrame) -> None:
    missing = [col for col in ALL_FEATURES if col not in ranked.columns]
    if missing:
        raise RuntimeError(f"Ranked candidate rows are missing required features: {missing}")


def _generate_assistant_fallback_reasons(row: pd.Series, parsed: ShoppingQuery) -> list[str]:
    reasons: list[str] = []
    normalized_category = normalize_category(parsed.category) if parsed.category else None

    if normalized_category and str(row.get("category", "")).lower() == normalized_category.lower():
        reasons.append(f"Matches your requested category: {parsed.category}")

    if parsed.max_price is not None and float(row.get("price", 0)) <= parsed.max_price:
        reasons.append(f"Within your budget of Rs. {parsed.max_price:.0f}")

    if parsed.brand and str(row.get("brand", "")).lower() == parsed.brand.lower():
        reasons.append(f"Matches your preferred brand: {parsed.brand}")

    if row.get("category_affinity", 0) == 1:
        reasons.append("Matches your historical category preference")

    if row.get("click_through_proxy", 0) > 0.3:
        reasons.append("You showed strong recent engagement in similar items")

    if row.get("browsing_time_last_7d_mins", 0) > 8:
        reasons.append("You recently spent more time browsing similar products")

    if not reasons:
        reasons.append("Recommended from your recent browsing and purchase patterns")

    return reasons[:3]


def _resolve_assistant_reasons(
    row: pd.Series,
    parsed: ShoppingQuery,
    shap_reasons: list | None,
    row_index: int,
) -> list[str]:
    if shap_reasons and row_index < len(shap_reasons) and shap_reasons[row_index]:
        return shap_reasons[row_index]
    return _generate_assistant_fallback_reasons(row, parsed)


def run_assistant(
    user_id: str,
    user_message: str,
    session: ConversationSession | None = None,
) -> tuple[AssistantResponse, ConversationSession]:
    if session is None:
        session = ConversationSession(user_id=user_id)

    try:
        _ensure_runtime_resources()
    except RuntimeError as exc:
        response = AssistantResponse(
            user_id=user_id,
            interpreted_query=ShoppingQuery(),
            recommendations=[],
            assistant_message=str(exc),
        )
        _append_turn(session, user_message, response.assistant_message)
        return response, session

    try:
        extractor = _get_query_extractor()
        history = session.memory
        parsed: ShoppingQuery = extractor.invoke({
            "user_message": user_message,
            "history": history,
        })
    except RuntimeError as exc:
        response = AssistantResponse(
            user_id=user_id,
            interpreted_query=ShoppingQuery(),
            recommendations=[],
            assistant_message=str(exc),
        )
        _append_turn(session, user_message, response.assistant_message)
        return response, session

    log.info("Parsed query: %s", parsed.model_dump())

    candidate_products = get_candidate_products(
        category=parsed.category,
        max_price=parsed.max_price,
        brand=parsed.brand,
    )
    candidate_products = filter_products_by_query_text(candidate_products, parsed.category)

    ranked = get_ranked_candidates_for_user(
        user_id=user_id,
        candidate_products=candidate_products,
    )

    if ranked.empty:
        response = AssistantResponse(
            user_id=user_id,
            interpreted_query=parsed,
            recommendations=[],
            assistant_message="I could not find matching products for your request.",
        )
        _append_turn(session, user_message, response.assistant_message)
        return response, session

    ranked = ranked.copy()
    _validate_ranked_features(ranked)

    proba = model_pipeline.predict_proba(ranked[ALL_FEATURES].copy())
    if len(proba.shape) != 2 or proba.shape[1] < 2 or len(proba) != len(ranked):
        raise RuntimeError("Model predict_proba output has an unexpected shape for assistant scoring.")
    ranked["score"] = proba[:, 1]

    normalized_category = normalize_category(parsed.category) if parsed.category else None
    if normalized_category:
        ranked = ranked[ranked["category"].fillna("").str.lower() == normalized_category.lower()]
    if parsed.max_price is not None:
        ranked = ranked[ranked["price"] <= parsed.max_price]
    if parsed.brand:
        ranked = ranked[ranked["brand"].fillna("").str.lower() == parsed.brand.lower()]

    if ranked.empty:
        response = AssistantResponse(
            user_id=user_id,
            interpreted_query=parsed,
            recommendations=[],
            assistant_message="No products matched your filters. Try relaxing the constraints.",
        )
        _append_turn(session, user_message, response.assistant_message)
        return response, session

    ranked = ranked.sort_values("score", ascending=False).head(3).reset_index(drop=True)

    try:
        shap_reasons = explain_predictions(
            pipeline=model_pipeline,
            explainer=_shap_explainer,
            X_input=ranked[ALL_FEATURES],
        )
    except Exception as exc:
        log.warning("SHAP failed in assistant; using fallback reasons. Error: %s", exc)
        shap_reasons = []

    recommendations: list[RecommendationResult] = []
    for i, (_, row) in enumerate(ranked.iterrows()):
        reasons = _resolve_assistant_reasons(row, parsed, shap_reasons, i)
        recommendations.append(
            RecommendationResult(
                product_id=row["product_id"],
                title=row["title"],
                category=row["category"],
                brand=row["brand"],
                price=round(float(row["price"]), 2),
                score=round(float(row["score"]), 4),
                reasons=reasons,
            )
        )

    lines = ["Here are your top recommendations:"]
    for idx, rec in enumerate(recommendations, 1):
        lines.append(
            f"{idx}. {rec.title} ({rec.brand}) — Rs. {rec.price:.0f}  [score: {rec.score:.3f}]"
        )
        for reason in rec.reasons:
            lines.append(f"   • {reason}")

    response = AssistantResponse(
        user_id=user_id,
        interpreted_query=parsed,
        recommendations=recommendations,
        assistant_message="\n".join(lines),
    )

    _append_turn(session, user_message, response.assistant_message)
    return response, session