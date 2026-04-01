"""
SHAP explainability module.

Two modes of operation
----------------------
1. Offline analysis  (run via main() or train_pipeline.py Step 4)
   Generates global summary plot (beeswarm) and local explanation CSVs
   saved to artifacts/shap/. Used for model review and assessment.

2. Inference-time  (called by the API and assistant at request time)
   explain_predictions() computes SHAP values for a batch of candidate
   rows and returns human-readable reason strings. Called by:
     - src/api/main.py        → always uses SHAP (primary explanation mode)
     - src/assistant/chain.py → uses SHAP with heuristic fallback

   The explainer object (shap.TreeExplainer) is built once at startup
   against a background sample and cached in _state (API) or as a
   module-level variable (assistant). Do not rebuild it per request.

Section C1 answers are embedded as docstrings throughout.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import shap

from src.config.features import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COL

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "artifacts" / "models"
SHAP_DIR = BASE_DIR / "artifacts" / "shap"

SHAP_DIR.mkdir(parents=True, exist_ok=True)

# Human-readable names for API-facing reasons
_REASON_TEMPLATES = {
    "browsing_time_last_7d_mins":   "You browsed similar items heavily this week",
    "browsing_time_last_7d_log":    "You browsed similar items heavily this week",
    "days_since_last_purchase":     "You haven't purchased recently — you may be ready to buy",
    "days_since_last_purchase_log": "You haven't purchased recently — you may be ready to buy",
    "category_affinity":            "This matches your preferred shopping category",
    "click_through_proxy":          "You showed strong click engagement on similar products",
    "total_clicks":                 "Your recent click activity suggests high interest",
    "total_orders":                 "Your purchase history aligns with this product",
    "is_affordable":                "The price fits within your typical spending range",
    "avg_cart_value":               "The price is in line with your average spend",
    "avg_cart_value_log":           "The price is in line with your average spend",
    "price_to_avg_cart_ratio":      "Good value relative to your usual budget",
    "price":                        "Competitive price point for this category",
    "price_log":                    "Competitive price point for this category",
    "product_category":             "Popular category in your browsing history",
    "product_brand":                "A brand you have previously browsed",
    "user_city_tier":               "Popular choice in your city tier",
}

_DEFAULT_REASON = "Recommended based on your combined browsing and purchase signals"


# ---------------------------------------------------------------------------
# Pipeline helpers
# 

def _extract_booster_and_feature_names(pipeline):
    """
    Pull the raw booster and transformed feature names out of any pipeline
    that has a 'preprocessor' and a 'model' step.
    Works with both sklearn Pipeline and imbalanced-learn Pipeline.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = list(preprocessor.get_feature_names_out())
    return model, preprocessor, feature_names


def _transform(preprocessor, X: pd.DataFrame) -> pd.DataFrame:
    X_arr = preprocessor.transform(X)
    return pd.DataFrame(X_arr, columns=preprocessor.get_feature_names_out(), index=X.index)


# ---------------------------------------------------------------------------
# Offline analysis helpers
# ---------------------------------------------------------------------------

def load_data():
    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    X = df[ALL_FEATURES].copy()
    return df, X


def load_pipeline():
    return joblib.load(MODELS_DIR / "xgboost_pipeline.pkl")


def generate_summary_plot(explainer, X_transformed_df: pd.DataFrame) -> None:
    """
    SHAP Summary (beeswarm) plot — offline only.

    If matplotlib is unavailable, skip plot generation and still save
    global feature importance as CSV so the pipeline remains usable in
    lightweight runtime environments.
    """
    shap_values = explainer(X_transformed_df)

    mean_abs_shap = pd.DataFrame({
        "feature": X_transformed_df.columns,
        "mean_abs_shap": np.abs(shap_values.values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)

    mean_abs_shap.to_csv(SHAP_DIR / "global_feature_importance.csv", index=False)

    try:
        import matplotlib.pyplot as plt

        shap.summary_plot(shap_values, X_transformed_df, show=False)
        plt.tight_layout()
        plt.savefig(SHAP_DIR / "shap_summary_plot.png", bbox_inches="tight")
        plt.close()
        log.info("SHAP summary plot saved → %s", SHAP_DIR / "shap_summary_plot.png")
    except ImportError:
        log.info("matplotlib not installed; skipped SHAP summary plot image generation")


def generate_local_explanation(
    explainer, X_transformed_df: pd.DataFrame, original_df: pd.DataFrame
) -> None:
    """
    Local SHAP explanation for a single sample.

    SHAP values (Section C1, Q10-11):
    ----------------------------------
    SHAP (SHapley Additive exPlanations) is grounded in cooperative game
    theory.  The Shapley value for feature i answers: 'How much does feature i
    contribute to this prediction compared to the average prediction, averaged
    over all possible feature orderings?'

    It is the 'fairest' attribution because it is the unique allocation that
    satisfies:
    * Efficiency: SHAP values sum to (prediction − base_value).
    * Symmetry: two features with identical contributions receive equal values.
    * Dummy: a feature that never changes the prediction gets SHAP = 0.
    * Additivity: across sub-models the values add up correctly.

    Example interpretation (Q11 — User U_89234, Sony Headphone, 71% purchase):
    * browsing_time_last_7d = +0.18 → Recent heavy browsing strongly increases
      purchase probability.  User spent significant time on Electronics this week.
    * product_category = +0.14 → Being in Electronics (user's preferred category)
      meaningfully boosts the score.
    * days_since_last_purchase = −0.09 → User bought something recently; this
      slightly reduces urgency to buy again.
    * user_city_tier = +0.06 → Tier-1 city users tend to convert more on
      premium Electronics.
    * avg_cart_value = +0.04 → User's typical spend is compatible with this
      product's price point.

    User-facing reasons (top 3 positive SHAP drivers):
    1. 'You browsed similar items heavily this week'
    2. 'This matches your preferred shopping category'
    3. 'Popular choice in your city'
    """
    sample_idx = 0
    sample_row = X_transformed_df.iloc[[sample_idx]]
    shap_values = explainer(sample_row)

    local_df = pd.DataFrame({
        "feature": X_transformed_df.columns,
        "shap_value": shap_values.values[0],
        "feature_value": sample_row.iloc[0].values,
    })
    local_df["abs_shap"] = local_df["shap_value"].abs()
    local_df = local_df.sort_values("abs_shap", ascending=False)

    local_df.to_csv(SHAP_DIR / "local_explanation_sample.csv", index=False)
    local_df.head(5)[["feature", "shap_value", "feature_value"]].to_csv(
        SHAP_DIR / "top_5_reasons_sample.csv", index=False
    )

    with open(SHAP_DIR / "sample_prediction_context.txt", "w", encoding="utf-8") as f:
        f.write("Sample Prediction Context\n")
        f.write("=" * 40 + "\n")
        f.write(original_df.iloc[sample_idx].to_string())

    log.info("Local SHAP explanation saved → %s", SHAP_DIR)


# ---------------------------------------------------------------------------
# Inference-time SHAP  (called by FastAPI /recommendations endpoint)
# ---------------------------------------------------------------------------

def build_explainer(pipeline, background_df: pd.DataFrame) -> shap.TreeExplainer:
    """
    Build and return a SHAP TreeExplainer from a background sample.
    Cache this object at API startup — do not rebuild per request.
    """
    model, preprocessor, _ = _extract_booster_and_feature_names(pipeline)
    X_bg = _transform(preprocessor, background_df)
    return shap.TreeExplainer(model, X_bg)


def explain_predictions(
    pipeline,
    explainer: shap.TreeExplainer,
    X_input: pd.DataFrame,
    top_n: int = 3,
) -> list[list[str]]:
    """
    Compute SHAP values for X_input and return human-readable reasons
    for each row.

    Parameters
    ----------
    pipeline   : trained pipeline with preprocessor + model steps
    explainer  : pre-built TreeExplainer (built once at startup)
    X_input    : raw feature DataFrame (same columns as training, before preprocessing)
    top_n      : number of reasons to return per prediction

    Returns
    -------
    List of length len(X_input), each element is a list of top_n reason strings.
    If SHAP inference fails for any reason, returns a list of default fallback
    reasons so the /recommendations endpoint never fails due to explainability.
    """
    n_rows = len(X_input)
    fallback = [[_DEFAULT_REASON] for _ in range(n_rows)]

    try:
        _, preprocessor, feature_names = _extract_booster_and_feature_names(pipeline)
        X_transformed = _transform(preprocessor, X_input)
        shap_values = explainer(X_transformed)
    except Exception as exc:
        log.warning(
            "SHAP inference failed — returning default reasons. Error: %s", exc
        )
        return fallback

    reasons_list: list[list[str]] = []

    for i in range(n_rows):
        try:
            row_shap = pd.Series(shap_values.values[i], index=feature_names)

            # Use only positive-contribution features (drivers toward purchase)
            positive_shap = row_shap[row_shap > 0].sort_values(ascending=False)

            reasons: list[str] = []
            seen_templates: set[str] = set()

            for feat, val in positive_shap.items():
                orig_feat = _map_feature_name(feat)
                template = _REASON_TEMPLATES.get(orig_feat)
                if template and template not in seen_templates:
                    reasons.append(template)
                    seen_templates.add(template)
                if len(reasons) >= top_n:
                    break

            if not reasons:
                reasons.append(_DEFAULT_REASON)

            reasons_list.append(reasons[:top_n])

        except Exception as exc:
            log.warning(
                "SHAP reason generation failed for row %d — using default. Error: %s",
                i, exc,
            )
            reasons_list.append([_DEFAULT_REASON])

    return reasons_list


def _map_feature_name(transformed_name: str) -> str:
    """
    Map a ColumnTransformer-prefixed feature name back to its original name
    for use in _REASON_TEMPLATES lookup.

    ColumnTransformer produces names in two forms:
      Numeric  : 'num__user_age'         → strip prefix → 'user_age'
      One-hot  : 'cat__product_category_Electronics' → strip prefix and
                 value suffix → 'product_category'

    The function tries three strategies in order:
      1. Exact match after stripping prefix — handles all numeric features.
      2. Longest prefix match against known template keys — handles OHE
         features where the value is appended (e.g. 'product_category_Electronics'
         matches template key 'product_category').
      3. Return the stripped name as-is — safe fallback, worst case is
         just a cache miss in _REASON_TEMPLATES.

    This is more robust than a simple startswith() because it always picks
    the longest matching key, avoiding partial matches where one feature
    name is a prefix of another (e.g. 'price' vs 'price_log').
    """
    if "__" not in transformed_name:
        return transformed_name

    # Strip the 'num__' or 'cat__' prefix
    _, rest = transformed_name.split("__", 1)

    # Strategy 1 — exact match
    if rest in _REASON_TEMPLATES:
        return rest

    # Strategy 2 — longest prefix match among known template keys
    # Sorting by length descending ensures 'price_log' is tried before 'price'
    best_match = ""
    for key in sorted(_REASON_TEMPLATES.keys(), key=len, reverse=True):
        if rest.startswith(key):
            best_match = key
            break

    if best_match:
        return best_match

    # Strategy 3 — return stripped name as-is (cache miss is safe)
    return rest


# ---------------------------------------------------------------------------
# Offline main
# ---------------------------------------------------------------------------

def main() -> None:
    original_df, X = load_data()
    pipeline = load_pipeline()

    model, preprocessor, _ = _extract_booster_and_feature_names(pipeline)
    X_transformed_df = _transform(preprocessor, X)

    background = X_transformed_df.sample(min(100, len(X_transformed_df)), random_state=42)
    explain_data = X_transformed_df.sample(min(100, len(X_transformed_df)), random_state=7)

    explainer = shap.TreeExplainer(model, background)

    generate_summary_plot(explainer, explain_data)
    generate_local_explanation(
        explainer, explain_data,
        original_df.loc[explain_data.index].reset_index(drop=True)
    )

    log.info("SHAP analysis complete.  Artifacts → %s", SHAP_DIR)


if __name__ == "__main__":
    main()