from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

_products_df: pd.DataFrame | None = None
_features_df: pd.DataFrame | None = None


def get_products_df() -> pd.DataFrame:
    global _products_df
    if _products_df is None:
        _products_df = pd.read_csv(RAW_DIR / "products.csv")
    return _products_df


def get_features_df() -> pd.DataFrame:
    global _features_df
    if _features_df is None:
        _features_df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    return _features_df


def normalize_category(category: str | None) -> str | None:
    if category is None:
        return None

    c = category.lower().strip()

    mapping = {
        "headphones": "Electronics",
        "earbuds": "Electronics",
        "speaker": "Electronics",
        "speakers": "Electronics",
        "electronics": "Electronics",
        "shoes": "Fashion",
        "running shoes": "Fashion",
        "fashion": "Fashion",
        "fmcg": "FMCG",
        "soap": "FMCG",
        "shampoo": "FMCG",
    }
    return mapping.get(c)


def get_candidate_products(
    category: str | None = None,
    max_price: float | None = None,
    brand: str | None = None,
) -> pd.DataFrame:
    df = get_products_df().copy()

    normalized = normalize_category(category)
    if normalized:
        df = df[df["category"].fillna("").str.lower() == normalized.lower()]

    if max_price is not None:
        df = df[df["price"] <= max_price]

    if brand:
        df = df[df["brand"].fillna("").str.lower() == brand.lower()]

    return df.reset_index(drop=True)


def _affordable_flag(category: str, price: float) -> int:
    category = str(category)
    if category == "FMCG" and price < 800:
        return 1
    if category == "Fashion" and price < 3000:
        return 1
    if category == "Electronics" and price < 12000:
        return 1
    return 0


def _build_fallback_rows(user_rows: pd.DataFrame, candidate_products: pd.DataFrame) -> pd.DataFrame:
    """
    Cold-start fallback: synthesises scoreable feature rows for user-product
    pairs that do not exist in the feature store.

    BONUS: Cold-start problem and hybrid recommendation strategy
    ============================================================

    The cold-start problem occurs when a new user has no interaction
    history — no purchases, clicks, or browsing events. A pure
    collaborative filtering model cannot score them because it relies
    entirely on historical user-product signals that do not yet exist.

    This project handles cold-start at two levels:

    1. Partial cold-start (known user, unknown product)
    ---------------------------------------------------
    When a user exists in the feature store but the requested product_id
    has no precomputed row, this function synthesises a feature row by:
      - Taking the user's most recent event row as the user-side context
        (age, city tier, browsing time, avg cart value, etc.)
      - Injecting the candidate product's attributes (category, brand,
        price) directly into that row
      - Recomputing derived features (price_log, price_to_avg_cart_ratio,
        category_affinity, is_affordable) from the combined row

    This gives the XGBoost model a complete, scoreable feature vector
    even for products the user has never interacted with. It is a
    content-based fallback layered inside a collaborative framework —
    the first step toward a hybrid approach.

    2. Proposed hybrid strategy (full cold-start — new user)
    ---------------------------------------------------------
    For a completely new user with no history, the fallback above
    cannot provide user-side signals. The recommended hybrid approach
    is a two-phase blend:

    Phase 1 — Content-based only (0 interactions):
      Recommend products based purely on product attributes — category
      popularity, price tier, and catalog trends. No user signals are
      used. This is equivalent to a "top picks in Electronics under
      Rs. 5000" strategy driven by global product metadata.

    Phase 2 — Collaborative blend (interactions >= threshold):
      As the user accumulates events (clicks, purchases, session time),
      collaborative signals are progressively blended in. The weight
      of content-based vs collaborative features shifts as a function
      of interaction count:

        score = α * content_score + (1 - α) * collaborative_score
        α = max(0, 1 - (interactions / threshold))

      With threshold=10, a user with 3 interactions gets
      α=0.7 (mostly content), a user with 10+ gets α=0 (fully
      collaborative). This ensures smooth degradation rather than
      a hard switch between strategies.

    This hybrid approach directly mirrors how production recommendation
    systems (YouTube, Spotify, Amazon) handle new users — serve
    content-based recommendations immediately, retire them gradually
    as personalisation data accumulates.
    """
    latest_user_row = (
        user_rows.sort_values("event_time", ascending=False)
        .iloc[0]
        .copy()
    )

    if "product_category" in user_rows.columns and not user_rows["product_category"].isna().all():
        inferred_preferred_category = (
            user_rows["product_category"]
            .dropna()
            .value_counts()
            .idxmax()
        )
    else:
        inferred_preferred_category = ""

    rows: list[pd.Series] = []
    for _, product in candidate_products.iterrows():
        row = latest_user_row.copy()

        row["product_id"] = product["product_id"]
        row["title"] = product["title"]
        row["category"] = product["category"]
        row["brand"] = product["brand"]
        row["price"] = float(product["price"])

        row["product_category"] = product["category"]
        row["product_brand"] = product["brand"]

        row["price_log"] = np.log1p(row["price"])

        row["category_affinity"] = int(
            str(inferred_preferred_category).lower() == str(product["category"]).lower()
        )

        row["is_affordable"] = _affordable_flag(product["category"], float(product["price"]))

        avg_cart = float(row.get("avg_cart_value", 1.0))
        if avg_cart <= 0:
            avg_cart = 1.0
        row["price_to_avg_cart_ratio"] = row["price"] / avg_cart

        rows.append(row)

    return pd.DataFrame(rows)


def get_ranked_candidates_for_user(
    user_id: str,
    candidate_products: pd.DataFrame,
) -> pd.DataFrame:
    features_df = get_features_df()
    user_rows = features_df[features_df["user_id"] == user_id].copy()

    if user_rows.empty or candidate_products.empty:
        return pd.DataFrame()

    candidate_ids = candidate_products["product_id"].tolist()

    ranked = user_rows[user_rows["product_id"].isin(candidate_ids)].copy()

    if ranked.empty:
        ranked = _build_fallback_rows(user_rows, candidate_products)

    ranked = ranked.merge(
        get_products_df()[["product_id", "title", "category", "brand", "price"]],
        on="product_id",
        how="left",
        suffixes=("", "_catalog"),
    )

    if "title_catalog" in ranked.columns:
        ranked["title"] = ranked["title_catalog"].combine_first(ranked.get("title"))
    if "category_catalog" in ranked.columns:
        ranked["category"] = ranked["category_catalog"].combine_first(ranked.get("category"))
        ranked["product_category"] = ranked["category_catalog"].combine_first(ranked.get("product_category"))
    if "brand_catalog" in ranked.columns:
        ranked["brand"] = ranked["brand_catalog"].combine_first(ranked.get("brand"))
        ranked["product_brand"] = ranked["brand_catalog"].combine_first(ranked.get("product_brand"))
    if "price_catalog" in ranked.columns:
        ranked["price"] = ranked["price_catalog"].combine_first(ranked.get("price"))

    ranked["price"] = ranked["price"].astype(float)
    ranked["price_log"] = np.log1p(ranked["price"])

    avg_cart = ranked["avg_cart_value"].fillna(1.0).replace(0, 1.0)
    ranked["price_to_avg_cart_ratio"] = ranked["price"] / avg_cart
    ranked["is_affordable"] = ranked.apply(
        lambda row: _affordable_flag(row["product_category"], float(row["price"])),
        axis=1,
    )

    ranked["category_affinity"] = (
        ranked["product_category"].fillna("").str.lower()
        == user_rows["product_category"].dropna().mode().iloc[0].lower()
        if "product_category" in user_rows.columns and not user_rows["product_category"].dropna().empty
        else False
    ).astype(int)

    ranked = ranked.drop(
        columns=[c for c in ranked.columns if c.endswith("_catalog")],
        errors="ignore",
    )

    ranked = ranked.drop_duplicates(subset=["user_id", "product_id"]).reset_index(drop=True)

    return ranked


def filter_products_by_query_text(candidate_products: pd.DataFrame, raw_category: str | None) -> pd.DataFrame:
    if raw_category is None:
        return candidate_products

    q = raw_category.lower().strip()

    keyword_map = {
        "headphones": ["headphone", "headphones", "earbuds", "earbud"],
        "earbuds": ["earbud", "earbuds", "headphone", "headphones"],
        "running shoes": ["shoe", "shoes", "sneaker", "sneakers"],
        "shoes": ["shoe", "shoes", "sneaker", "sneakers"],
        "soap": ["soap"],
        "shampoo": ["shampoo"],
    }

    keywords = keyword_map.get(q)
    if not keywords:
        return candidate_products

    mask = candidate_products["title"].fillna("").str.lower().apply(
        lambda title: any(k in title for k in keywords)
    )
    filtered = candidate_products[mask].copy()
    return filtered.reset_index(drop=True)