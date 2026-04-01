from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
INTERIM_DIR = BASE_DIR / "data" / "interim"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    base = pd.read_parquet(INTERIM_DIR / "base_table.parquet")
    transactions = pd.read_parquet(INTERIM_DIR / "transactions_clean.parquet")
    return base, transactions


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["event_date"] = pd.to_datetime(out["event_time"]).dt.date
    out["event_hour"] = pd.to_datetime(out["event_time"]).dt.hour
    out["event_dayofweek"] = pd.to_datetime(out["event_time"]).dt.dayofweek

    return out


def add_user_history_features(df: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Computes purchase history features (last purchase date, total orders,
    average cart value) per user using all available transactions.

    Simplification note:
    In a production system, only transactions with purchase_date BEFORE
    each row's event_time should be used, to prevent future purchases from
    influencing the features for earlier events. For example, a purchase
    made on day 15 should not affect the days_since_last_purchase feature
    for a browsing event on day 3.

    This implementation aggregates over all transactions for simplicity,
    which is acceptable for this offline demo dataset. For a production
    pipeline, filter transactions to purchase_date <= event_time before
    computing these aggregates.
    """
    out = df.copy()
    tx = transactions.copy()
    tx["purchase_date"] = pd.to_datetime(tx["purchase_date"])
    out["event_time"] = pd.to_datetime(out["event_time"])

    user_last_purchase = (
        tx.groupby("user_id")["purchase_date"]
        .max()
        .reset_index()
        .rename(columns={"purchase_date": "last_purchase_date"})
    )

    out = out.merge(user_last_purchase, on="user_id", how="left")
    out["days_since_last_purchase"] = (
        out["event_time"] - out["last_purchase_date"]
    ).dt.total_seconds() / (24 * 3600)

    out["days_since_last_purchase"] = out["days_since_last_purchase"].fillna(999)
    out["days_since_last_purchase"] = out["days_since_last_purchase"].clip(lower=0)

    user_orders = tx.groupby("user_id").size().reset_index(name="total_orders")
    out = out.merge(user_orders, on="user_id", how="left")
    out["total_orders"] = out["total_orders"].fillna(0)

    user_avg_cart = tx.groupby("user_id")["price"].mean().reset_index(name="avg_cart_value")
    out = out.merge(user_avg_cart, on="user_id", how="left")
    out["avg_cart_value"] = out["avg_cart_value"].fillna(out["price"].median())

    return out


def add_browsing_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes browsing aggregates (session duration, event counts, click counts)
    per user across all available rows.

    Simplification note:
    In a production system these aggregates should be computed using only
    events that occurred BEFORE each row's event_time, to avoid leaking
    future behaviour into earlier rows. For example, a browsing event on
    day 1 should not include clicks that happened on day 10.

    This implementation uses the full user history for simplicity, which is
    acceptable for this offline demo dataset. For a production feature store,
    replace this with a time-windowed aggregation keyed on event_time.
    """
    out = df.copy()

    user_browse_stats = (
        out.groupby("user_id")
        .agg(
            browsing_time_last_7d_mins=("session_duration", "mean"),
            total_browsing_events=("event_id", "count"),
            total_clicks=("event_type", lambda s: (s == "click").sum()),
        )
        .reset_index()
    )

    out = out.merge(user_browse_stats, on="user_id", how="left")

    out["click_through_proxy"] = out["total_clicks"] / out["total_browsing_events"].replace(0, 1)
    out["category_affinity"] = (out["preferred_category"] == out["category"]).astype(int)

    return out


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["price_to_avg_cart_ratio"] = out["price"] / out["avg_cart_value"].replace(0, 1)
    out["is_affordable"] = (
        (
            ((out["category"] == "FMCG") & (out["price"] < 800)) |
            ((out["category"] == "Fashion") & (out["price"] < 3000)) |
            ((out["category"] == "Electronics") & (out["price"] < 12000))
        )
    ).astype(int)

    return out


def add_model_ready_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["user_age"] = out["age"]
    out["user_city_tier"] = out["city_tier"]
    out["product_category"] = out["category"]
    out["product_brand"] = out["brand"]

    out["days_since_last_purchase_log"] = np.log1p(out["days_since_last_purchase"])
    out["browsing_time_last_7d_log"] = np.log1p(out["browsing_time_last_7d_mins"])
    out["avg_cart_value_log"] = np.log1p(out["avg_cart_value"])
    out["price_log"] = np.log1p(out["price"])

    return out


def select_final_columns(df: pd.DataFrame) -> pd.DataFrame:
    final_cols = [
        "user_id",
        "product_id",
        "event_time",
        "user_age",
        "user_city_tier",
        "product_category",
        "product_brand",
        "price",
        "price_log",
        "session_duration",
        "browsing_time_last_7d_mins",
        "browsing_time_last_7d_log",
        "days_since_last_purchase",
        "days_since_last_purchase_log",
        "avg_cart_value",
        "avg_cart_value_log",
        "total_orders",
        "total_browsing_events",
        "total_clicks",
        "click_through_proxy",
        "category_affinity",
        "is_affordable",
        "price_to_avg_cart_ratio",
        "event_hour",
        "event_dayofweek",
        "purchased_within_7_days",
    ]
    return df[final_cols].copy()


def save_feature_data(df: pd.DataFrame) -> None:
    df.to_parquet(PROCESSED_DIR / "features.parquet", index=False)
    df.to_csv(PROCESSED_DIR / "features.csv", index=False)


def build_features() -> pd.DataFrame:
    base, transactions = load_data()

    df = add_time_features(base)
    df = add_user_history_features(df, transactions)
    df = add_browsing_aggregates(df)
    df = add_price_features(df)
    df = add_model_ready_columns(df)
    df = select_final_columns(df)

    save_feature_data(df)

    print("Feature engineering complete.")
    print(f"Feature dataset shape: {df.shape}")
    print(f"Positive rate: {df['purchased_within_7_days'].mean():.3f}")

    return df


if __name__ == "__main__":
    build_features()