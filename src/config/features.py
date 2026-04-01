"""
Single source of truth for all feature column definitions.

Every module that needs feature lists imports from here.
Adding, removing, or renaming a feature only requires a change in this file.
"""
from __future__ import annotations

TARGET_COL = "purchased_within_7_days"

NUMERIC_FEATURES: list[str] = [
    "user_age",
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
]

CATEGORICAL_FEATURES: list[str] = [
    "user_city_tier",
    "product_category",
    "product_brand",
]

ALL_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES