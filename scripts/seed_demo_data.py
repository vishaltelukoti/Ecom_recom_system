# scripts/seed_demo_data.py

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
random.seed(SEED)
np.random.seed(SEED)


@dataclass
class Config:
    n_users: int = 45
    n_products: int = 75
    n_browsing_events: int = 260
    purchase_rate_target: float = 0.20
    start_date: str = "2025-09-01"
    end_date: str = "2026-03-20"


CONFIG = Config()


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def random_dates(start: datetime, end: datetime, n: int) -> list[datetime]:
    delta = end - start
    return [start + timedelta(seconds=random.randint(0, int(delta.total_seconds()))) for _ in range(n)]


def build_users(n_users: int) -> pd.DataFrame:
    user_ids = [f"U_{i:03d}" for i in range(1, n_users + 1)]

    ages = np.random.randint(18, 61, size=n_users)

    city_tiers = np.random.choice(
        ["Tier-1", "Tier-2", "Tier-3"],
        size=n_users,
        p=[0.34, 0.33, 0.33],
    )

    genders = np.random.choice(
        ["Male", "Female", "Other"],
        size=n_users,
        p=[0.47, 0.47, 0.06],
    )

    preferred_categories = np.random.choice(
        ["Fashion", "Electronics", "FMCG"],
        size=n_users,
        p=[0.34, 0.33, 0.33],
    )

    users = pd.DataFrame(
        {
            "user_id": user_ids,
            "age": ages,
            "city_tier": city_tiers,
            "gender": genders,
            "preferred_category": preferred_categories,
        }
    )
    return users


def build_products(n_products: int) -> pd.DataFrame:
    categories = (
        ["Fashion"] * 25 +
        ["Electronics"] * 25 +
        ["FMCG"] * 25
    )
    categories = categories[:n_products]
    random.shuffle(categories)

    fashion_brands = ["StyleHub", "UrbanWeave", "TrendCraft", "ModaLeaf"]
    electronics_brands = ["Sony", "Samsung", "Boat", "JBL", "Xiaomi"]
    fmcg_brands = ["Nestle", "Dove", "Patanjali", "HUL", "ITC"]

    product_rows = []
    for idx in range(1, n_products + 1):
        category = categories[idx - 1]

        if category == "Fashion":
            brand = random.choice(fashion_brands)
            price = round(np.random.uniform(500, 4500), 2)
            title = random.choice(
                ["Running Shoes", "T-Shirt", "Jeans", "Jacket", "Sneakers", "Kurta"]
            )
        elif category == "Electronics":
            brand = random.choice(electronics_brands)
            price = round(np.random.uniform(1500, 25000), 2)
            title = random.choice(
                ["Headphones", "Smart Watch", "Bluetooth Speaker", "Power Bank", "Earbuds"]
            )
        else:
            brand = random.choice(fmcg_brands)
            price = round(np.random.uniform(80, 1200), 2)
            title = random.choice(
                ["Shampoo", "Face Wash", "Protein Bar", "Tea Pack", "Soap", "Toothpaste"]
            )

        product_rows.append(
            {
                "product_id": f"P_{idx:03d}",
                "title": f"{brand} {title}",
                "category": category,
                "brand": brand,
                "price": price,
            }
        )

    return pd.DataFrame(product_rows)


def choose_product_for_user(products: pd.DataFrame, preferred_category: str) -> pd.Series:
    same_cat = products[products["category"] == preferred_category]
    other_cat = products[products["category"] != preferred_category]

    if random.random() < 0.60:
        return same_cat.sample(1, random_state=np.random.randint(0, 10_000)).iloc[0]
    return other_cat.sample(1, random_state=np.random.randint(0, 10_000)).iloc[0]


def session_duration_by_category(category: str) -> float:
    if category == "Electronics":
        value = np.random.exponential(scale=9.0) + 2.0
    elif category == "Fashion":
        value = np.random.exponential(scale=7.0) + 1.5
    else:
        value = np.random.exponential(scale=4.0) + 0.5
    return round(min(value, 45.0), 2)


def event_type_from_duration(duration: float) -> str:
    click_prob = 0.18
    if duration > 10:
        click_prob = 0.35
    if duration > 18:
        click_prob = 0.50
    return np.random.choice(["view", "click"], p=[1 - click_prob, click_prob])


def build_browsing_events(
    users: pd.DataFrame,
    products: pd.DataFrame,
    n_events: int,
    start_dt: datetime,
    end_dt: datetime,
) -> pd.DataFrame:
    rows = []
    event_times = random_dates(start_dt, end_dt, n_events)

    for i in range(n_events):
        user = users.sample(1, random_state=np.random.randint(0, 10_000)).iloc[0]
        product = choose_product_for_user(products, user["preferred_category"])
        duration = session_duration_by_category(product["category"])
        event_type = event_type_from_duration(duration)

        rows.append(
            {
                "event_id": f"E_{i + 1:04d}",
                "user_id": user["user_id"],
                "product_id": product["product_id"],
                "event_type": event_type,
                "event_time": event_times[i].strftime("%Y-%m-%d %H:%M:%S"),
                "session_duration": duration,
            }
        )

    browsing = pd.DataFrame(rows).sort_values("event_time").reset_index(drop=True)
    return browsing


def build_transactions(
    browsing: pd.DataFrame,
    users: pd.DataFrame,
    products: pd.DataFrame,
    purchase_rate_target: float,
) -> pd.DataFrame:
    user_map = users.set_index("user_id").to_dict("index")
    product_map = products.set_index("product_id").to_dict("index")

    transaction_rows = []

    for _, row in browsing.iterrows():
        user = user_map[row["user_id"]]
        product = product_map[row["product_id"]]

        affinity = 1 if user["preferred_category"] == product["category"] else 0
        duration = float(row["session_duration"])
        clicked = 1 if row["event_type"] == "click" else 0

        if product["category"] == "Electronics":
            category_weight = 0.18
        elif product["category"] == "Fashion":
            category_weight = 0.14
        else:
            category_weight = 0.12

        price = float(product["price"])
        affordable = 1 if (
            (product["category"] == "FMCG" and price < 800)
            or (product["category"] == "Fashion" and price < 3000)
            or (product["category"] == "Electronics" and price < 12000)
        ) else 0

        score = (
            -2.2
            + 0.95 * clicked
            + 0.08 * min(duration, 20)
            + 0.70 * affinity
            + 0.45 * affordable
            + category_weight
        )

        probability = 1 / (1 + np.exp(-score))

        # Calibrate towards small-demo-friendly purchase rate
        probability = max(0.03, min(probability * 0.42, 0.75))

        purchase_flag = np.random.binomial(1, probability)

        if purchase_flag == 1:
            event_time = pd.to_datetime(row["event_time"])
            purchase_date = event_time + timedelta(
                hours=random.randint(2, 96)
            )

            quantity = 1
            if product["category"] == "FMCG":
                quantity = np.random.choice([1, 2, 3], p=[0.55, 0.30, 0.15])

            transaction_rows.append(
                {
                    "transaction_id": f"T_{len(transaction_rows) + 1:04d}",
                    "user_id": row["user_id"],
                    "product_id": row["product_id"],
                    "purchase_date": purchase_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "price": price,
                    "quantity": int(quantity),
                }
            )

    transactions = pd.DataFrame(transaction_rows)

    # If purchase count drifts too low/high, lightly rebalance by sampling
    desired_count = int(len(browsing) * purchase_rate_target)
    if len(transactions) > desired_count + 15:
        transactions = transactions.sample(desired_count + 15, random_state=SEED).sort_values("purchase_date")
    elif len(transactions) < max(25, desired_count - 10):
        needed = max(25, desired_count - 10) - len(transactions)
        non_purchased = browsing.merge(
            transactions[["user_id", "product_id"]],
            on=["user_id", "product_id"],
            how="left",
            indicator=True,
        )
        non_purchased = non_purchased[non_purchased["_merge"] == "left_only"].drop(columns="_merge")

        extras = []
        for _, row in non_purchased.sample(min(needed, len(non_purchased)), random_state=SEED).iterrows():
            product = product_map[row["product_id"]]
            event_time = pd.to_datetime(row["event_time"])
            purchase_date = event_time + timedelta(hours=random.randint(2, 72))
            extras.append(
                {
                    "transaction_id": f"T_{len(transactions) + len(extras) + 1:04d}",
                    "user_id": row["user_id"],
                    "product_id": row["product_id"],
                    "purchase_date": purchase_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "price": float(product["price"]),
                    "quantity": 1,
                }
            )
        if extras:
            transactions = pd.concat([transactions, pd.DataFrame(extras)], ignore_index=True)

    transactions = transactions.drop_duplicates(subset=["user_id", "product_id", "purchase_date"]).reset_index(drop=True)
    return transactions


def validate_balance(users: pd.DataFrame, products: pd.DataFrame, browsing: pd.DataFrame, transactions: pd.DataFrame) -> None:
    print("\n===== DATA QUALITY SUMMARY =====")
    print(f"Users: {len(users)}")
    print(f"Products: {len(products)}")
    print(f"Browsing events: {len(browsing)}")
    print(f"Transactions: {len(transactions)}")

    print("\nUser city tier distribution:")
    print(users["city_tier"].value_counts(normalize=True).round(3))

    print("\nUser preferred category distribution:")
    print(users["preferred_category"].value_counts(normalize=True).round(3))

    print("\nProduct category distribution:")
    print(products["category"].value_counts(normalize=True).round(3))

    print("\nBrowsing event distribution:")
    print(browsing["event_type"].value_counts(normalize=True).round(3))

    merged = browsing.merge(
        transactions[["user_id", "product_id"]].assign(purchased=1),
        on=["user_id", "product_id"],
        how="left",
    )
    merged["purchased"] = merged["purchased"].fillna(0).astype(int)

    print("\nPurchase rate over browsing rows:")
    print(round(merged["purchased"].mean(), 3))

    purchase_by_tier = (
        merged.merge(users[["user_id", "city_tier"]], on="user_id", how="left")
        .groupby("city_tier")["purchased"]
        .mean()
        .round(3)
    )
    print("\nPurchase rate by city tier:")
    print(purchase_by_tier)

    purchase_by_pref = (
        merged.merge(users[["user_id", "preferred_category"]], on="user_id", how="left")
        .groupby("preferred_category")["purchased"]
        .mean()
        .round(3)
    )
    print("\nPurchase rate by preferred category:")
    print(purchase_by_pref)

    print("\nSession duration summary:")
    print(browsing["session_duration"].describe().round(2))


def save_dataframes(users: pd.DataFrame, products: pd.DataFrame, browsing: pd.DataFrame, transactions: pd.DataFrame) -> None:
    users.to_csv(RAW_DIR / "users.csv", index=False)
    products.to_csv(RAW_DIR / "products.csv", index=False)
    browsing.to_csv(RAW_DIR / "browsing_events.csv", index=False)
    transactions.to_csv(RAW_DIR / "transactions.csv", index=False)
    print(f"\nFiles saved to: {RAW_DIR}")


def main() -> None:
    start_dt = datetime.strptime(CONFIG.start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(CONFIG.end_date, "%Y-%m-%d")

    users = build_users(CONFIG.n_users)
    products = build_products(CONFIG.n_products)
    browsing = build_browsing_events(
        users=users,
        products=products,
        n_events=CONFIG.n_browsing_events,
        start_dt=start_dt,
        end_dt=end_dt,
    )
    transactions = build_transactions(
        browsing=browsing,
        users=users,
        products=products,
        purchase_rate_target=CONFIG.purchase_rate_target,
    )

    validate_balance(users, products, browsing, transactions)
    save_dataframes(users, products, browsing, transactions)


if __name__ == "__main__":
    main()