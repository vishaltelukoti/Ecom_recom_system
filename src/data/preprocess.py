from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw"
INTERIM_DIR = BASE_DIR / "data" / "interim"
INTERIM_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_data(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    users = pd.read_csv(raw_dir / "users.csv")
    products = pd.read_csv(raw_dir / "products.csv")
    browsing = pd.read_csv(raw_dir / "browsing_events.csv")
    transactions = pd.read_csv(raw_dir / "transactions.csv")
    return users, products, browsing, transactions


def clean_users(users: pd.DataFrame) -> pd.DataFrame:
    df = users.copy()

    df = df.drop_duplicates(subset=["user_id"]).reset_index(drop=True)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["age"] = df["age"].fillna(df["age"].median()).clip(lower=18, upper=80)

    df["city_tier"] = df["city_tier"].fillna("Tier-2")
    df["gender"] = df["gender"].fillna("Unknown")
    df["preferred_category"] = df["preferred_category"].fillna("Fashion")

    return df


def clean_products(products: pd.DataFrame) -> pd.DataFrame:
    df = products.copy()

    df = df.drop_duplicates(subset=["product_id"]).reset_index(drop=True)
    df["title"] = df["title"].fillna("Unknown Product")
    df["category"] = df["category"].fillna("Fashion")
    df["brand"] = df["brand"].fillna("Unknown")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["price"] = df["price"].fillna(df["price"].median()).clip(lower=1)

    return df


def clean_browsing_events(browsing: pd.DataFrame) -> pd.DataFrame:
    df = browsing.copy()

    df = df.drop_duplicates(subset=["event_id"]).reset_index(drop=True)
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df = df.dropna(subset=["user_id", "product_id", "event_time"]).reset_index(drop=True)

    df["event_type"] = df["event_type"].fillna("view").str.lower()
    df.loc[~df["event_type"].isin(["view", "click"]), "event_type"] = "view"

    df["session_duration"] = pd.to_numeric(df["session_duration"], errors="coerce")
    df["session_duration"] = df["session_duration"].fillna(df["session_duration"].median())
    df["session_duration"] = df["session_duration"].clip(lower=0.1, upper=60)

    return df.sort_values("event_time").reset_index(drop=True)


def clean_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    df = transactions.copy()

    df = df.drop_duplicates(subset=["transaction_id"]).reset_index(drop=True)
    df["purchase_date"] = pd.to_datetime(df["purchase_date"], errors="coerce")
    df = df.dropna(subset=["user_id", "product_id", "purchase_date"]).reset_index(drop=True)

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["price"] = df["price"].fillna(df["price"].median()).clip(lower=1)

    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["quantity"] = df["quantity"].fillna(1).clip(lower=1).astype(int)

    return df.sort_values("purchase_date").reset_index(drop=True)


def validate_foreign_keys(
    users: pd.DataFrame,
    products: pd.DataFrame,
    browsing: pd.DataFrame,
    transactions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_users = set(users["user_id"])
    valid_products = set(products["product_id"])

    browsing = browsing[
        browsing["user_id"].isin(valid_users) & browsing["product_id"].isin(valid_products)
    ].reset_index(drop=True)

    transactions = transactions[
        transactions["user_id"].isin(valid_users) & transactions["product_id"].isin(valid_products)
    ].reset_index(drop=True)

    return browsing, transactions


def build_base_table(
    users: pd.DataFrame,
    products: pd.DataFrame,
    browsing: pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    df = browsing.merge(users, on="user_id", how="left")
    df = df.merge(products, on="product_id", how="left", suffixes=("", "_product"))

    # target: whether this browsed product was purchased by the same user within 7 days
    tx = transactions[["user_id", "product_id", "purchase_date", "price", "quantity"]].copy()

    df = df.merge(
        tx,
        on=["user_id", "product_id"],
        how="left",
        suffixes=("", "_txn"),
    )

    df["days_to_purchase"] = (df["purchase_date"] - df["event_time"]).dt.total_seconds() / (24 * 3600)

    df["purchased_within_7_days"] = (
        df["days_to_purchase"].between(0, 7, inclusive="both")
    ).fillna(False).astype(int)

    return df


def save_outputs(
    users: pd.DataFrame,
    products: pd.DataFrame,
    browsing: pd.DataFrame,
    transactions: pd.DataFrame,
    base_table: pd.DataFrame,
) -> None:
    users.to_parquet(INTERIM_DIR / "users_clean.parquet", index=False)
    products.to_parquet(INTERIM_DIR / "products_clean.parquet", index=False)
    browsing.to_parquet(INTERIM_DIR / "browsing_clean.parquet", index=False)
    transactions.to_parquet(INTERIM_DIR / "transactions_clean.parquet", index=False)
    base_table.to_parquet(INTERIM_DIR / "base_table.parquet", index=False)


def run_preprocessing() -> pd.DataFrame:
    users, products, browsing, transactions = load_raw_data()

    users = clean_users(users)
    products = clean_products(products)
    browsing = clean_browsing_events(browsing)
    transactions = clean_transactions(transactions)

    browsing, transactions = validate_foreign_keys(users, products, browsing, transactions)

    base_table = build_base_table(users, products, browsing, transactions)
    save_outputs(users, products, browsing, transactions, base_table)

    print("Preprocessing complete.")
    print(f"Users: {users.shape}")
    print(f"Products: {products.shape}")
    print(f"Browsing: {browsing.shape}")
    print(f"Transactions: {transactions.shape}")
    print(f"Base table: {base_table.shape}")
    print(f"Positive rate: {base_table['purchased_within_7_days'].mean():.3f}")

    return base_table


if __name__ == "__main__":
    run_preprocessing()