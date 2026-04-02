# E-Commerce Recommendation Engine

This project implements a production-style **AI-powered recommendation system** for an e-commerce platform.

The system predicts purchase probability, ranks products, and provides **explainable recommendations using SHAP**, along with a **LangChain-powered query parsing module** for natural language constraint extraction.

It integrates *Machine Learning*, *Explainable AI (SHAP)*, *LLM-based query parsing*, *REST APIs*, and *Docker deployment* to simulate a real-world recommendation system.

---

# Key Features

* Purchase propensity prediction using XGBoost
* Feature engineering with Scikit-learn pipelines
* SHAP-based explainability ("Why we recommend this")
* LangChain + Groq powered query parsing for structured constraint extraction
* FastAPI-based REST service
* Docker & Docker Compose deployment
* Unit testing with pytest

---

## Assessment Requirement Mapping

| Question | File | Location |
|----------|------|----------|
| Purchase propensity model | `src/models/train_xgboost.py` | `train()` function |
| Reproducible pipeline | `src/pipelines/train_pipeline.py` | `run()` function |
| SHAP explanations | `src/explainability/shap_explainer.py` | module |
| Conversational assistant (module) | `src/assistant/chain.py` | `run_assistant()` |
| FastAPI deployment | `src/api/main.py` | module |
| Docker Compose deployment | `docker-compose.yml` | — |
| B2: GOSS & EFB explanation | `src/pipelines/train_pipeline.py` | module docstring |
| Q15: Memory type comparison | `src/assistant/chain.py` | `_append_turn` docstring |
| Q16: Hallucination safeguards | `src/assistant/chain.py` | `_validate_recommendation_products` |
| Q17: Latency design (<200ms) | `src/api/main.py` | `get_recommendations` docstring |
| Bonus: Cold-start strategy | `src/assistant/tools.py` | `_build_fallback_rows` docstring |

---

# Architecture Overview

```
Client
│
│ POST /recommendations
▼
FastAPI (src/api/main.py)
│
├── Feature Store (precomputed features)
│
├── ML Pipeline (XGBoost)
│      ├ preprocessing (ColumnTransformer)
│      ├ SMOTE (class imbalance)
│      └ purchase probability prediction
│
└── SHAP Explainer
       └ generates top feature-based reasons


LangChain Assistant (standalone module)
│
├── src/assistant/chain.py     — run_assistant()
├── src/assistant/tools.py     — product retrieval + cold-start fallback
├── src/assistant/schemas.py   — ShoppingQuery, AssistantResponse
└── src/assistant/demo.py      — demo runner
```

---

# Project Structure

```
ecom-recommendation/
│
├── src/
│   ├── api/                # FastAPI app
│   ├── assistant/          # LangChain assistant module
│   ├── config/             # Feature schema
│   ├── data/               # Preprocessing
│   ├── explainability/     # SHAP logic
│   ├── features/           # Feature engineering
│   ├── models/             # Training & evaluation
│   └── pipelines/          # End-to-end pipeline
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── artifacts/
│   ├── models/
│   ├── shap/
│   └── metrics/
│
├── notebooks/
│   ├── linear_algebra.ipynb
│   └── stats_prob_distributions.ipynb
│
├── tests/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

# Machine Learning Pipeline

## Purchase Propensity Model

The system predicts the probability that a user will purchase a product within 7 days.

Model: `XGBoost Classifier`

| Metric  | Score  | Purpose                      |
|---------|--------|------------------------------|
| ROC-AUC | 0.9333 | ranking performance          |
| PR-AUC  | 0.6273 | quality at low positive rate |

## Why XGBoost over LightGBM

XGBoost (AUC 0.9333) was selected over LightGBM (AUC 0.9111) despite LightGBM training ~60% faster. The deciding factor is explainability: SHAP `TreeExplainer` produces exact Shapley values against XGBoost's tree structure, whereas LightGBM requires approximated SHAP methods — unsuitable for the user-facing "Why we recommend this" feature under regulatory review.

For the full GOSS & EFB analysis see the module docstring in `src/pipelines/train_pipeline.py`.

---

## Feature Engineering

Features include:

* user demographics (age, city tier)
* browsing behaviour (session duration, clicks, browsing time)
* purchase history (days since last purchase, total orders, avg cart value)
* product attributes (category, brand, price)

Pipeline includes:

* `ColumnTransformer` (numeric + categorical)
* `StandardScaler` for numeric, `OneHotEncoder` for categorical
* log transformations for right-skewed features
* `SMOTE` for class imbalance (~2.8% positive rate)
* `RandomizedSearchCV` with stratified 5-fold CV for hyperparameter tuning

---

# Explainability (SHAP)

SHAP (SHapley Additive exPlanations) explains each prediction by attributing the model output fairly across features using Shapley values from cooperative game theory.

For each recommendation the system returns the top 3 positive-contribution features as human-readable reasons:

```
"You browsed similar items heavily this week"
"Matches your preferred shopping category"
"Fits within your typical spending range"
```

The `TreeExplainer` is built once at API startup against a background sample and cached — it is never rebuilt per request.

---

# API Endpoints

## Health Check

```
GET /health
```

Response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "shap_explainer_ready": true,
  "rows_in_feature_store": { "value": 500, "explanation": "..." },
  "unique_users": { "value": 50, "explanation": "..." },
  "unique_products": { "value": 100, "explanation": "..." }
}
```

---

## Sample Payload

```
GET /debug/sample-payload
```

Returns a ready-to-use `user_id` and `product_ids` from the live feature store for quick testing.

---

## Recommendations

```
POST /recommendations
```

### Request

```json
{
  "user_id": "U_030",
  "product_ids": ["P_007", "P_061", "P_040"]
}
```

### Response

```json
{
  "user_id": "U_030",
  "total_recommendations": 3,
  "summary": "3 product(s) scored and ranked for user 'U_030'.",
  "recommendations": [
    {
      "product_id": "P_007",
      "rank": 1,
      "recommendation_score": {
        "value": 0.008,
        "label": "Low",
        "explanation": "Probability that this user will purchase this product within 7 days, as predicted by the XGBoost model."
      },
      "reasons": [
        "Competitive price point for this category",
        "The price fits within your typical spending range",
        "Popular choice in your city tier"
      ]
    },
    {
      "product_id": "P_061",
      "rank": 2,
      "recommendation_score": {
        "value": 0.0071,
        "label": "Low",
        "explanation": "Probability that this user will purchase this product within 7 days, as predicted by the XGBoost model."
      },
      "reasons": [
        "Popular choice in your city tier",
        "You browsed similar items heavily this week",
        "Your recent click activity suggests high interest"
      ]
    },
    {
      "product_id": "P_040",
      "rank": 3,
      "recommendation_score": {
        "value": 0.0065,
        "label": "Low",
        "explanation": "Probability that this user will purchase this product within 7 days, as predicted by the XGBoost model."
      },
      "reasons": [
        "Competitive price point for this category",
        "Popular choice in your city tier",
        "You browsed similar items heavily this week"
      ]
    }
  ]
}
```

### Score labels

| Label     | Probability range |
|-----------|-------------------|
| Very High | >= 0.75           |
| High      | >= 0.60           |
| Medium    | >= 0.40           |
| Low       | < 0.40            |

---

# LangChain Assistant (Standalone Module)

The conversational assistant is implemented as a standalone module in `src/assistant/` and is not exposed as a REST endpoint in the current API. It can be run directly via the demo script.

## Running the assistant

```bash
python src/assistant/demo.py
```

## What it does

1. Parses natural language query into structured constraints (category, price, brand)
2. Retrieves candidate products from catalog
3. Ranks using XGBoost propensity model
4. Generates SHAP explanations per product
5. Validates all product IDs against catalog (hallucination safeguard)
6. Returns top 3 recommendations with reasons

## Example

Input:
```
"Show me running shoes under Rs. 3000"
```

Parsed constraints:
```json
{
  "category": "running shoes",
  "max_price": 3000.0,
  "brand": null,
  "intent": "recommend"
}
```

Output:
```
Here are your top recommendations:
1. Nike Air Zoom Running Shoes (Nike) — Rs. 2499  [score: 0.081]
   • Matches your requested category: running shoes
   • Within your budget of Rs. 3000
   • Matches your historical category preference
2. Adidas Ultraboost Lite (Adidas) — Rs. 2899  [score: 0.074]
   • Within your budget of Rs. 3000
   • You browsed similar items heavily this week
   • Competitive price point for this category
3. Puma Softride Running Shoes (Puma) — Rs. 1999  [score: 0.068]
   • Within your budget of Rs. 3000
   • The price fits within your typical spending range
   • Popular choice in your city tier
```

## Memory

Uses `ConversationBufferMemory` (via `MessagesPlaceholder`) to preserve exact constraints across turns. Price limits, brand preferences, and category filters are never paraphrased or summarised away between turns. Follow-ups like "make it cheaper" or "only Nike" correctly inherit all previous constraints.

---

# Local Setup

## Clone repository

```bash
git clone https://github.com/vishaltelukoti/Ecom_recom_system.git
cd ecom-recommendation
```

## Create virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Mac / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Configure environment variables

Copy `.env.example` to `.env` and fill in your key:

```
GROQ_API_KEY=your_api_key_here
```

## Run training pipeline

```bash
python -m src.pipelines.train_pipeline
```

## Run API locally

```bash
uvicorn src.api.main:app --reload
```

API available at `http://localhost:8000` — interactive docs at `http://localhost:8000/docs`.

---

# Running with Docker

## Build and run

```bash
docker build -t ecommerce-api .
docker run --env-file .env -p 8000:8000 ecommerce-api
```

# Running with Docker Compose

```bash
docker compose up --build   # start
docker compose down         # stop
```

---

# Testing the API

## Health check

```bash
# Linux / Mac
curl http://localhost:8000/health

# Windows PowerShell
Invoke-RestMethod http://localhost:8000/health | ConvertTo-Json
```

## Recommendations endpoint

```bash
# Linux / Mac
curl -X POST http://localhost:8000/recommendations \
-H "Content-Type: application/json" \
-d '{"user_id": "U_030", "product_ids": ["P_007", "P_061", "P_040"]}'

# Windows PowerShell
Invoke-RestMethod -Method POST "http://localhost:8000/recommendations" `
-ContentType "application/json" `
-Body '{"user_id":"U_030","product_ids":["P_007","P_061","P_040"]}' | ConvertTo-Json -Depth 10
```

---

# Running Tests

```bash
pytest -q
```

Tests cover:

* API endpoints
* feature engineering
* model pipeline
* assistant logic

---

# Technologies Used

| Component        | Technology           |
|------------------|----------------------|
| API              | FastAPI              |
| Machine Learning | XGBoost              |
| Explainability   | SHAP                 |
| LLM              | Groq (LLaMA 3.1)     |
| Orchestration    | LangChain            |
| Data Processing  | pandas, scikit-learn |
| Containerization | Docker               |
| Testing          | pytest               |

---

# Assumptions & Limitations

* Conversational assistant (`src/assistant/chain.py`) is implemented as a standalone module — no REST endpoint is exposed for it in the current API
* Cold-start for known users handled via feature synthesis (`_build_fallback_rows`); hybrid content-based fallback proposed for fully new users — see `src/assistant/tools.py`
* Feature store is precomputed offline; real-time feature ingestion not implemented
* Assistant supports constraints on category, price, and brand only
* SHAP explanations reflect feature importance at inference time — quality depends on feature engineering coverage
* Demo dataset is small (100 users, ~100 products); model scores will be low on this data — the architecture scales to production volumes

---

# Author

Vishal Telukoti