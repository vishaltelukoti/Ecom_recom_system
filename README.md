
# E-Commerce Recommendation Engine

This project implements a production-style **AI-powered recommendation system** for an e-commerce platform.

The system predicts purchase probability, ranks products, and provides **explainable recommendations using SHAP**, along with a **LangChain-based conversational assistant**.

It integrates *Machine Learning*, *Explainable AI (SHAP)*, *LLM-based interaction*, *REST APIs*, and *Docker deployment* to simulate a real-world recommendation system.

---

# Key Features

* Purchase propensity prediction using XGBoost  
* Feature engineering with Scikit-learn pipelines  
* SHAP-based explainability ("Why we recommend this")  
* Conversational assistant using LangChain + Groq  
* FastAPI-based REST service  
* Docker & Docker Compose deployment  
* Unit testing with pytest  

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
├── SHAP Explainer
│      └ generates top feature-based reasons
│
└── LangChain Assistant
├ query parsing
├ product retrieval
└ recommendation generation

```

---

# Project Structure

```

ecom-recommendation/
│
├── src/
│   ├── api/                # FastAPI app
│   ├── assistant/          # LangChain assistant
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

*Model used:*
```

XGBoost Classifier

```

## Why XGBoost

* Handles non-linear relationships  
* Strong performance on tabular data  
* Works well with imbalanced datasets  

---

## Feature Engineering

Features include:

* user demographics  
* browsing behavior  
* purchase history  
* product attributes  

Pipeline includes:

* ColumnTransformer (numeric + categorical)  
* encoding strategies (OHE / ordinal)  
* log transformations for skewed features  
* SMOTE for class imbalance  

---

## Model Evaluation

| Metric    | Purpose                  |
|----------|--------------------------|
| AUC      | ranking performance      |
| Precision| recommendation quality   |
| Recall   | capturing potential buyers|

---

# Explainability (SHAP)

SHAP (SHapley Additive exPlanations) is used to explain predictions.

For each recommendation, the system returns:

* top contributing features  
* human-readable reasons  

### Example

```

"You browsed similar items heavily this week"
"Matches your preferred category"
"Fits within your typical spending range"

```

SHAP ensures:

* fair attribution of features  
* interpretable model behavior  
* explainable recommendations  

---

# Conversational Assistant

Users can interact using natural language:

```

"Show me running shoes under 3000"

```

## Pipeline

1. Parse query → structured constraints  
2. Retrieve candidate products  
3. Rank using ML model  
4. Generate SHAP explanations  
5. Return top recommendations  

## Memory Support

* Maintains conversation context  
* Supports follow-ups like:  
  * "make it cheaper"  

---

# API Endpoints

## Health Check

```

GET /health

````

Response:

```json
{
  "status": "ok"
}
````

---

## Sample Payload

```
GET /debug/sample-payload
```

---

## Recommendations

```
POST /recommendations
```

### Request

```json
{
  "user_id": "U_001",
  "product_ids": ["P_001", "P_002", "P_003"]
}
```

### Response

```json
[
  {
    "product_id": "P_001",
    "score": 0.82,
    "reasons": [
      "You browsed similar items heavily this week",
      "Matches your preferred category",
      "Fits your budget"
    ]
  }
]
```

---

# Local Setup

## Clone repository

```bash
git clone https://github.com/vishaltelukoti/Ecom_recom_system.git
cd ecom-recommendation
```

---

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

---

## Install dependencies

```bash
pip install -r requirements.txt
```

---

## Configure environment variables

Create `.env`:

```
GROQ_API_KEY=your_api_key_here
```

---

## Run training pipeline

```bash
python -m src.pipelines.train_pipeline
```

---

## Run API locally

```bash
uvicorn src.api.main:app --reload
```

Server:

```
http://localhost:8000
```

---

# Running with Docker

## Build image

```bash
docker build -t ecommerce-api .
```

---

## Run container

```bash
docker run --env-file .env -p 8000:8000 ecommerce-api
```

---

# Running with Docker Compose

## Start

```bash
docker compose up --build
```

---

## Stop

```bash
docker compose down
```

---

# Testing the API

## Health

```bash
curl http://localhost:8000/health
```

---

## Recommendations

```bash
curl -X POST http://localhost:8000/recommendations \
-H "Content-Type: application/json" \
-d '{"user_id":"U_001","product_ids":["P_001","P_002","P_003"]}'
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

| Component        | Technology       |
| ---------------- | ---------------- |
| API              | FastAPI          |
| Machine Learning | XGBoost          |
| Explainability   | SHAP             |
| LLM              | Groq + LangChain |
| Data Processing  | pandas           |
| Containerization | Docker           |
| Testing          | pytest           |

---

# Assumptions & Limitations

* Limited handling of cold-start users
* Feature store is precomputed
* Assistant supports basic constraints (category, price, brand)
* SHAP explanations depend on feature quality

---

# Author

Vishal Telukoti

```
