"""
End-to-end training pipeline orchestrator.

Run this single script to go from raw CSVs → trained model → SHAP artifacts.

Steps executed in order:
  1. Preprocessing     — clean raw data, build base table, save interim parquets
  2. Feature engineering — compute all feature columns, save processed parquet
  3. Model training    — XGBoost + SMOTE + RandomizedSearchCV, save pipeline.pkl
  4. SHAP analysis     — global summary plot + local explanation artifacts
  5. Model comparison  — LightGBM head-to-head, write comparison report

Usage:
    python -m src.pipelines.train_pipeline            # full run
    python -m src.pipelines.train_pipeline --skip-shap  # skip slow SHAP step

B2: Why LightGBM is faster — GOSS and EFB
==========================================
XGBoost scans all N samples × all F features at every split (exact greedy).
On 10M rows with ~300 post-OHE features, this drives the 4.2hr training cost.
LightGBM cuts this with two algorithms:

GOSS (Gradient-based One-Side Sampling): samples with large gradients are
under-fitted and information-rich; small-gradient samples contribute little.
GOSS retains all top-a% high-gradient samples, randomly keeps b% of the rest,
and upweights them by (1-a)/b to correct for bias. With a=0.2, b=0.1 on 10M
rows: ~2.8M effective samples per split vs 10M — a 3.5× data reduction.

EFB (Exclusive Feature Bundling): OHE produces many mutually exclusive sparse
features (is_Electronics / is_Fashion / is_FMCG are never non-zero together).
EFB bundles them via greedy graph colouring, encoding each into a distinct value
range within the bundle. Reduces ~300 post-OHE columns to ~90 bundles — 3×
fewer features per split, zero information loss for exclusive features.

Combined: ~0.09× compute cost per split vs XGBoost. At 10M rows LightGBM
trains in ~40% of XGBoost's time. XGBoost is still chosen for production
because SHAP TreeExplainer produces exact Shapley values against its tree
structure — LightGBM requires approximated SHAP, which is unsuitable for
the user-facing "Why we recommend this" explanations under regulatory review.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
)
log = logging.getLogger("train_pipeline")


def run(skip_shap: bool = False, skip_compare: bool = False) -> None:
    total_start = time.perf_counter()

    # Step 1 — Preprocessing
    log.info("=" * 60)
    log.info("STEP 1/5  Preprocessing raw data")
    log.info("=" * 60)
    t0 = time.perf_counter()
    from src.data.preprocess import run_preprocessing
    run_preprocessing()
    log.info("Preprocessing done in %.1fs", time.perf_counter() - t0)

    # Step 2 — Feature engineering
    log.info("=" * 60)
    log.info("STEP 2/5  Feature engineering")
    log.info("=" * 60)
    t0 = time.perf_counter()
    from src.features.build_features import build_features
    build_features()
    log.info("Feature engineering done in %.1fs", time.perf_counter() - t0)

    # Step 3 — Model training (XGBoost + SMOTE + RandomizedSearchCV)
    log.info("=" * 60)
    log.info("STEP 3/5  Training XGBoost model (n_iter=20, cv=5)")
    log.info("=" * 60)
    t0 = time.perf_counter()
    from src.models.train_xgboost import train
    best_pipeline, metrics, best_params = train(n_iter=20, cv_folds=5)
    log.info(
        "Model training done in %.1fs  |  Test ROC-AUC: %.4f",
        time.perf_counter() - t0,
        metrics["roc_auc"],
    )

    # Step 4 — SHAP analysis
    if skip_shap:
        log.info("STEP 4/5  SHAP analysis  [SKIPPED]")
    else:
        log.info("=" * 60)
        log.info("STEP 4/5  SHAP explainability analysis")
        log.info("=" * 60)
        t0 = time.perf_counter()
        from src.explainability.shap_explainer import main as run_shap
        run_shap()
        log.info("SHAP analysis done in %.1fs", time.perf_counter() - t0)

    # Step 5 — Model comparison (LightGBM vs XGBoost)
    if skip_compare:
        log.info("STEP 5/5  LightGBM comparison  [SKIPPED]")
    else:
        log.info("=" * 60)
        log.info("STEP 5/5  LightGBM vs XGBoost comparison")
        log.info("=" * 60)
        t0 = time.perf_counter()
        from src.models.evaluate import main as run_compare
        run_compare()
        log.info("Model comparison done in %.1fs", time.perf_counter() - t0)

    
    # Summary
    log.info("=" * 60)
    log.info("PIPELINE COMPLETE in %.1fs", time.perf_counter() - total_start)
    log.info("Artifacts:")
    base = Path(__file__).resolve().parents[2]
    log.info("  Model    → %s", base / "artifacts" / "models" / "xgboost_pipeline.pkl")
    log.info("  Metrics  → %s", base / "artifacts" / "metrics")
    log.info("  SHAP     → %s", base / "artifacts" / "shap")
    log.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full training pipeline.")
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP analysis step")
    parser.add_argument("--skip-compare", action="store_true", help="Skip LightGBM comparison step")
    args = parser.parse_args()
    run(skip_shap=args.skip_shap, skip_compare=args.skip_compare)


if __name__ == "__main__":
    main()