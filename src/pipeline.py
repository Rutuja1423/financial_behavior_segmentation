"""
End-to-end pipeline for the financial behavior segmentation project.

Usage
-----
    python -m src.pipeline                # from the project root
    python -m src.pipeline --data path    # custom data file
"""

import argparse
import os
import sys
from pathlib import Path

# Prevent matplotlib font-cache warnings
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib_cache"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import pandas as pd

from .config import (
    ASSETS_DIR,
    CLUSTER_FEATURE_COLS,
    COMPOSITE_SCORE_COLS,
    DATA_PATH,
    KMEANS_K_RANGE,
    PCA_N_COMPONENTS,
    REGRESSION_COEF_DISPLAY,
    REGRESSION_FORMULA,
    SURVEY_ITEMS,
)
from .data_loader import load_data, validate_schema
from .feature_engineering import compute_composite_scores, compute_financial_ratios
from .reliability import reliability_report
from .segmentation import fit_kmeans, find_best_k, run_pca, scale_features
from .statistics import correlation_matrix, run_anova, run_regression
from .visualization import (
    plot_clusters,
    plot_heatmap,
    plot_pca_variance,
    plot_regression_coefs,
)


def _banner(title: str) -> None:
    """Print a visually distinct stage header."""
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def run_full_pipeline(data_path: Path = DATA_PATH, assets_dir: Path = ASSETS_DIR) -> dict:
    """
    Execute every analysis stage and generate all visual assets.

    Returns a summary dict with key metrics for downstream use.
    """
    assets_dir.mkdir(exist_ok=True)

    # Stage 1: Load & validate
    _banner("Stage 1 - Data Loading & Validation")
    df = load_data(data_path)
    validate_schema(df)

    # Stage 2: Feature engineering
    _banner("Stage 2 - Feature Engineering")
    df = compute_composite_scores(df)
    df = compute_financial_ratios(df)

    # Stage 3: Reliability testing
    _banner("Stage 3 - Reliability Testing (Cronbach's Alpha)")
    rel_report = reliability_report(df, SURVEY_ITEMS)

    # Stage 4: Correlation analysis
    _banner("Stage 4 - Correlation Analysis")
    corr = correlation_matrix(df, CLUSTER_FEATURE_COLS)
    plot_heatmap(corr, assets_dir / "heatmap.png")

    # Stage 5: Hypothesis testing (ANOVA)
    _banner("Stage 5 - Hypothesis Testing (ANOVA)")
    anova_result = run_anova(df, "employment_status", "savings_rate")

    # Stage 6: Regression modelling
    _banner("Stage 6 - Regression Modelling")
    model, reg_summary = run_regression(df, REGRESSION_FORMULA)
    plot_regression_coefs(model, REGRESSION_COEF_DISPLAY, assets_dir / "regression_coefficients.png")

    # Stage 7: Dimensionality reduction (PCA)
    _banner("Stage 7 - Dimensionality Reduction (PCA)")
    X_scaled, scaler = scale_features(df, CLUSTER_FEATURE_COLS)
    pca_components, pca_obj = run_pca(X_scaled, PCA_N_COMPONENTS)
    plot_pca_variance(pca_obj.explained_variance_ratio_, assets_dir / "pca_explained_variance.png")

    # Stage 8: K-Means clustering
    _banner("Stage 8 - K-Means Clustering")
    sil_scores = find_best_k(X_scaled, KMEANS_K_RANGE)
    best_k = max(sil_scores, key=sil_scores.get)
    labels, kmeans_model = fit_kmeans(X_scaled, best_k)
    df["cluster"] = labels

    pca_df = pd.DataFrame(pca_components, columns=["PC1", "PC2"])
    pca_df["cluster"] = labels
    plot_clusters(pca_df, best_k, assets_dir / "cluster_plot.png")

    # Stage 9: Segment profiling
    _banner("Stage 9 - Segment Profiling")
    profile = df.groupby("cluster")[COMPOSITE_SCORE_COLS + ["savings_rate"]].mean().round(3)
    print(profile.to_string())

    # Summary
    _banner("Pipeline Complete")
    summary = {
        "n_rows": len(df),
        "n_clusters": best_k,
        "best_silhouette": round(sil_scores[best_k], 3),
        "r_squared": reg_summary["r_squared"],
        "anova_p": anova_result["p_value"],
        "cluster_sizes": df["cluster"].value_counts().to_dict(),
    }
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return summary


# CLI entry point
def main() -> None:
    parser = argparse.ArgumentParser(description="Run the financial behavior segmentation pipeline")
    parser.add_argument(
        "--data", type=Path, default=DATA_PATH,
        help="Path to the CSV dataset (default: synthetic_financial_behavior_data.csv)",
    )
    parser.add_argument(
        "--assets", type=Path, default=ASSETS_DIR,
        help="Directory to save generated charts (default: assets/)",
    )
    args = parser.parse_args()
    run_full_pipeline(data_path=args.data, assets_dir=args.assets)


if __name__ == "__main__":
    main()
