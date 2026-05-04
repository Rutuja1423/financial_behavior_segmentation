"""
Visualization functions for the financial behavior segmentation pipeline.
Each function creates a single publication-ready chart and saves it to disk.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")


def _save(fig: plt.Figure, path: Path) -> None:
    """Save a figure and close it."""
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


def plot_heatmap(corr: pd.DataFrame, save_path: Path) -> None:
    """Correlation heatmap of financial behavior variables."""
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        corr, annot=True, cmap="RdBu_r", center=0,
        fmt=".2f", linewidths=0.5, ax=ax,
    )
    ax.set_title("Correlation Heatmap of Financial Behavior Variables", pad=18)
    _save(fig, save_path)


def plot_pca_variance(explained_var, save_path: Path) -> None:
    """Bar chart of PCA explained variance per component."""
    labels = [f"PC{i+1}" for i in range(len(explained_var))]
    var_df = pd.DataFrame({"Component": labels, "Explained Variance": explained_var})

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=var_df, x="Component", y="Explained Variance",
        hue="Component", palette="Blues_d", legend=False, ax=ax,
    )
    ax.set_ylim(0, 0.55)
    ax.set_title("PCA Explained Variance", pad=18)
    ax.set_ylabel("Variance Explained")
    ax.set_xlabel("")
    for idx, value in enumerate(explained_var):
        ax.text(idx, value + 0.015, f"{value:.1%}", ha="center", fontweight="bold")
    ax.text(
        0.5, 0.49, f"Total explained variance: {sum(explained_var):.1%}",
        ha="center", fontsize=13,
    )
    _save(fig, save_path)


def plot_clusters(pca_df: pd.DataFrame, best_k: int, save_path: Path) -> None:
    """Scatter plot of K-Means clusters projected onto PCA space."""
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        data=pca_df, x="PC1", y="PC2", hue="cluster",
        palette="Set2", s=90, alpha=0.85,
        edgecolor="white", linewidth=0.5, ax=ax,
    )
    ax.set_title(f"K-Means User Segments on PCA Space (k={best_k})", pad=18)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend(title="Cluster")
    _save(fig, save_path)


def plot_regression_coefs(
    model, coef_names: list, save_path: Path
) -> None:
    """Horizontal bar chart of regression coefficients for savings rate."""
    coef_df = (
        model.params.loc[coef_names]
        .rename_axis("Variable")
        .reset_index(name="Coefficient")
        .sort_values("Coefficient")
    )
    colors = ["#D95F5F" if v < 0 else "#3C8D6E" for v in coef_df["Coefficient"]]
    color_map = dict(zip(coef_df["Variable"], colors))

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=coef_df, x="Coefficient", y="Variable",
        hue="Variable", palette=color_map, legend=False, ax=ax,
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title(
        f"Regression Coefficients for Savings Rate (R² = {model.rsquared:.3f})",
        pad=18,
    )
    ax.set_xlabel("Coefficient")
    ax.set_ylabel("")
    _save(fig, save_path)
