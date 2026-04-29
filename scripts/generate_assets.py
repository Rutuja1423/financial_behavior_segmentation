import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


ASSETS_DIR = ROOT / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")

df = pd.read_csv(ROOT / "synthetic_financial_behavior_data.csv")

score_cols = [
    "financial_literacy_score",
    "impulse_buying_score",
    "stress_spending_score",
    "saving_discipline_score",
    "budgeting_habit_score",
    "savings_rate",
    "expense_to_income_ratio",
]

cluster_features = score_cols
X = df[cluster_features].copy()
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2, random_state=42)
pca_components = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(pca_components, columns=["PC1", "PC2"])

silhouette_scores = {}
for k in range(2, 8):
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)
    silhouette_scores[k] = silhouette_score(X_scaled, labels)

best_k = max(silhouette_scores, key=silhouette_scores.get)
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)
pca_df["cluster"] = df["cluster"]

model = smf.ols(
    "savings_rate ~ financial_literacy_score + impulse_buying_score + "
    "stress_spending_score + saving_discipline_score + budgeting_habit_score + "
    "age + C(gender) + C(employment_status) + C(city_tier)",
    data=df,
).fit()


def save_current_figure(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close()


plt.figure(figsize=(12, 8))
corr = df[score_cols].corr()
sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0, fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Financial Behavior Variables", pad=18)
save_current_figure("heatmap.png")


plt.figure(figsize=(10, 6))
variance_df = pd.DataFrame(
    {
        "Component": ["PC1", "PC2"],
        "Explained Variance": pca.explained_variance_ratio_,
    }
)
sns.barplot(
    data=variance_df,
    x="Component",
    y="Explained Variance",
    hue="Component",
    palette="Blues_d",
    legend=False,
)
plt.ylim(0, 0.55)
plt.title("PCA Explained Variance", pad=18)
plt.ylabel("Variance Explained")
plt.xlabel("")
for idx, value in enumerate(variance_df["Explained Variance"]):
    plt.text(idx, value + 0.015, f"{value:.1%}", ha="center", fontweight="bold")
plt.text(
    0.5,
    0.49,
    f"Total explained variance: {pca.explained_variance_ratio_.sum():.1%}",
    ha="center",
    fontsize=13,
)
save_current_figure("pca_explained_variance.png")


plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="cluster",
    palette="Set2",
    s=90,
    alpha=0.85,
    edgecolor="white",
    linewidth=0.5,
)
plt.title(f"K-Means User Segments on PCA Space (k={best_k})", pad=18)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")
save_current_figure("cluster_plot.png")


coef_names = [
    "financial_literacy_score",
    "impulse_buying_score",
    "stress_spending_score",
    "saving_discipline_score",
    "budgeting_habit_score",
]
coef_df = (
    model.params.loc[coef_names]
    .rename_axis("Variable")
    .reset_index(name="Coefficient")
    .sort_values("Coefficient")
)

plt.figure(figsize=(10, 6))
colors = ["#D95F5F" if value < 0 else "#3C8D6E" for value in coef_df["Coefficient"]]
color_map = dict(zip(coef_df["Variable"], colors))
sns.barplot(
    data=coef_df,
    x="Coefficient",
    y="Variable",
    hue="Variable",
    palette=color_map,
    legend=False,
)
plt.axvline(0, color="black", linewidth=1)
plt.title(f"Regression Coefficients for Savings Rate (R-squared = {model.rsquared:.3f})", pad=18)
plt.xlabel("Coefficient")
plt.ylabel("")
save_current_figure("regression_coefficients.png")
