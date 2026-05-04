"""
Segmentation: feature scaling, PCA, and K-Means clustering.
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE


def scale_features(
    df: pd.DataFrame, cols: list
) -> Tuple[np.ndarray, StandardScaler]:
    """Standardize features to zero mean and unit variance."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[cols])
    print(f"  Scaled {len(cols)} features")
    return X_scaled, scaler


def run_pca(
    X_scaled: np.ndarray, n_components: int = 2
) -> Tuple[np.ndarray, PCA]:
    """
    Apply PCA and return the transformed components and the fitted PCA object.
    """
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    components = pca.fit_transform(X_scaled)
    total_var = pca.explained_variance_ratio_.sum()
    print(f"  PCA: {n_components} components explain {total_var:.1%} of variance")
    return components, pca


def find_best_k(
    X_scaled: np.ndarray, k_range: range
) -> Dict[int, float]:
    """
    Evaluate silhouette scores for each k in the given range.
    Returns a dict mapping k → silhouette_score.
    """
    scores = {}
    for k in k_range:
        labels = KMeans(
            n_clusters=k, random_state=RANDOM_STATE, n_init=10
        ).fit_predict(X_scaled)
        scores[k] = silhouette_score(X_scaled, labels)
    best = max(scores, key=scores.get)
    print(f"  Silhouette search: best k={best} (score={scores[best]:.3f})")
    return scores


def fit_kmeans(
    X_scaled: np.ndarray, k: int
) -> Tuple[np.ndarray, KMeans]:
    """Fit K-Means with the chosen k and return labels + model."""
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    print(f"  K-Means fitted with k={k}: cluster sizes = {np.bincount(labels).tolist()}")
    return labels, kmeans
