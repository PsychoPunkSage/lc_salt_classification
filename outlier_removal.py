"""DBSCAN-based outlier removal operating on PCA feature space (not t-SNE).

Running DBSCAN on t-SNE coordinates was non-deterministic: t-SNE's stochastic
layout changed per run, which changed which points DBSCAN removed, which changed
the training set, and ultimately caused classifier accuracy to vary between runs.
Fix: DBSCAN now operates on a low-dim PCA projection (deterministic, distance-
preserving) so the cleaned dataset is identical on every run.
"""

from __future__ import annotations

import numpy as np
from collections import Counter
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def compute_initial_tsne(
    features: np.ndarray,
    pca_components: int = 50,
    perplexity: int = 30,
    n_iter: int = 1000,
) -> np.ndarray:
    """PCA -> t-SNE (2D) on the full (pre-DBSCAN) feature matrix (visualisation only).

    Returns
    -------
    tsne_coords : ndarray, shape (N, 2)
    """
    print(f"\n  Running initial t-SNE 2D (perplexity={perplexity}, n_iter={n_iter})...")
    pca = PCA(n_components=min(pca_components, features.shape[1]), random_state=42)
    features_pca = pca.fit_transform(features)
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter,
                random_state=42, method="exact")
    return tsne.fit_transform(features_pca)


def remove_outliers_dbscan(
    features: np.ndarray,
    labels: list[str],
    eps: float = 2.0,
    min_samples: int = 5,
    warn_threshold: float = 0.15,
    min_class_size: int = 35,
    pca_components: int = 10,
) -> tuple[np.ndarray, list[str], int]:
    """Keep all non-noise points (cluster_label != -1) from each salt class.

    DBSCAN runs in PCA space (standardised, 10 components) — fully deterministic
    and independent of t-SNE layout, so the cleaned dataset is identical every run.

    Returns
    -------
    features_clean      : ndarray (M, D)
    labels_clean        : list of M strings
    adjusted_perplexity : safe t-SNE perplexity for the cleaned dataset
    """
    # Project to low-dim PCA and standardise so Euclidean distances are meaningful
    n_comp = min(pca_components, features.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    features_pca = StandardScaler().fit_transform(pca.fit_transform(features))

    print("\n  Removing scattered outlier points (DBSCAN on PCA space)...")
    print("  " + "=" * 58)

    salt_classes = sorted(set(labels))
    cleaned_indices: list[int] = []

    for salt in salt_classes:
        mask = np.array(labels) == salt
        class_pca = features_pca[mask]
        class_idx = np.where(mask)[0]
        original_count = len(class_idx)

        if original_count > 10:
            db = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = db.fit_predict(class_pca)
            valid = cluster_labels[cluster_labels != -1]

            if len(valid) > 0:
                keep_mask = cluster_labels != -1
                kept = keep_mask.sum()
                removed = original_count - kept
                removal_pct = removed / original_count
                cleaned_indices.extend(class_idx[keep_mask])
                status = "WARNING: >15% removed" if removal_pct > warn_threshold else "OK"
                print(f"  {salt:15s}: {original_count:3d} -> {kept:3d}  "
                      f"(removed {removed:2d}, {removal_pct*100:.1f}%)  [{status}]")
            else:
                cleaned_indices.extend(class_idx)
                print(f"  {salt:15s}: {original_count:3d} -> {original_count:3d}  "
                      f"(no clusters found, kept all)")
        else:
            cleaned_indices.extend(class_idx)
            print(f"  {salt:15s}: {original_count:3d} -> {original_count:3d}  "
                  f"(too few points, kept all)")

    print("  " + "=" * 58)

    total_removed = len(labels) - len(cleaned_indices)
    print("  " + "=" * 58)
    print(f"  Total kept: {len(cleaned_indices)}, Total removed: {total_removed}")

    features_clean = features[cleaned_indices]
    labels_clean = [labels[i] for i in cleaned_indices]

    # Validate no class fell below minimum
    post_counts = Counter(labels_clean)
    for salt, count in post_counts.items():
        assert count >= min_class_size, (
            f"Salt '{salt}' has only {count} samples after DBSCAN "
            f"- below minimum {min_class_size}"
        )

    print("\n  Post-DBSCAN class distribution:")
    for s in sorted(salt_classes):
        print(f"    {s:15s}: {post_counts.get(s, 0):3d} samples")

    adjusted_perplexity = min(30, len(features_clean) // 3)
    print(f"\n  t-SNE perplexity adjusted to: {adjusted_perplexity}")

    return features_clean, labels_clean, adjusted_perplexity
