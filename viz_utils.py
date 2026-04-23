"""Shared color scheme and all dimensionality-reduction visualisations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap as umap_lib
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - registers 3d projection
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import MDS, TSNE
from sklearn.metrics import silhouette_score

# ---------------------------------------------------------------------------
# Colour scheme - built once, imported everywhere
# ---------------------------------------------------------------------------

SALT_ORDER: list[str] = sorted([
    "azlocillin", "carbenicillin", "oxacillin",
    "penicillin", "pipercilin", "ticarcillin",
])
_TAB20 = plt.cm.tab20(np.linspace(0, 1, len(SALT_ORDER)))
SALT_COLOR_MAP: dict[str, tuple] = {
    salt: tuple(_TAB20[i]) for i, salt in enumerate(SALT_ORDER)
}

# ---------------------------------------------------------------------------
# t-SNE helpers
# ---------------------------------------------------------------------------

def _run_pca_tsne(
    features: np.ndarray,
    n_components: int,
    pca_components: int,
    perplexity: int,
    n_iter: int,
) -> np.ndarray:
    pca = PCA(n_components=min(pca_components, features.shape[1]), random_state=42)
    features_pca = pca.fit_transform(features)
    tsne = TSNE(n_components=n_components, perplexity=perplexity,
                max_iter=n_iter, random_state=42)
    return tsne.fit_transform(features_pca)


def plot_tsne_2d(
    features: np.ndarray,
    labels: list[str],
    title: str,
    output_path: str | Path,
    pca_components: int = 50,
    perplexity: int = 30,
    n_iter: int = 1000,
) -> np.ndarray:
    print(f"\n  Running t-SNE 2D (perplexity={perplexity}, n_iter={n_iter})...")
    coords = _run_pca_tsne(features, 2, pca_components, perplexity, n_iter)

    plt.figure(figsize=(12, 8))
    for salt in [s for s in SALT_ORDER if s in labels]:
        mask = np.array(labels) == salt
        plt.scatter(coords[mask, 0], coords[mask, 1],
                    c=[SALT_COLOR_MAP[salt]], label=salt,
                    alpha=0.7, s=40, edgecolors="black", linewidth=0.6)
    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel("t-SNE Component 1", fontsize=12)
    plt.ylabel("t-SNE Component 2", fontsize=12)
    plt.legend(title="Antibiotic Salt", loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")
    return coords


def plot_tsne_3d(
    features: np.ndarray,
    labels: list[str],
    title: str,
    output_path: str | Path,
    pca_components: int = 50,
    perplexity: int = 30,
    n_iter: int = 1000,
) -> np.ndarray:
    print(f"\n  Running t-SNE 3D (perplexity={perplexity}, n_iter={n_iter})...")
    coords = _run_pca_tsne(features, 3, pca_components, perplexity, n_iter)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    for salt in [s for s in SALT_ORDER if s in labels]:
        mask = np.array(labels) == salt
        ax.scatter(coords[mask, 0], coords[mask, 1], coords[mask, 2],
                   c=[SALT_COLOR_MAP[salt]], label=salt,
                   alpha=0.7, s=40, edgecolors="black", linewidth=0.4)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("t-SNE Component 1", fontsize=10)
    ax.set_ylabel("t-SNE Component 2", fontsize=10)
    ax.set_zlabel("t-SNE Component 3", fontsize=10)
    ax.legend(title="Antibiotic Salt", loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")
    return coords


def plot_tsne_from_coords(
    tsne_coords: np.ndarray,
    labels: list[str],
    title: str,
    output_path: str | Path,
) -> None:
    """Replot using pre-computed 2D t-SNE coordinates - avoids re-running t-SNE."""
    plt.figure(figsize=(14, 10))
    for salt in [s for s in SALT_ORDER if s in labels]:
        mask = np.array(labels) == salt
        plt.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1],
                    c=[SALT_COLOR_MAP[salt]], label=salt,
                    alpha=0.7, s=50, edgecolors="black", linewidth=0.6)
    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel("t-SNE Component 1", fontsize=12)
    plt.ylabel("t-SNE Component 2", fontsize=12)
    plt.legend(title="Antibiotic Salt", loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_tsne_3d_from_coords(
    tsne_coords: np.ndarray,
    labels: list[str],
    title: str,
    output_path: str | Path,
) -> None:
    """Replot using pre-computed 3D t-SNE coordinates."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    for salt in [s for s in SALT_ORDER if s in labels]:
        mask = np.array(labels) == salt
        ax.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1], tsne_coords[mask, 2],
                   c=[SALT_COLOR_MAP[salt]], label=salt,
                   alpha=0.7, s=50, edgecolors="black", linewidth=0.4)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("t-SNE Component 1", fontsize=10)
    ax.set_ylabel("t-SNE Component 2", fontsize=10)
    ax.set_zlabel("t-SNE Component 3", fontsize=10)
    ax.legend(title="Antibiotic Salt", loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")

# ---------------------------------------------------------------------------
# MDS
# ---------------------------------------------------------------------------

def plot_mds(
    features: np.ndarray,
    labels: list[str],
    output_path: str | Path,
    pca_components: int = 50,
) -> np.ndarray:
    print("\n  Computing MDS (preserves inter-group distances)...")
    pca = PCA(n_components=min(pca_components, features.shape[1]), random_state=42)
    features_pca = pca.fit_transform(features)
    mds = MDS(n_components=2, random_state=42, dissimilarity="euclidean")
    coords = mds.fit_transform(features_pca)

    salts_present = [s for s in SALT_ORDER if s in labels]
    centroids = []

    plt.figure(figsize=(14, 10))
    for salt in salts_present:
        mask = np.array(labels) == salt
        plt.scatter(coords[mask, 0], coords[mask, 1],
                    c=[SALT_COLOR_MAP[salt]], label=salt,
                    alpha=0.6, s=50, edgecolors="black", linewidth=0.5)
        centroids.append(coords[mask].mean(axis=0))

    centroids = np.array(centroids)
    for i in range(len(centroids) - 1):
        dist = np.linalg.norm(centroids[i + 1] - centroids[i])
        mid = (centroids[i] + centroids[i + 1]) / 2
        plt.plot([centroids[i, 0], centroids[i + 1, 0]],
                 [centroids[i, 1], centroids[i + 1, 1]],
                 "r-", linewidth=2, alpha=0.7)
        plt.annotate(f"d={dist:.1f}", mid, fontsize=9, color="red",
                     fontweight="bold",
                     bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.title("MDS: True Distance Relationships Between Antibiotic Salt Classes",
              fontsize=15, fontweight="bold")
    plt.xlabel("MDS Dimension 1", fontsize=12)
    plt.ylabel("MDS Dimension 2", fontsize=12)
    plt.legend(title="Antibiotic Salt", loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")

    # Full inter-centroid distance matrix
    dist_matrix = np.zeros((len(salts_present), len(salts_present)))
    for i in range(len(salts_present)):
        for j in range(len(salts_present)):
            dist_matrix[i, j] = np.linalg.norm(centroids[i] - centroids[j])
    dist_df = pd.DataFrame(dist_matrix,
                           index=[s[:6] for s in salts_present],
                           columns=[s[:6] for s in salts_present])

    intra_radii = []
    for salt in salts_present:
        mask = np.array(labels) == salt
        pts = coords[mask]
        intra_radii.append(np.linalg.norm(pts - pts.mean(axis=0), axis=1).mean())

    all_inter = [
        np.linalg.norm(centroids[i] - centroids[j])
        for i in range(len(centroids)) for j in range(i + 1, len(centroids))
    ]
    mean_inter = np.mean(all_inter)
    mean_intra = np.mean(intra_radii)
    snr = mean_inter / mean_intra

    print("\n  MDS EMBEDDING METRICS")
    print(f"    Mean Inter-Centroid Distance : {mean_inter:.4f}")
    print(f"    Mean Intra-Cluster Radius    : {mean_intra:.4f}")
    print(f"    SNR                          : {snr:.4f}")
    print("\n  Inter-Centroid Distance Matrix (MDS):")
    print(dist_df.round(3).to_string())

    return coords

# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------

def plot_umap(
    features: np.ndarray,
    labels: list[str],
    output_path: str | Path,
    pca_components: int = 50,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
) -> np.ndarray:
    print("\n  Computing UMAP...")
    pca = PCA(n_components=min(pca_components, features.shape[1]), random_state=42)
    features_pca = pca.fit_transform(features)

    reducer = umap_lib.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist,
        n_components=2, metric="euclidean", random_state=42,
    )
    coords = reducer.fit_transform(features_pca)

    salts_present = [s for s in SALT_ORDER if s in labels]
    centroids = []

    plt.figure(figsize=(14, 10))
    for salt in salts_present:
        mask = np.array(labels) == salt
        plt.scatter(coords[mask, 0], coords[mask, 1],
                    c=[SALT_COLOR_MAP[salt]], label=salt,
                    alpha=0.6, s=50, edgecolors="white", linewidth=0.5)
        centroids.append(coords[mask].mean(axis=0))

    centroids = np.array(centroids)
    for i, salt in enumerate(salts_present):
        plt.annotate(salt[:5], centroids[i], fontsize=9, fontweight="bold",
                     ha="center",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    plt.title("UMAP - Antibiotic Salt Classes (Meaningful Inter-Class Distances)",
              fontsize=16, fontweight="bold")
    plt.xlabel("UMAP Component 1", fontsize=12)
    plt.ylabel("UMAP Component 2", fontsize=12)
    plt.legend(title="Antibiotic Salt", loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")

    # UMAP SNR
    intra_radii = [
        np.linalg.norm(coords[np.array(labels) == s] -
                       coords[np.array(labels) == s].mean(axis=0), axis=1).mean()
        for s in salts_present
    ]
    inter = [
        np.linalg.norm(centroids[i] - centroids[i + 1])
        for i in range(len(centroids) - 1)
    ]
    snr = np.mean(inter) / np.mean(intra_radii)
    print(f"\n  UMAP SNR: {snr:.3f}")

    return coords

# ---------------------------------------------------------------------------
# LDA
# ---------------------------------------------------------------------------

def plot_lda(
    features: np.ndarray,
    labels: list[str],
    output_dir: str | Path,
) -> np.ndarray:
    print("\n  Computing LDA...")
    output_dir = Path(output_dir)
    n_components = min(5, len(SALT_ORDER) - 1)
    lda = LinearDiscriminantAnalysis(n_components=n_components, solver="svd")
    coords = lda.fit_transform(features, np.array(labels))
    print(f"  LDA Explained Variance Ratio: {lda.explained_variance_ratio_}")

    salts_present = [s for s in SALT_ORDER if s in labels]
    var = lda.explained_variance_ratio_

    # LD1 vs LD2
    plt.figure(figsize=(14, 10))
    for salt in salts_present:
        mask = np.array(labels) == salt
        plt.scatter(coords[mask, 0], coords[mask, 1],
                    c=[SALT_COLOR_MAP[salt]], label=salt,
                    alpha=0.7, s=60, edgecolors="black", linewidth=0.8)
    var2 = var[1] if len(var) > 1 else 0.0
    plt.title("LDA: Supervised Antibiotic Salt Class Separation (LD1 vs LD2)",
              fontsize=15, fontweight="bold")
    plt.xlabel(f"LD1 ({var[0]:.1%} variance)", fontsize=12)
    plt.ylabel(f"LD2 ({var2:.1%} variance)", fontsize=12)
    plt.legend(title="Antibiotic Salt", loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    plt.axvline(0, color="black", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "lda_ld1_ld2.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'lda_ld1_ld2.png'}")

    # LD1 vs LD3
    if coords.shape[1] >= 3:
        plt.figure(figsize=(14, 10))
        for salt in salts_present:
            mask = np.array(labels) == salt
            plt.scatter(coords[mask, 0], coords[mask, 2],
                        c=[SALT_COLOR_MAP[salt]], label=salt,
                        alpha=0.7, s=60, edgecolors="black", linewidth=0.8)
        plt.title("LDA: Supervised Antibiotic Salt Class Separation (LD1 vs LD3)",
                  fontsize=15, fontweight="bold")
        plt.xlabel(f"LD1 ({var[0]:.1%} variance)", fontsize=12)
        plt.ylabel(f"LD3 ({var[2]:.1%} variance)", fontsize=12)
        plt.legend(title="Antibiotic Salt", loc="best", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.axhline(0, color="black", linewidth=0.5, alpha=0.5)
        plt.axvline(0, color="black", linewidth=0.5, alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_dir / "lda_ld1_ld3.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {output_dir / 'lda_ld1_ld3.png'}")

    return coords

# ---------------------------------------------------------------------------
# Silhouette + centroid overlay
# ---------------------------------------------------------------------------

def compute_cluster_metrics(
    tsne_clean: np.ndarray,
    features: np.ndarray,
    labels: list[str],
) -> dict:
    salts_present = sorted(set(labels))

    sil_tsne = silhouette_score(tsne_clean, labels)
    sil_feat = silhouette_score(features, labels)

    intra, inter = [], []
    for salt in salts_present:
        mask = np.array(labels) == salt
        pts = tsne_clean[mask]
        if len(pts) > 1:
            intra.append(pdist(pts).mean())
        for other in salts_present:
            if other != salt:
                omask = np.array(labels) == other
                c1 = pts.mean(axis=0)
                c2 = tsne_clean[omask].mean(axis=0)
                inter.append(np.linalg.norm(c1 - c2))

    sep_ratio = np.mean(inter) / np.mean(intra)

    print("\n  CLUSTER SEPARABILITY VALIDATION")
    print("  " + "=" * 58)
    print(f"  Silhouette Score (t-SNE)    : {sil_tsne:.4f}")
    print(f"  Silhouette Score (features) : {sil_feat:.4f}")
    print(f"  Avg Intra-class Distance    : {np.mean(intra):.2f}")
    print(f"  Avg Inter-class Distance    : {np.mean(inter):.2f}")
    print(f"  Separation Ratio            : {sep_ratio:.2f}  (>3 = excellent)")

    return {
        "sil_tsne": sil_tsne,
        "sil_feat": sil_feat,
        "separation_ratio": sep_ratio,
    }


def plot_centroid_overlay(
    tsne_clean: np.ndarray,
    labels: list[str],
    output_path: str | Path,
) -> None:
    salts_present = [s for s in SALT_ORDER if s in labels]
    centroids = np.array([
        tsne_clean[np.array(labels) == s].mean(axis=0) for s in salts_present
    ])

    plt.figure(figsize=(14, 10))
    for salt in salts_present:
        mask = np.array(labels) == salt
        plt.scatter(tsne_clean[mask, 0], tsne_clean[mask, 1],
                    c=[SALT_COLOR_MAP[salt]], label=salt,
                    alpha=0.4, s=30, edgecolors="none")

    for i, salt in enumerate(salts_present):
        plt.scatter(centroids[i, 0], centroids[i, 1],
                    c=[SALT_COLOR_MAP[salt]], s=250, zorder=10,
                    edgecolors="black", linewidth=2, marker="*")
        plt.annotate(salt, xy=centroids[i], xytext=(0, 14),
                     textcoords="offset points", ha="center", va="bottom",
                     fontsize=10, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                               alpha=0.8, edgecolor="gray"))

    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            plt.plot([centroids[i, 0], centroids[j, 0]],
                     [centroids[i, 1], centroids[j, 1]],
                     "k--", linewidth=0.5, alpha=0.25)

    plt.title("t-SNE Cluster Map: Antibiotic Salt Classes with Centroids",
              fontsize=15, fontweight="bold")
    plt.xlabel("t-SNE Component 1", fontsize=12)
    plt.ylabel("t-SNE Component 2", fontsize=12)
    plt.legend(title="Antibiotic Salt", loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")
