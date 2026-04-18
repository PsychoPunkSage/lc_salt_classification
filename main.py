"""
LC Salt Classification Pipeline
================================
End-to-end pipeline: data loading → ResNet-50 feature extraction →
DBSCAN outlier removal → dimensionality reduction → classification →
optional OOD detection.

Usage
-----
    python main.py
    python main.py --data-dir data/salts --ood-dir data/OOD
    python main.py --skip-viz --run-ablation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import config
from config import (
    BATCH_SIZE, CONFIDENCE_THRESHOLD, DBSCAN_EPS, DBSCAN_MIN_CLASS_SIZE,
    DBSCAN_MIN_SAMPLES, DBSCAN_REMOVAL_WARN_THRESHOLD, N_TOP_FEATURES,
    PCA_COMPONENTS_TSNE, PCA_COMPONENTS_UMAP, RF_N_ESTIMATORS,
    RF_RANDOM_STATE, RF_TEST_SIZE, SALT_LIST, TSNE_ITERATIONS,
    TSNE_PERPLEXITY, UMAP_MIN_DIST, UMAP_N_NEIGHBORS, device,
)
from classification import (
    make_split, plot_model_comparison, run_mlp_ablation, run_pairwise_rf,
    train_gbm, train_mlp, train_random_forest, train_svm,
)
from data_loader import load_images_from_zips, plot_class_distribution
from feature_extractor import (
    build_resnet_extractor, extract_features, select_top_features,
)
from ood_detection import run_ood_detection
from outlier_removal import compute_initial_tsne, remove_outliers_dbscan
from viz_utils import (
    compute_cluster_metrics, plot_centroid_overlay, plot_lda, plot_mds,
    plot_tsne_2d, plot_tsne_3d, plot_tsne_3d_from_coords,
    plot_tsne_from_coords, plot_umap,
)


def _section(title: str) -> None:
    print(f"\n{'='*64}")
    print(f"  {title}")
    print(f"{'='*64}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Antibiotic salt classification from LC birefringence images."
    )
    p.add_argument(
        "--data-dir", default="data/salts",
        help="Path to folder containing the 6 salt zip files  [default: data/salts]",
    )
    p.add_argument(
        "--output-dir", default="outputs",
        help="Directory for saved plots and CSVs  [default: outputs/]",
    )
    p.add_argument(
        "--ood-dir", default="data/OOD",
        help="Path to folder with OOD images (JPG/PNG, recursive)  [default: data/OOD]",
    )
    p.add_argument(
        "--skip-viz", action="store_true",
        help="Skip dimensionality-reduction plots (faster pipeline run)",
    )
    p.add_argument(
        "--run-ablation", action="store_true",
        help="Run MLP architecture ablation study (slow)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility
    config.set_seeds(42)
    print(f"  Using device: {device}")

    # ------------------------------------------------------------------ #
    _section("1. DATA LOADING")
    # ------------------------------------------------------------------ #
    images, labels = load_images_from_zips(
        data_dir=args.data_dir,
        salt_list=SALT_LIST,
        target_size=(224, 224),
        pixel_norm=255.0,
    )
    plot_class_distribution(labels, SALT_LIST,
                            output_dir / "class_distribution.png")

    # ------------------------------------------------------------------ #
    _section("2. RESNET-50 FEATURE EXTRACTION")
    # ------------------------------------------------------------------ #
    resnet = build_resnet_extractor(device)
    features = extract_features(resnet, images, device, BATCH_SIZE)
    features, selector, selected_indices, _ = select_top_features(
        features, labels, k=N_TOP_FEATURES
    )

    # ------------------------------------------------------------------ #
    _section("3. INITIAL t-SNE (PRE-DBSCAN)")
    # ------------------------------------------------------------------ #
    tsne_before = compute_initial_tsne(
        features,
        pca_components=PCA_COMPONENTS_TSNE,
        perplexity=TSNE_PERPLEXITY,
        n_iter=TSNE_ITERATIONS,
    )
    if not args.skip_viz:
        plot_tsne_from_coords(
            tsne_before, labels,
            "t-SNE 2D — Antibiotic Salt Classes (Before Outlier Removal)",
            output_dir / "tsne_before_dbscan_2d.png",
        )

    # ------------------------------------------------------------------ #
    _section("4. DBSCAN OUTLIER REMOVAL")
    # ------------------------------------------------------------------ #
    features, labels, tsne_perplexity = remove_outliers_dbscan(
        features, tsne_before, labels,
        eps=DBSCAN_EPS,
        min_samples=DBSCAN_MIN_SAMPLES,
        warn_threshold=DBSCAN_REMOVAL_WARN_THRESHOLD,
        min_class_size=DBSCAN_MIN_CLASS_SIZE,
    )

    # ------------------------------------------------------------------ #
    _section("5. DIMENSIONALITY REDUCTION (CLEAN DATA)")
    # ------------------------------------------------------------------ #
    if not args.skip_viz:
        tsne_clean_2d = plot_tsne_2d(
            features, labels,
            "t-SNE 2D — Antibiotic Salt Classes (After Outlier Removal)",
            output_dir / "tsne_clean_2d.png",
            pca_components=PCA_COMPONENTS_TSNE,
            perplexity=tsne_perplexity,
            n_iter=TSNE_ITERATIONS,
        )
        tsne_clean_3d = plot_tsne_3d(
            features, labels,
            "t-SNE 3D — Antibiotic Salt Classes (After Outlier Removal)",
            output_dir / "tsne_clean_3d.png",
            pca_components=PCA_COMPONENTS_TSNE,
            perplexity=tsne_perplexity,
            n_iter=TSNE_ITERATIONS,
        )
        plot_mds(features, labels, output_dir / "mds.png",
                 pca_components=PCA_COMPONENTS_UMAP)
        plot_umap(features, labels, output_dir / "umap.png",
                  pca_components=PCA_COMPONENTS_UMAP,
                  n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST)
        plot_lda(features, labels, output_dir)
        metrics = compute_cluster_metrics(tsne_clean_2d, features, labels)
        plot_centroid_overlay(tsne_clean_2d, labels,
                              output_dir / "tsne_centroid_overlay.png")
    else:
        # Still need t-SNE coords for cluster metrics — compute silently
        tsne_clean_2d = plot_tsne_2d(
            features, labels,
            "t-SNE 2D (clean)",
            output_dir / "tsne_clean_2d.png",
            pca_components=PCA_COMPONENTS_TSNE,
            perplexity=tsne_perplexity,
            n_iter=TSNE_ITERATIONS,
        )
        metrics = compute_cluster_metrics(tsne_clean_2d, features, labels)

    # ------------------------------------------------------------------ #
    _section("6. CLASSIFICATION")
    # ------------------------------------------------------------------ #
    X_train, X_test, y_train, y_test = make_split(
        features, labels, test_size=RF_TEST_SIZE, random_state=RF_RANDOM_STATE
    )

    rf_model, rf_acc = train_random_forest(
        X_train, X_test, y_train, y_test,
        selected_indices, output_dir,
        n_estimators=RF_N_ESTIMATORS, random_state=RF_RANDOM_STATE,
    )
    mlp_model, mlp_acc = train_mlp(
        X_train, X_test, y_train, y_test, output_dir, rf_acc
    )
    svm_model, svm_acc = train_svm(
        X_train, X_test, y_train, y_test, output_dir, rf_acc, mlp_acc
    )
    gbm_model, gbm_acc = train_gbm(
        X_train, X_test, y_train, y_test, output_dir, rf_acc, mlp_acc, svm_acc
    )

    plot_model_comparison(
        {"Random Forest": rf_acc, "MLP": mlp_acc,
         "SVM": svm_acc, "Gradient Boosting": gbm_acc},
        output_dir,
    )

    # ------------------------------------------------------------------ #
    _section("7. PAIRWISE CLASSIFICATION (15 PAIRS)")
    # ------------------------------------------------------------------ #
    results_df = run_pairwise_rf(
        features, labels, output_dir,
        n_estimators=RF_N_ESTIMATORS,
        test_size=RF_TEST_SIZE,
        random_state=RF_RANDOM_STATE,
    )

    # ------------------------------------------------------------------ #
    _section("FINAL PIPELINE SUMMARY")
    # ------------------------------------------------------------------ #
    print(f"\n  [DATASET]")
    print(f"    Salt classes        : {', '.join(sorted(set(labels)))}")
    print(f"    Images after DBSCAN : {len(labels)}")
    print(f"\n  [CLUSTER SEPARABILITY]")
    print(f"    Silhouette (t-SNE)  : {metrics['sil_tsne']:.4f}")
    print(f"    Silhouette (feats)  : {metrics['sil_feat']:.4f}")
    print(f"    Separation Ratio    : {metrics['separation_ratio']:.2f}")
    print(f"\n  [6-CLASS CLASSIFICATION]")
    print(f"    Random Forest       : {rf_acc*100:.1f}%")
    print(f"    MLP                 : {mlp_acc*100:.1f}%")
    print(f"    SVM                 : {svm_acc*100:.1f}%")
    print(f"    Gradient Boosting   : {gbm_acc*100:.1f}%")
    print(f"\n  [PAIRWISE (15 pairs)]")
    print(f"    Mean accuracy       : {results_df['Test_Accuracy'].mean():.4f}")
    print(f"    Best pair           : {results_df.iloc[0]['Pair']}"
          f"  ({results_df.iloc[0]['Test_Accuracy']:.1%})")
    print(f"    Worst pair          : {results_df.iloc[-1]['Pair']}"
          f"  ({results_df.iloc[-1]['Test_Accuracy']:.1%})")
    print(f"\n  Outputs saved to: {output_dir.resolve()}")
    print("\n  Pipeline complete.")

    # ------------------------------------------------------------------ #
    # Optional: OOD detection
    # ------------------------------------------------------------------ #
    if args.ood_dir and Path(args.ood_dir).exists():
        _section("8. OOD DETECTION")
        run_ood_detection(
            ood_dir=args.ood_dir,
            resnet=resnet,
            selector=selector,
            models_dict={"Random Forest": rf_model, "MLP": mlp_model},
            device=device,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            output_dir=output_dir,
        )

    # ------------------------------------------------------------------ #
    # Optional: MLP ablation
    # ------------------------------------------------------------------ #
    if args.run_ablation:
        _section("9. MLP ARCHITECTURE ABLATION")
        run_mlp_ablation(X_train, X_test, y_train, y_test, output_dir)


if __name__ == "__main__":
    main()
