"""All classifiers: RF, MLP, SVM, GBM, pairwise RF, and MLP ablation."""

from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from tqdm import tqdm

from viz_utils import SALT_ORDER


# ---------------------------------------------------------------------------
# Train / test split (shared)
# ---------------------------------------------------------------------------

def make_split(
    features: np.ndarray,
    labels: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    return train_test_split(
        features, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )


# ---------------------------------------------------------------------------
# Random Forest — 6-class
# ---------------------------------------------------------------------------

def train_random_forest(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: list[str],
    y_test: list[str],
    selected_feature_indices: np.ndarray,
    output_dir: Path,
    n_estimators: int = 100,
    random_state: int = 42,
) -> tuple[RandomForestClassifier, float]:
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=random_state,
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = rf.score(X_test, y_test)

    print("\n  " + "=" * 58)
    print("  6-CLASS RANDOM FOREST CLASSIFICATION RESULTS")
    print("  " + "=" * 58)
    print(f"  Training samples : {len(X_train)}")
    print(f"  Test samples     : {len(X_test)}")
    print(f"  Test Accuracy    : {accuracy:.4f} ({accuracy*100:.1f}%)")
    print()
    print(classification_report(y_test, y_pred, target_names=SALT_ORDER))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=SALT_ORDER)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=SALT_ORDER, yticklabels=SALT_ORDER,
                ax=ax, linewidths=0.5, linecolor="gray")
    ax.set_title("Confusion Matrix — 6-Class Antibiotic Salt Classification",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("True Salt Class", fontsize=12)
    ax.set_xlabel("Predicted Salt Class", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "rf_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'rf_confusion_matrix.png'}")

    # Feature importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:20]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(20), importances[indices],
           color=plt.cm.viridis(np.linspace(0.2, 0.9, 20)),
           edgecolor="black", linewidth=0.6)
    ax.set_title("Top 20 Feature Importances — Random Forest",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Feature Rank (by importance)", fontsize=11)
    ax.set_ylabel("Feature Importance (Gini)", fontsize=11)
    ax.set_xticks(range(20))
    ax.set_xticklabels(
        [f"F{selected_feature_indices[i]}" for i in indices],
        rotation=45, ha="right", fontsize=8,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "rf_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'rf_feature_importance.png'}")

    return rf, accuracy


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

def train_mlp(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: list[str],
    y_test: list[str],
    output_dir: Path,
    rf_accuracy: float,
) -> tuple[MLPClassifier, float]:
    mlp = MLPClassifier(
        hidden_layer_sizes=(512, 256),
        activation="relu",
        solver="adam",
        max_iter=2000,
        tol=1e-4,
        n_iter_no_change=20,
        random_state=42,
    )
    mlp.fit(X_train, list(y_train))
    y_pred = mlp.predict(X_test)
    accuracy = mlp.score(X_test, list(y_test))

    print("\n  " + "=" * 58)
    print("  NEURAL NETWORK (MLP) CLASSIFIER RESULTS")
    print("  " + "=" * 58)
    print(f"  Test Accuracy : {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"  RF Accuracy   : {rf_accuracy:.4f}  "
          f"(improvement: {(accuracy - rf_accuracy)*100:+.1f}%)")
    print()
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=SALT_ORDER)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                xticklabels=SALT_ORDER, yticklabels=SALT_ORDER)
    plt.title("Confusion Matrix: Neural Network (MLP) Classifier",
              fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Salt")
    plt.ylabel("True Salt")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "mlp_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'mlp_confusion_matrix.png'}")

    return mlp, accuracy


# ---------------------------------------------------------------------------
# SVM
# ---------------------------------------------------------------------------

def train_svm(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: list[str],
    y_test: list[str],
    output_dir: Path,
    rf_accuracy: float,
    mlp_accuracy: float,
) -> tuple[SVC, float]:
    svm = SVC(
        kernel="rbf", C=10, gamma="scale",
        probability=True, class_weight="balanced", random_state=42,
    )
    svm.fit(X_train, list(y_train))
    y_pred = svm.predict(X_test)
    accuracy = svm.score(X_test, list(y_test))

    print("\n  " + "=" * 58)
    print("  SVM CLASSIFIER RESULTS")
    print("  " + "=" * 58)
    print(f"  Test Accuracy : {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"  RF  Accuracy  : {rf_accuracy:.4f}")
    print(f"  MLP Accuracy  : {mlp_accuracy:.4f}")
    print()
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=SALT_ORDER)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=SALT_ORDER, yticklabels=SALT_ORDER)
    plt.title("Confusion Matrix: SVM Classifier", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Salt")
    plt.ylabel("True Salt")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "svm_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'svm_confusion_matrix.png'}")

    return svm, accuracy


# ---------------------------------------------------------------------------
# Gradient Boosting
# ---------------------------------------------------------------------------

def train_gbm(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: list[str],
    y_test: list[str],
    output_dir: Path,
    rf_accuracy: float,
    mlp_accuracy: float,
    svm_accuracy: float,
) -> tuple[GradientBoostingClassifier, float]:
    gbm = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1,
        max_depth=4, subsample=0.8, random_state=42,
    )
    gbm.fit(X_train, list(y_train))
    y_pred = gbm.predict(X_test)
    accuracy = gbm.score(X_test, list(y_test))

    print("\n  " + "=" * 58)
    print("  GRADIENT BOOSTING CLASSIFIER RESULTS")
    print("  " + "=" * 58)
    print(f"  Test Accuracy : {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"  RF  Accuracy  : {rf_accuracy:.4f}")
    print(f"  MLP Accuracy  : {mlp_accuracy:.4f}")
    print(f"  SVM Accuracy  : {svm_accuracy:.4f}")
    print()
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=SALT_ORDER)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=SALT_ORDER, yticklabels=SALT_ORDER)
    plt.title("Confusion Matrix: Gradient Boosting Classifier",
              fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Salt")
    plt.ylabel("True Salt")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "gbm_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'gbm_confusion_matrix.png'}")

    return gbm, accuracy


# ---------------------------------------------------------------------------
# Model comparison bar chart
# ---------------------------------------------------------------------------

def plot_model_comparison(
    accuracies: dict[str, float],
    output_dir: Path,
) -> None:
    model_names = list(accuracies.keys())
    model_accs  = list(accuracies.values())
    colors = ["steelblue", "darkorange", "mediumpurple", "mediumseagreen"]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(model_names, [a * 100 for a in model_accs],
                   color=colors[:len(model_names)], edgecolor="black", width=0.5)
    plt.ylim(70, 100)
    plt.ylabel("Test Accuracy (%)", fontsize=12)
    plt.title("Model Comparison: Test Accuracy", fontsize=14, fontweight="bold")
    for bar, acc in zip(bars, model_accs):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{acc*100:.1f}%", ha="center", va="bottom",
                 fontweight="bold", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'model_comparison.png'}")


# ---------------------------------------------------------------------------
# Pairwise RF (all 15 pairs)
# ---------------------------------------------------------------------------

def run_pairwise_rf(
    features: np.ndarray,
    labels: list[str],
    output_dir: Path,
    n_estimators: int = 100,
    test_size: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    all_pairs = list(itertools.combinations(SALT_ORDER, 2))
    print(f"\n  Total pairwise combinations (6C2): {len(all_pairs)}")

    pairwise_results: list[dict] = []
    n_cols = 3
    n_rows = (len(all_pairs) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 5))
    axes_flat = axes.flatten()

    for idx, (salt1, salt2) in enumerate(tqdm(all_pairs, desc="  Pairwise RF")):
        mask = np.array([(y == salt1 or y == salt2) for y in labels])
        X_pair = features[mask]
        y_pair = np.array(labels)[mask]

        if len(np.unique(y_pair)) < 2:
            print(f"  Skipping {salt1} vs {salt2}: insufficient samples")
            continue

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_pair, y_pair,
            test_size=test_size, random_state=random_state, stratify=y_pair,
        )
        rf_pair = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            random_state=random_state,
        )
        rf_pair.fit(X_tr, y_tr)
        y_pred_pair = rf_pair.predict(X_te)
        acc = np.mean(y_pred_pair == y_te)

        pairwise_results.append({
            "Salt_1": salt1, "Salt_2": salt2,
            "Pair": f"{salt1} vs {salt2}",
            "Test_Accuracy": round(acc, 4),
            "Test_Samples": len(y_te),
            "Train_Samples": len(y_tr),
        })

        cm_pair = confusion_matrix(y_te, y_pred_pair, labels=[salt1, salt2])
        ax = axes_flat[idx]
        sns.heatmap(cm_pair, annot=True, fmt="d", cmap="Blues",
                    xticklabels=[salt1[:6], salt2[:6]],
                    yticklabels=[salt1[:6], salt2[:6]],
                    ax=ax, linewidths=0.5)
        ax.set_title(f"{salt1[:6]} vs {salt2[:6]}\nAcc: {acc:.1%}",
                     fontsize=10, fontweight="bold")
        ax.set_ylabel("True", fontsize=9)
        ax.set_xlabel("Predicted", fontsize=9)

    for j in range(len(all_pairs), len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle("Confusion Matrices — All 15 Pairwise Antibiotic Salt Classifications",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / "pairwise_confusion_matrices.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'pairwise_confusion_matrices.png'}")

    results_df = pd.DataFrame(pairwise_results)
    results_df = results_df.sort_values("Test_Accuracy", ascending=False).reset_index(drop=True)
    results_df.index += 1
    results_df.index.name = "Rank"

    csv_path = output_dir / "pairwise_classification_results.csv"
    results_df.to_csv(csv_path)
    print(f"  Saved: {csv_path}")

    print("\n  PAIRWISE RESULTS — RANKED BY ACCURACY")
    print("  " + "=" * 68)
    print(results_df[["Pair", "Test_Accuracy", "Test_Samples"]].to_string())
    print(f"\n  Best pair  : {results_df.iloc[0]['Pair']}  "
          f"(acc={results_df.iloc[0]['Test_Accuracy']:.1%})")
    print(f"  Worst pair : {results_df.iloc[-1]['Pair']}  "
          f"(acc={results_df.iloc[-1]['Test_Accuracy']:.1%})")
    print(f"  Mean accuracy : {results_df['Test_Accuracy'].mean():.4f}")
    print(f"  Std  accuracy : {results_df['Test_Accuracy'].std():.4f}")

    # Bar plot
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    bar_colors = plt.cm.RdYlGn(results_df["Test_Accuracy"].values)
    bars = ax2.barh(results_df["Pair"], results_df["Test_Accuracy"],
                    color=bar_colors, edgecolor="black", linewidth=0.6)
    mean_acc = results_df["Test_Accuracy"].mean()
    ax2.axvline(0.5, color="red", linestyle="--", linewidth=1.5, label="50% baseline")
    ax2.axvline(mean_acc, color="navy", linestyle=":", linewidth=1.5,
                label=f"Mean ({mean_acc:.1%})")
    for bar, acc in zip(bars, results_df["Test_Accuracy"]):
        ax2.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{acc:.1%}", va="center", fontsize=9)
    ax2.set_title("Pairwise Classification Accuracy — All 15 Antibiotic Salt Pairs",
                  fontsize=14, fontweight="bold")
    ax2.set_xlabel("Test Accuracy", fontsize=12)
    ax2.set_ylabel("Salt Pair", fontsize=12)
    ax2.set_xlim(0, 1.1)
    ax2.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "pairwise_accuracy_barplot.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'pairwise_accuracy_barplot.png'}")

    return results_df


# ---------------------------------------------------------------------------
# MLP ablation (optional)
# ---------------------------------------------------------------------------

def run_mlp_ablation(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: list[str],
    y_test: list[str],
    output_dir: Path,
) -> None:
    layer_configs = {
        1: [(64,), (128,), (256,), (512,)],
        2: [(64, 32), (128, 64), (256, 128), (512, 256)],
        3: [(64, 32, 16), (128, 64, 32), (256, 128, 64), (512, 256, 128)],
        4: [(128, 64, 32, 16), (256, 128, 64, 32), (512, 256, 128, 64)],
        5: [(256, 128, 64, 32, 16), (512, 256, 128, 64, 32)],
    }
    results = []
    for n_layers, configs in layer_configs.items():
        for config in configs:
            mlp_temp = MLPClassifier(
                hidden_layer_sizes=config, activation="relu", solver="adam",
                max_iter=2000, tol=1e-4, n_iter_no_change=20, random_state=42,
            )
            mlp_temp.fit(X_train, list(y_train))
            acc = mlp_temp.score(X_test, list(y_test))
            results.append({
                "layers": n_layers, "config": config,
                "neurons": sum(config), "acc": round(acc, 4),
            })

    COLORS = {1: "#378ADD", 2: "#1D9E75", 3: "#E24B4A",
              4: "#EF9F27", 5: "#7F77DD"}
    max_neurons = max(r["neurons"] for r in results)
    best = max(results, key=lambda r: r["acc"])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    plt.subplots_adjust(bottom=0.18, right=0.78)
    rng = np.random.RandomState(0)

    for r in results:
        jitter = rng.uniform(-0.08, 0.08)
        size = 80 + (r["neurons"] / max_neurons) * 1200
        is_best = r is best
        ax.scatter(r["layers"] + jitter, r["acc"],
                   s=size, color=COLORS[r["layers"]],
                   edgecolors="black" if is_best else "white",
                   linewidths=2 if is_best else 0.6,
                   alpha=0.92 if is_best else 0.75,
                   zorder=5 if is_best else 3)
        if is_best:
            ax.annotate(f"acc = {best['acc']:.4f}",
                        xy=(r["layers"] + jitter, r["acc"]),
                        xytext=(14, 4), textcoords="offset points",
                        fontsize=8.5, color="black", fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

    ax.set_xlabel("Number of hidden layers", fontsize=11)
    ax.set_ylabel("Test accuracy", fontsize=11)
    ax.set_title("MLP architecture search — accuracy vs depth vs width\n"
                 "(bubble size = total neurons)",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlim(0.4, 5.6)
    ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    color_handles = [
        mpatches.Patch(color=COLORS[L], label=f"{L} layer{'s' if L > 1 else ''}")
        for L in range(1, 6)
    ]
    leg1 = ax.legend(handles=color_handles, title="layer depth",
                     loc="upper left", bbox_to_anchor=(0.0, -0.16),
                     ncol=5, fontsize=9, title_fontsize=9, frameon=False)
    ax.add_artist(leg1)

    REF_NEURONS = (64, 256, 512, 992)
    size_handles = [
        ax.scatter([], [], s=((80 + (n / max_neurons) * 1200) * 0.55),
                   color="gray", alpha=0.45, label=f"{n}")
        for n in REF_NEURONS
    ]
    ax.legend(handles=size_handles, title="total neurons",
              loc="upper left", bbox_to_anchor=(1.03, 1.0),
              ncol=1, fontsize=9, title_fontsize=9, frameon=False,
              labelspacing=2.0, handletextpad=1.2)

    plt.savefig(output_dir / "mlp_ablation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'mlp_ablation.png'}")
    print(f"  Best: {best['config']}  {best['layers']} layers  acc={best['acc']:.4f}")
