"""Out-of-distribution detection via softmax confidence thresholding."""

from __future__ import annotations

import glob
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_selection import SelectKBest

from viz_utils import SALT_ORDER


def run_ood_detection(
    ood_dir: str | Path,
    resnet: nn.Module,
    selector: SelectKBest,
    models_dict: dict,           # {"Random Forest": rf_model, "MLP": mlp_model, ...}
    device: torch.device,
    target_size: tuple[int, int] = (224, 224),
    pixel_norm: float = 255.0,
    confidence_threshold: float = 0.5,
    output_dir: str | Path = "outputs",
) -> None:
    """Load OOD images from a plain directory (JPG/PNG) and flag low-confidence samples."""
    ood_dir = Path(ood_dir)
    output_dir = Path(output_dir)

    ood_images = sorted(
        glob.glob(str(ood_dir / "**" / "*.jpg"), recursive=True) +
        glob.glob(str(ood_dir / "**" / "*.png"), recursive=True)
    )
    if not ood_images:
        print(f"  No OOD images found in {ood_dir}. Skipping OOD detection.")
        return

    print(f"\n  Found {len(ood_images)} OOD image(s)")

    all_probas: dict[str, list] = {name: [] for name in models_dict}
    img_names: list[str] = []

    for img_path in ood_images:
        img_name = os.path.basename(img_path)
        folder_name = os.path.basename(os.path.dirname(img_path))
        img_names.append(f"{folder_name}/{img_name}")

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        img_array = np.array(img, dtype=np.float32) / pixel_norm
        img_tensor = (torch.FloatTensor(img_array)
                      .permute(2, 0, 1).unsqueeze(0).to(device))

        with torch.no_grad():
            feat = resnet(img_tensor)
            feat = feat.squeeze(-1).squeeze(-1).cpu().numpy()

        feat_selected = selector.transform(feat)

        for name, model in models_dict.items():
            all_probas[name].append(model.predict_proba(feat_selected)[0])

    for name in all_probas:
        all_probas[name] = np.array(all_probas[name])

    max_confs = {name: all_probas[name].max(axis=1) for name in all_probas}

    # Confidence histogram
    n_models = len(max_confs)
    hist_colors = ["darkorange", "mediumseagreen", "steelblue", "mediumpurple"]
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    for ax, (name, max_conf), color in zip(axes, max_confs.items(), hist_colors):
        ax.hist(max_conf, bins=15, color=color, edgecolor="black", alpha=0.8)
        ax.axvline(confidence_threshold, color="red", linestyle="--", linewidth=2,
                   label=f"Threshold = {confidence_threshold}")
        ax.set_title(f"{name} — Max Confidence Distribution\n"
                     f"over {len(ood_images)} OOD Images", fontweight="bold")
        ax.set_xlabel("Max Predicted Probability")
        ax.set_ylabel("Number of Images")
        ax.legend()
        ax.set_xlim(0, 1)
    plt.suptitle("OOD Confidence Distribution per Model", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / "ood_confidence_histogram.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'ood_confidence_histogram.png'}")

    # Per-image class probability bar chart (first image)
    IMG_IDX = 0
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 4), sharey=True)
    if n_models == 1:
        axes = [axes]
    for ax, (name, probas), color in zip(axes, all_probas.items(), hist_colors):
        probs = probas[IMG_IDX]
        bars = ax.bar(SALT_ORDER, probs, color=color, edgecolor="black", alpha=0.85)
        ax.axhline(confidence_threshold, color="red", linestyle="--", linewidth=1.5,
                   label=f"Threshold={confidence_threshold}")
        ax.set_ylim(0, 1)
        ax.set_title(f"{name}\nPredicted: {SALT_ORDER[np.argmax(probs)]}",
                     fontweight="bold")
        ax.set_xlabel("Salt Class")
        ax.set_ylabel("Probability")
        ax.tick_params(axis="x", rotation=30)
        for bar, p in zip(bars, probs):
            ax.text(bar.get_x() + bar.get_width() / 2, p + 0.02,
                    f"{p:.2f}", ha="center", fontsize=8)
    plt.suptitle(f"Class Probability Distribution — {img_names[IMG_IDX]}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "ood_prob_distribution.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'ood_prob_distribution.png'}")

    # OOD verdict pie charts
    palette = [["#4CAF50", "#F44336"], ["#FF9800", "#F44336"],
               ["#9C27B0", "#F44336"], ["#009688", "#F44336"]]
    n = len(ood_images)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    for ax, (name, max_conf), pie_colors in zip(axes, max_confs.items(), palette):
        ood_count = (max_conf < confidence_threshold).sum()
        counts = [ood_count, n - ood_count]
        labels_pie = [
            f"Flagged as OOD\n({ood_count}/{n})",
            f"Misclassified\nwith high confidence\n({n - ood_count}/{n})",
        ]
        ax.pie(counts, labels=labels_pie, colors=pie_colors, autopct="%1.1f%%",
               startangle=90, textprops={"fontsize": 11})
        ax.set_title(f"{name}\n(threshold={confidence_threshold})", fontweight="bold")
    plt.suptitle("OOD Flagging Verdict", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / "ood_verdict_pie.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'ood_verdict_pie.png'}")

    # Summary printout
    print("\n  " + "=" * 58)
    print(f"  OOD DETECTION RESULTS  (threshold τ = {confidence_threshold})")
    print("  " + "=" * 58)
    for name in models_dict:
        max_conf = max_confs[name]
        ood_count = (max_conf < confidence_threshold).sum()
        print(f"\n  {name}")
        print(f"    OOD flagged   : {ood_count}/{n} ({100*ood_count/n:.1f}%)")
        print(f"    Mean max conf : {max_conf.mean():.4f}")
        print(f"    Std  max conf : {max_conf.std():.4f}")
        print(f"    Min  max conf : {max_conf.min():.4f}")
        print(f"    Max  max conf : {max_conf.max():.4f}")
