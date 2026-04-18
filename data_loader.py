"""Load antibiotic salt images directly from zip files — no extraction to disk."""

from __future__ import annotations

import zipfile
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_images_from_zips(
    data_dir: str | Path,
    salt_list: list[str],
    target_size: tuple[int, int] = (224, 224),
    pixel_norm: float = 255.0,
) -> tuple[np.ndarray, list[str]]:
    """Read every salt zip in-memory, decode JPEGs, and return normalised arrays.

    Returns
    -------
    images : ndarray, shape (N, H, W, 3), dtype float32, range [0, 1]
    labels : list of N salt-name strings
    """
    data_dir = Path(data_dir)
    image_array: list[np.ndarray] = []
    salt_labels: list[str] = []

    for salt in salt_list:
        zip_path = data_dir / f"{salt}.zip"
        if not zip_path.exists():
            raise FileNotFoundError(f"Zip not found: {zip_path}")

        count = 0
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                raw = zf.read(name)
                buf = np.frombuffer(raw, dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, target_size)
                image_array.append(img)
                salt_labels.append(salt)
                count += 1

        print(f"  Loaded {count:3d} images for salt: {salt}")

    images = np.array(image_array, dtype=np.float32) / pixel_norm

    # Assertions
    assert images.shape[1:] == (*target_size, 3), \
        f"Unexpected image shape: {images.shape[1:]}"
    assert len(set(salt_labels)) == len(salt_list), \
        f"Expected {len(salt_list)} salt classes, got {len(set(salt_labels))}"

    print(f"\n  Total images loaded: {images.shape[0]}  |  Classes: {len(set(salt_labels))}")
    return images, salt_labels


def plot_class_distribution(
    labels: list[str],
    salt_list: list[str],
    output_path: str | Path,
) -> None:
    counts = Counter(labels)
    salts_sorted = sorted(counts.keys())
    values = [counts[s] for s in salts_sorted]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        salts_sorted, values,
        color=plt.cm.tab10(np.linspace(0, 1, len(salt_list))),
        edgecolor="black", linewidth=0.8,
    )
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            str(v), ha="center", va="bottom", fontsize=11, fontweight="bold",
        )
    ax.set_title("Class Distribution — Antibiotic Salt Images (1 mM)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Antibiotic Salt", fontsize=12)
    ax.set_ylabel("Number of Images", fontsize=12)
    ax.set_ylim(0, max(values) + 10)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")

    print("\n  CLASS DISTRIBUTION SUMMARY")
    print("  " + "=" * 44)
    for s in salts_sorted:
        pct = counts[s] / len(labels) * 100
        print(f"  {s:15s}: {counts[s]:3d} images ({pct:.1f}%)")
    print(f"  {'TOTAL':15s}: {len(labels):3d} images")
    max_c, min_c = max(values), min(values)
    ratio = max_c / min_c
    print(f"\n  Imbalance ratio (max/min): {ratio:.2f}")
    if ratio > 1.5:
        print("  WARNING: class imbalance detected — using class_weight='balanced'.")
