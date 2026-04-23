# LC Salt Classification

End-to-end machine learning pipeline for classifying six structurally similar beta-lactam antibiotic salts from liquid crystal (LC) birefringence microscopy images.

**ResNet-50** features → **DBSCAN** outlier removal → **t-SNE / MDS / UMAP / LDA** visualisation → **Random Forest / MLP / SVM / Gradient Boosting** classification → optional **OOD detection**.

---

## Dataset

Six antibiotic salt classes, each at 1 mM concentration, imaged under cross-polarised microscopy:

| Salt | Images (raw) | After DBSCAN |
|---|---|---|
| Azlocillin | 67 | 65 |
| Carbenicillin | 64 | 62 |
| Oxacillin | 73 | 67 |
| Penicillin | 55 | 48 |
| Pipercilin | 47 | 36 |
| Ticarcillin | 60 | 60 |
| **Total** | **366** | **338** |

Each zip file unpacks to a flat folder of `square_XXXX.jpg` images (224 × 224 px, RGB). The pipeline reads images **directly from the zip files** — no manual extraction needed.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
# Basic run — point at the folder containing the 6 zip files
python main.py --data-dir /path/to/salts/

# Skip dimensionality-reduction plots (much faster)
python main.py --data-dir /path/to/salts/ --skip-viz

# With OOD detection (provide a folder of unknown images)
python main.py --data-dir /path/to/salts/ --ood-dir /path/to/ood_images/

# Full run including MLP architecture ablation study
python main.py --data-dir /path/to/salts/ --run-ablation

# Custom output directory
python main.py --data-dir /path/to/salts/ --output-dir my_results/
```

All arguments:

| Argument | Default | Description |
|---|---|---|
| `--data-dir` | `../salts` | Folder containing the 6 `{salt}.zip` files |
| `--output-dir` | `outputs/` | Directory for saved plots and CSVs |
| `--ood-dir` | *(none)* | Optional folder with OOD images (JPG/PNG) |
| `--skip-viz` | False | Skip dimensionality-reduction plots |
| `--run-ablation` | False | Run MLP depth/width ablation study |

---

## Outputs

All files are saved to `--output-dir` (default: `outputs/`):

| File | Description |
|---|---|
| `class_distribution.png` | Bar chart of per-class image counts |
| `tsne_before_dbscan_2d.png` | t-SNE before outlier removal |
| `tsne_clean_2d.png` / `tsne_clean_3d.png` | t-SNE after DBSCAN cleanup |
| `mds.png` | MDS with inter-centroid distances |
| `umap.png` | UMAP embedding |
| `lda_ld1_ld2.png` / `lda_ld1_ld3.png` | LDA projections |
| `tsne_centroid_overlay.png` | t-SNE with centroid scatter overlay |
| `rf_confusion_matrix.png` | Random Forest confusion matrix |
| `rf_feature_importance.png` | Top-20 Gini feature importances |
| `mlp_confusion_matrix.png` | MLP confusion matrix |
| `svm_confusion_matrix.png` | SVM confusion matrix |
| `gbm_confusion_matrix.png` | Gradient Boosting confusion matrix |
| `model_comparison.png` | Side-by-side accuracy bar chart |
| `pairwise_confusion_matrices.png` | 5 × 3 grid of all 15 pairwise CMs |
| `pairwise_accuracy_barplot.png` | Ranked horizontal bar chart |
| `pairwise_classification_results.csv` | Full pairwise results table |
| `ood_confidence_histogram.png` | OOD confidence histograms *(if --ood-dir)* |
| `ood_prob_distribution.png` | Per-class probabilities for first OOD image |
| `ood_verdict_pie.png` | OOD flagging verdict pie charts |
| `mlp_ablation.png` | Bubble chart of MLP arch search *(if --run-ablation)* |

---

## Key Results

| Model | Test Accuracy |
|---|---|
| Random Forest | 88.4% |
| MLP (512-256) | 92.8% |
| **Pairwise RF (mean)** | **94.4%** |

---

## Project Structure

```
lc_salt_classification/
├── main.py                  # CLI entry point
├── config.py                # All hyperparameters
├── data_loader.py           # In-memory zip loading
├── feature_extractor.py     # ResNet-50 + SelectKBest
├── outlier_removal.py       # DBSCAN in t-SNE space
├── viz_utils.py             # All dimensionality-reduction plots
├── classification.py        # RF, MLP, SVM, GBM, pairwise, ablation
├── ood_detection.py         # Softmax-confidence OOD flagging
├── requirements.txt
└── outputs/                 # Auto-created; all generated files land here
```

---

## Citation

Abhinav Prakash, *Liquid Crystal-Based Chemical Sensing: Quantitative H⁺ Concentration Detection and Multi-Analyte Antibiotic Salt Classification via Machine Learning*, IIT Kharagpur, 2026.
