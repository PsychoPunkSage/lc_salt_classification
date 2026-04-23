import random
import numpy as np
import torch

# ============================================================
# CONFIGURATION - edit these values to reconfigure the pipeline
# ============================================================

SALT_LIST = [
    "azlocillin", "carbenicillin", "oxacillin",
    "penicillin", "pipercilin", "ticarcillin"
]
EXPECTED_TOTAL = 366

# Image preprocessing
TARGET_SIZE  = (224, 224)
PIXEL_NORM   = 255.0

# Feature extraction
BATCH_SIZE     = 32
N_TOP_FEATURES = 100   # SelectKBest k

# Dimensionality reduction
PCA_COMPONENTS_TSNE = 50
PCA_COMPONENTS_UMAP = 50
TSNE_PERPLEXITY     = 30
TSNE_ITERATIONS     = 1000

# DBSCAN outlier removal
DBSCAN_EPS                    = 3.5
DBSCAN_MIN_SAMPLES            = 5
DBSCAN_REMOVAL_WARN_THRESHOLD = 0.15
DBSCAN_MIN_CLASS_SIZE         = 35

# UMAP
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST    = 0.1

# Classification
RF_N_ESTIMATORS = 100
RF_TEST_SIZE    = 0.2
RF_RANDOM_STATE = 42

# OOD
CONFIDENCE_THRESHOLD = 0.5

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
