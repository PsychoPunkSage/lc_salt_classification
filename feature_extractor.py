"""ResNet-50 feature extraction and ANOVA-based feature selection."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
from sklearn.feature_selection import SelectKBest, f_classif
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def build_resnet_extractor(device: torch.device) -> nn.Module:
    """Return pretrained ResNet-50 with the final FC layer removed."""
    resnet = tv_models.resnet50(pretrained=True)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval()
    resnet = resnet.to(device)
    return resnet


def extract_features(
    resnet: nn.Module,
    images: np.ndarray,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Run images through frozen ResNet-50 and return (N, 2048) feature matrix.

    Parameters
    ----------
    images : ndarray, shape (N, H, W, 3), float32, range [0, 1]
    """
    # (N, H, W, C) -> (N, C, H, W)
    tensor = torch.FloatTensor(images).permute(0, 3, 1, 2)
    dataset = TensorDataset(tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    features_list: list[torch.Tensor] = []
    with torch.no_grad():
        for (batch,) in tqdm(loader, desc="ResNet-50 feature extraction"):
            batch = batch.to(device)
            feats = resnet(batch)
            feats = feats.squeeze(-1).squeeze(-1)   # (B, 2048, 1, 1) -> (B, 2048)
            if feats.dim() == 1:
                feats = feats.unsqueeze(0)
            features_list.append(feats.cpu())

    extracted = torch.cat(features_list, dim=0).numpy()
    torch.cuda.empty_cache()

    print(f"\n  Extracted features shape: {extracted.shape}  (images x 2048-dim vectors)")
    return extracted


def select_top_features(
    features: np.ndarray,
    labels: list[str],
    k: int = 100,
) -> tuple[np.ndarray, SelectKBest, np.ndarray, np.ndarray]:
    """Apply ANOVA F-statistic SelectKBest to reduce (N, 2048) -> (N, k).

    Returns
    -------
    features_selected   : ndarray (N, k)
    selector            : fitted SelectKBest (needed for OOD transform)
    selected_indices    : ndarray of chosen feature indices in original 2048-space
    feature_scores      : F-scores for the selected features
    """
    selector = SelectKBest(score_func=f_classif, k=k)
    features_selected = selector.fit_transform(features, labels)

    selected_indices = selector.get_support(indices=True)
    feature_scores = selector.scores_[selected_indices]

    print("\n  " + "=" * 58)
    print("  RESNET-50 FEATURE EXTRACTION SUMMARY")
    print("  " + "=" * 58)
    print(f"  Total images processed        : {features_selected.shape[0]}")
    print(f"  Feature vector size           : {features_selected.shape[1]} dimensions")
    print(f"  Feature range                 : [{features_selected.min():.4f}, {features_selected.max():.4f}]")
    print(f"  Mean / Std                    : {features_selected.mean():.4f} / {features_selected.std():.4f}")
    print(f"  No NaNs                       : {not np.isnan(features_selected).any()}")
    print(f"\n  Top-5 selected feature indices: {selected_indices[:5]}")
    print(f"  Top-5 F-scores                : {[f'{s:.2f}' for s in feature_scores[:5]]}")

    return features_selected, selector, selected_indices, feature_scores
