"""Utility functions for NEDO Challenge."""

import os
import random
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
import albumentations as A
from albumentations.pytorch import ToTensorV2


def set_random_seed(seed: int = 42, deterministic: bool = False):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic


def get_transforms(image_size: int):
    """Get data augmentation transforms."""
    train_transform = A.Compose([
        A.Resize(p=1.0, height=image_size, width=image_size),
        ToTensorV2(p=1.0)
    ])
    val_transform = A.Compose([
        A.Resize(p=1.0, height=image_size, width=image_size),
        ToTensorV2(p=1.0)
    ])
    return train_transform, val_transform


def compute_mcrmse_original(y_pred, y_test, player_id_list):
    """Compute MCRMSE score (original implementation)."""
    rmses = []
    for player_id in np.unique(player_id_list):
        y_pred_i = y_pred[player_id_list == player_id]
        y_test_i = y_test[player_id_list == player_id]
        rmse_xyz = [
            mean_squared_error(y_pred_i[:, i::3], y_test_i[:, i::3])
            for i in range(3)
        ]
        rmse = np.sqrt(np.sum(rmse_xyz, axis=0))
        rmses.append(rmse)
        
    mcrmse = np.mean(rmses)
    return mcrmse


def compute_mcrmse(y_pred, y_test, player_id_list):
    """Compute MCRMSE score (improved implementation)."""
    rmses = []
    for player_id in np.unique(player_id_list):
        y_pred_i = y_pred[player_id_list == player_id]
        y_test_i = y_test[player_id_list == player_id]
        for idx in range(y_pred_i.shape[0]):
            mse_xyz = [
                (y_pred_i[idx, i::3] - y_test_i[idx, i::3])**2
                for i in range(3)
            ]
            rmse = np.sqrt(np.sum(mse_xyz, axis=0).mean())
            rmses.append(rmse)
        
    mcrmse = np.mean(rmses)
    return mcrmse


def get_player_data_indices():
    """Get the number of trials per player and cumulative indices."""
    n_trial_list = [0, 319, 300, 320, 320]
    n_trial_cumsum_list = np.cumsum(n_trial_list)
    return n_trial_list, n_trial_cumsum_list
