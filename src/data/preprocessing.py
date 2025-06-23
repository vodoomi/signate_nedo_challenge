"""Data preprocessing functions for NEDO Challenge."""

import numpy as np
import torch


def preprocess_dataset(mat_data, ref_flag=False):
    """
    Preprocess dataset from MAT file format.
    
    Args:
        mat_data: Dictionary containing MAT data
        ref_flag: If True, data contains reference format with 
                 (train_emg, train_cog, pred_emg, pred_cog, board_stance)
    
    Returns:
        np.ndarray: Preprocessed EMG data
    """
    # Set indices based on data format
    if ref_flag:
        emg_idx = 2
        board_idx = 4
    else:
        emg_idx = 0
        board_idx = 4
    
    # Get player keys
    player_keys = [key for key in mat_data.keys() if key.isdigit()]
    
    # Handle board stance - flip left/right for goofy stance
    if not ref_flag:
        for p_key in player_keys:
            if mat_data[p_key][0][0][board_idx][0] == "goofy":
                n_channel = mat_data[p_key][0][0][emg_idx].shape[1]
                # Reorder channels: 1, 0, 3, 2, 5, 4, ...
                new_order_list = np.array([
                    [i, i-1] for i in range(1, n_channel, 2)
                ]).flatten().tolist()
                mat_data[p_key][0][0][emg_idx] = mat_data[p_key][0][0][emg_idx][:, new_order_list]
    
    # Extract EMG data
    emg_data = np.concatenate([
        mat_data[p_key][0][0][emg_idx] for p_key in player_keys
    ])
    
    return emg_data


def extract_labels(mat_data, ref_flag=False):
    """
    Extract labels from MAT data.
    
    Args:
        mat_data: Dictionary containing MAT data
        ref_flag: If True, extract reference labels
    
    Returns:
        np.ndarray: Flattened labels
    """
    if ref_flag:
        # Reference data: extract prediction targets
        player_keys = ["0005"]  # Only player 5 in reference
        labels = np.concatenate([
            mat_data[p_key][0][0][3] for p_key in player_keys
        ])
        # Apply sign corrections for reference data
        labels *= -1
        labels[:, 2::3] = labels[:, 2::3] * -1
    else:
        # Training data: extract COG velocity
        player_keys = [f"000{player_id}" for player_id in range(1, 5)]
        labels = np.concatenate([
            mat_data[p_key][0][0][1] for p_key in player_keys
        ])
    
    # Flatten labels for model training
    labels = labels.transpose(0, 2, 1).reshape(labels.shape[0], -1)
    
    return labels


def create_channel_flip_indices():
    """Create indices for flipping left-right channels."""
    # Assumes 16 channels, paired as: (0,1), (2,3), (4,5), ...
    # Returns: [1, 0, 3, 2, 5, 4, ...]
    return np.array([
        [i, i-1] for i in range(1, 16, 2)
    ]).flatten().tolist()


def prepare_augmented_data(data_array, labels):
    """
    Prepare augmented data with left-right flips.
    
    Args:
        data_array: Original data array
        labels: Original labels
    
    Returns:
        tuple: (augmented_data, augmented_labels, is_reverse_flags)
    """
    # Create reversed data
    reverse_array = data_array.clone()
    new_order_list = create_channel_flip_indices()
    reverse_array = reverse_array[:, new_order_list]
    
    # Combine original and reversed data
    augmented_data = torch.cat([data_array, reverse_array], dim=0)
    
    # Duplicate labels
    augmented_labels = np.concatenate([labels, labels])
    
    # Create flags indicating original (0) vs reversed (1) data
    is_reverse = np.repeat(np.arange(2), len(data_array))
    
    return augmented_data, augmented_labels, is_reverse
