"""Data preprocessing script for NEDO Challenge."""

import os
import argparse
import scipy.io as sio
import torch

from src.models.cwt import CWT
from src.data.preprocessing import preprocess_dataset
from src.utils.config import Config


def preprocess_data(config: Config, data_type: str = "train"):
    """
    Preprocess data and apply CWT transform.
    
    Args:
        config: Configuration object
        data_type: Type of data to process ("train", "test", or "reference")
    """
    print(f"Processing {data_type} data...")
    
    # Load data
    if data_type == "train":
        data_path = os.path.join(config.original_data_dir, "train.mat")
        ref_flag = False
    elif data_type == "test":
        data_path = os.path.join(config.original_data_dir, "test.mat")
        ref_flag = False
    elif data_type == "reference":
        data_path = os.path.join(config.original_data_dir, "reference.mat")
        ref_flag = True
    else:
        raise ValueError(f"Unknown data_type: {data_type}")
    
    mat_data = sio.loadmat(data_path)
    
    # Preprocess EMG data
    emg_data = preprocess_dataset(mat_data, ref_flag=ref_flag)
    print(f"EMG data shape: {emg_data.shape}")
    
    # Initialize CWT
    cwt = CWT(
        wavelet_width=config.wavelet_width,
        fs=config.fs,
        lower_freq=config.lower_freq,
        upper_freq=config.upper_freq,
        n_scales=config.n_scales,
        size_factor=config.size_factor,
        border_crop=config.border_crop,
        stride=config.stride
    )
    
    # Apply CWT transform
    emg_tensor = torch.tensor(emg_data, dtype=torch.float32)
    cwt_data = cwt(emg_tensor)
    print(f"CWT data shape: {cwt_data.shape}")    # Save processed data
    if data_type == "reference":
        output_dir = config.ref_dir
        output_filename = "ref_cwt.pt"
    elif data_type == "test":
        output_dir = config.test_dir
        output_filename = "test_cwt.pt"
    else:  # train
        output_dir = config.data_dir
        output_filename = f"{data_type}_cwt.pt"
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    torch.save(cwt_data, output_path)
    print(f"Saved CWT data to {output_path}")


def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(description="Preprocess NEDO Challenge data")
    parser.add_argument("--data_type", type=str, choices=["train", "test", "reference"],
                       default="train", help="Type of data to process")
    parser.add_argument("--data_dir", type=str, default="./input",
                       help="Directory containing input data")
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = Config()
    config.original_data_dir = args.data_dir
    
    # Process data
    preprocess_data(config, args.data_type)


if __name__ == "__main__":
    main()
