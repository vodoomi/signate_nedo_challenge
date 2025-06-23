"""Test data preprocessing script for NEDO Challenge."""

import os
import argparse
import numpy as np
import scipy.io as sio
import torch

from src.models.cwt import CWT


def preprocess_dataset(mat_data, ref_flag=False):
    """
    Preprocess dataset from MATLAB format.
    
    Args:
        mat_data: MATLAB data dictionary
        ref_flag: Whether this is reference data
        
    Returns:
        np.array: Preprocessed EMG data
    """
    # mat_data: {"0001": [[np.array(n_trial, n_channel, n_timepoint), np.array(n_tiral, n_axis, n_timepoint), np.array(board_stance)]], "0002": ...}
    # ref_flag = Trueのときは、学習側筋電位, 学習側重心速度, 予測側筋電位, 予測側重心速度, ボードスタンスと並んでいる
    
    # 筋電位とボードスタンスデータのインデックスを指定
    if ref_flag:
        emg_idx = 2
    else:
        emg_idx = 0
    player_keys = [key for key in mat_data.keys() if key.isdigit()]
    
    # 筋電位データの取得
    emg_data = np.concatenate([mat_data[p_key][0][0][emg_idx] for p_key in player_keys])

    return emg_data


def preprocess_test_data(input_dir: str, output_dir: str):
    """
    Preprocess test data and save CWT transformed data.
    
    Args:
        input_dir: Directory containing test.mat
        output_dir: Directory to save processed data
    """
    print("Loading test data...")
    test_path = os.path.join(input_dir, "test.mat")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found: {test_path}")
    
    test = sio.loadmat(test_path)
    
    print("Preprocessing test data...")
    test_array = preprocess_dataset(test, ref_flag=False)
    
    print(f"Test data shape: {test_array.shape}")
    
    # CWT変換の実行
    print("Applying CWT transformation...")
    cwt_transform = CWT(
        wavelet_width=7.0,
        fs=200,
        lower_freq=0.5,
        upper_freq=40,
        n_scales=40,
        size_factor=1.0,
        border_crop=0,
        stride=1
    )
    
    test_cwt = cwt_transform(torch.tensor(test_array, dtype=torch.float32))
    
    print(f"CWT transformed data shape: {test_cwt.shape}")
    
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    # CWT変換済みデータの保存
    output_path = os.path.join(output_dir, "test_cwt.pt")
    torch.save(test_cwt, output_path)
    
    print(f"Test CWT data saved to: {output_path}")
    
    return test_cwt


def main():
    """Main function for test data preprocessing."""
    parser = argparse.ArgumentParser(description="Preprocess test data for NEDO Challenge")
    parser.add_argument("--input_dir", type=str, default="./input", 
                       help="Directory containing test.mat")
    parser.add_argument("--output_dir", type=str, default="./input/nedo-challenge-cwt-for-test",
                       help="Directory to save processed CWT data")
    
    args = parser.parse_args()
    
    print("=== NEDO Challenge Test Data Preprocessing ===")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    
    try:
        test_cwt = preprocess_test_data(args.input_dir, args.output_dir)
        print("Test data preprocessing completed successfully!")
        print(f"Final CWT data shape: {test_cwt.shape}")
        
    except Exception as e:
        print(f"Error during test data preprocessing: {str(e)}")
        raise


if __name__ == "__main__":
    main()
