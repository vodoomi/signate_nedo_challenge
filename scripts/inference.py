"""Inference script for NEDO Challenge."""

import os
import argparse
import json
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.config import Config
from src.utils.utils import get_transforms, compute_mcrmse, get_player_data_indices
from src.data.preprocessing import extract_labels
from src.data.dataset import NEDOSpecDataset
from src.models.nedo_model import NEDOSpecModel


def inference_reference(config: Config, model_dir: str):
    """
    Run inference on reference data for validation.
    
    Args:
        config: Configuration object
        model_dir: Directory containing trained models
        
    Returns:
        tuple: (predictions, mcrmse_score)
    """
    print("Running inference on reference data...")
    
    # Load reference data
    reference = sio.loadmat(os.path.join(config.original_data_dir, "reference.mat"))
    ref_array = torch.load(os.path.join(config.ref_dir, "ref_cwt.pt"), weights_only=True)
    ref_array = ref_array.reshape(ref_array.size(0), -1, ref_array.size(3))
    
    # Extract labels and player IDs
    labels = extract_labels(reference, ref_flag=True)
    player_id_array = np.concatenate([
        np.repeat(player_id, len(reference[f"000{player_id}"][0][0][3]))
        for player_id in range(5, 6)
    ])
    
    # Get transforms
    _, val_transform = get_transforms(config.image_size)
    
    # Run inference for each model
    predictions = []
    device = torch.device(config.device)
    
    # Use player 0 models for reference inference (as in original notebook)
    for seed in range(config.seed, config.seed + 5):
        print(f"Processing seed {seed}...")
        
        # Create dataset
        ref_dataset = NEDOSpecDataset(
            ref_array.numpy(), labels, transform=val_transform
        )
        ref_loader = DataLoader(
            ref_dataset, batch_size=config.batch_size,
            num_workers=4, shuffle=False, drop_last=False
        )
        
        # Load model
        model_path = os.path.join(model_dir, f"model_seed{seed}_player0.pth")
        model = NEDOSpecModel(
            model_name=config.model_name,
            pretrained=True,
            n_timepoint=config.n_timepoint,
            in_channels=config.in_channels,
            out_channels=config.out_channels
        )
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        model.to(device)
        model.eval()
        
        # Generate predictions
        pred_list = []
        with torch.no_grad():
            for batch in tqdm(ref_loader, desc=f"Seed {seed}"):
                x = batch["data"].to(device, dtype=torch.float32)
                expand_x = batch["reverse_data"].to(device, dtype=torch.float32)
                
                y, _ = model(x)
                expand_y, _ = model(expand_x)
                
                # Test Time Augmentation
                pred = (y.detach().cpu().numpy() + expand_y.detach().cpu().numpy()) / 2
                pred_list.append(pred)
        
        pred_arr = np.concatenate(pred_list)
        predictions.append(pred_arr)
    
    # Ensemble predictions and apply corrections
    mean_pred = np.mean(predictions, axis=0) * -1
    mean_pred[:, 2::3] = mean_pred[:, 2::3] * -1
    
    # Calculate score
    score = compute_mcrmse(mean_pred, labels, player_id_array)
    print(f"Reference MCRMSE: {score:.4f}")
    
    return mean_pred, score


def inference_test(config: Config, model_dir: str, output_path: str = "submission.json"):
    """
    Run inference on test data and generate submission file.
    
    Args:
        config: Configuration object
        model_dir: Directory containing trained models
        output_path: Path for output submission file
        
    Returns:
        dict: Submission dictionary
    """
    print("Running inference on test data...")
    
    # Load test data
    test_array = torch.load(os.path.join(config.test_dir, "test_cwt.pt"), weights_only=True)
    test_array = test_array.reshape(test_array.size(0), -1, test_array.size(3))
    
    # Get player data indices
    n_trial_list, n_trial_cumsum_list = get_player_data_indices()
    
    # Get transforms
    _, val_transform = get_transforms(config.image_size)
    
    device = torch.device(config.device)
    mean_preds = []
    
    # Process each player
    for player_id in range(4):
        print(f"Processing player {player_id + 1}...")
        
        start = n_trial_cumsum_list[player_id]
        end = n_trial_cumsum_list[player_id + 1]
        
        # Get player-specific test data
        test_array_i = test_array.numpy()[start:end]
        
        predictions = []
        
        # Run inference for each seed
        for seed in range(config.seed, config.seed + 5):
            print(f"  Seed {seed}...")
            
            # Create dataset (dummy labels for test data)
            test_dataset = NEDOSpecDataset(
                test_array_i,
                np.zeros((len(test_array_i), 90)),
                transform=val_transform
            )
            test_loader = DataLoader(
                test_dataset, batch_size=config.batch_size,
                num_workers=4, shuffle=False, drop_last=False
            )
            
            # Load player-specific model
            model_path = os.path.join(model_dir, f"model_seed{seed}_player{player_id}.pth")
            model = NEDOSpecModel(
                model_name=config.model_name,
                pretrained=True,
                n_timepoint=config.n_timepoint,
                in_channels=config.in_channels,
                out_channels=config.out_channels
            )
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
            model.to(device)
            model.eval()
            
            # Generate predictions
            pred_list = []
            with torch.no_grad():
                for batch in tqdm(test_loader, desc=f"Player {player_id + 1}, Seed {seed}"):
                    x = batch["data"].to(device, dtype=torch.float32)
                    expand_x = batch["reverse_data"].to(device, dtype=torch.float32)
                    
                    y, _ = model(x)
                    expand_y, _ = model(expand_x)
                    
                    # Test Time Augmentation
                    pred = (y.detach().cpu().numpy() + expand_y.detach().cpu().numpy()) / 2
                    pred_list.append(pred)
            
            pred_arr = np.concatenate(pred_list)
            predictions.append(pred_arr)
        
        # Ensemble predictions and apply corrections for prediction target data
        mean_pred = np.mean(predictions, axis=0) * -1
        mean_pred[:, 2::3] = mean_pred[:, 2::3] * -1
        
        mean_preds.append(mean_pred)
    
    # Combine all player predictions
    mean_pred = np.concatenate(mean_preds)
    
    # Reshape and create submission format
    n_trial_list = [319, 300, 320, 320]  # Test data trial counts
    mean_pred = mean_pred.reshape(mean_pred.shape[0], 30, 3)
    preds_xyz_list = np.split(mean_pred, np.cumsum(n_trial_list)[:-1], axis=0)
    
    # Create submission dictionary
    preds_xyz_dict = {
        f"sub{player_id}": {
            f"trial{trial_no}": pred.tolist()
            for trial_no, pred in zip(range(1, len(preds) + 1), preds)
        }
        for player_id, preds in zip(range(1, 5), preds_xyz_list)
    }
    
    # Save submission file
    with open(output_path, 'w') as f:
        json.dump(preds_xyz_dict, f, indent=2)
    
    print(f"Submission saved to {output_path}")
    return preds_xyz_dict


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Run inference for NEDO Challenge")
    parser.add_argument("--mode", type=str, choices=["reference", "test", "both"],
                       default="both", help="Inference mode")
    parser.add_argument("--model_dir", type=str, required=True,
                       help="Directory containing trained models")
    parser.add_argument("--data_dir", type=str, default="/kaggle/input",
                       help="Directory containing input data")
    parser.add_argument("--output_path", type=str, default="submission.json",
                       help="Output path for submission file")
    parser.add_argument("--batch_size", type=int, default=128,
                       help="Batch size for inference")
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = Config()
    config.original_data_dir = args.data_dir
    config.data_dir = os.path.join(args.data_dir, "nedo-challenge-cwt-for-train")
    config.ref_dir = os.path.join(args.data_dir, "nedo-challenge-cwt-for-ref")
    config.batch_size = args.batch_size
    
    # Run inference
    if args.mode in ["reference", "both"]:
        ref_preds, ref_score = inference_reference(config, args.model_dir)
        np.save("reference_predictions.npy", ref_preds)
        print("Reference predictions saved to reference_predictions.npy")
    
    if args.mode in ["test", "both"]:
        submission = inference_test(config, args.model_dir, args.output_path)
        with open(args.output_path, 'w') as f:
            json.dump(submission, f)
        print(f"Test inference completed. Submission file: {args.output_path}")


if __name__ == "__main__":
    main()
