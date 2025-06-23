"""Evaluation script for NEDO Challenge."""

import os
import argparse
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader

from src.utils.config import Config
from src.utils.utils import get_transforms, compute_mcrmse
from src.data.preprocessing import extract_labels
from src.data.dataset import NEDOSpecDataset
from src.models.nedo_model import NEDOSpecModel


def evaluate_model(config: Config, model_paths: list, test_data_path: str = None):
    """
    Evaluate trained models.
    
    Args:
        config: Configuration object
        model_paths: List of paths to trained models
        test_data_path: Path to test data (if None, use reference data)
    """
    print("Evaluating models...")
    
    # Load reference data for evaluation
    reference = sio.loadmat(os.path.join(config.original_data_dir, "reference.mat"))
    ref_array = torch.load(os.path.join(config.ref_dir, "ref_cwt.pt"), weights_only=True)
    ref_array = ref_array.reshape(ref_array.size(0), -1, ref_array.size(3))
    
    # Extract reference labels
    ref_labels = extract_labels(reference, ref_flag=True)
    
    # Create player ID array for reference data
    player_id_array = np.concatenate([
        np.repeat(player_id, len(reference[f"000{player_id}"][0][0][3]))
        for player_id in range(5, 6)
    ])
    
    # Get transforms
    _, val_transform = get_transforms(config.image_size)
      # Create dataset
    val_dataset = NEDOSpecDataset(
        ref_array.numpy(),
        ref_labels,
        np.zeros(len(ref_array)),
        val_transform,
        config.log_clip_min,
        config.log_clip_max
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        num_workers=4, shuffle=False, drop_last=False
    )
    
    # Load and evaluate each model
    all_preds = []
    device = torch.device(config.device)
    
    for i, model_path in enumerate(model_paths):
        print(f"Evaluating model {i+1}/{len(model_paths)}: {model_path}")
          # Initialize model
        model = NEDOSpecModel(
            model_name=config.model_name,
            pretrained=False,
            n_timepoint=config.n_timepoint,
            in_channels=config.in_channels,
            out_channels=config.out_channels
        )
        
        # Load weights
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        model.to(device)
        model.eval()
        
        # Generate predictions
        preds = []
        targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch["data"].to(device, dtype=torch.float32)
                t = batch["target"].to(device, dtype=torch.float32)
                expand_x = batch["reverse_data"].to(device, dtype=torch.float32)
                
                # Forward pass with TTA
                y, _ = model(x)
                expand_y, _ = model(expand_x)
                
                # Average predictions (Test Time Augmentation)
                pred = (y.cpu().numpy() + expand_y.cpu().numpy()) / 2
                preds.append(pred)
                targets.append(t.cpu().numpy())
        
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        
        # Calculate individual model score
        score = compute_mcrmse(preds, targets, player_id_array)
        print(f"  Model {i+1} MCRMSE: {score:.4f}")
        
        all_preds.append(preds)
    
    # Ensemble predictions
    ensemble_preds = np.mean(all_preds, axis=0)
    ensemble_score = compute_mcrmse(ensemble_preds, ref_labels, player_id_array)
    print(f"Ensemble MCRMSE: {ensemble_score:.4f}")
    
    # Save ensemble predictions
    np.save("ensemble_predictions.npy", ensemble_preds)
    print("Saved ensemble predictions to ensemble_predictions.npy")
    
    return ensemble_preds, ensemble_score


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate NEDO Challenge model")
    parser.add_argument("--model_dir", type=str, default="./",
                       help="Directory containing trained models")
    parser.add_argument("--model_pattern", type=str, default="model_seed*.pth",
                       help="Pattern to match model files")
    parser.add_argument("--data_dir", type=str, default="/kaggle/input",
                       help="Directory containing input data")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for evaluation")
    
    args = parser.parse_args()
    
    # Find model files
    import glob
    model_paths = glob.glob(os.path.join(args.model_dir, args.model_pattern))
    
    if not model_paths:
        print(f"No models found matching pattern: {args.model_pattern}")
        return
    
    print(f"Found {len(model_paths)} models to evaluate")
    
    # Initialize configuration
    config = Config()
    config.original_data_dir = args.data_dir
    config.ref_dir = os.path.join(args.data_dir, "nedo-challenge-cwt-for-ref")
    config.batch_size = args.batch_size
    
    # Evaluate models
    evaluate_model(config, model_paths)


if __name__ == "__main__":
    main()
