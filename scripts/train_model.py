"""Training script for NEDO Challenge."""

import os
import argparse
import numpy as np
import scipy.io as sio
import torch

from src.utils.config import Config
from src.utils.utils import get_transforms, get_player_data_indices, compute_mcrmse
from src.data.preprocessing import extract_labels, prepare_augmented_data
from src.training.trainer import NEDOTrainer


def load_data(config: Config):
    """Load and prepare training data."""
    # Load original data
    train = sio.loadmat(os.path.join(config.original_data_dir, "train.mat"))
    reference = sio.loadmat(os.path.join(config.original_data_dir, "reference.mat"))
    
    # Load CWT-transformed data
    train_array = torch.load(os.path.join(config.data_dir, "train_cwt.pt"), weights_only=True)
    ref_array = torch.load(os.path.join(config.ref_dir, "ref_cwt.pt"), weights_only=True)
    
    # Reshape data: (n_data, n_channel, height, width) -> (n_data, height, width)
    train_array = train_array.reshape(train_array.size(0), -1, train_array.size(3))
    ref_array = ref_array.reshape(ref_array.size(0), -1, ref_array.size(3))
    
    # Extract labels
    train_labels = extract_labels(train, ref_flag=False)
    ref_labels = extract_labels(reference, ref_flag=True)
    
    # Create player ID array for reference data
    player_id_array = np.concatenate([
        np.repeat(player_id, len(reference[f"000{player_id}"][0][0][3]))
        for player_id in range(5, 6)
    ])
    
    return train_array, ref_array, train_labels, ref_labels, player_id_array


def train_full_model(config: Config):
    """Train full model with data augmentation."""
    print("Training full model...")
    
    # Load data
    train_array, ref_array, train_labels, ref_labels, player_id_array = load_data(config)
    
    # Prepare augmented data
    augmented_data, augmented_labels, is_reverse = prepare_augmented_data(train_array, train_labels)
    
    # Get transforms
    train_transform, val_transform = get_transforms(config.image_size)    
    # Initialize trainer
    trainer = NEDOTrainer(config)
    
    # Train multiple seeds
    preds = []
    for seed in range(config.seed, config.seed + 5):
        print(f"Training with seed {seed}")
        pred = trainer.train_one_fold(
            seed=seed,
            train_array=augmented_data.numpy(),
            valid_array=ref_array.numpy(),
            train_labels=augmented_labels,
            valid_labels=ref_labels,
            is_reverse=is_reverse,
            player_id_list=player_id_array,
            train_transform=train_transform,
            val_transform=val_transform
        )
        preds.append(pred)
    
    # Ensemble predictions
    oof = np.mean(preds, axis=0)
    np.save("oof.npy", oof)
    
    # Calculate final score
    score = compute_mcrmse(oof, ref_labels, player_id_array)
    print(f"Final MCRMSE: {score:.4f}")
    
    return oof, score


def train_player_specific_models(config: Config, pretrained_dir: str):
    """Train player-specific models with fine-tuning."""
    print("Training player-specific models...")
    
    # Load data
    train_array, ref_array, train_labels, ref_labels, player_id_array = load_data(config)
    
    # Get player data indices
    n_trial_list, n_trial_cumsum_list = get_player_data_indices()
    
    # Get transforms
    train_transform, val_transform = get_transforms(config.image_size)    
    # Initialize trainer
    trainer = NEDOTrainer(config)
    
    # Train for each player
    all_preds = []
    
    for player_id in range(4):
        print(f"Training for player {player_id}")
        
        # Get player-specific data
        start = n_trial_cumsum_list[player_id]
        end = n_trial_cumsum_list[player_id + 1]
        
        player_train_array = train_array[start:end]
        player_train_labels = train_labels[start:end]
        
        # Prepare augmented data for this player
        player_augmented_data, player_augmented_labels, player_is_reverse = prepare_augmented_data(
            player_train_array, player_train_labels
        )
        
        # Reshape for model input
        player_augmented_data = player_augmented_data.reshape(
            player_augmented_data.size(0), -1, player_augmented_data.size(3)
        )
        
        # Train multiple seeds for this player
        player_preds = []
        for seed in range(config.seed, config.seed + 5):
            print(f"  Training with seed {seed}")
            
            # Load pretrained model
            pretrained_model_path = os.path.join(pretrained_dir, f"model_fold{seed}.pth")
            
            pred = trainer.train_one_fold(
                seed=seed,
                train_array=player_augmented_data.numpy(),
                valid_array=ref_array.numpy(),
                train_labels=player_augmented_labels,
                valid_labels=ref_labels,
                is_reverse=player_is_reverse,
                player_id_list=player_id_array,
                train_transform=train_transform,
                val_transform=val_transform,
                pretrained_model_path=pretrained_model_path
            )
            player_preds.append(pred)
            
            # Save player-specific model
            os.rename(f'model_seed{seed}.pth', f'model_seed{seed}_player{player_id}.pth')
        
        all_preds.extend(player_preds)
    
    # Ensemble all predictions
    oof = np.mean(all_preds, axis=0)
    np.save("oof_player_specific.npy", oof)
    
    # Calculate final score
    score = compute_mcrmse(oof, ref_labels, player_id_array)
    print(f"Player-specific MCRMSE: {score:.4f}")
    
    return oof, score


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train NEDO Challenge model")
    parser.add_argument("--mode", type=str, choices=["full", "player_specific"],
                       default="full", help="Training mode")
    parser.add_argument("--data_dir", type=str, default="/kaggle/input",
                       help="Directory containing input data")
    parser.add_argument("--pretrained_dir", type=str, default="./",
                       help="Directory containing pretrained models (for player_specific mode)")
    parser.add_argument("--max_epoch", type=int, default=20,
                       help="Maximum number of epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=8.0e-04,
                       help="Learning rate")
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = Config()
    config.original_data_dir = args.data_dir
    config.data_dir = os.path.join(args.data_dir, "nedo-challenge-cwt-for-train")
    config.ref_dir = os.path.join(args.data_dir, "nedo-challenge-cwt-for-ref")
    config.max_epoch = args.max_epoch
    config.batch_size = args.batch_size
    config.lr = args.lr
    
    # Train model
    if args.mode == "full":
        train_full_model(config)
    elif args.mode == "player_specific":
        train_player_specific_models(config, args.pretrained_dir)


if __name__ == "__main__":
    main()
