"""Training logic for NEDO Challenge."""

import numpy as np
import torch
from torch import optim, amp
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from time import time

from ..models.nedo_model import NEDOSpecModel
from ..training.loss import CombinedLoss
from ..utils.utils import compute_mcrmse, set_random_seed
from ..data.dataset import NEDOSpecDataset


class NEDOTrainer:
    """Trainer class for NEDO model."""
    
    def __init__(self, config):
        """
        Initialize trainer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.device = torch.device(config.device)
        
    def train_one_fold(
        self,
        seed: int,
        train_array: np.ndarray,
        valid_array: np.ndarray,
        train_labels: np.ndarray,
        valid_labels: np.ndarray,
        is_reverse: np.ndarray,
        player_id_list: np.ndarray,
        train_transform,
        val_transform,
        pretrained_model_path: str = None
    ):
        """
        Train model for one fold.
        
        Args:
            seed: Random seed
            train_array: Training data
            valid_array: Validation data
            train_labels: Training labels
            valid_labels: Validation labels
            is_reverse: Flags for augmented data
            player_id_list: Player ID array for validation
            train_transform: Training transforms
            val_transform: Validation transforms
            pretrained_model_path: Path to pretrained model
            
        Returns:
            np.ndarray: Best validation predictions
        """
        # Set random seed
        set_random_seed(seed, deterministic=self.config.deterministic)
          # Create datasets
        train_dataset = NEDOSpecDataset(
            train_array, train_labels, is_reverse, train_transform,
            self.config.log_clip_min, self.config.log_clip_max
        )
        val_dataset = NEDOSpecDataset(
            valid_array, valid_labels, np.zeros(len(valid_array)), val_transform,
            self.config.log_clip_min, self.config.log_clip_max
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size,
            num_workers=4, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size,
            num_workers=4, shuffle=False, drop_last=False
        )
          # Initialize model
        model = NEDOSpecModel(
            model_name=self.config.model_name,
            pretrained=True if pretrained_model_path is None else False,
            n_timepoint=self.config.n_timepoint,
            in_channels=self.config.in_channels,
            out_channels=self.config.out_channels
        )
        
        # Load pretrained weights if provided
        if pretrained_model_path is not None:
            model.load_state_dict(torch.load(pretrained_model_path, map_location=self.device, weights_only=False))
            
        model.to(self.device)
        
        # Setup optimizer and scheduler
        optimizer = optim.AdamW(
            params=model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        
        num_train_steps = int(len(train_dataset) / self.config.batch_size * self.config.max_epoch)
        num_warmup_steps = int(num_train_steps * 0)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
        )
        
        # Setup loss function
        loss_func = CombinedLoss(aux_weight=self.config.aux_loss_weight)
        loss_func.to(self.device)
        
        # Setup mixed precision training
        use_amp = self.config.enable_amp
        scaler = amp.GradScaler(self.config.device, enabled=use_amp, init_scale=2**14)
        
        # Training loop
        best_val_score = 1.0e+09
        best_preds = None
        
        for epoch in range(1, self.config.max_epoch + 1):
            epoch_start = time()
            
            # Training phase
            model.train()
            train_loss = 0
            
            for batch in train_loader:
                x = batch["data"].to(self.device, dtype=torch.float32)
                t = batch["target"].to(self.device, dtype=torch.float32)
                r = batch["is_reverse"].to(self.device, dtype=torch.long)
                
                optimizer.zero_grad()
                
                with amp.autocast(self.config.device, enabled=use_amp):
                    y, y_r = model(x)
                    loss = loss_func(y, t, y_r, r)
                    
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                train_loss += loss.item()
                
            train_loss /= len(train_loader)
                
            # Validation phase
            model.eval()
            preds = []
            targets = []
            
            for batch in val_loader:
                x = batch["data"].to(self.device, dtype=torch.float32)
                t = batch["target"].to(self.device, dtype=torch.float32)
                expand_x = batch["reverse_data"].to(self.device, dtype=torch.float32)
                
                with torch.no_grad(), amp.autocast(self.config.device, enabled=use_amp):
                    y, _ = model(x)
                    expand_y, _ = model(expand_x)
                
                # Test Time Augmentation (TTA)
                pred = (y.detach().cpu().numpy() + expand_y.detach().cpu().numpy()) / 2
                preds.append(pred)
                targets.append(t.detach().cpu().numpy())
            
            preds = np.concatenate(preds)
            targets = np.concatenate(targets)
            val_score = compute_mcrmse(preds, targets, player_id_list)
            
            if val_score < best_val_score:
                best_val_score = val_score
                best_preds = preds
            
            elapsed_time = time() - epoch_start
            print(
                f"[epoch {epoch}] train loss: {train_loss:.6f}, "
                f"val mcrmse: {val_score:.6f}, elapsed_time: {elapsed_time:.3f}"
            )
        
        # Save model
        torch.save(model.state_dict(), f'model_seed{seed}.pth')
        
        return best_preds
