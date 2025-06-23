"""Loss functions for NEDO Challenge."""

import torch
import torch.nn as nn


class CustomLoss(nn.Module):
    """Custom RMSE loss for NEDO Challenge."""
    
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, input, target):
        """
        Compute custom RMSE loss.
        
        Args:
            input: Predicted values of shape (batch_size, sequence_length)
            target: Target values of shape (batch_size, sequence_length)
            
        Returns:
            torch.Tensor: Mean RMSE loss across batch
        """
        loss = torch.zeros(input.size(0), device=input.device)
        for trial in range(input.size(0)):
            loss[trial] = torch.sqrt(torch.mean((input[trial] - target[trial]) ** 2))
        return torch.mean(loss)


class CombinedLoss(nn.Module):
    """Combined loss with main task loss and auxiliary loss."""
    
    def __init__(self, aux_weight: float = 0.1):
        """
        Initialize combined loss.
        
        Args:
            aux_weight: Weight for auxiliary loss
        """
        super(CombinedLoss, self).__init__()
        self.main_loss = CustomLoss()
        self.aux_loss = nn.CrossEntropyLoss()
        self.aux_weight = aux_weight
    
    def forward(self, main_pred, main_target, aux_pred, aux_target):
        """
        Compute combined loss.
        
        Args:
            main_pred: Main task predictions
            main_target: Main task targets
            aux_pred: Auxiliary task predictions
            aux_target: Auxiliary task targets
            
        Returns:
            torch.Tensor: Combined loss
        """
        main_loss = self.main_loss(main_pred, main_target)
        aux_loss = self.aux_loss(aux_pred, aux_target)
        return main_loss + self.aux_weight * aux_loss
