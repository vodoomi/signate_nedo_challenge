"""Dataset classes for NEDO Challenge."""

import numpy as np
from torch.utils.data import Dataset
import albumentations as A


class NEDOSpecDataset(Dataset):
    """Dataset for NEDO spectrogram data."""
    
    def __init__(
        self,
        spec_arrays: np.ndarray,
        labels: np.ndarray,
        is_reverse: np.ndarray,
        transform: A.Compose,
        log_clip_min: float = -7,
        log_clip_max: float = -2,
    ):
        """
        Initialize dataset.
        
        Args:
            spec_arrays: Spectrogram arrays
            labels: Target labels
            is_reverse: Flags indicating if data is horizontally flipped
            transform: Albumentations transform
            log_clip_min: Minimum value for log clipping
            log_clip_max: Maximum value for log clipping
        """
        self.spec_arrays = spec_arrays
        self.labels = labels
        self.is_reverse = is_reverse
        self.transform = transform
        self.log_clip_min = log_clip_min
        self.log_clip_max = log_clip_max

    def __len__(self):
        return len(self.spec_arrays)

    def __getitem__(self, index: int):
        img = self.spec_arrays[index]
        label = self.labels[index]
        is_reverse = self.is_reverse[index]
        
        # Apply log transform with clipping
        img = np.clip(img, np.exp(self.log_clip_min), np.exp(self.log_clip_max))
        img = np.log(img)
        
        # Normalize per image
        eps = 1e-6
        img_mean = img.mean(axis=(0, 1))
        img = img - img_mean
        img_std = img.std(axis=(0, 1))
        img = img / (img_std + eps)
        
        # Create horizontally flipped version
        img_copy = img.copy()
        img_copy = img_copy.reshape(16, -1, img.shape[1])
        # Reorder channels for horizontal flip: 1, 0, 3, 2, 5, 4, ...
        new_order_list = np.array([
            [i, i-1] for i in range(1, 16, 2)
        ]).flatten().tolist()
        img_reverse = img_copy[new_order_list].reshape(img.shape[0], img.shape[1])
        
        # Add channel dimension and apply transforms
        img = img[..., None]  # shape: (Hz, Time) -> (Hz, Time, Channel)
        img = self._apply_transform(img)
        
        img_reverse = img_reverse[..., None]
        img_reverse = self._apply_transform(img_reverse)
        
        return {
            "data": img,
            "reverse_data": img_reverse,
            "target": label,
            "is_reverse": is_reverse
        }

    def _apply_transform(self, img: np.ndarray):
        """Apply transform to image."""
        transformed = self.transform(image=img)
        return transformed["image"]
