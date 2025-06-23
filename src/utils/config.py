"""Configuration settings for NEDO Challenge."""

from dataclasses import dataclass


@dataclass
class Config:
    """Configuration class for NEDO Challenge experiments."""    # Data directories
    original_data_dir: str = "./input"
    data_dir: str = "./input/nedo-challenge-cwt-for-train"
    test_dir: str = "./input/nedo-challenge-cwt-for-test"
    ref_dir: str = "./input/nedo-challenge-cwt-for-ref"
    
    # Model configuration
    model_name: str = "resnet34d"
    
    # Training configuration
    fold: int = 5
    image_size: int = 224
    max_epoch: int = 20
    batch_size: int = 32
    lr: float = 8.0e-04
    weight_decay: float = 1.0e-04
    es_patience: int = 5
    
    # Experiment configuration
    seed: int = 88
    deterministic: bool = True
    enable_amp: bool = True
    device: str = "cuda"
    
    # CWT configuration
    wavelet_width: float = 7.0
    fs: int = 200
    lower_freq: float = 0.5
    upper_freq: float = 40
    n_scales: int = 40
    size_factor: float = 1.0
    border_crop: int = 0
    stride: int = 1
    
    # Model architecture
    n_timepoint: int = 30
    in_channels: int = 1
    out_channels: int = 3
    
    # Loss configuration
    aux_loss_weight: float = 0.1
    
    # Preprocessing configuration
    log_clip_min: float = -7
    log_clip_max: float = -2
