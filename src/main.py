"""Main entry point for NEDO Challenge."""

import argparse
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from scripts.preprocess_data import main as preprocess_main
from scripts.train_model import main as train_main
from scripts.evaluate_model import main as evaluate_main
from scripts.inference import main as inference_main


def main():
    """Main function with subcommands."""
    parser = argparse.ArgumentParser(description="NEDO Challenge Pipeline")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
      # Preprocessing command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess data')
    preprocess_parser.add_argument("--data_type", type=str, 
                                 choices=["train", "test", "reference"],
                                 default="train", help="Type of data to process")
    preprocess_parser.add_argument("--data_dir", type=str, default="./input",
                                 help="Directory containing input data")
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument("--mode", type=str, choices=["full", "player_specific"],
                            default="full", help="Training mode")
    train_parser.add_argument("--data_dir", type=str, default="./input",
                            help="Directory containing input data")
    train_parser.add_argument("--pretrained_dir", type=str, default="./",
                            help="Directory containing pretrained models")
    train_parser.add_argument("--max_epoch", type=int, default=20,
                            help="Maximum number of epochs")
    train_parser.add_argument("--batch_size", type=int, default=32,
                            help="Batch size")
    train_parser.add_argument("--lr", type=float, default=8.0e-04,
                            help="Learning rate")
    
    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    eval_parser.add_argument("--model_dir", type=str, default="./",
                           help="Directory containing trained models")
    eval_parser.add_argument("--model_pattern", type=str, default="model_seed*.pth",
                           help="Pattern to match model files")
    eval_parser.add_argument("--data_dir", type=str, default="./input",
                           help="Directory containing input data")
    eval_parser.add_argument("--batch_size", type=int, default=32,
                           help="Batch size for evaluation")
    
    # Inference command
    inference_parser = subparsers.add_parser('inference', help='Run inference')
    inference_parser.add_argument("--mode", type=str, choices=["reference", "test", "both"],
                                default="both", help="Inference mode")
    inference_parser.add_argument("--model_dir", type=str, required=True,
                                help="Directory containing trained models")
    inference_parser.add_argument("--data_dir", type=str, default="./input",
                                help="Directory containing input data")
    inference_parser.add_argument("--output_path", type=str, default="submission.json",
                                help="Output path for submission file")
    inference_parser.add_argument("--batch_size", type=int, default=128,
                                help="Batch size for inference")
    
    args = parser.parse_args()
    
    if args.command == 'preprocess':
        # Set up arguments for preprocessing
        sys.argv = ['preprocess_data.py', '--data_type', args.data_type, 
                   '--data_dir', args.data_dir]
        preprocess_main()
        
    elif args.command == 'train':
        # Set up arguments for training
        sys.argv = ['train_model.py', '--mode', args.mode,
                   '--data_dir', args.data_dir,
                   '--pretrained_dir', args.pretrained_dir,
                   '--max_epoch', str(args.max_epoch),
                   '--batch_size', str(args.batch_size),
                   '--lr', str(args.lr)]
        train_main()
        
    elif args.command == 'evaluate':
        # Set up arguments for evaluation
        sys.argv = ['evaluate_model.py', '--model_dir', args.model_dir,
                   '--model_pattern', args.model_pattern,
                   '--data_dir', args.data_dir,
                   '--batch_size', str(args.batch_size)]
        evaluate_main()
        
    elif args.command == 'inference':
        # Set up arguments for inference
        sys.argv = ['inference.py', '--mode', args.mode,
                   '--model_dir', args.model_dir,
                   '--data_dir', args.data_dir,
                   '--output_path', args.output_path,
                   '--batch_size', str(args.batch_size)]
        inference_main()
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
