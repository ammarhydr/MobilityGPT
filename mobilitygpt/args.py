import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='MobilityGPT Training')
    
    # System arguments
    parser.add_argument('--dataset', type=str, default='SF',
                        help='Dataset to use for training')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use for training (default: auto)')
    
    # Model arguments
    parser.add_argument('--lora', action='store_true',
                        help='Enable LoRA layers')
    parser.add_argument('--model-path', type=str, required='pretrain' not in sys.argv,
                        help='Path to pretrained model (required for finetuning)')
    
    # Training arguments
    parser.add_argument('--mode', type=str, default='pretrain',
                        choices=['pretrain', 'supervised', 'dpo', 'ppo'],
                        help='Training mode')
    parser.add_argument('--create-rl-dataset', action='store_true',
                        help='Create dataset for RL training')
    parser.add_argument('--create-dpo-dataset', action='store_true',
                        help='Create dataset for DPO training')
    parser.add_argument('--dp-training', action='store_true',
                        help='Enable differential privacy training')
    parser.add_argument('--random-trajs', action='store_true',
                        help='Use random trajectories for training')
    parser.add_argument('--num-samples', type=int, default=5000,
                        help='Number of synthetic samples to generate')
    
    return parser.parse_args() 