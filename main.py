import torch
from mobilitygpt.model import GPT, get_lora_model
from mobilitygpt.config_utils import set_seed, setup_logging
from mobilitygpt.args import parse_args
from mobilitygpt.config import get_config_from_args
from mobilitygpt.synthetic import generate_synthetic_trajectories
from datasets.data_utils import load_geo_data, load_trajectory_data, od_pair_to_adjacency_matrix, create_comparison_dataset_ls
from mobilitygpt.trainer_utils import (
    train_base_model,
    train_supervised,
    train_ppo,
    train_dpo
)
from datasets.char_dataset import CharDataset
from datasets.pairwise_dataset import PairwiseDataset
from datasets.prompt_dataset import PromptDataset
from datasets.data_utils import create_rl_dataset, create_dpo_dataset

def main():
    # Parse arguments and get config
    args = parse_args()
    config = get_config_from_args(args)
    
    # Setup
    setup_logging(config)
    set_seed(config.system.seed)
    
    # Load all data
    text = load_trajectory_data(config)
    geo, geo_ids, rel, od_list = load_geo_data(config.data.dataset)
    
    # Create dataset
    train_dataset = CharDataset(config.data, text, geo_ids)
    
    # Initialize model and get adjacency matrix
    model, adj_matrix = initialize_model(config, train_dataset, od_list)

    
    # Training
    if config.training.mode == 'pretrain':
        train_base_model(model, train_dataset, config, adj_matrix)

    # Create datasets if needed
    if config.training.create_rl_dataset:
        create_rl_dataset(model, train_dataset, geo, config)
    elif config.training.create_dpo_dataset:
        create_dpo_dataset(model, train_dataset, config)

    
    if config.training.mode == 'supervised':
        pairs = create_comparison_dataset_ls(config)
        finetune_dataset = PairwiseDataset(config.data, text, geo_ids, pairs)
        train_supervised(model, finetune_dataset, config)
    
    elif config.training.mode == 'ppo':
        prompt_dataset = PromptDataset(config.data, text, geo_ids, config.training.prompt_size)
        train_ppo(model, prompt_dataset, config)
    
    elif config.training.mode == 'dpo':
        path = f"{config.system.work_dir}/preference_dataset_dpo"
        reward_dataset = torch.load(path)
        train_dpo(model, reward_dataset, config)

    generate_synthetic_trajectories(model, train_dataset, config)

def initialize_model(config, dataset, od_list):
    # Update model config with dataset properties
    config.model.vocab_size = dataset.get_vocab_size()
    config.model.block_size = dataset.get_block_size()
    
    # Create adjacency matrix if needed
    if config.model.use_adjacency:
        adj_matrix = od_pair_to_adjacency_matrix(od_list, config.system.device)
    else:
        adj_matrix = None
    
    # Initialize model with updated config
    model = GPT(config.model, adj_matrix=adj_matrix)
    
    if config.model.use_lora:
        model = get_lora_model(model)
    
    # Load pretrained model for all finetuning stages
    if config.training.mode != 'pretrain':
        if not config.model.load_path:
            raise ValueError(f"Pretrained model path must be provided for {config.training.mode} finetuning")
        path = f"{config.system.work_dir}/{config.model.load_path}.pt"
        model.load_state_dict(torch.load(path))
        print(f"Loaded pretrained model from {config.model.load_path}")
        
    return model.to(config.system.device), adj_matrix

if __name__ == '__main__':
    main()
