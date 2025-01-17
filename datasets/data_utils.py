import torch
import pickle
import random
import pandas as pd
import numpy as np
from tqdm import tqdm


def load_geo_data(dataset_name):
    """Load geographical data for the dataset"""
    # Load geographical data
    geo_path = f"{dataset_name}-Taxi/roadmap.geo"
    geo = pd.read_csv(geo_path)
    geo_ids = geo['geo_id'].apply(str).tolist()
    
    # Load relational data
    rel_path = f"{dataset_name}-Taxi/roadmap.rel"
    rel = pd.read_csv(rel_path)
    rel['combined'] = rel.apply(lambda x: list([x['origin_id'], x['destination_id']]), axis=1)
    od_list = rel['combined'].tolist()
    
    return geo, geo_ids, rel, od_list

def load_trajectory_data(config):
    """Load trajectory data based on configuration"""
    # Determine file path based on config
    file_name = f'Trajs_{config.data.dataset}_random.txt' if config.data.random_trajs else f'Trajs_{config.data.dataset}.txt'
    
    # Load trajectory data
    with open(file_name, 'r') as f:
        text = f.read()
        
    return text 

def od_pair_to_adjacency_matrix(od_pair_list, device):
    """
    Converts an od pair to an adjacency matrix.
    
    Args:
        od_pair_list: List of [origin, destination] pairs
        device: Device to place tensor on
        
    Returns:
        torch.Tensor: Adjacency matrix on specified device
    """
    # Find the maximum index in the OD pair
    max_index = np.max(od_pair_list)
    
    # Initialize the adjacency matrix with zeros
    adjacency_matrix = torch.zeros((max_index + 1, max_index + 1), dtype=int)
    
    # Set the elements in the adjacency matrix based on the OD pair
    for origin, destination in od_pair_list:
        adjacency_matrix[origin, destination] = 1

    # Add two new rows of ones and columns at the end of adjacency matrix
    adjacency_matrix = torch.cat((adjacency_matrix, torch.ones(1, adjacency_matrix.size(0))), 0)
    adjacency_matrix = torch.cat((adjacency_matrix, torch.ones(adjacency_matrix.size(0), 1)), 1)

    return adjacency_matrix.to(device) 


def create_comparison_dataset_ls(config):
    file = open(config.system.work_dir+'/preference_dataset','rb')
    data = pickle.load(file)
    
    sequence_length = config.data.max_length
    EOS_TOKEN = '</S>'
    
    pairs = []
    for sample in data:
        chosen = None
        rejected = None
        if sample['choice'] == 0:
            chosen =  sample['candidate_0']
            rejected = sample['candidate_1']
        else:
            chosen = sample['candidate_1']
            rejected = sample['candidate_0']

        if len(chosen) < sequence_length:
            chosen = chosen + [EOS_TOKEN] * (sequence_length - len(chosen))
        if len(rejected) < sequence_length:
            rejected = rejected + [EOS_TOKEN] * (sequence_length - len(rejected))
        pair = {
            'chosen': chosen[:sequence_length],
            'rejected': rejected[:sequence_length]
        }
        pairs.append(pair)
    return pairs



def create_rl_dataset(model, train_dataset, geo, config):
    """Create dataset for RL training"""
    print('Creating RL dataset')
    preference_data = []
    
    pbar = tqdm(range(config.training.num_samples))
    while len(preference_data) < config.training.num_samples:
        # Sample random trajectory
        traj = random.sample(train_dataset.trajs, 1)[0]
        traj_first_n = traj[:config.training.prompt_size]
        
        # Convert prompt to tensor
        x = torch.tensor([train_dataset.stoi[s] for s in traj_first_n], 
                        dtype=torch.long)[None,...].to(config.system.device)
        
        try:
            # Calculate original trajectory length
            traj_length = geo[geo['geo_id'].isin([int(t) for t in traj[1:-1]])].length.sum()
            
            # Generate two candidates
            y = model.generate_test(x, train_dataset.itos, train_dataset.EOS_TOKEN, 
                                  temperature=1.0, do_sample=True, top_k=None)[0]
            candidate_0 = [train_dataset.itos[int(i)] for i in y]
            # Skip if candidate contains only EOS tokens
            if all(t == train_dataset.EOS_TOKEN for t in candidate_0[1:]):
                continue
            length_0 = geo[geo['geo_id'].isin([int(t) for t in candidate_0[1:]])].length.sum()
            
            y = model.generate_test(x, train_dataset.itos, train_dataset.EOS_TOKEN, 
                                  temperature=1.0, do_sample=True, top_k=None)[0]
            candidate_1 = [train_dataset.itos[int(i)] for i in y]
            # Skip if candidate contains only EOS tokens
            if all(t == train_dataset.EOS_TOKEN for t in candidate_1[1:]):
                continue
            length_1 = geo[geo['geo_id'].isin([int(t) for t in candidate_1[1:]])].length.sum()
            
            # Choose based on length difference
            choice = np.argmin([abs(traj_length - length_0), abs(traj_length - length_1)])
            
            d = {
                'input': traj,
                'candidate_0': candidate_0,
                'candidate_1': candidate_1,
                'choice': choice
            }
            preference_data.append(d)
            pbar.update(1)
        except ValueError:
            # Skip this iteration if we encounter invalid values
            print('ValueError')
            continue
    
    pbar.close()
    
    # Save dataset
    file_path = f"{config.system.work_dir}/preference_dataset"
    with open(file_path, 'wb') as f:
        pickle.dump(preference_data, f)
        

def create_dpo_dataset(model, train_dataset, config):
    """Create dataset for DPO training"""
    print("Creating DPO dataset")
    preference_data = []
    
    for _ in tqdm(range(config.training.num_samples)):
        # Sample random trajectory
        traj = random.sample(train_dataset.trajs, 1)[0]
        traj_first_n = traj[:config.training.prompt_size]
        
        # Generate candidate
        x = torch.tensor([train_dataset.stoi[s] for s in traj_first_n], 
                        dtype=torch.long)[None,...].to(config.system.device)
        y = model.generate_test(x, train_dataset.itos, train_dataset.EOS_TOKEN, 
                              temperature=1.0, do_sample=True, top_k=None)[0]
        candidate_0 = [train_dataset.itos[int(i)] for i in y]
        
        # Create token tensors
        prompt_tokens = torch.tensor([train_dataset.stoi[s] for s in traj_first_n], 
                                   dtype=torch.long)[None,...].to(config.system.device)
        chosen_response_tokens = torch.tensor([train_dataset.stoi[s] for s in traj], 
                                            dtype=torch.long)[None,...].to(config.system.device)
        rejected_response_tokens = torch.tensor([train_dataset.stoi[s] for s in candidate_0], 
                                              dtype=torch.long)[None,...].to(config.system.device)
        
        # Combine prompts and responses
        prompt_chosen_tokens = torch.cat([prompt_tokens, chosen_response_tokens], dim=1)
        prompt_rejected_tokens = torch.cat([prompt_tokens, rejected_response_tokens], dim=1)
        
        # Create loss masks
        chosen_loss_mask = torch.cat([
            torch.zeros(prompt_tokens.shape), 
            torch.ones(chosen_response_tokens.shape)
        ], dim=1)
        rejected_loss_mask = torch.cat([
            torch.zeros(prompt_tokens.shape), 
            torch.ones(rejected_response_tokens.shape)
        ], dim=1)
        
        dataset_example = {
            'prompt_chosen_tokens': prompt_chosen_tokens.squeeze(),
            'prompt_rejected_tokens': prompt_rejected_tokens.squeeze(),
            'chosen_loss_mask': chosen_loss_mask.squeeze(),
            'rejected_loss_mask': rejected_loss_mask.squeeze()
        }
        preference_data.append(dataset_example)

    # Save dataset
    file_path = f"{config.system.work_dir}/preference_dataset_dpo"
    torch.save(preference_data, file_path)
