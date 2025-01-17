import torch
from tqdm import tqdm
import os
import pickle
import random

def generate_synthetic_trajectories(model, dataset, config):
    """Generate synthetic trajectories using the trained model"""
    print('Creating synthetic trajectories')
    synthetic_links = []
    
    for _ in tqdm(range(config.training.num_samples)):
        origin = random.sample(dataset.origins, 1)[0]
        context = [dataset.EOS_TOKEN, origin]
        
        x = torch.tensor([dataset.stoi[s] for s in context], 
                        dtype=torch.long)[None,...].to(config.system.device)
        
        y = model.generate_test(
            x, 
            dataset.itos, 
            dataset.EOS_TOKEN, 
            max_token=config.data.max_length,
            temperature=1.0,
            do_sample=True,
            top_k=None
        )[0]
        
        trajectory = []
        for i in y[1:]:
            if dataset.itos[int(i)] == dataset.EOS_TOKEN:
                break
            trajectory.append(int(dataset.itos[int(i)]))
        
        synthetic_links.append(trajectory)
    
    # Save trajectories
    filename = 'test_trajectories_lora.txt' if config.model.use_lora else 'test_trajectories_'+config.training.mode+'_.txt'
    file_path = os.path.join(config.system.work_dir, filename)
    
    with open(file_path, 'wb') as f:
        pickle.dump(synthetic_links, f)
        