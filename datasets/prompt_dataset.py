import torch
from torch.utils.data import Dataset
import numpy as np

class PromptDataset(Dataset):
    """Dataset for PPO training with prompts"""
    
    def __init__(self, config, data, vocab, prompt_size):
        self.config = config
        self.EOS_TOKEN = '</S>'
        
        # Process raw data
        lines = data.strip().split('\n\n') 
        line_words = [[self.EOS_TOKEN]+l.strip().split(',')+[self.EOS_TOKEN] for l in lines]
        words = [item for sublist in line_words for item in sublist]
        data_size, vocab_size = len(words), len(vocab)
        
        # Create vocabulary mappings
        self.stoi = {ch:i for i,ch in enumerate(vocab)}
        self.itos = {i:ch for i,ch in enumerate(vocab)}
        self.stoi[self.EOS_TOKEN] = len(vocab)
        self.itos[len(vocab)] = self.EOS_TOKEN
        self.vocab_size = vocab_size + 1
        
        # Create prompt dataset
        all_input_ids = [[self.stoi[s] for s in traj] for traj in line_words]
        self.prompts_input_ids = np.array([item[:prompt_size] for item in all_input_ids if len(item)>prompt_size])
        
    def __len__(self):
        return len(self.prompts_input_ids)
        
    def __getitem__(self, ix):
        return self.prompts_input_ids[ix] 