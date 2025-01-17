import torch
from torch.utils.data import Dataset

class CharDataset(Dataset):
    """Dataset for character-level trajectory data"""

    def __init__(self, config, data, vocab):
        self.config = config
        self.EOS_TOKEN = '</S>'
        
        # Process raw data
        lines = data.strip().split('\n\n') 
        line_words = [[self.EOS_TOKEN]+l.strip().split(',')+[self.EOS_TOKEN] for l in lines]
        words = [item for sublist in line_words for item in sublist]
        origins = [s[1] for s in line_words]
        data_size, vocab_size = len(words), len(vocab)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        # Create vocabulary mappings
        self.stoi = {ch:i for i,ch in enumerate(vocab)}
        self.itos = {i:ch for i,ch in enumerate(vocab)}
        
        self.stoi[self.EOS_TOKEN] = len(vocab)
        self.itos[len(vocab)] = self.EOS_TOKEN
        self.vocab_size = vocab_size + 1 
    
        # Store dataset attributes
        self.num_trajs = len(lines)
        self.data = words
        self.trajs = line_words
        self.origins = origins

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.data) - self.config.block_size

    def __getitem__(self, idx):
        # Get chunk of data
        chunk = self.data[idx:idx + self.config.block_size + 1]
        
        # Encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        
        # Return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y 