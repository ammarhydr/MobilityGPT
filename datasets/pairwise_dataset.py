import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class PairwiseDataset(Dataset):
    """Dataset for pairwise comparison training (used in PPO reward and supervised training)"""


    def __init__(self, config, data, vocab, pairs):
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
        
        # Process pairs into input ids
        self.chosen_input_ids = []
        self.rejected_input_ids = []
        
        for pair in tqdm(pairs):
            chosen, rejected = pair["chosen"], pair["rejected"]
            chosen_ids = [self.stoi[c] for c in chosen]
            rejected_ids = [self.stoi[c] for c in rejected]
            
            self.chosen_input_ids.append(chosen_ids)
            self.rejected_input_ids.append(rejected_ids)

        # Store dataset attributes
        self.num_trajs = len(lines)
        self.data = words
        self.trajs = line_words
        self.origins = origins
        
    def __len__(self):
        return len(self.chosen_input_ids)

    def get_block_size(self):
        return self.config.block_size

    def get_vocab_size(self):
        return self.vocab_size
    
    def __getitem__(self, idx):
        x = torch.tensor([self.chosen_input_ids[idx], self.rejected_input_ids[idx]])        
        y = torch.tensor([0, 1])  # 0 for chosen, 1 for rejected
        return x, y 