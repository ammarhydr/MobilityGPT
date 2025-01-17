from abc import ABC, abstractmethod
import torch
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
import numpy as np
import pandas as pd

class BaseFinetuner(ABC):
    def __init__(self, config, model, dataset):
        self.config = config
        self.model = model
        self.dataset = dataset
        
    def create_data_samplers(self, validation_split=0.2, shuffle_dataset=True, random_seed=42):
        return self._create_random_samplers(validation_split, shuffle_dataset, random_seed)
                
    def _create_random_samplers(self, validation_split, shuffle_dataset, random_seed):
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        
        if shuffle_dataset:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
            
        train_indices, val_indices = indices[split:], indices[:split]
        
        return (SubsetRandomSampler(train_indices),
                SubsetRandomSampler(val_indices))
    
    @abstractmethod
    def train(self):
        pass 