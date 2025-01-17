"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader


class RewardTrainer:
    """A trainer for reward models using pairwise preference data"""
        
    def __init__(self, config, model, train_dataset, train_sampler=None, val_sampler=None):
        self.config = config
        self.model = model
        self.max_iters = 3000
        self.callbacks = defaultdict(list)
        self.train_dataset = train_dataset
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=config.betas,
            weight_decay=config.weight_decay
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            shuffle=train_sampler is None,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers
        )
        
        self.callbacks = defaultdict(list)

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config
        
        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(self.train_loader)
        
        while True:
            try:
                batch = next(data_iter)
            except StopIteration:
                break
                
            # Forward pass
            x, y = batch  # x shape: [batch_size, 2, seq_len], y shape: [batch_size, 2]
            batch_size = x.shape[0]
            
            # Reshape for processing both chosen and rejected sequences
            x = x.view(-1, x.size(-1))  # [batch_size * 2, seq_len]
            
            # Forward pass
            logits, _ = model(x)
            # Compute pairwise loss
            chosen_rewards = logits[::2].squeeze(-1)  # [batch_size]
            rejected_rewards = logits[1::2].squeeze(-1)  # [batch_size]
            self.loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
            
            # Backward pass
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()
            
            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if self.max_iters is not None and self.iter_num >= self.max_iters:
                break
