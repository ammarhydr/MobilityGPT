from .base_finetuner import BaseFinetuner
import torch
from mobilitygpt.reward_trainer import RewardTrainer
import os

class SupervisedFinetuner(BaseFinetuner):
    def __init__(self, config, model, dataset, dp_epsilon=10):
        super().__init__(config, model, dataset)
        self.dp_epsilon = dp_epsilon
        
    def train(self):
        train_sampler, val_sampler = self.create_data_samplers()
        
        trainer = RewardTrainer(
            self.config.training,
            self.model,
            self.dataset,
            train_sampler=train_sampler,
            val_sampler=val_sampler
        )
        
        def batch_end_callback(trainer):
            if trainer.iter_num % 10 == 0:
                print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

            if trainer.iter_num % 500 == 0:
                self.model.eval()
                ckpt_path = os.path.join(self.config.system.work_dir, "model.pt")
                torch.save(self.model.state_dict(), ckpt_path)
                self.model.train()


        trainer.set_callback('on_batch_end', batch_end_callback)        
        trainer.run()