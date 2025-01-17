from .base_finetuner import BaseFinetuner
import torch
from tqdm import tqdm
import random
from torch.utils.data import Dataset
from mobilitygpt.dpo_trainer import DPOTrainer

class DPOFinetuner(BaseFinetuner):
    def __init__(self, config, model, reference_model, dataset):
        super().__init__(config, model, dataset)
        self.reference_model = reference_model
        
    def train(self):
        """
        Train the model using DPO (Direct Preference Optimization)
        
        """

        trainer = DPOTrainer(
            config=self.config,
            model=self.model,
            reference_model=self.reference_model,
            train_dataset=self.dataset,
        )
        
        def batch_end_callback(trainer):
            if trainer.iter_num % 10 == 0:
                print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: "
                      f"train loss {trainer.loss.item():.5f} "
                      f"chosen reward {trainer.chosen_reward.item():.5f} "
                      f"rejected reward {trainer.rejected_reward.item():.5f}")

        trainer.set_callback('on_batch_end', batch_end_callback)
        trainer.run()
        