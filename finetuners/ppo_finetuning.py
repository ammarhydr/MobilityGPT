import os
import torch
from mobilitygpt.ppo_policy_trainer import PolicyTrainer
from .base_finetuner import BaseFinetuner


class PPOFinetuner(BaseFinetuner):
    """Finetuner for PPO training"""
    
    def __init__(self, config, policy_model, ref_model, dataset, reward_model, prompt_size=4):
        super().__init__(config, policy_model.model, dataset)
        self.prompt_size = prompt_size
        self.reward_model = reward_model
        self.policy_model = policy_model
        self.ref_model = ref_model
        
    def train(self):
        """Train the model using PPO"""
        print("Starting PPO training")
        
        # Initialize trainer
        trainer = PolicyTrainer(
            policy_config=self.config.policy,
            system_config=self.config.system,
            reward_model=self.reward_model,
            prompt_dataset=self.dataset
        )
        
        # Initialize optimizer
        trainer.init_optimizer(self.policy_model)
        
        # Training loop
        best_score = float('-inf')
        for epoch in range(self.config.policy.epochs):
            # Initialize stats for this epoch
            epoch_stats = {
                'losses': [],
                'rewards': []
            }
            
            # Generate experience
            rollouts, score = trainer.make_experience(
                self.policy_model,
                self.ref_model,
                self.config.policy.num_rollouts
            )
            
            # Update policy
            for batch in rollouts:
                for _ in range(self.config.policy.ppo_epochs):
                    trainer.optimizer.zero_grad()
                    loss, reward = trainer.compute_loss(batch[0], self.policy_model)
                    loss.backward()
                    trainer.optimizer.step()
                    
                    # Track stats
                    epoch_stats['losses'].append(loss.item())
                    epoch_stats['rewards'].append(reward)
            
            # Calculate summary statistics
            avg_loss = sum(epoch_stats['losses']) / len(epoch_stats['losses'])
            avg_reward = sum(epoch_stats['rewards']) / len(epoch_stats['rewards'])
            print(f"Epoch {epoch+1}, Score: {score:.3f}, Avg Loss: {avg_loss:.3f}, Avg Reward: {avg_reward:.3f}")
            
            # Save best model
            if score > best_score:
                best_score = score
                self._save_checkpoint(self.policy_model.model)
                print("Best model saved")
                
    def _save_checkpoint(self, model):
        """Save model checkpoint"""
        ckpt_path = os.path.join(self.config.system.work_dir, "model_ppo.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")
    
    