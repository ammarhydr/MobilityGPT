import torch
import os
import numpy as np
import pandas as pd
from mobilitygpt.model import GPT
from mobilitygpt.trainer import Trainer
from finetuners.supervised_finetuning import SupervisedFinetuner
from finetuners.dpo_finetuning import DPOFinetuner
from finetuners.ppo_finetuning import PPOFinetuner
from mobilitygpt.ppo_policy_trainer import Agent
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler

def train_base_model(model, dataset, config, adj_matrix=None):
    """Train the base model"""
    print("Starting base model training")
    
    # Create samplers
    train_sampler, val_sampler = create_samplers(dataset, config)
    
    # Initialize trainer
    trainer = Trainer(
        config=config.training,
        model=model,
        train_dataset=dataset,
        train_sampler=train_sampler,
        val_sampler=val_sampler,
        DP=config.training.dp_training,
        eps=config.training.dp_epsilon
    )
    
    # Add callbacks
    def batch_end_callback(trainer):
        if trainer.iter_num % 10 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: "
                  f"train loss {trainer.loss.item():.5f}")
        if trainer.iter_num % 500 == 0:
            save_checkpoint(trainer.model, config, "model_pretrain.pt")
            
    def validation_end_callback(trainer):
        print(f"Validation loss: {trainer.val_loss:.5f}")
    
    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.add_callback('validation_end', validation_end_callback)
    
    # Run training
    trainer.run()
    
    # Save final model
    save_checkpoint(model, config, "model_pretrain.pt")

def train_supervised(model, dataset, config):
    """Train with supervised finetuning"""
    print("Starting supervised finetuning")
    supervised_finetuner = SupervisedFinetuner(
        config=config,
        model=model,
        dataset=dataset,
        gravity_sampling=config.training.gravity_sampling,
        dp_epsilon=config.training.dp_epsilon
    )
    supervised_finetuner.train()
    save_checkpoint(model, config, "model_supervised.pt")

def train_ppo(model, dataset, config):
    """Train with PPO finetuning"""
    print("Starting PPO training")
    
    # Create reward model with same config as base model
    reward_model = GPT(config.model)
    # Try to load supervised model as reward model, if not exists train one
    supervised_path = f"{config.system.work_dir}/model_supervised.pt"
    
    if os.path.exists(supervised_path):
        print("Loading existing supervised model as reward model...")
        reward_model.load_state_dict(torch.load(supervised_path))
    else:
        print("No supervised model found. Performing supervised finetuning for reward model first...")
        # Create a copy of model for supervised training
        reward_model.load_state_dict(model.state_dict())
        supervised_finetuner = SupervisedFinetuner(
            config=config,
            model=reward_model,
            dataset=dataset,
            gravity_sampling=config.training.gravity_sampling,
            dp_epsilon=config.training.dp_epsilon
        )
        supervised_finetuner.train()
        save_checkpoint(reward_model, config, "model_supervised.pt")
    # Set reward model mode after loading
    reward_model.reward_model = True
    reward_model.eval()
    
    # Create policy and reference models
    policy_model = Agent(model, trainable=True)

    # Create new instance for reference model
    ref_model_base = GPT(config.model, adj_matrix=model.adj_matrix)
    ref_model_base.load_state_dict(model.state_dict())
    ref_model_base = ref_model_base.to(config.system.device)
    ref_model = Agent(ref_model_base, trainable=False)
    
    # PPO training
    ppo_finetuner = PPOFinetuner(
        config=config,
        policy_model=policy_model,
        ref_model=ref_model,
        dataset=dataset,
        reward_model=reward_model,
        gravity_sampling=config.training.gravity_sampling,
        prompt_size=config.training.prompt_size
    )
    ppo_finetuner.train()
    save_checkpoint(policy_model.model, config, "model_ppo.pt")

def train_dpo(model, dataset, config):
    """Train with DPO finetuning"""
    print("Starting DPO training")

    # Create reward model with same config as base model
    reference_model = GPT(config.model)
    # Try to load supervised model as reward model, if not exists train one
    supervised_path = f"{config.system.work_dir}/model_supervised.pt"
    
    if os.path.exists(supervised_path):
        print("Loading existing supervised model as reward model...")
        reference_model.load_state_dict(torch.load(supervised_path))
    else:
        print("No supervised model found. Performing supervised finetuning for reward model first...")
        # Create a copy of model for supervised training
        reference_model.load_state_dict(model.state_dict())
        supervised_finetuner = SupervisedFinetuner(
            config=config,
            model=reference_model,
            dataset=dataset,
            gravity_sampling=config.training.gravity_sampling,
            dp_epsilon=config.training.dp_epsilon
        )
        supervised_finetuner.train()
        save_checkpoint(reference_model, config, "model_supervised.pt")
    # Set reward model mode after loading
    reference_model.eval()


    dpo_finetuner = DPOFinetuner(
        config=config,
        model=model,
        reference_model=reference_model,
        dataset=dataset,
    )
    
    # Train
    dpo_finetuner.train()
    save_checkpoint(model, config, "model_dpo.pt")
    
def create_samplers(dataset, config):
    """Create train and validation samplers"""
    if config.training.gravity_sampling:
        return create_gravity_samplers(dataset, config)
    return create_random_samplers(dataset, config)

def create_gravity_samplers(dataset, config):
    """Create samplers with gravity-based weighting"""
    gravity = pd.read_csv(f"{config.data.dataset}-Taxi/{config.data.dataset}_Taxi_trajectory_train_w_gravity.csv").gravity.values
    gravity = torch.Tensor(gravity)
    minimum, maximum = torch.min(gravity), torch.max(gravity)    
    gravity = (gravity-minimum)/(maximum-minimum) 
    
    num_trajs = len(dataset.trajs)
    indices = list(range(num_trajs))
    split = int(np.floor(config.training.validation_split * num_trajs))
    
    if config.training.shuffle_dataset:
        np.random.seed(config.training.random_seed)
        np.random.shuffle(indices)
        
    train_indices, val_indices = indices[split:], indices[:split]
    train_gravity, val_gravity = gravity[train_indices], gravity[val_indices]
    
    return (WeightedRandomSampler(train_gravity, len(train_gravity)),
            WeightedRandomSampler(val_gravity, len(val_gravity)))

def create_random_samplers(dataset, config):
    """Create samplers with random sampling"""
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(config.training.validation_split * dataset_size))
    
    if config.training.shuffle_dataset:
        np.random.seed(config.training.random_seed)
        np.random.shuffle(indices)
        
    train_indices, val_indices = indices[split:], indices[:split]
    
    return (SubsetRandomSampler(train_indices),
            SubsetRandomSampler(val_indices))

def save_checkpoint(model, config, filename):
    """Save model checkpoint"""
    ckpt_path = os.path.join(config.system.work_dir, filename)
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}") 