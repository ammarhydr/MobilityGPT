import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def logprobs_from_logits(logits, labels):
    """Get logprobs from logits and labels"""
    logprobs = F.log_softmax(logits, dim=-1)
    logprobs_labels = torch.gather(logprobs, dim=-1, index=labels.unsqueeze(-1))
    return logprobs_labels.squeeze(-1)

def gae(config, values, rewards):
    """Compute Generalized Advantage Estimation"""
    advantages = torch.zeros_like(rewards, device=rewards.device)
    last_advantage = 0
    last_value = 0
    with torch.no_grad():
        for t in reversed(range(rewards.shape[0])):
            delta = rewards[t] + config.gamma * last_value - values[t]
            last_advantage = delta + config.gamma * config.lam * last_advantage
            advantages[t] = last_advantage
            last_value = values[t]
        returns = advantages + values
    return advantages, returns

def ppo_loss(config, logprobs, values, old_logprobs, old_values, advantages, returns):
    """Compute PPO loss"""
    values_clipped = torch.clamp(
        values,
        old_values - config.cliprange_value,
        old_values + config.cliprange_value,
    )
    n = len(values)
    vf_loss1 = (values - returns) ** 2
    vf_loss2 = (values_clipped - returns) ** 2
    vf_loss = 0.5 * torch.sum(torch.max(vf_loss1, vf_loss2)) / n
    
    log_ratio = (logprobs - old_logprobs)
    ratio = torch.exp(log_ratio)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(
        ratio, 
        1.0 - config.cliprange, 
        1.0 + config.cliprange
    )
    pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2)) / n
    loss = pg_loss + config.vf_coef * vf_loss
    return loss

class Agent(nn.Module):
    """PPO Agent wrapping a GPT model"""
    
    def __init__(self, model, trainable=False):
        super().__init__()
        self.trainable = trainable
        self.model = model
        
        if not self.trainable:
            self.model = self.model.eval()
            self.model.requires_grad_(False)
        else:
            n_embd = self.model.lm_head.in_features
            self.value_head = nn.Sequential(
                nn.LayerNorm(n_embd),
                nn.GELU(),
                nn.Linear(n_embd, 4*n_embd),
                nn.GELU(),
                nn.Linear(4*n_embd, 1),
            )
            # Move value head to same device as model
            self.value_head = self.value_head.to(next(model.parameters()).device)
    
    def generate(self, input_ids, pad_id, **kwargs):
        """Generate responses with temperature adjustment on failure"""
        responses = []
        start_len = input_ids.cpu().shape[1]
        
        for idx in range(len(input_ids)):
            response = None
            kwargs['temperature'] = 1.0
            
            for i in range(10):  # Try up to 10 times
                response = self.model.generate_test(
                    input_ids[idx].reshape(1,-1), 
                    **kwargs
                ).tolist()[0]
                
                if len(response) > start_len:
                    break
                    
                kwargs['temperature'] -= 0.1
                print(f"Warning: Empty response generated, retrying (attempt {i+1})")
                
                if i == 8:  # After 9 failed attempts
                    if idx > 0:
                        input_ids[idx] = input_ids[idx-1]
                    else:
                        input_ids[idx] = input_ids[idx+1]
                    kwargs['temperature'] = 1.0
                    
            response = response + [pad_id.item()] * (kwargs['max_token'] - len(response))
            responses.append(response)
            
        return responses

    def forward(self, input_ids, attention_mask=None):
        """Forward pass through model and value head if trainable"""
        lm_logits, last_hidden_state = self.model.policy(input_ids)
        
        if self.trainable:
            value = self.value_head(last_hidden_state).squeeze(-1)
            return lm_logits, value
        else:
            return lm_logits

class CustomPromptDataGenerator:
    """Iterator for generating prompt batches"""
    
    def __init__(self, prompt_dataset, prompt_batch_size):
        self.prompt_dataset = prompt_dataset
        self.prompt_batch_size = prompt_batch_size
    
    def __iter__(self):
        self.dataset_indices = np.arange(len(self.prompt_dataset))
        return self
    
    def __next__(self):
        if len(self.dataset_indices) >= self.prompt_batch_size:
            # Sample without replacement
            picked_indices = np.random.choice(
                np.arange(len(self.dataset_indices)),
                self.prompt_batch_size,
                replace=False
            )
            samples = self.prompt_dataset[self.dataset_indices[picked_indices]]
            self.dataset_indices = np.delete(self.dataset_indices, picked_indices)
            return torch.tensor(samples)
        raise StopIteration

class PolicyTrainer:
    """Trainer for PPO policy optimization"""
    
    def __init__(self, policy_config, system_config, reward_model, prompt_dataset):
        self.policy_config = policy_config
        self.system_config = system_config
        self.prompt_dataset = prompt_dataset
        self.reward_model = reward_model
        
        # Initialize prompt generator
        self.prompt_generator = CustomPromptDataGenerator(
            self.prompt_dataset, 
            policy_config.prompt_batch_size
        )
        self.prompt_iterator = iter(self.prompt_generator)
        
        self.optimizer = None
        
        # Set up generation parameters
        self.generate_kwargs = {
            'max_token': self.prompt_dataset.config.max_length,
            'itos': self.prompt_dataset.itos,
            'end_token': self.prompt_dataset.EOS_TOKEN,
            'temperature': 1.0,
            'do_sample': False,
            'top_k': None,
        }

    def init_optimizer(self, model):
        """Initialize optimizer with model parameters"""
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.policy_config.learning_rate,
            betas=(0.9, 0.95)
        )
        return self.optimizer

    def reward_fn(self, samples):
        """Compute rewards using reward model"""
        with torch.no_grad():
            logits, _ = self.reward_model(samples)
            rewards = torch.sigmoid(logits*self.policy_config.temperature)[:,0]
            rewards = rewards.cpu().tolist()
        return rewards

    def make_experience(self, model, ref_model, num_rollouts=128):
        """Generate experience using current policy"""
        all_rollouts = []
        device = self.system_config.device
        
        while len(all_rollouts) < num_rollouts:
            try:
                batch = next(self.prompt_iterator).to(device)
            except StopIteration:
                self.prompt_generator = CustomPromptDataGenerator(
                    self.prompt_dataset, 
                    self.policy_config.prompt_batch_size
                )
                self.prompt_iterator = iter(self.prompt_generator)
                batch = next(self.prompt_iterator).to(device)
            
            # Generate responses
            eos_id = torch.tensor(self.prompt_dataset.stoi[self.prompt_dataset.EOS_TOKEN]).to(device)
            query_response_list = model.generate(batch, eos_id, **self.generate_kwargs)
            query_response_pad = torch.tensor(query_response_list).to(device)
            response = query_response_pad[:, batch.shape[1]:]
            attention_mask = query_response_pad.not_equal(eos_id).long()

            # Get model outputs
            with torch.no_grad():
                logits, values = model(query_response_pad)
                ref_logits = ref_model(query_response_pad)
                
            # Process outputs
            logprobs = logprobs_from_logits(logits[:, :-1, :], query_response_pad[:, 1:])
            ref_logprobs = logprobs_from_logits(ref_logits[:, :-1, :], query_response_pad[:, 1:])
            values = values[:, :-1]

            # Calculate rewards and process trajectories
            start = batch.shape[1]
            ends = start + attention_mask[:, start:].sum(1)
            # Get rewards from reward model
            scores = self.reward_fn(query_response_pad)
            rewards = -self.policy_config.kl_coef * (logprobs - ref_logprobs)
            scores_tensor = torch.tensor(scores, device=device)
            for i in range(len(batch)):
                rs = rewards[i][start  : ends[i]]
                rs[-1] = scores[i]
                rollout = [
                    {
                        'query_tensors': batch[i],
                        'response_tensors':response[i],
                        'logprobs': logprobs[i, start:ends[i]],
                        'values': values[i, start:ends[i]],
                        'rewards': rs,
                    }
                ]
                all_rollouts.append(rollout)

        score = scores_tensor.mean().item()
        return all_rollouts, score

    def compute_loss(self, batch, model):
        """Compute PPO loss for a batch of experience"""
        # Ensure optimizer exists
        
        trajectories = torch.hstack([batch['query_tensors'], batch['response_tensors']])
        logits, values_pred = model(trajectories.reshape(1,-1))
        
        values_pred = values_pred[:, :-1]
        logprobs = logprobs_from_logits(logits[0, :-1, :], trajectories[1:])
        
        start = batch['query_tensors'].shape[0] - 1
        end = start + len(batch['rewards'])
        logprobs = logprobs[start:end]
        values_pred = values_pred[0, start:end]
        
        advantages, returns = gae(self.policy_config, batch['values'], batch['rewards'])
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        loss = ppo_loss(
            config=self.policy_config,
            logprobs=logprobs,
            values=values_pred,
            old_logprobs=batch['logprobs'],
            old_values=batch['values'],
            advantages=advantages,
            returns=returns,
        )
        
        return loss, batch['rewards'][-1].mean().item()
