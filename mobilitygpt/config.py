from mobilitygpt.config_utils import CfgNode as CN
import torch

def get_base_config():
    """Base configuration for all training modes"""
    C = CN()
    
    # System
    C.system = CN()
    C.system.seed = 160
    C.system.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    C.system.work_dir = None  # Will be set based on dataset
    
    # Policy (PPO) settings
    C.policy = CN()
    C.policy.seq_length = 81
    C.policy.batch_size = 128
    C.policy.prompt_size = 30
    C.policy.prompt_batch_size = 128
    C.policy.num_rollouts = 128
    C.policy.epochs = 100
    C.policy.ppo_epochs = 4
    C.policy.kl_coef = 0.01
    C.policy.gamma = 1
    C.policy.lam = 0.95
    C.policy.cliprange = 0.2
    C.policy.cliprange_value = 0.2
    C.policy.vf_coef = 1
    C.policy.learning_rate = 5e-3
    C.policy.temperature = 0.3
    
    # Data
    C.data = CN()
    C.data.dataset = "SF"  # Dataset name
    C.data.block_size = 81
    C.data.max_length = 81
    C.data.random_trajs = False
    
    # Model
    C.model = CN()
    C.model.model_type = 'gpt-mobility'
    C.model.use_lora = False  
    C.model.load_path = None
    C.model.use_adjacency = True  # Use adjacency matrix by default
    C.model.vocab_size = None  # Will be set based on dataset
    C.model.block_size = None  # Will be set based on dataset
    C.model.n_layer = None
    C.model.n_head = None
    C.model.n_embd = None
    C.model.embd_pdrop = 0.1
    C.model.resid_pdrop = 0.1
    C.model.attn_pdrop = 0.1
    C.model.bias = False
    # LoRA parameters
    C.model.lora_rank = 8
    C.model.lora_alpha = 16.0
    C.model.lora_dropout = 0.0
    # Model type configurations
    C.model.model_configs = {
        'openai-gpt':   dict(n_layer=12, n_head=12, n_embd=768),
        'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
        'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
        'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
        'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
        'gopher-44m':   dict(n_layer=8, n_head=16, n_embd=512),
        'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192),
        'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128),
        'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),
        'gpt-mobility': dict(n_layer=6, n_head=4, n_embd=64),
    }
    # Training
    C.training = CN()
    C.training.mode = 'pretrain'  # ['pretrain', 'supervised', 'dpo', 'ppo']
    C.training.create_rl_dataset = False
    C.training.create_dpo_dataset = False
    C.training.device = 'auto'
    C.training.gravity_sampling = True
    C.training.dp_training = False
    C.training.dp_epsilon = 10
    C.training.validation_split = 0.2
    C.training.shuffle_dataset = True
    C.training.random_seed = 42
    C.training.num_samples = int(5e3)
    C.training.prompt_size = 4
    C.training.learning_rate = 5e-3
    C.training.weight_decay = 0.1
    C.training.max_iters = 3000
    C.training.batch_size = 8
    C.training.betas = (0.9, 0.95)
    C.training.grad_norm_clip = 1.0
    C.training.eps = 5.0
    C.training.delta = 1e-5
    C.training.num_workers = 16

    return C

def get_config_from_args(args):
    """Update base config with command line arguments"""
    config = get_base_config()
    
    # Update system settings
    config.system.work_dir = f'./Trajs_{args.dataset}_synthetic/gpt-mobility'
    
    # Update data settings
    config.data.dataset = args.dataset
    config.data.random_trajs = args.random_trajs
    
    # Update model settings
    config.model.use_lora = not args.lora
    if args.model_path:
        config.model.load_path = args.model_path
    
    # Update training settings
    config.training.mode = args.mode
    config.training.create_rl_dataset = args.create_rl_dataset
    config.training.create_dpo_dataset = args.create_dpo_dataset
    if args.mode == 'supervised' and args.dp_training:
        config.training.gravity_sampling = False
        config.training.dp_training = True
    
    return config 