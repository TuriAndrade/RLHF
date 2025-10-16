from dataclasses import dataclass
from typing import Tuple


@dataclass
class TrainConfig:
    # Models
    policy_name: str = "gpt2"
    rm_name: str = "lvwerra/distilbert-imdb"

    # Device
    device: str = "cuda"

    # Tokenization / generation
    max_prompt_len: int = 128
    gen_len: int = 64
    top_p: float = 0.9
    top_k: int = 50
    temperature: float = 1.0

    # PPO training
    outer_steps: int = 10
    batch_size: int = 64  # prompts per rollout step
    ppo_epochs: int = 4
    mini_batch_size: int = 16
    lr: float = 5e-6
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    # PPO/GAE params
    gamma: float = 1.0
    lam: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: float = 0.2
    vf_coef: float = 0.1
    ent_coef: float = 0.02
    kl_coef: float = 1  # β

    # Repro
    seed: int = 0

    # Output
    ckpt_dir: str = "ckpts"
    log_interval: int = 1


@dataclass
class EvalConfig:
    policy_path: str = "ckpts/policy_lm"
    ref_name: str = "gpt2"
    rm_name: str = "lvwerra/distilbert-imdb"
    device: str = "cuda"
    max_prompt_len: int = 128
    gen_len: int = 64
    top_p: float = 0.9
    top_k: int = 50
    temperature: float = 1.0
    kl_coef: float = 0.1
    n_samples: int = 500
    batch_size: int = 32
    seed: int = 0
    out_csv: str = "eval_results.csv"
