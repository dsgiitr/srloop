import os
import json
from dataclasses import dataclass, asdict

@dataclass
class SRDiffELTConfig:
    img_size: int = 32           
    patch_size: int = 4          
    in_channels: int = 3         
    cond_channels: int = 3       
    hidden_dim: int = 256        
    num_heads: int = 4           
    mlp_dim: int = 1024          
    num_blocks: int = 6          
    max_loops: int = 3           
    min_loops: int = 1           
    num_rrdb: int = 1            

    num_timesteps: int = 1000
    schedule_cosine_s: float = 0.008

    batch_size: int = 128        
    epochs: int = 500
    lr: float = 2e-4
    weight_decay: float = 0.01
    ema_decay: float = 0.999
    eval_freq: int = 50          
    save_freq: int = 50          
    data_dir: str = "./data"

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=4)

    @classmethod
    def load(cls, path: str) -> "SRDiffELTConfig":
        if not os.path.exists(path):
            return cls()
        with open(path, "r") as f:
            return cls(**json.load(f))
