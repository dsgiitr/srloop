from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    unique_blocks: int = 6
    loops: int = 3
    image_size: int = 32
    patch_size: int = 2
    in_channels: int = 4
    hidden_size: int = 256
    num_heads: int = 4
    mlp_ratio: float = 4.0
    class_dropout_prob: float = 0.1
    num_classes: int = 10
    learn_sigma: bool = True

@dataclass
class LossConfig:
    diffusion_weight: float = 1.0
    ilsd_weight: float = 1.0
    use_s3_sampling: bool = False

@dataclass
class SamplingConfig:
    inference_loops: int = 3

@dataclass
class ELTConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

def get_config():
    return ELTConfig()
