from dataclasses import dataclass, field


@dataclass
class ModuleSharingConfig:
    """Configuration for basis sharing on a single module category."""

    module_name: str
    group_size: int = 2
    compression_ratio: float = 0.2
    enabled: bool = True


@dataclass
class BSConfig:
    """Top-level configuration for a Basis Sharing compression."""

    model_id: str  # HF hub model ID, e.g. "meta-llama/Llama-3.2-3B-Instruct"
    module_cfgs: list[ModuleSharingConfig] = field(default_factory=list)
