from dataclasses import dataclass, field
import torch.nn as nn


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

    def target_modules(self) -> set[str]:
        """Returns a set of module names that are targeted for sharing."""
        target_names = set()
        for cfg in self.module_cfgs:
            if cfg.enabled:
                target_names.add(cfg.module_name)
        return target_names


def get_groups(model: nn.Module, bs_config: BSConfig) -> dict[str, dict[str, any]]:
    """
    Scans the model for modules matching the BSConfig and organizes them into groups of consecutive layers.
    Single point of truth for how layers are grouped for basis sharing, used by both `compressor.py` and `bs_mixin.py`.
    """
    groups = {}
    for cfg in bs_config.module_cfgs:
        if not cfg.enabled:
            continue
        matched = [
            n for n, _ in model.named_modules() if n.split(".")[-1] == cfg.module_name
        ]
        for i, name in enumerate(matched):
            uid = f"{cfg.module_name}_{cfg.group_size}_{i // cfg.group_size}"
            groups.setdefault(uid, {"layers": [], "cfg": cfg})
            groups[uid]["layers"].append(name)
    return groups
