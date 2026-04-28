import torch
import torch.nn as nn
import os
from tqdm import tqdm
from basissharing.bs_config import BSConfig, get_groups
import torch.nn.functional as F


class SharedLinear(nn.Module):
    def __init__(self, basis_registry, uid, coeffs, device, bias=None):
        super().__init__()
        object.__setattr__(self, "basis_registry", basis_registry)
        self.uid = uid
        self.coeffs = nn.Parameter(coeffs.to(device))
        self.bias = nn.Parameter(bias.to(device)) if bias is not None else None
        self._cached_weight: torch.Tensor | None = None

    def _invalidate_cache(self):
        self._cached_weight = None

    def _get_weight(self) -> torch.Tensor:
        if not self.training and self._cached_weight is not None:
            return self._cached_weight
        basis = self.basis_registry.get_basis(self.uid)  # [d_in, k]
        out = torch.matmul(
            basis, self.coeffs
        ).T  # [d_in, k] @ [k, d_out] -> [d_in, d_out] -> transpose to [d_out, d_in] for F.linear
        if not self.training:
            self._cached_weight = out.detach()  # Cache the weight for inference
        return out

    def train(self, mode: bool = True):
        if mode:
            self._invalidate_cache()
        return super().train(mode)

    def forward(self, X: torch.Tensor):
        return F.linear(X, self._get_weight(), self.bias)


def init_basissharing(model: nn.Module, bs_config: BSConfig):
    """Injects BasisSharingMixin into the model instance."""
    model.__class__ = type(
        model.__class__.__name__, (model.__class__, BasisSharingMixin), {}
    )
    model.init_mixin(bs_config)


class BasisRegistry(nn.Module):
    def __init__(self):
        super().__init__()
        self.bases = nn.ParameterDict()

    def add_basis(self, uid: str, tensor: torch.Tensor):
        self.bases[uid] = nn.Parameter(tensor)

    def get_basis(self, uid: str) -> torch.Tensor:
        return self.bases[uid]


class BasisSharingMixin:
    def init_mixin(self, bs_config: BSConfig):
        self.bs_config = bs_config
        self.basis_registry = BasisRegistry()
        self.groups = get_groups(self, bs_config)

    def _replace_module(self, name: str, uid: str, coeffs: torch.Tensor):
        *path, target = name.split(".")
        parent = self.get_submodule(".".join(path)) if path else self
        old_mod = getattr(parent, target)
        new_mod = SharedLinear(
            self.basis_registry,
            uid,
            coeffs,
            old_mod.weight.device,
            bias=old_mod.bias.data.clone() if old_mod.bias is not None else None,
        )

        # Free memory immediately
        old_mod.weight.data = torch.empty(0)
        if old_mod.bias is not None:
            old_mod.bias.data = torch.empty(0)

        setattr(parent, target, new_mod)

    def apply_compression(self, weight_dir: str):
        for uid, group in tqdm(self.groups.items(), desc="Applying shared weights"):
            basis_and_coeffs = torch.load(
                os.path.join(weight_dir, f"{uid}.pt"), map_location="cpu"
            )

            device = self.get_submodule(group["layers"][0]).weight.device
            self.basis_registry.add_basis(uid, basis_and_coeffs["basis"].to(device))

            for i, name in enumerate(group["layers"]):
                old_mod = self.get_submodule(name)
                d2 = old_mod.weight.shape[0]
                coeffs = basis_and_coeffs["coeffs"][:, i * d2 : (i + 1) * d2]
                self._replace_module(name, uid, coeffs)

    def save_compressed_weights(self, path: str):
        torch.save(self.state_dict(), path)

    def load_compressed_weights(self, path: str):
        state_dict = torch.load(path, map_location="cpu")
        for uid, group in self.groups.items():
            target_device = self.get_submodule(group["layers"][0]).weight.device
            self.basis_registry.add_basis(
                uid, state_dict[f"basis_registry.bases.{uid}"].to(target_device)
            )

            for i, name in enumerate(group["layers"]):
                if (coeffs_key := f"{name}.coeffs") in state_dict:
                    self._replace_module(name, uid, state_dict[coeffs_key])
        self.load_state_dict(state_dict)
