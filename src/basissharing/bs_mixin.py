import torch
import torch.nn as nn
import os
from tqdm import tqdm
from basissharing.bs_config import BSConfig, get_groups

import torch.nn.functional as F


class SharedLinear(nn.Module):
    def __init__(self, org_module, basis_registry, uid, coeffs):
        super().__init__()
        object.__setattr__(
            self, "basis_registry", basis_registry
        )  # Avoids registration as a submodule
        self.uid = uid
        self.coeffs = nn.Parameter(coeffs)
        self.bias = (
            nn.Parameter(org_module.bias.data.clone())
            if org_module.bias is not None
            else None
        )

    def forward(self, X: torch.Tensor):
        basis = self.basis_registry.get_basis(self.uid)  # [d_in, k]
        weight = torch.matmul(
            basis, self.coeffs
        ).T  # [d_in, k] @ [k, d_out] -> [d_in, d_out] -> transpose to [d_out, d_in] for F.linear
        return F.linear(X, weight, self.bias)


def init_basissharing(model: nn.Module, bs_config: BSConfig):
    """Injects BasisSharingMixin into the model instance."""
    model.__class__ = type(
        model.__class__.__name__, (model.__class__, BasisSharingMixin), {}
    )
    model.init_mixin(bs_config)


class BasisRegistry(nn.Module):
    """Registry for the shared basis tensors."""

    def __init__(self):
        super().__init__()
        self.bases = nn.ParameterDict()

    def add_basis(self, uid: int, tensor: torch.Tensor):
        self.bases[str(uid)] = nn.Parameter(tensor)

    def get_basis(self, uid: int):
        return self.bases[str(uid)]


class BasisSharingMixin:
    def init_mixin(self, bs_config: BSConfig):
        self.bs_config = bs_config
        self.basis_registry = BasisRegistry()
        self.groups = get_groups(self, bs_config)

    def apply_from_shelf(self, weight_dir: str):
        """Swaps nn.Linear for SharedLinear using weights from the shelf."""
        for uid, group in tqdm(self.groups.items(), desc="Applying shared weights"):
            data = torch.load(os.path.join(weight_dir, f"{uid}.pt"), map_location="cpu")
            self.basis_registry.add_basis(uid, data["basis"])

            for i, name in enumerate(group["layers"]):
                old_mod = self.get_submodule(name)
                d2 = old_mod.weight.shape[0]
                coeffs = data["coeffs"][:, i * d2 : (i + 1) * d2]
                self._replace_module(name, uid, coeffs)

    def _replace_module(self, name: str, uid: str, coeffs: torch.Tensor):
        *path, target = name.split(".")
        parent = self.get_submodule(".".join(path)) if path else self
        old_mod = getattr(parent, target)

        new_mod = SharedLinear(old_mod, self.basis_registry, uid, coeffs)
        new_mod.to(old_mod.weight.device)

        # Free memory immediately
        old_mod.weight.data = torch.empty(0)
        if old_mod.bias is not None:
            old_mod.bias.data = torch.empty(0)
        setattr(parent, target, new_mod)

    def save_compressed_weights(self, path: str):
        torch.save(self.state_dict(), path)

    def load_compressed_weights(self, path: str, device: str = "cpu"):
        state_dict = torch.load(path, map_location=device)
        # Restore Registry
        for key, param in state_dict.items():
            if "basis_registry.bases." in key:
                self.basis_registry.add_basis(
                    key.split("basis_registry.bases.")[-1], param
                )
        # Structural replacement
        for uid, group in self.groups.items():
            for name in group["layers"]:
                if f"{name}.coeffs" in state_dict:
                    self._replace_module(name, uid, state_dict[f"{name}.coeffs"])
        self.load_state_dict(state_dict)
