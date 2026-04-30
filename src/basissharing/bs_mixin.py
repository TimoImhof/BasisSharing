import torch
import torch.nn as nn
import os
from tqdm import tqdm
from basissharing.bs_config import BSConfig, get_groups


class SharedLinear(nn.Module):
    def __init__(self, basis_registry, uid, coeffs, device, dtype, bias=None):
        super().__init__()
        object.__setattr__(self, "basis_registry", basis_registry)
        self.uid = uid
        k, d_out = coeffs.shape
        self.coeff_proj = nn.Linear(
            k, d_out, bias=bias is not None, device=device, dtype=dtype
        )
        with torch.no_grad():
            self.coeff_proj.weight.copy_(coeffs.T)  # Linear expects [d_out, k]
            if bias is not None:
                self.coeff_proj.bias.copy_(bias)

    def forward(self, X: torch.Tensor):
        basis_linear = self.basis_registry.get_basis(self.uid)
        hidden = basis_linear(X)  # [b*s, d_in] → [b*s, k]
        return self.coeff_proj(hidden)  # [b*s, k] → [b*s, d_out]


def init_basissharing(model: nn.Module, bs_config: BSConfig):
    """Injects BasisSharingMixin into the model instance."""
    model.__class__ = type(
        model.__class__.__name__, (model.__class__, BasisSharingMixin), {}
    )
    model.init_mixin(bs_config)


class BasisRegistry(nn.Module):
    def __init__(self):
        super().__init__()
        self.bases = nn.ModuleDict()

    def add_basis(self, uid: str, tensor: torch.Tensor, device, dtype):
        """tensor: [d_in, k]"""
        d_in, k = tensor.shape
        linear = nn.Linear(d_in, k, bias=False, device=device, dtype=dtype)
        with torch.no_grad():
            linear.weight.copy_(tensor.T)  # Linear expects [k, d_in]
        self.bases[uid] = linear

    def get_basis(self, uid: str) -> nn.Linear:
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
            bias=old_mod.bias.data.clone() if old_mod.bias is not None else None,
            device=old_mod.weight.device,
            dtype=old_mod.weight.dtype,
        )

        setattr(parent, target, new_mod)
        del old_mod

    def apply_compression(self, weight_dir: str):
        for uid, group in tqdm(self.groups.items(), desc="Applying shared weights"):
            basis_and_coeffs = torch.load(
                os.path.join(weight_dir, f"{uid}.pt"), map_location="cpu"
            )
            basis, coeffs = basis_and_coeffs["basis"], basis_and_coeffs["coeffs"]

            # Add basis layer to registry
            target = self.get_submodule(group["layers"][0]).weight
            device, dtype = (
                target.device,
                target.dtype,
            )
            self.basis_registry.add_basis(uid, basis, device=device, dtype=dtype)

            # Replace layers in group with SharedLinear
            for i, name in enumerate(group["layers"]):
                d2, _ = self.get_submodule(name).weight.shape  # [d2, d1]
                self._replace_module(
                    name,
                    uid,
                    coeffs[:, i * d2 : (i + 1) * d2],
                )

    def save_compressed_weights(self, weight_dir: str):
        os.makedirs(weight_dir, exist_ok=True)
        for uid, group in self.groups.items():
            # basis linear weight is [k, d_in], transpose back to [d_in, k] for storage
            basis = self.basis_registry.get_basis(uid).weight.T.detach().cpu()
            coeffs_list = []
            for name in group["layers"]:
                shared = self.get_submodule(name)
                # coeff_proj weight is [d_out, k], transpose back to [k, d_out] for storage
                coeffs_list.append(shared.coeff_proj.weight.T.detach().cpu())
            torch.save(
                {"basis": basis, "coeffs": torch.cat(coeffs_list, dim=1)},
                os.path.join(weight_dir, f"{uid}.pt"),
            )

    def load_compressed_weights(self, weight_dir: str):
        self.apply_compression(weight_dir)
