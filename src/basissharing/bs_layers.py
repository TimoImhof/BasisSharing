import torch
import torch.nn as nn
import torch.nn.functional as F


class BasisRegistry(nn.Module):
    """Registry for the shared basis tensors."""

    def __init__(self):
        super().__init__()
        self.bases = nn.ParameterDict()

    def add_basis(self, uid: int, tensor: torch.Tensor):
        self.bases[str(uid)] = nn.Parameter(tensor)

    def get_basis(self, uid: int):
        return self.bases[str(uid)]


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
