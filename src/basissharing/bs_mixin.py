import torch
import torch.nn as nn
from basissharing.bs_layers import SharedLinear, BasisRegistry
from basissharing.bs_configs import BSConfig

from contextlib import contextmanager


def init_basissharing(model: nn.Module, bs_config: BSConfig):
    """Initialize basis sharing support."""
    model.__class__ = type(
        model.__class__.__name__, (model.__class__, BasisSharingMixin), {}
    )
    model.init_mixin(bs_config)


class BasisSharingMixin:
    def init_mixin(self, bs_config: BSConfig):
        self.bs_config = bs_config
        self.basis_registry = BasisRegistry()

    @contextmanager
    def _attach_hooks(self, target_name: str):
        """Attaches forward hooks to all modules matching target_name.
        Yields dict: module_name -> {'weight': tensor, 'XtX': tensor}
        """
        collected = {}
        hooks = []

        for module_name, module in self.named_modules():
            if module_name.split(".")[-1] != target_name:
                continue

            collected[module_name] = {
                "weight": module.weight.data.clone(),
                "XtX": None,
            }

            def make_hook(name):
                def hook(mod, inp, _out):
                    X = (
                        inp[0].detach().float().reshape(-1, inp[0].shape[-1])
                    )  # [tokens, d]
                    if collected[name]["XtX"] is None:
                        collected[name]["XtX"] = X.T @ X
                    else:
                        collected[name]["XtX"] += (
                            X.T @ X
                        )  # X_concat.T @ X_concat = X(1).T @ X(1)  +  X(2).T @ X(2)  + ... +  X(n).T @ X(n)

                return hook

            hooks.append(module.register_forward_hook(make_hook(module_name)))

        try:
            yield collected
        finally:
            for h in hooks:
                h.remove()

    def _collect_group_inputs_and_weights(
        self, samples: list[torch.Tensor]
    ) -> dict[str, dict]:
        """Runs forward passes and returns data grouped by uid.
        Returns: uid -> {'layers': [(module_name, weight, XtX), ...], 'cfg': ModuleSharingConfig}
        """
        groups = {}

        for cfg in self.bs_config.module_cfgs:
            if not cfg.enabled:
                continue

            with self._attach_hooks(cfg.module_name) as collected:
                self.eval()
                with torch.no_grad():
                    for sample in samples:
                        self(sample)

            # Group collected data by uid (consecutive groups of group_size)
            matched = list(collected.items())  # preserves named_modules() order
            for i, (module_name, data) in enumerate(matched):
                group_num = i // cfg.group_size
                uid = f"{cfg.module_name}_{cfg.group_size}_{group_num}"
                groups.setdefault(uid, {"layers": [], "cfg": cfg})
                groups[uid]["layers"].append((module_name, data["weight"], data["XtX"]))

        return groups

    def execute_compression(self, samples: list[torch.Tensor]):
        groups = self._collect_group_inputs_and_weights(samples)

        for uid, group in groups.items():
            cfg, layers = group["cfg"], group["layers"]

            # Compute Scaling matrix S from sum of XtX across layers in this group
            XtX = sum(layer_XtX for _, _, layer_XtX in layers).float()
            eivals, eivecs = torch.linalg.eigh(XtX)
            eivals = eivals.clamp(min=1e-8)
            S = eivecs @ torch.diag(torch.sqrt(eivals))

            # Scale and concatenate weights: S @ W_i.T for each layer i, then concat along columns
            W_list = []
            for _, layer_W, _ in layers:
                W_list.append(S @ layer_W.T.float())
            W_scaled_concat = torch.cat(W_list, dim=1)  # [d1, n * d2]

            # Perform SVD and truncate to rank k
            U, Sigma, Vh = torch.linalg.svd(
                W_scaled_concat, full_matrices=False
            )  # U: [d1, d1], Sigma: [d1,], Vh: [d1, n * d2]
            d_1 = W_scaled_concat.shape[
                0
            ]  # W_scaled is [d1, n * d2], computed in math convention
            k = max(1, int(d_1 * (1 - cfg.compression_ratio)))

            # Compute basis B' and coefficients C' from truncated SVD, then apply inverse scaling to B'
            B_prime = U[:, :k] @ torch.diag(Sigma[:k])  # [d_1, k]
            C_prime_concat = Vh[:k, :]  # [k, n * d_2]
            S_pinv = torch.linalg.pinv(
                S, rcond=1e-6
            )  # [d1, d1], pseudo-inverse for numerical stability
            B_prime_prime = S_pinv @ B_prime  # [d_1, k]

            # Store basis in registry and replace modules with SharedLinear with their share of coefficients
            d_2 = layers[0][1].shape[
                0
            ]  # org W is stored as [d_out, d_in] in PyTorch, so d_2 = d_out
            dtype = layers[0][1].dtype
            self.basis_registry.add_basis(uid, B_prime_prime.to(dtype))
            for i, (module_name, _, _) in enumerate(layers):
                # Slice out the coefficients for this specific layer
                C_i = C_prime_concat[:, i * d_2 : (i + 1) * d_2]  # [k, d_2]
                self._replace_module(module_name, uid, C_i.to(dtype))

    def _replace_module(self, name, uid, coeffs):
        *path, target = name.split(".")
        parent = self.get_submodule(".".join(path)) if path else self
        old_module = getattr(parent, target)
        new_module = SharedLinear(old_module, self.basis_registry, uid, coeffs)
        setattr(parent, target, new_module)
