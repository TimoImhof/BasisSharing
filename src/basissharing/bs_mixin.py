import torch
import torch.nn as nn
from basissharing.bs_layers import SharedLinear, BasisRegistry
from basissharing.bs_configs import BSConfig

from contextlib import contextmanager
from tqdm import tqdm
import gc


def init_basissharing(
    model: nn.Module, bs_config: BSConfig, compression_on_cpu: bool = True
):
    """Initialize basis sharing support."""
    model.__class__ = type(
        model.__class__.__name__, (model.__class__, BasisSharingMixin), {}
    )
    model.init_mixin(bs_config, compression_on_cpu)


class BasisSharingMixin:
    def init_mixin(self, bs_config: BSConfig, compression_on_cpu: bool = True):
        self.bs_config = bs_config
        self.basis_registry = BasisRegistry()
        self.compression_on_cpu = (
            compression_on_cpu  # Flag to control CPU offloading during compression
        )

    @contextmanager
    def _attach_hooks(self, target_name: str):
        """Attaches forward hooks to all modules matching target_name.
        Yields dict: module_name -> XtX (on CPU)
        """
        collected = {}
        hooks = []

        for module_name, module in self.named_modules():
            if module_name.split(".")[-1] != target_name:
                continue

            collected[module_name] = None

            def make_hook(name):
                def hook(mod, inp, _out):
                    X = (
                        inp[0].detach().float().reshape(-1, inp[0].shape[-1])
                    )  # [tokens, d]
                    XtX_step = (X.T @ X).cpu() if self.compression_on_cpu else X.T @ X
                    if collected[name] is None:
                        collected[name] = XtX_step
                    else:
                        collected[name] += (
                            XtX_step  # X_concat.T @ X_concat = X(1).T @ X(1)  +  X(2).T @ X(2)  + ... +  X(n).T @ X(n)
                        )

                return hook

            hooks.append(module.register_forward_hook(make_hook(module_name)))
        try:
            yield collected
        finally:
            for h in hooks:
                h.remove()

    def _collect_group_inputs(self, samples: list[torch.Tensor]) -> dict[str, dict]:
        """Runs forward passes and returns data grouped by uid.
        Returns: uid -> {'layers': [(module_name, XtX), ...], 'cfg': ModuleSharingConfig}
        """
        groups = {}

        for cfg in self.bs_config.module_cfgs:
            if not cfg.enabled:
                continue

            with self._attach_hooks(cfg.module_name) as collected_activations:
                self.eval()
                with torch.no_grad():
                    for sample in tqdm(
                        samples, desc=f"Hooking {cfg.module_name}", leave=False
                    ):
                        self(sample)

            matched = list(collected_activations.items())
            for i, (module_name, XtX) in enumerate(matched):
                group_num = i // cfg.group_size
                uid = f"{cfg.module_name}_{cfg.group_size}_{group_num}"
                groups.setdefault(uid, {"layers": [], "cfg": cfg})
                groups[uid]["layers"].append((module_name, XtX))

        return groups

    @staticmethod
    def _total_model_bytes(model: nn.Module) -> int:
        return sum(p.numel() * p.element_size() for p in model.parameters())

    def execute_compression(self, samples: list[torch.Tensor]):
        total_before = self._total_model_bytes(self)
        groups = self._collect_group_inputs(samples)

        with tqdm(
            groups.items(), desc="Compressing", unit="group", total=len(groups)
        ) as group_bar:
            for uid, group in group_bar:
                cfg, group_layers = group["cfg"], group["layers"]
                group_bar.set_postfix(group=uid)

                # Compute Scaling matrix S from sum of XtX across layers in this group
                XtX = sum(layer_XtX for _, layer_XtX in group_layers).float()
                eivals, eivecs = torch.linalg.eigh(XtX)
                eivals = eivals.clamp(min=1e-8)
                S = eivecs @ torch.diag(torch.sqrt(eivals))

                # Scale and concatenate weights: S @ W_i.T for each layer i, then concat along columns
                W_list = []
                for layer_id, _ in group_layers:
                    layer_W = (
                        self.get_submodule(layer_id).weight.detach().float()
                    )  # [d2, d1]
                    scaled = S.to(layer_W.device) @ layer_W.T
                    W_list.append(scaled.cpu() if self.compression_on_cpu else scaled)
                W_scaled_concat = torch.cat(W_list, dim=1)  # [d1, n * d2]

                # Compute basis B' and coefficients C' from truncated SVD, then apply inverse scaling to B'
                U, Sigma, Vh = torch.linalg.svd(
                    W_scaled_concat, full_matrices=False
                )  # U: [d1, d1], Sigma: [d1,], Vh: [d1, n * d2]
                d_1, n_d2 = W_scaled_concat.shape  # W_scaled is [d1, n * d2]
                k = max(1, int(d_1 * (1 - cfg.compression_ratio)))
                if k >= n_d2:
                    raise ValueError(
                        f"Rank inflation detected for group {uid}: k={k} >= n*d2={n_d2}. "
                        f"Compression only works for square or upward projection matrices."
                    )
                B_prime = U[:, :k] @ torch.diag(Sigma[:k])  # [d_1, k]
                C_prime_concat = Vh[:k, :]  # [k, n * d_2]
                S_pinv = torch.linalg.pinv(
                    S.cpu() if self.compression_on_cpu else S, rcond=1e-6
                )  # [d1, d1], pseudo-inverse for numerical stability
                B_prime_prime = S_pinv @ B_prime  # [d_1, k]

                # Store B_prime_prime in registry and replace modules with SharedLinear
                dtype, d_2 = layer_W.dtype, layer_W.shape[0]
                self.basis_registry.add_basis(uid, B_prime_prime.to(dtype))
                for i, (module_name, _) in enumerate(group_layers):
                    # Slice out the coefficients for this specific layer
                    C_i = C_prime_concat[:, i * d_2 : (i + 1) * d_2]  # [k, d_2]
                    self._replace_module(module_name, uid, C_i.to(dtype))

                # Clear now redundant large tensors to free GPU memory before next group
                del XtX, S, W_list, W_scaled_concat, U, Sigma, Vh, C_prime_concat
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        self._report_compression(total_before, groups)

    def _replace_module(self, name, uid, coeffs):
        *path, target = name.split(".")
        parent = self.get_submodule(".".join(path)) if path else self
        old_module = getattr(parent, target)

        new_module = SharedLinear(old_module, self.basis_registry, uid, coeffs)
        new_module.to(old_module.weight.device)

        # Explicitly clear old weight data to free GPU memory immediately
        old_module.weight.data = torch.empty(0)
        if old_module.bias is not None:
            old_module.bias.data = torch.empty(0)

        setattr(parent, target, new_module)

    @staticmethod
    def _fmt_bytes(n: int) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if n < 1024:
                return f"{n:.1f} {unit}"
            n /= 1024
        return f"{n:.1f} TB"

    def _report_compression(self, total_before, groups):
        bytes_before, bytes_after = 0, 0
        for uid, group in groups.items():
            basis = self.basis_registry.get_basis(uid)
            d1, k = basis.shape
            bpe = torch.finfo(basis.dtype).bits // 8

            bytes_after += d1 * k * bpe  # Shared basis

            for module_name, _ in group["layers"]:
                mod = self.get_submodule(module_name)
                ck, d2 = mod.coeffs.shape
                bytes_before += d1 * d2 * bpe
                bytes_after += ck * d2 * bpe

        saved = bytes_before - bytes_after
        pct = 100 * saved / bytes_before if bytes_before else 0
        total_after = total_before - saved
        total_pct = 100 * saved / total_before if total_before else 0
        n_layers = sum(len(g["layers"]) for g in groups.values())
        tqdm.write(
            f"✓ Compression complete — {len(groups)} groups, {n_layers} layers replaced.\n"
            f"  Target layers: {self._fmt_bytes(bytes_before)} → {self._fmt_bytes(bytes_after)} "
            f"({self._fmt_bytes(saved)} saved, {pct:.1f}% reduction)\n"
            f"  Full model:    {self._fmt_bytes(total_before)} → {self._fmt_bytes(total_after)} "
            f"({self._fmt_bytes(saved)} saved, {total_pct:.1f}% reduction)"
        )
