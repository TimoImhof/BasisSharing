import torch
import torch.nn as nn
import os
import gc
from tqdm import tqdm
from basissharing.bs_config import BSConfig, get_groups
import numpy as np


class WeightCompressor:
    """Mathematical engine for SVD-based basis sharing compression."""

    def __init__(self, bs_config: BSConfig, compression_on_cpu: bool = True):
        self.bs_config = bs_config
        self.compression_on_cpu = compression_on_cpu

    def compress(self, model: nn.Module, xtx_dir: str, weight_dir: str):
        """Perform SVD on collected XtX and save basis/coeffs to disk.
        `xtx_dir` expects the absolute path where the pre-computed XtX matrices are stored, named as `{layer_name}.npy`."""
        os.makedirs(weight_dir, exist_ok=True)
        groups = get_groups(model, self.bs_config)

        for uid, group in tqdm(groups.items(), desc="Optimizing weights"):
            cfg, layers = group["cfg"], group["layers"]
            target_device = model.get_submodule(layers[0]).weight.device

            # Compute Scaling matrix S from sum of XtX
            XtX = sum(
                torch.from_numpy(np.load(os.path.join(xtx_dir, f"{n}.npy"))).to(
                    torch.float32
                )
                for n in layers
            )  # [d1, d1]
            if not self.compression_on_cpu:
                XtX = XtX.to(target_device)
            eivals, eivecs = torch.linalg.eigh(XtX)  # [d1,], [d1, d1]
            S = eivecs @ torch.diag(torch.sqrt(eivals.clamp(min=1e-8)))  # [d1, d1]

            # Scale and concatenate weights: S @ W_i.T for each layer i, then concat along columns
            W_list = []
            for name in layers:
                W = model.get_submodule(name).weight.detach().to(torch.float32)
                scaled = (
                    S.to(W.device) @ W.T
                )  # [d1, d1] @ [d1, d2]  = [d1, d2] | always on GPU
                W_list.append(scaled.cpu() if self.compression_on_cpu else scaled)
            W_scaled_concat = torch.cat(W_list, dim=1)  # [d1, n * d2]

            # Compute basis B' and coefficients C' from truncated SVD, then apply inverse scaling to B'
            U, Sigma, Vh = torch.linalg.svd(
                W_scaled_concat, full_matrices=False
            )  # U: [d1, d1], Sigma: [d1,], Vh: [d1, n * d2] (! if d1 <= n*d2)
            d1, n_d2 = W_scaled_concat.shape
            k = max(1, int(d1 * (1 - cfg.compression_ratio)))
            if k >= n_d2:
                raise ValueError(
                    f"Rank inflation detected for group {uid}: k={k} >= n*d2={n_d2}. "
                    f"Compression only works for square or upward projection matrices."
                )
            B_prime = U[:, :k] @ torch.diag(Sigma[:k])  # [d1, k]
            C_prime_concat = Vh[:k, :]  # [k, n * d_2]
            S_pinv = torch.linalg.pinv(
                S.cpu() if self.compression_on_cpu else S, rcond=1e-6
            )  # [d1, d1], pseudo-inverse for numerical stability
            B_prime_prime = S_pinv @ B_prime  # [d1, k]

            dtype = model.get_submodule(layers[0]).weight.dtype
            torch.save(
                {
                    "basis": B_prime_prime.to(dtype).cpu(),
                    "coeffs": C_prime_concat.to(dtype).cpu(),
                },
                os.path.join(weight_dir, f"{uid}.pt"),
            )

            del (
                XtX,
                S,
                W_list,
                W_scaled_concat,
                U,
                Sigma,
                Vh,
                B_prime,
                C_prime_concat,
                S_pinv,
                B_prime_prime,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
