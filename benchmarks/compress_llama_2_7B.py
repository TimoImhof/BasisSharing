from basissharing import WeightCompressor, BSConfig, ModuleSharingConfig
from transformers import LlamaForCausalLM
import torch
import os


def main():
    bs_config = BSConfig(
        model_id="meta-llama/Llama-2-7b-hf",
        module_cfgs=[
            ModuleSharingConfig(
                module_name="q_proj", group_size=2, compression_ratio=0.2
            ),
            ModuleSharingConfig(
                module_name="k_proj", group_size=2, compression_ratio=0.2
            ),
            ModuleSharingConfig(
                module_name="v_proj", group_size=2, compression_ratio=0.2
            ),
            ModuleSharingConfig(
                module_name="gate_proj", group_size=2, compression_ratio=0.2
            ),
            ModuleSharingConfig(
                module_name="up_proj", group_size=2, compression_ratio=0.2
            ),
        ],
    )

    comp = WeightCompressor(
        bs_config=bs_config,
        compression_on_cpu=False,
    )

    comp.compress(
        model=LlamaForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16, device_map="auto"
        ),
        xtx_dir=os.path.join(os.getcwd(), "benchmarks/llama_2_7B/xtx_data"),
        weight_dir=os.path.join(
            os.getcwd(), "benchmarks/llama_2_7B/compressed_weights"
        ),
    )


def check_bases_and_coeffs():
    weight_dir = os.path.join(os.getcwd(), "benchmarks/llama_2_7B/compressed_weights")

    issues = []

    for fname in os.listdir(weight_dir):
        if not fname.endswith(".pt"):
            continue

        path = os.path.join(weight_dir, fname)
        data = torch.load(path, map_location="cpu")
        basis = data["basis"]  # [d1, k], float32
        coeffs = data["coeffs"]  # [k, n*d2], float32

        # --- dtype check ---
        if basis.dtype != torch.float32:
            issues.append(f"[{fname}] basis dtype is {basis.dtype}, expected float32")
        if coeffs.dtype != torch.float32:
            issues.append(f"[{fname}] coeffs dtype is {coeffs.dtype}, expected float32")

        # --- nan / inf in saved weights ---
        if torch.isnan(basis).any() or torch.isinf(basis).any():
            issues.append(f"[{fname}] basis contains nan/inf")
        if torch.isnan(coeffs).any() or torch.isinf(coeffs).any():
            issues.append(f"[{fname}] coeffs contains nan/inf")

        # --- value range for float16 safety ---
        basis_max = basis.abs().max().item()
        coeffs_max = coeffs.abs().max().item()
        if basis_max > 65504:
            issues.append(f"[{fname}] basis max {basis_max:.1f} overflows float16")
        if coeffs_max > 65504:
            issues.append(f"[{fname}] coeffs max {coeffs_max:.1f} overflows float16")

        # --- weight reconstruction in float32 ---
        W_recon = basis @ coeffs  # [d1, n*d2], float32
        recon_max = W_recon.abs().max().item()
        recon_mean = W_recon.abs().mean().item()
        if torch.isnan(W_recon).any() or torch.isinf(W_recon).any():
            issues.append(f"[{fname}] reconstructed weight contains nan/inf")
        if recon_max > 65504:
            issues.append(
                f"[{fname}] reconstructed weight max {recon_max:.1f} overflows float16"
            )

        # --- float16 cast ---
        W_fp16 = W_recon.to(torch.float16)
        if torch.isnan(W_fp16).any():
            issues.append(f"[{fname}] reconstructed weight has nan after float16 cast")
        if torch.isinf(W_fp16).any():
            n_inf = torch.isinf(W_fp16).sum().item()
            issues.append(
                f"[{fname}] reconstructed weight has {n_inf} infs after float16 cast"
            )

        print(
            f"[{fname}] "
            f"basis=({basis.shape}, max={basis_max:.4f})  "
            f"coeffs=({coeffs.shape}, max={coeffs_max:.4f})  "
            f"W_recon max={recon_max:.4f} mean={recon_mean:.4f}  "
            f"fp16 nan={torch.isnan(W_fp16).any().item()} inf={torch.isinf(W_fp16).any().item()}"
        )

    print("\n--- Summary ---")
    if issues:
        for issue in issues:
            print(f"  ❌ {issue}")
    else:
        print("  ✅ All checks passed")


if __name__ == "__main__":
    check_bases_and_coeffs()
