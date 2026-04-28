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
        xtx_dir=os.path.join(os.getcwd() + "/benchmarks/llama_2_7B/xtx_data"),
        weight_dir="./benchmarks/llama_2_7B/compressed_weights",
    )


if __name__ == "__main__":
    main()
