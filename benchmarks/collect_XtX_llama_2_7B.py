from basissharing import (
    BSConfig,
    ModuleSharingConfig,
    InputCollector,
)
from transformers import LlamaForCausalLM, LlamaTokenizer
from _data_utils import prepare_data
import torch
from torch.utils.data import DataLoader


def main():
    model = LlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    train_dataset, _, _, _ = prepare_data(
        dataset_name="wikitext",
        tokenizer=tokenizer,
        context_length=2048,
    )
    calibration_samples = DataLoader(
        [
            torch.tensor(s["input_ids"], dtype=torch.long)
            for s in train_dataset.select(range(256))
        ],
        batch_size=2,
        shuffle=False,
        collate_fn=lambda batch: torch.stack(batch),
    )

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

    collector = InputCollector(
        model=model,
        target_nn_modules=bs_config.target_modules(),
        save_dir="./benchmarks/llama_2_7B/xtx_data",
        dram_limit_gb=12,
    )
    collector.collect(calibration_samples)


if __name__ == "__main__":
    main()
