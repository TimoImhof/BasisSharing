from basissharing import BSConfig, ModuleSharingConfig, init_basissharing
from _data_utils import prepare_data
from _benchmark_utils import _compress_model, compute_ppl
import torch
from transformers import OPTForCausalLM, AutoTokenizer

from torch.utils.data import DataLoader


def main():

    bs_config = BSConfig(
        model_id="facebook/opt-6.7b",
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
            ModuleSharingConfig(module_name="fc1", group_size=2, compression_ratio=0.2),
        ],
    )

    model = OPTForCausalLM.from_pretrained(
        "facebook/opt-6.7b", torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-6.7b")

    train_dataset, _, test_dataset, _ = prepare_data(
        dataset_name="wikitext",
        tokenizer=tokenizer,
        context_length=2048,
    )
    ppl_compressed = compute_ppl(
        max_length=2048,
        stride=512,
        data=test_dataset,
        model=model,
    )
    print(f"Original Model Perplexity: {ppl_compressed:.4f}")
    calibration_samples = DataLoader(
        [
            torch.tensor(s["input_ids"], dtype=torch.long)
            for s in train_dataset.select(range(256))
        ],
        batch_size=2,
        shuffle=False,
        collate_fn=lambda batch: torch.stack(batch),
    )

    _compress_model(
        model=model,
        bs_config=bs_config,
        calibration_samples=calibration_samples,
        xtx_save_dir="./benchmarks/opt_6.7B/xtx_data",
        compressed_weight_save_dir="./benchmarks/opt_6.7B/compressed_weights",
    )

    init_basissharing(model=model, bs_config=bs_config)
    model.apply_compression(weight_dir="./benchmarks/opt_6.7B/compressed_weights")
    ppl_compressed = compute_ppl(
        max_length=2048,
        stride=512,
        data=test_dataset,
        model=model,
    )
    print(f"Compressed Model Perplexity: {ppl_compressed:.4f}")


if __name__ == "__main__":
    main()
