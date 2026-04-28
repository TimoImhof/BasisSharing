from basissharing import init_basissharing, BSConfig, ModuleSharingConfig
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
from _eval_utils import compute_ppl
from _data_utils import prepare_data


def main():
    model = LlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    _, _, test_dataset, _ = prepare_data(
        dataset_name="wikitext",
        tokenizer=tokenizer,
        context_length=2048,
    )

    ppl = compute_ppl(
        max_length=2048,
        stride=512,  # Stride for sliding window evaluation
        data=test_dataset,
        model=model,
    )
    print(f"Original Model Perplexity: {ppl:.4f}")

    init_basissharing(
        model=model,
        bs_config=BSConfig(
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
        ),
    )

    model.apply_compression(
        weight_dir="./benchmarks/llama_2_7B/compressed_weights",
    )
    ppl_compressed = compute_ppl(
        max_length=2048,
        stride=512,  # Stride for sliding window evaluation
        data=test_dataset,
        model=model,
    )
    print(f"Compressed Model Perplexity: {ppl_compressed:.4f}")


if __name__ == "__main__":
    main()
