from basissharing import init_basissharing, BSConfig, ModuleSharingConfig
from transformers import LlamaForCausalLM, LlamaTokenizer
from _data_utils import prepare_data
from _eval_utils import compute_ppl
import torch


def main():
    model = LlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16, device_map="auto"
    )
    # The paper logic: try LlamaTokenizer specifically, but catch errors
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    _, val_dataset, test_dataset, _ = prepare_data(
        dataset_name="wikitext",
        tokenizer=tokenizer,
        context_length=2048,
    )
    calibration_samples = [
        torch.tensor(s["input_ids"]).unsqueeze(0).to("cuda")
        for s in val_dataset.select(range(128))
    ]
    # ppl = compute_ppl(
    #     max_length=2048,
    #     stride=512,  # Stride for sliding window evaluation
    #     data=test_dataset,
    #     model=model,
    # )
    # print(f"Original Model Perplexity: {ppl:.4f}")

    init_basissharing(
        model=model,
        bs_config=BSConfig(
            # model_id="meta-llama/Llama-3.2-1B-Instruct",
            model_id="meta-llama/Llama-2-7b-hf",
            module_cfgs=[
                ModuleSharingConfig(
                    module_name="q_proj", group_size=2, compression_ratio=0.2
                ),
                # ModuleSharingConfig(
                #     module_name="o_proj", group_size=2, compression_ratio=0.2
                # ),
                ModuleSharingConfig(
                    module_name="gate_proj", group_size=2, compression_ratio=0.2
                ),
                ModuleSharingConfig(
                    module_name="up_proj", group_size=2, compression_ratio=0.2
                ),
            ],
        ),
    )
    model.execute_compression(samples=calibration_samples)
    # 5. Compute Perplexity on the COMPRESSED model
    # Note: test_dataset from prepare_data is one giant tensor
    ppl = compute_ppl(
        max_length=2048,
        stride=512,  # Stride for sliding window evaluation
        data=test_dataset,
        model=model,
        device="cuda",
    )
    print(f"Compressed Model Perplexity: {ppl:.4f}")


if __name__ == "__main__":
    main()
