from basissharing import init_basissharing, BSConfig, ModuleSharingConfig
from transformers import LlamaForCausalLM, LlamaTokenizer


def _prep_samples(tokenizer):
    input_ids = [
        tokenizer(
            "The quick brown fox jumps over the lazy dog.",
            return_tensors="pt",
        ).input_ids.to("cuda"),
        tokenizer(
            "I am a student at the University of Darmstadt, Germany.",
            return_tensors="pt",
        ).input_ids.to("cuda"),
        tokenizer(
            "The mitochondria is the powerhouse of the cell.",
            return_tensors="pt",
        ).input_ids.to("cuda"),
    ]
    return input_ids


def main():
    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct").to(
        "cuda"
    )
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

    init_basissharing(
        model=model,
        bs_config=BSConfig(
            model_id="meta-llama/Llama-3.2-1B-Instruct",
            module_cfgs=[
                ModuleSharingConfig(
                    module_name="q_proj", group_size=8, compression_ratio=0.4
                ),
                ModuleSharingConfig(
                    module_name="o_proj", group_size=8, compression_ratio=0.4
                ),
                ModuleSharingConfig(
                    module_name="gate_proj", group_size=8, compression_ratio=0.4
                ),
                ModuleSharingConfig(
                    module_name="up_proj", group_size=8, compression_ratio=0.4
                ),
            ],
        ),
    )
    model.execute_compression(samples=_prep_samples(tokenizer))


if __name__ == "__main__":
    main()
