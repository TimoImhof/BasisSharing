# Basis Sharing

A generalized implementation of [Basis Sharing: Cross-Layer Parameter Sharing for Large Language Model Compression](https://arxiv.org/abs/2410.03765), compatible with any HuggingFace model that uses `nn.Linear` for its attention and FFN components. The original paper implementation has basis-sharing-specific model classes; this version works on any HuggingFace model which fulfills the mentioned requirements.

## Quickstart

```bash
git clone https://github.com/TimoImhof/basissharing.git
cd basissharing
make install # sync uv + install pre-commit rules
```

`basissharing` is installed as a package together with its dependencies in `.venv`. Run pre-configured [benchmarks](./benchmarks/), or write your own script; the general flow is always something like this:

```python
from transformers import AutoModelForCausalLM
from basissharing import InputCollector, WeightCompressor, BSConfig, ModuleSharingConfig, init_basissharing

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16, device_map="auto")

bs_config = BSConfig(
    model_id="meta-llama/Llama-2-7b-hf",
    module_cfgs=[
        ModuleSharingConfig(module_name="q_proj", group_size=2, compression_ratio=0.2),
        ModuleSharingConfig(module_name="k_proj", ...),
        ...
    ],
)

# Stage 1: collect activation statistics
InputCollector(model, bs_config.target_modules(), xtx_dir).collect(calibration_samples)

# Stage 2: compress weights
WeightCompressor(bs_config).compress(model, xtx_dir, weight_dir)

# Stage 3: apply and run for your task
init_basissharing(model, bs_config)
model.apply_compression(weight_dir)
out = model(input_ids)
```

## Benchmark Results | Perplexity

Compression ratio: 20%, group size: 2, calibration: 256 WikiText-2 training samples, evaluation: WikiText-2 test set with stride 512.

| Model | Original PPL | Compressed PPL | command
|---|---|---|---
| Llama-2 7B | 4.86 | 6.07 | `uv run benchmarks/llama_2_7B.py`
| Mistral 7B | 4.69 | 5.85 | `uv run benchmarks/mistral_7B.py`
| OPT 6.7B | 9.31 | 10.02 | `uv run benchmarks/opt_6_dot_7B.py`

## Reference

```bibtex
@misc{parametersharing2024,
    title={Basis Sharing: Cross-Layer Parameter Sharing for Large Language Model Compression},
    author={Jingcun Wang and Yu-Guang Chen and Ing-Chao Lin and Bing Li and Grace Li Zhang},
    archivePrefix={arXiv},
    year={2024}
}
```