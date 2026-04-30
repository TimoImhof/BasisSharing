from tqdm import tqdm
import torch
import os
from basissharing import InputCollector, WeightCompressor, BSConfig


def _compress_model(
    model: torch.nn.Module,
    bs_config: BSConfig,
    calibration_samples: torch.utils.data.DataLoader,
    xtx_save_dir: str,
    compressed_weight_save_dir: str,
):
    # Input collection
    collector = InputCollector(
        model=model,
        target_nn_modules=bs_config.target_modules(),
        save_dir=xtx_save_dir,
        dram_limit_gb=12,
    )
    collector.collect(calibration_samples)

    # Weight compression
    comp = WeightCompressor(
        bs_config=bs_config,
        compression_on_cpu=False,
    )
    comp.compress(
        model=model,
        xtx_dir=os.path.join(os.getcwd(), xtx_save_dir),
        weight_dir=os.path.join(os.getcwd(), compressed_weight_save_dir),
    )


def compute_ppl(max_length, stride, data, model):
    device = next(model.parameters()).device
    model = model.eval()
    seq_len = data.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = data.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            output = model(input_ids, labels=target_ids)

            neg_log_likelihood = output.loss
        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl
