import pytest
import torch
from basissharing import init_basissharing, BSConfig, ModuleSharingConfig
from basissharing.collector import InputCollector
from basissharing.compressor import WeightCompressor
from tests.conftest import MinimalModel


@pytest.fixture
def bs_config():
    return BSConfig(
        model_id="minimal",
        module_cfgs=[
            ModuleSharingConfig("q", group_size=2, compression_ratio=0.5),
            ModuleSharingConfig("up", group_size=2, compression_ratio=0.5),
        ],
    )


@pytest.fixture
def dummy_samples():
    return [torch.randn(1, 10, 64) for _ in range(2)]


def test_full_pipeline_integration(bs_config, dummy_samples, tmp_path):
    # Dirs
    xtx_dir = str(tmp_path / "xtx")
    weight_dir = str(tmp_path / "weights")

    model = MinimalModel()
    collector = InputCollector(model, ["q", "up"], xtx_dir)
    collector.collect(dummy_samples)

    optimizer = WeightCompressor(bs_config)
    optimizer.compress(model, xtx_dir, weight_dir)

    init_basissharing(model, bs_config)
    model.apply_from_shelf(weight_dir)

    model.eval()
    with torch.no_grad():
        out = model(dummy_samples[0])

    assert out.shape == (1, 10, 64)
    assert type(model.blocks[0].attn.q).__name__ == "SharedLinear"
    assert type(model.blocks[0].ffn.up).__name__ == "SharedLinear"


def test_save_load_persistence(bs_config, dummy_samples, tmp_path):
    xtx_dir = str(tmp_path / "xtx_p")
    weight_dir = str(tmp_path / "weights_p")
    save_path = str(tmp_path / "final_model.pt")

    # Full cycle
    model = MinimalModel()
    InputCollector(model, ["q", "up"], xtx_dir).collect(dummy_samples)
    WeightCompressor(bs_config).compress(model, xtx_dir, weight_dir)
    init_basissharing(model, bs_config)
    model.apply_from_shelf(weight_dir)

    # Get reference output
    model.eval()
    with torch.no_grad():
        ref_out = model(dummy_samples[0])

    # Save
    model.save_compressed_weights(save_path)

    # Load into fresh model
    fresh_model = MinimalModel()
    init_basissharing(fresh_model, bs_config)
    fresh_model.load_compressed_weights(save_path)

    # Compare
    fresh_model.eval()
    with torch.no_grad():
        new_out = fresh_model(dummy_samples[0])

    assert torch.allclose(ref_out, new_out, atol=1e-6)
