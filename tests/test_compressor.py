import pytest
import torch
import os
from basissharing import WeightCompressor, BSConfig, ModuleSharingConfig
import numpy as np
from tests.conftest import MinimalModel


@pytest.fixture
def xtx_dir(tmp_path):
    d = tmp_path / "xtx_data"
    os.makedirs(d, exist_ok=True)
    return str(d)


@pytest.fixture
def weight_dir(tmp_path):
    d = tmp_path / "weights"
    os.makedirs(d, exist_ok=True)
    return str(d)


def test_optimizer_creates_valid_files(xtx_dir, weight_dir):
    model = MinimalModel(num_layers=1)
    cfg = BSConfig(
        model_id="test",
        module_cfgs=[ModuleSharingConfig("q", group_size=1, compression_ratio=0.5)],
    )
    np.save(arr=np.eye(64), file=os.path.join(xtx_dir, "blocks.0.attn.q.npy"))

    opt = WeightCompressor(cfg)
    opt.compress(model, xtx_dir, weight_dir)

    # uid format: {module_name}_{group_size}_{index}
    weight_file = os.path.join(weight_dir, "q_1_0.pt")
    assert os.path.exists(weight_file)

    data = torch.load(weight_file)
    assert "basis" in data
    assert "coeffs" in data
    # Basis for q should be (hidden_dim, hidden_dim * compression_ratio) = (64, 64 * 0.5) = (64, 32)
    assert data["basis"].shape == (64, 32)


def test_optimizer_grouping(xtx_dir, weight_dir):

    model = MinimalModel(num_layers=2)
    bs_config = BSConfig(
        model_id="test",
        module_cfgs=[
            ModuleSharingConfig("gate", group_size=2, compression_ratio=0.5),
        ],
    )

    # popularize xtx dir
    np.save(file=os.path.join(xtx_dir, "blocks.0.ffn.gate.npy"), arr=np.eye(64))
    np.save(file=os.path.join(xtx_dir, "blocks.1.ffn.gate.npy"), arr=np.eye(64))

    compr = WeightCompressor(bs_config=bs_config, compression_on_cpu=True)
    compr.compress(model, xtx_dir, weight_dir)

    # Should produce 1 file for the group of 2
    assert os.path.exists(os.path.join(weight_dir, "gate_2_0.pt"))
    data = torch.load(os.path.join(weight_dir, "gate_2_0.pt"))
    # Coeffs should be (k, 2 * d2) = (hidden_dim * 0.5, 2 * ffn_dim) = (32, 256)
    assert data["coeffs"].shape == (32, 256)
    # Should not produce any other files since only 'gate' is configured for sharing
    assert len(os.listdir(weight_dir)) == 1
