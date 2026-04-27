import pytest
import os
from basissharing.collector import ShelfWriter, InputCollector
import torch
import torch.nn as nn
import numpy as np


@pytest.fixture
def dummy_model():
    torch.manual_seed(42)
    return nn.Sequential(
        nn.Linear(16, 16, bias=False),
        nn.Linear(16, 16, bias=False),
    )


@pytest.fixture
def save_dir(tmp_path):
    d = tmp_path / "xtx_shelf"
    os.makedirs(d, exist_ok=True)
    return str(d)


def load(save_dir, uid):
    """Helper: load a tensor from the npy shelf."""
    return torch.from_numpy(np.load(os.path.join(save_dir, f"{uid}.npy")))


def test_shelf_writer_merges_correctly(save_dir):
    with ShelfWriter(save_dir) as writer:
        uid = "test_layer"
        writer.flush({uid: torch.ones(4, 4)})
        writer.flush({uid: torch.ones(4, 4) * 2})

    result = load(save_dir, uid)
    assert torch.all(result == 3.0)


def test_collector_mathematical_accuracy(dummy_model, save_dir):
    """Verify that collector output matches manual XtX computation."""
    collector = InputCollector(
        dummy_model,
        target_nn_modules=["0", "1"],
        save_dir=save_dir,
        dram_limit_gb=1.0,
    )

    batches = [torch.randn(3, 4, 16) for _ in range(3)]

    expected_xtx = torch.zeros(16, 16)
    for b in batches:
        X = b.reshape(-1, 16)
        expected_xtx += X.T @ X

    collector.collect(batches)

    actual_xtx = load(save_dir, "0")
    assert torch.allclose(expected_xtx, actual_xtx, atol=1e-5)


def test_collector_dram_limit_trigger(dummy_model, save_dir):
    """Verify collector still works correctly when forced to flush multiple times."""
    collector = InputCollector(
        dummy_model,
        target_nn_modules=["0", "1"],
        save_dir=save_dir,
        dram_limit_gb=0.000001,
    )

    samples = [torch.randn(1, 16) for _ in range(10)]

    expected_xtx = torch.zeros(16, 16)
    for s in samples:
        expected_xtx += s.T @ s

    collector.collect(samples)

    actual_xtx = load(save_dir, "0")
    assert torch.allclose(expected_xtx, actual_xtx, atol=1e-5)


def test_writer_context_manager_closes_on_exception(save_dir):
    """Thread must not leak when an exception fires inside the with block."""
    writer = ShelfWriter(save_dir)
    try:
        with writer:
            writer.flush({"x": torch.ones(2, 2)})
            raise RuntimeError("simulated failure")
    except RuntimeError:
        pass

    assert not writer.thread.is_alive(), "Worker thread leaked after exception"
