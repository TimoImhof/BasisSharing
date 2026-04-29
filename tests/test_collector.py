import pytest
import os

from torch import nn
from basissharing.collector import ShelfWriter, InputCollector
import torch
import numpy as np


@pytest.fixture
def save_dir(tmp_path):
    d = tmp_path / "xtx_shelf"
    os.makedirs(d, exist_ok=True)
    return str(d)


def load(save_dir, uid):
    """Helper: load a tensor from the npy shelf."""
    return torch.from_numpy(np.load(os.path.join(save_dir, f"{uid}.npy")))


class TestShelfWriter:
    def test_shelf_writer_merges_correctly(self, save_dir):
        with ShelfWriter(save_dir) as writer:
            uid = "test_layer"
            writer.flush({uid: np.ones((4, 4))})
            writer.flush({uid: np.ones((4, 4)) * 2})

        result = load(save_dir, uid)
        assert torch.all(result == 3.0)

    def test_writer_context_manager_closes_on_exception(self, save_dir):
        """Thread must not leak when an exception fires inside the with block."""
        writer = ShelfWriter(save_dir)
        try:
            with writer:
                writer.flush({"x": np.ones((2, 2))})
                raise RuntimeError("simulated failure")
        except RuntimeError:
            pass

        assert not writer.thread.is_alive(), "Worker thread leaked after exception"


class TestCollector:
    def test_expected_files_created(self, model, save_dir, batches):
        """Verify that collector creates .npy files for each target module."""
        target_modules = ["q", "k", "o", "up", "down"]
        collector = InputCollector(
            model, target_nn_modules=target_modules, save_dir=save_dir
        )
        collector.collect(batches)

        created_files = set(os.listdir(save_dir))
        expected_files = {
            f"{i[0]}.npy"
            for i in model.named_modules()
            if i[0].split(".")[-1] in target_modules
        }
        assert expected_files.issubset(created_files), (
            f"Expected files {expected_files} not found in {created_files}"
        )

    def test_collector_math_matches_manual_computation(self, save_dir, batches):
        """Verify that collector output matches manual XtX computation for the first layer."""
        collector = InputCollector(
            nn.Sequential(
                nn.Linear(64, 64, bias=False),  # "0"
                nn.Linear(64, 64, bias=False),  # "1"
            ),
            target_nn_modules=["0"],
            save_dir=save_dir,
        )
        collector.collect(batches)
        actual_xtx = load(save_dir, "0")

        expected_xtx = torch.zeros(64, 64)
        for b in batches:
            X = b.reshape(-1, 64)  # (batch_size * seq_len, hidden_dim)
            XtX = X.T @ X
            expected_xtx += XtX
        assert torch.allclose(expected_xtx, actual_xtx, atol=1e-5)

    def test_xtx_accumulates_across_batches(self, model, tmp_path, samples):
        """More samples should produce larger XtX (more activation energy)."""
        xtx_dir_2 = str(tmp_path / "xtx_2")
        xtx_dir_8 = str(tmp_path / "xtx_8")

        samples_2 = samples[:2]
        samples_8 = samples

        InputCollector(model, ["q"], xtx_dir_2).collect(samples_2)
        InputCollector(model, ["q"], xtx_dir_8).collect(samples_8)

        xtx_2 = np.load(os.path.join(xtx_dir_2, "blocks.0.attn.q.npy"))
        xtx_8 = np.load(os.path.join(xtx_dir_8, "blocks.0.attn.q.npy"))
        assert np.linalg.norm(xtx_8) > np.linalg.norm(xtx_2)

    def test_xtx_stored_as_float32(self, model, batches, save_dir):
        InputCollector(model, ["q"], save_dir).collect(batches)
        xtx = np.load(os.path.join(save_dir, "blocks.0.attn.q.npy"))
        assert xtx.dtype == np.float32, f"Expected float32, got {xtx.dtype}"

    def test_xtx_is_symmetric(self, model, batches, save_dir):
        InputCollector(model, ["q", "k", "v", "up", "gate", "down"], save_dir).collect(
            batches
        )
        for xtx_file in os.listdir(save_dir):
            xtx = np.load(os.path.join(save_dir, xtx_file))
            assert np.allclose(xtx, xtx.T, atol=1e-5), f"{xtx_file} is not symmetric"
