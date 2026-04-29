import torch
import os
from basissharing import WeightCompressor, InputCollector, BSConfig, ModuleSharingConfig


class TestCompressor:
    def _run_compression(self, model, samples, bs_config, tmp_path):
        xtx_dir = str(tmp_path / "xtx")
        weight_dir = str(tmp_path / "weights")
        InputCollector(
            model,
            bs_config.target_modules(),  # ["q", "up"]
            xtx_dir,
        ).collect(samples)
        WeightCompressor(bs_config).compress(model, xtx_dir, weight_dir)
        return xtx_dir, weight_dir

    def test_optimizer_creates_valid_files(self, model, batches, bs_config, tmp_path):
        _, weight_dir = self._run_compression(model, batches, bs_config, tmp_path)

        # check that we have 4 files for the 4 q and up modules (2 q's and 2 up's, each with group_size=2)
        assert len(os.listdir(weight_dir)) == 4
        assert os.path.exists(os.path.join(weight_dir, "q_2_0.pt"))
        assert os.path.exists(os.path.join(weight_dir, "q_2_1.pt"))
        assert os.path.exists(os.path.join(weight_dir, "up_2_0.pt"))
        assert os.path.exists(os.path.join(weight_dir, "up_2_1.pt"))
        # check we have weights for basis and coeffs
        weight_file = os.path.join(weight_dir, "q_2_0.pt")
        data = torch.load(weight_file)
        assert "basis" in data
        assert "coeffs" in data

    def test_saved_weights_are_float32(self, model, batches, bs_config, tmp_path):
        _, weight_dir = self._run_compression(model, batches, bs_config, tmp_path)
        data = torch.load(os.path.join(weight_dir, "q_2_0.pt"))
        assert data["basis"].dtype == torch.float32, (
            f"Basis should be float32, got {data['basis'].dtype}"
        )
        assert data["coeffs"].dtype == torch.float32, (
            f"Coeffs should be float32, got {data['coeffs'].dtype}"
        )

    def test_no_nan_or_inf_in_saved_weights(self, model, batches, bs_config, tmp_path):
        _, weight_dir = self._run_compression(model, batches, bs_config, tmp_path)
        for fname in os.listdir(weight_dir):
            data = torch.load(os.path.join(weight_dir, fname))
            assert not torch.isnan(data["basis"]).any()
            assert not torch.isnan(data["coeffs"]).any()
            assert not torch.isinf(data["basis"]).any()
            assert not torch.isinf(data["coeffs"]).any()

    def test_basis_shape(self, model, batches, bs_config, tmp_path):
        _, weight_dir = self._run_compression(model, batches, bs_config, tmp_path)
        data = torch.load(os.path.join(weight_dir, "q_2_0.pt"))
        d1, k = data["basis"].shape
        assert d1 == 64
        assert d1 > k > 0

    def test_coeffs_shape(self, model, batches, bs_config, tmp_path):
        _, weight_dir = self._run_compression(model, batches, bs_config, tmp_path)
        data = torch.load(os.path.join(weight_dir, "q_2_0.pt"))
        k, n_d2 = data["coeffs"].shape
        assert n_d2 == 2 * 64

    def test_compressed_param_count_matches_ratio(
        self, model, batches, bs_config, tmp_path
    ):
        _, weight_dir = self._run_compression(model, batches, bs_config, tmp_path)
        # q group: 2 layers, each [64, 64] → original = 2 * 64 * 64 = 8192
        data = torch.load(os.path.join(weight_dir, "q_2_0.pt"))
        basis, coeffs = data["basis"], data["coeffs"]

        d1, k = basis.shape
        k2, n_d2 = coeffs.shape
        assert k == k2, "Basis and coeffs rank mismatch"

        original_params = 2 * 64 * 64  # 2 layers * d1 * d2
        compressed_params = d1 * k + k * n_d2
        actual_ratio = 1 - compressed_params / original_params

        assert abs(actual_ratio - 0.5) < 0.01, (
            f"Compression ratio off: expected ~0.5, got {actual_ratio:.3f}"
        )

    def test_minimum_rank_is_one(self, model, batches, tmp_path):
        aggressive_config = BSConfig(
            model_id="minimal",
            module_cfgs=[
                ModuleSharingConfig("q", group_size=2, compression_ratio=0.99)
            ],
        )
        _, weight_dir = self._run_compression(
            model, batches, aggressive_config, tmp_path
        )
        data = torch.load(os.path.join(weight_dir, "q_2_0.pt"))
        k = data["basis"].shape[1]
        assert k >= 1
