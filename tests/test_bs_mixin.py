import pytest
import copy
import torch
from basissharing import init_basissharing, BSConfig, ModuleSharingConfig
from basissharing.collector import InputCollector
from basissharing.compressor import WeightCompressor
from basissharing.bs_config import get_groups
from tests.conftest import MinimalModel


@pytest.fixture
def compressed_model(model, batches, bs_config, tmp_path):
    """Full pipeline fixture: returns a compressed model ready for inference."""
    xtx_dir = str(tmp_path / "xtx")
    weight_dir = str(tmp_path / "weights")
    InputCollector(model, bs_config.target_modules(), xtx_dir).collect(batches)
    WeightCompressor(bs_config).compress(model, xtx_dir, weight_dir)
    init_basissharing(model, bs_config)
    model.apply_compression(weight_dir)
    return model, weight_dir


class TestGetGroups:
    def test_groups_cover_all_matched_layers(self, model, bs_config):
        groups = get_groups(model, bs_config)
        all_layers = [layer for g in groups.values() for layer in g["layers"]]
        # 4 blocks × 1 q each = 4 q layers, 4 up layers
        q_layers = [layer for layer in all_layers if layer.endswith(".q")]
        up_layers = [layer for layer in all_layers if layer.endswith(".up")]
        assert len(q_layers) == 4
        assert len(up_layers) == 4

    def test_group_sizes_are_correct(self, model, bs_config):
        groups = get_groups(model, bs_config)
        for uid, group in groups.items():
            assert len(group["layers"]) == group["cfg"].group_size

    def test_no_layer_appears_in_multiple_groups(self, model, bs_config):
        groups = get_groups(model, bs_config)
        all_layers = [layer for g in groups.values() for layer in g["layers"]]
        assert len(all_layers) == len(set(all_layers))

    def test_disabled_module_not_included(self, model):
        config = BSConfig(
            model_id="minimal",
            module_cfgs=[
                ModuleSharingConfig(
                    "q", group_size=2, compression_ratio=0.5, enabled=False
                ),
                ModuleSharingConfig("up", group_size=2, compression_ratio=0.5),
            ],
        )
        groups = get_groups(model, config)
        for group in groups.values():
            for layer in group["layers"]:
                assert not layer.endswith(".q")


class TestSharedLinear:
    def test_output_shape(self, compressed_model, batches):
        model, _ = compressed_model
        model.eval()
        x = batches[0]
        with torch.no_grad():
            out = model(x)
        assert out.shape == batches[0].shape

    def test_no_nan_or_inf_in_output(self, compressed_model, batches):
        model, _ = compressed_model
        model.eval()
        with torch.no_grad():
            out = model(batches[0])
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_output_is_deterministic_in_eval(self, compressed_model, batches):
        model, _ = compressed_model
        model.eval()
        x = batches[0]
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)


class TestBasisSharingMixin:
    def test_targeted_layers_replaced(self, compressed_model):
        model, _ = compressed_model
        for name, module in model.named_modules():
            if name.endswith(".q") or name.endswith(".up"):
                assert type(module).__name__ == "SharedLinear"

    def test_non_targeted_layers_unchanged(self, compressed_model):
        model, _ = compressed_model
        for name, module in model.named_modules():
            if not (name.endswith(".q") or name.endswith(".up")):
                assert type(module).__name__ != "SharedLinear"

    def test_layers_in_group_share_basis_tensor(self, compressed_model):
        model, _ = compressed_model
        q0 = model.blocks[0].attn.q
        q1 = model.blocks[1].attn.q
        assert q0.uid == q1.uid
        b0 = q0.basis_registry.get_basis(q0.uid)
        b1 = q1.basis_registry.get_basis(q1.uid)
        # Ensure both references point to the exact same basis object
        assert b0 is b1

    def test_layers_in_different_groups_have_different_basis(self, compressed_model):
        model, _ = compressed_model
        q0 = model.blocks[0].attn.q  # group q_2_0
        q2 = model.blocks[2].attn.q  # group q_2_1
        assert q0.uid != q2.uid
        b0 = q0.basis_registry.get_basis(q0.uid)
        b2 = q2.basis_registry.get_basis(q2.uid)
        assert b0 is not b2

    def test_compressed_output_close_to_original(
        self, model, batches, bs_config, tmp_path
    ):
        original = copy.deepcopy(model)
        original.eval()
        x = batches[0]
        with torch.no_grad():
            orig_out = original(x)

        xtx_dir = str(tmp_path / "xtx")
        weight_dir = str(tmp_path / "weights")
        InputCollector(model, bs_config.target_modules(), xtx_dir).collect(batches)
        WeightCompressor(bs_config).compress(model, xtx_dir, weight_dir)
        init_basissharing(model, bs_config)
        model.apply_compression(weight_dir)
        model.eval()
        with torch.no_grad():
            comp_out = model(x)

        rel_err = (orig_out - comp_out).norm() / orig_out.norm()
        assert rel_err < 1.0, f"Output diverged too much: rel_err={rel_err:.3f}"


class TestSaveLoad:
    def test_save_load_produces_identical_output(
        self, compressed_model, batches, tmp_path
    ):
        model, _ = compressed_model
        save_dir = str(tmp_path / "saved_weights")
        model.eval()
        x = batches[0]
        with torch.no_grad():
            ref_out = model(x)

        model.save_compressed_weights(save_dir)

        fresh = MinimalModel(num_layers=4, hidden=64, ffn_dim=128)
        init_basissharing(fresh, model.bs_config)
        fresh.load_compressed_weights(save_dir)
        fresh.eval()
        with torch.no_grad():
            loaded_out = fresh(x)

        assert torch.allclose(ref_out, loaded_out, atol=1e-5)

    def test_loaded_model_has_shared_linear_layers(self, compressed_model, tmp_path):
        model, _ = compressed_model
        save_dir = str(tmp_path / "saved_weights")
        model.save_compressed_weights(save_dir)

        fresh = MinimalModel(num_layers=4, hidden=64, ffn_dim=128)
        init_basissharing(fresh, model.bs_config)
        fresh.load_compressed_weights(save_dir)

        assert type(fresh.blocks[0].attn.q).__name__ == "SharedLinear"
        assert type(fresh.blocks[0].ffn.up).__name__ == "SharedLinear"
