import pytest
from basissharing import init_basissharing, BSConfig, ModuleSharingConfig
import torch


@pytest.fixture
def bs_config():
    return BSConfig(
        model_id="minimal-model",
        module_cfgs=[
            ModuleSharingConfig(
                module_name="q", group_size=2, enabled=True, compression_ratio=0.5
            ),
            ModuleSharingConfig(
                module_name="up", group_size=2, enabled=True, compression_ratio=0.5
            ),
        ],
    )


@pytest.fixture
def dummy_samples():
    # Hardcoded dim to match minimal_model's expected input dimension
    return [torch.rand(1, 256, 64) for _ in range(2)]


def test_minimal_model_forward(minimal_model, dummy_samples):
    """Sanity check that the minimal model can do a forward pass before compression."""
    with torch.no_grad():
        for sample in dummy_samples:
            out = minimal_model(sample)
            assert out.shape == sample.shape


def test_init(minimal_model, bs_config):
    init_basissharing(minimal_model, bs_config)
    assert hasattr(minimal_model, "basis_registry")
    assert hasattr(minimal_model, "bs_config")
    assert minimal_model.bs_config == bs_config


def test_compression_calculates_bases_and_replaces_layers(
    minimal_model, bs_config, dummy_samples
):
    init_basissharing(minimal_model, bs_config)
    minimal_model.execute_compression(samples=dummy_samples)

    assert (
        len(minimal_model.basis_registry.bases) == 4
    )  # (2 target modules: 'q', 'up') * (Group size 2 in a 4-layer model)
    for cfg in bs_config.module_cfgs:
        matching_keys = [
            k
            for k in minimal_model.basis_registry.bases.keys()
            if k.startswith(f"{cfg.module_name}_")  # uid = f"{cfg.module_name}_..."
        ]
        assert len(matching_keys) == 2

    target_names = [cfg.module_name for cfg in bs_config.module_cfgs]
    for name, module in minimal_model.named_modules():
        if name.split(".")[-1] in target_names:
            assert type(module).__name__ == "SharedLinear", (
                f"Module {name} was not replaced!"
            )
            assert hasattr(module, "basis_registry")


def test_model_fwd_and_bwd_after_compression(minimal_model, bs_config, dummy_samples):
    init_basissharing(minimal_model, bs_config)
    minimal_model.execute_compression(samples=dummy_samples)

    # fwd
    sample = dummy_samples[0]
    out = minimal_model(sample)
    assert out.shape == sample.shape

    # bwd
    loss = out.pow(2).mean()
    loss.backward()

    # Check shared bases in the registry received gradients
    for name, param in minimal_model.basis_registry.bases.items():
        assert param.grad is not None, f"Basis {name} is disconnected from the graph!"
        assert torch.any(param.grad != 0), f"Basis {name} received zero gradients."

    # Check unique layer coefficients received gradients
    for name, module in minimal_model.named_modules():
        if type(module).__name__ == "SharedLinear":
            assert module.coeffs.grad is not None, (
                f"Coefficients in {name} are disconnected!"
            )
