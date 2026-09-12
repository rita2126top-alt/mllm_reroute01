from dataclasses import asdict

import pytest
import torch

from cold_ghost.checkpoint import inspect_checkpoint, load_checkpoint, save_checkpoint
from cold_ghost.ghost import GhostConfig, GhostModules


def modules_and_metadata():
    modules = GhostModules(8, 4, GhostConfig(bottleneck_dim=2, num_prototypes=2, share_every=1))
    metadata = {"backbone": "synthetic", "hidden_size": 8, "num_layers": 4,
                "drop_layers": [0, 2], "keep_ratios": [0.5, 0.5], "ablation": "full",
                "ghost_config": asdict(modules.config), "manifest_sha256": "a" * 64}
    return modules, metadata


def test_strict_checkpoint_roundtrip_and_mismatch(tmp_path):
    modules, metadata = modules_and_metadata()
    target = tmp_path / "ghost.pt"
    original = {key: value.clone() for key, value in modules.state_dict().items()}
    save_checkpoint(target, modules, metadata, training_state={"complete": True, "smoke": False})
    with torch.no_grad():
        for parameter in modules.parameters():
            parameter.zero_()
    load_checkpoint(target, modules, expected_metadata=metadata)
    assert all(torch.equal(value, modules.state_dict()[key]) for key, value in original.items())
    with pytest.raises(ValueError, match="keep_ratios"):
        load_checkpoint(target, modules, expected_metadata={"keep_ratios": [0.1, 0.1]})
    assert "backbone" not in inspect_checkpoint(target)["state_dict"]


@pytest.mark.parametrize("state", [{"complete": False}, {"complete": True, "smoke": True}])
def test_formal_evaluation_rejects_partial_or_smoke(tmp_path, state):
    modules, metadata = modules_and_metadata()
    target = tmp_path / "ghost.pt"
    save_checkpoint(target, modules, metadata, training_state=state)
    with pytest.raises(ValueError, match="Formal evaluation"):
        load_checkpoint(target, modules)
    load_checkpoint(target, modules, require_complete=False)


def test_missing_provenance_rejected(tmp_path):
    modules, metadata = modules_and_metadata()
    del metadata["manifest_sha256"]
    with pytest.raises(ValueError, match="missing metadata"):
        save_checkpoint(tmp_path / "invalid.pt", modules, metadata)
