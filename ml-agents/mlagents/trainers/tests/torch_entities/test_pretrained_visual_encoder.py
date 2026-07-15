import pytest

from mlagents.torch_utils import torch
from mlagents.trainers.exception import UnityTrainerException
from mlagents.trainers.torch_entities.encoders import ResNetVisualEncoder
from mlagents.trainers.torch_entities.pretrained_visual_encoder import (
    load_pretrained_visual_encoders,
)
from mlagents.trainers.poca.optimizer_torch import TorchPOCAOptimizer
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.settings import (
    EncoderType,
    NetworkSettings,
    RewardSignalSettings,
    RewardSignalType,
)
from mlagents.trainers.tests import mock_brain
from mlagents.trainers.tests.dummy_config import poca_dummy_config
from mlagents.trainers.torch_entities.networks import SimpleActor


class VisualModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNetVisualEncoder(84, 84, 3, 512)
        self.head = torch.nn.Linear(512, 2)


def save_checkpoint(path, encoder):
    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "metadata": {
                "height": 84,
                "width": 84,
                "channels": 3,
                "output_size": 512,
            },
        },
        path,
    )


def test_loads_and_freezes_actor_and_critic(tmp_path):
    source = ResNetVisualEncoder(84, 84, 3, 512)
    checkpoint = tmp_path / "encoder.pt"
    save_checkpoint(checkpoint, source)
    actor, critic = VisualModule(), VisualModule()

    actor_count, critic_count, digest = load_pretrained_visual_encoders(
        actor, critic, str(checkpoint), True
    )

    assert (actor_count, critic_count) == (1, 1)
    assert len(digest) == 64
    assert all(not parameter.requires_grad for parameter in actor.encoder.parameters())
    assert all(not parameter.requires_grad for parameter in critic.encoder.parameters())
    assert all(parameter.requires_grad for parameter in actor.head.parameters())
    for expected, actual in zip(source.parameters(), actor.encoder.parameters()):
        assert torch.equal(expected, actual)


def test_rejects_incompatible_checkpoint(tmp_path):
    checkpoint = tmp_path / "encoder.pt"
    torch.save({"encoder_state_dict": {}, "metadata": {}}, checkpoint)
    with pytest.raises(UnityTrainerException, match="metadata mismatch"):
        load_pretrained_visual_encoders(
            VisualModule(), VisualModule(), str(checkpoint), True
        )


def test_poca_optimizer_excludes_and_reapplies_frozen_encoder(tmp_path):
    source = ResNetVisualEncoder(84, 84, 3, 512)
    checkpoint = tmp_path / "encoder.pt"
    save_checkpoint(checkpoint, source)
    settings = poca_dummy_config()
    settings.network_settings = NetworkSettings(
        hidden_units=512,
        num_layers=2,
        vis_encode_type=EncoderType.RESNET,
        pretrained_visual_encoder_path=str(checkpoint),
        freeze_visual_encoder=True,
    )
    settings.reward_signals = {
        RewardSignalType.EXTRINSIC: RewardSignalSettings(strength=1.0, gamma=0.99)
    }
    behavior_spec = mock_brain.setup_test_behavior_specs(
        use_discrete=False, use_visual=True, vector_action_space=2
    )
    policy = TorchPolicy(0, behavior_spec, settings.network_settings, SimpleActor, {})
    optimizer = TorchPOCAOptimizer(policy, settings)
    actor_encoder = next(
        module
        for module in policy.actor.modules()
        if isinstance(module, ResNetVisualEncoder)
    )
    optimized_ids = {
        id(parameter)
        for group in optimizer.optimizer.param_groups
        for parameter in group["params"]
    }
    assert all(id(parameter) not in optimized_ids for parameter in actor_encoder.parameters())

    with torch.no_grad():
        next(actor_encoder.parameters()).fill_(42.0)
    optimizer.reload_pretrained_visual_encoder()
    for expected, actual in zip(source.parameters(), actor_encoder.parameters()):
        assert torch.equal(expected, actual)
