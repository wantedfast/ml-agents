import hashlib
import inspect

import pytest

from mlagents.torch_utils import torch
from mlagents.trainers.exception import UnityTrainerException
from mlagents.trainers.torch_entities.encoders import ResNetVisualEncoder
from mlagents.trainers.torch_entities.pretrained_visual_encoder import (
    NAVIGATION_GEOMETRY_PADDING_SIZE,
    NAVIGATION_GEOMETRY_SIZE,
    NavigationGeometryAdapter,
    POSITION_STATE_PADDING_SIZE,
    POSITION_STATE_SIZE,
    ExactState28PositionAdapter,
    PositionStateAdapter,
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
from mlagents_envs.base_env import (
    ActionSpec,
    BehaviorSpec,
    DimensionProperty,
    ObservationSpec,
    ObservationType,
)


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
                "encoder_type": "resnet",
                "pretraining_schema": "world_only_v2",
                "qualified_for_rl": True,
                "dataset_manifest_sha256": "a" * 64,
            },
        },
        path,
    )


def save_probe_checkpoint(path, probe, encoder_checkpoint):
    encoder_digest = hashlib.sha256(encoder_checkpoint.read_bytes()).hexdigest()
    torch.save(
        {
            "probe_state_dict": probe.state_dict(),
            "metadata": {
                "probe_type": "linear",
                "encoder_checkpoint_sha256": encoder_digest,
                "dataset_manifest_sha256": "a" * 64,
                "station_order": ["Onion", "Dish", "Pot", "Delivery"],
                "grid_size": 16,
                "embedding_size": 512,
            },
        },
        path,
    )


def save_position_probe_checkpoint(path, probe, encoder_checkpoint):
    encoder_digest = hashlib.sha256(encoder_checkpoint.read_bytes()).hexdigest()
    torch.save(
        {
            "probe_state_dict": probe.state_dict(),
            "metadata": {
                "probe_type": "position_state_grid_classifier",
                "encoder_checkpoint_sha256": encoder_digest,
                "dataset_manifest_sha256": "a" * 64,
                "object_order": [
                    "Agent1", "Agent2", "Onion", "Dish", "Pot", "Counter"
                ],
                "grid_size": 16,
                "embedding_size": 512,
                "position_state_size": 12,
                "qualified_for_rl": True,
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


def test_rejects_legacy_non_world_only_checkpoint(tmp_path):
    source = ResNetVisualEncoder(84, 84, 3, 512)
    checkpoint = tmp_path / "legacy_encoder.pt"
    save_checkpoint(checkpoint, source)
    payload = torch.load(checkpoint, map_location="cpu")
    del payload["metadata"]["pretraining_schema"]
    torch.save(payload, checkpoint)

    with pytest.raises(UnityTrainerException, match="pretraining_schema"):
        load_pretrained_visual_encoders(
            VisualModule(), VisualModule(), str(checkpoint), True
        )


def test_rejects_unqualified_world_only_checkpoint(tmp_path):
    source = ResNetVisualEncoder(84, 84, 3, 512)
    checkpoint = tmp_path / "unqualified_encoder.pt"
    save_checkpoint(checkpoint, source)
    payload = torch.load(checkpoint, map_location="cpu")
    payload["metadata"]["qualified_for_rl"] = False
    torch.save(payload, checkpoint)

    with pytest.raises(UnityTrainerException, match="not qualified"):
        load_pretrained_visual_encoders(
            VisualModule(), VisualModule(), str(checkpoint), True
        )


def test_poca_optimizer_excludes_and_reapplies_frozen_encoder(tmp_path):
    source = ResNetVisualEncoder(84, 84, 3, 512)
    checkpoint = tmp_path / "encoder.pt"
    save_checkpoint(checkpoint, source)
    probe_path = tmp_path / "probe.pt"
    save_probe_checkpoint(probe_path, NavigationGeometryAdapter(), checkpoint)
    settings = poca_dummy_config()
    settings.network_settings = NetworkSettings(
        hidden_units=512,
        num_layers=2,
        vis_encode_type=EncoderType.RESNET,
        pretrained_visual_encoder_path=str(checkpoint),
        freeze_visual_encoder=True,
        use_navigation_probe_features=True,
        visual_navigation_probe_path=str(probe_path),
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
    critic_encoder = next(
        module
        for module in optimizer.critic.modules()
        if isinstance(module, ResNetVisualEncoder)
    )
    optimized_ids = {
        id(parameter)
        for group in optimizer.optimizer.param_groups
        for parameter in group["params"]
    }
    assert all(id(parameter) not in optimized_ids for parameter in actor_encoder.parameters())
    assert all(
        id(parameter) not in optimized_ids for parameter in critic_encoder.parameters()
    )

    frozen_before = {
        "actor": {
            name: value.detach().clone()
            for name, value in actor_encoder.state_dict().items()
        },
        "critic": {
            name: value.detach().clone()
            for name, value in critic_encoder.state_dict().items()
        },
    }
    optimizer.optimizer.zero_grad()
    trainable_loss = sum(
        parameter.sum()
        for group in optimizer.optimizer.param_groups
        for parameter in group["params"]
    )
    trainable_loss.backward()
    optimizer.optimizer.step()
    for name, value in actor_encoder.state_dict().items():
        assert torch.equal(frozen_before["actor"][name], value)
    for name, value in critic_encoder.state_dict().items():
        assert torch.equal(frozen_before["critic"][name], value)

    with torch.no_grad():
        next(actor_encoder.parameters()).fill_(42.0)
    optimizer.reload_pretrained_visual_encoder()
    for expected, actual in zip(source.parameters(), actor_encoder.parameters()):
        assert torch.equal(expected, actual)


def test_navigation_probe_adapter_exposes_geometry_and_preserves_width(tmp_path):
    source = ResNetVisualEncoder(84, 84, 3, 512)
    checkpoint = tmp_path / "encoder.pt"
    save_checkpoint(checkpoint, source)
    probe = NavigationGeometryAdapter()
    probe_path = tmp_path / "probe.pt"
    save_probe_checkpoint(probe_path, probe, checkpoint)
    actor, critic = VisualModule(), VisualModule()

    load_pretrained_visual_encoders(
        actor, critic, str(checkpoint), True, str(probe_path)
    )

    output = actor.encoder(torch.rand(2, 3, 84, 84))
    assert output.shape == (2, 512)
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output[:, NAVIGATION_GEOMETRY_SIZE:]) == 0
    assert (
        output[:, NAVIGATION_GEOMETRY_SIZE:].shape[1]
        == NAVIGATION_GEOMETRY_PADDING_SIZE
    )
    critic_output = critic.encoder(torch.rand(2, 3, 84, 84))
    assert torch.isfinite(critic_output).all()
    assert torch.count_nonzero(critic_output[:, NAVIGATION_GEOMETRY_SIZE:]) == 0
    assert hasattr(actor.encoder, "navigation_geometry_adapter")
    assert all(
        not parameter.requires_grad
        for parameter in actor.encoder.navigation_geometry_adapter.parameters()
    )

    # Reapplying the checkpoints must replace, rather than stack, forward hooks.
    first_output = output.clone()
    load_pretrained_visual_encoders(
        actor, critic, str(checkpoint), True, str(probe_path)
    )
    second_output = actor.encoder(torch.rand(2, 3, 84, 84))
    assert second_output.shape == first_output.shape


def test_navigation_probe_adapter_exports_with_onnx_opset_9(tmp_path):
    adapter = NavigationGeometryAdapter()
    output_path = tmp_path / "navigation_adapter.onnx"

    export_options = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_options["dynamo"] = False
    torch.onnx.export(
        adapter,
        torch.zeros((1, 512)),
        str(output_path),
        opset_version=9,
        input_names=["embedding"],
        output_names=["geometry"],
        dynamic_axes={"embedding": {0: "batch"}, "geometry": {0: "batch"}},
        **export_options,
    )

    assert output_path.is_file()


def test_navigation_probe_rejects_wrong_encoder_hash(tmp_path):
    source = ResNetVisualEncoder(84, 84, 3, 512)
    checkpoint = tmp_path / "encoder.pt"
    save_checkpoint(checkpoint, source)
    probe_path = tmp_path / "probe.pt"
    save_probe_checkpoint(probe_path, NavigationGeometryAdapter(), checkpoint)
    payload = torch.load(probe_path, map_location="cpu")
    payload["metadata"]["encoder_checkpoint_sha256"] = "b" * 64
    torch.save(payload, probe_path)

    with pytest.raises(UnityTrainerException, match="encoder_checkpoint_sha256"):
        load_pretrained_visual_encoders(
            VisualModule(), VisualModule(), str(checkpoint), True, str(probe_path)
        )


def test_position_state_adapter_exposes_positions_and_preserves_width(tmp_path):
    source = ResNetVisualEncoder(84, 84, 3, 512)
    checkpoint = tmp_path / "encoder.pt"
    save_checkpoint(checkpoint, source)
    probe = PositionStateAdapter()
    probe_path = tmp_path / "position_probe.pt"
    save_position_probe_checkpoint(probe_path, probe, checkpoint)
    actor, critic = VisualModule(), VisualModule()

    load_pretrained_visual_encoders(
        actor,
        critic,
        str(checkpoint),
        True,
        position_probe_path=str(probe_path),
    )

    output = actor.encoder(torch.rand(2, 3, 84, 84))
    critic_output = critic.encoder(torch.rand(2, 3, 84, 84))
    assert output.shape == (2, 512)
    assert torch.isfinite(output).all()
    assert torch.all((output[:, :POSITION_STATE_SIZE] >= 0.0))
    assert torch.all((output[:, :POSITION_STATE_SIZE] <= 1.0))
    assert torch.count_nonzero(output[:, POSITION_STATE_SIZE:]) == 0
    assert output[:, POSITION_STATE_SIZE:].shape[1] == POSITION_STATE_PADDING_SIZE
    assert torch.count_nonzero(critic_output[:, POSITION_STATE_SIZE:]) == 0
    assert all(
        not parameter.requires_grad
        for parameter in actor.encoder.position_state_adapter.parameters()
    )


def test_position_state_adapter_rejects_unqualified_probe(tmp_path):
    source = ResNetVisualEncoder(84, 84, 3, 512)
    checkpoint = tmp_path / "encoder.pt"
    save_checkpoint(checkpoint, source)
    probe_path = tmp_path / "position_probe.pt"
    save_position_probe_checkpoint(probe_path, PositionStateAdapter(), checkpoint)
    payload = torch.load(probe_path, map_location="cpu")
    payload["metadata"]["qualified_for_rl"] = False
    torch.save(payload, probe_path)

    with pytest.raises(UnityTrainerException, match="qualified_for_rl"):
        load_pretrained_visual_encoders(
            VisualModule(),
            VisualModule(),
            str(checkpoint),
            True,
            position_probe_path=str(probe_path),
        )


def test_exact_state28_adapter_outputs_only_positions_and_freezes_both_sides(tmp_path):
    source = ResNetVisualEncoder(84, 84, 3, 512)
    checkpoint = tmp_path / "encoder.pt"
    save_checkpoint(checkpoint, source)
    probe_path = tmp_path / "position_probe.pt"
    save_position_probe_checkpoint(
        probe_path, ExactState28PositionAdapter(), checkpoint
    )
    # Exact State28 consumes continuous expected coordinates. The historical
    # checkpoint missed the stricter hard-cell threshold but has low coordinate
    # MAE, so this mode intentionally does not claim hard-cell qualification.
    payload = torch.load(probe_path, map_location="cpu")
    payload["metadata"]["qualified_for_rl"] = False
    torch.save(payload, probe_path)
    actor, critic = VisualModule(), VisualModule()

    load_pretrained_visual_encoders(
        actor,
        critic,
        str(checkpoint),
        True,
        exact_state28_position_probe_path=str(probe_path),
    )

    actor_output = actor.encoder(torch.rand(2, 3, 84, 84))
    critic_output = critic.encoder(torch.rand(2, 3, 84, 84))
    assert actor_output.shape == (2, 12)
    assert critic_output.shape == (2, 12)
    assert torch.isfinite(actor_output).all()
    assert torch.all((actor_output >= 0.0) & (actor_output <= 1.0))
    assert all(
        not parameter.requires_grad
        for parameter in actor.encoder.position_state_adapter.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in critic.encoder.position_state_adapter.parameters()
    )


def test_poca_optimizer_builds_frozen_cnn_exact_state28_contract(tmp_path):
    source = ResNetVisualEncoder(84, 84, 3, 512)
    checkpoint = tmp_path / "encoder.pt"
    save_checkpoint(checkpoint, source)
    probe_path = tmp_path / "position_probe.pt"
    save_position_probe_checkpoint(
        probe_path, ExactState28PositionAdapter(), checkpoint
    )
    settings = poca_dummy_config()
    settings.network_settings = NetworkSettings(
        normalize=True,
        hidden_units=512,
        num_layers=2,
        vis_encode_type=EncoderType.RESNET,
        pretrained_visual_encoder_path=str(checkpoint),
        freeze_visual_encoder=True,
        visual_position_probe_path=str(probe_path),
        use_cnn_exact_state28=True,
    )
    settings.reward_signals = {
        RewardSignalType.EXTRINSIC: RewardSignalSettings(strength=1.0, gamma=0.99)
    }
    visual_spec = ObservationSpec(
        shape=(3, 84, 84),
        dimension_property=(
            DimensionProperty.NONE,
            DimensionProperty.TRANSLATIONAL_EQUIVARIANCE,
            DimensionProperty.TRANSLATIONAL_EQUIVARIANCE,
        ),
        observation_type=ObservationType.DEFAULT,
        name="01_OverallView",
    )
    vector_spec = ObservationSpec(
        shape=(17,),
        dimension_property=(DimensionProperty.UNSPECIFIED,),
        observation_type=ObservationType.DEFAULT,
        name="VectorSensor_size17",
    )
    behavior_spec = BehaviorSpec(
        [visual_spec, vector_spec], ActionSpec.create_continuous(2)
    )
    policy = TorchPolicy(0, behavior_spec, settings.network_settings, SimpleActor, {})

    optimizer = TorchPOCAOptimizer(policy, settings)

    actor_encoder = next(
        module
        for module in policy.actor.modules()
        if isinstance(module, ResNetVisualEncoder)
    )
    critic_encoder = next(
        module
        for module in optimizer.critic.modules()
        if isinstance(module, ResNetVisualEncoder)
    )
    optimized_ids = {
        id(parameter)
        for group in optimizer.optimizer.param_groups
        for parameter in group["params"]
    }
    for encoder in (actor_encoder, critic_encoder):
        assert all(not parameter.requires_grad for parameter in encoder.parameters())
        assert all(
            id(parameter) not in optimized_ids for parameter in encoder.parameters()
        )
        assert encoder(torch.rand(2, 3, 84, 84)).shape == (2, 12)
    assert policy.actor.network_body.observation_encoder.total_enc_size == 28
    assert optimizer.critic.network_body.observation_encoder.total_enc_size == 28
