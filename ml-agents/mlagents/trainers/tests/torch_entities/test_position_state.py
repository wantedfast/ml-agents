import inspect

import pytest

from mlagents.torch_utils import torch
from mlagents.trainers.exception import UnityTrainerException
from mlagents.trainers.settings import EncoderType
from mlagents.trainers.torch_entities.encoders import (
    OracleEgoState28Input,
    OracleEgoState28CompactInput,
    OraclePositionStateInput,
    VectorInput,
)
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents.trainers.torch_entities.networks import ObservationEncoder
from mlagents_envs.base_env import (
    DimensionProperty,
    ObservationSpec,
    ObservationType,
)


def vector_spec(name, size):
    return ObservationSpec(
        shape=(size,),
        dimension_property=(DimensionProperty.UNSPECIFIED,),
        observation_type=ObservationType.DEFAULT,
        name=name,
    )


def visual_spec(name="01_OverallView"):
    return ObservationSpec(
        shape=(3, 84, 84),
        dimension_property=(
            DimensionProperty.NONE,
            DimensionProperty.TRANSLATIONAL_EQUIVARIANCE,
            DimensionProperty.TRANSLATIONAL_EQUIVARIANCE,
        ),
        observation_type=ObservationType.DEFAULT,
        name=name,
    )


def test_oracle_position_state_preserves_prefix_and_zero_pads():
    processor = OraclePositionStateInput()
    positions = torch.linspace(0.0, 1.0, 12).reshape(1, 12)
    output = processor(positions)

    assert output.shape == (1, 512)
    assert torch.equal(output[:, :12], positions)
    assert torch.count_nonzero(output[:, 12:]) == 0
    assert sum(parameter.numel() for parameter in processor.parameters()) == 0


def test_oracle_position_state_processor_is_named_and_not_normalized():
    specs = [
        vector_spec("01_OraclePositionState", 12),
        vector_spec("VectorSensor_size17", 17),
    ]
    processors, sizes = ModelUtils.create_input_processors(
        specs,
        h_size=512,
        vis_encode_type=EncoderType.SIMPLE,
        attention_embedding_size=128,
        normalize=True,
        use_oracle_position_state=True,
    )

    assert sizes == [512, 17]
    assert isinstance(processors[0], OraclePositionStateInput)
    assert isinstance(processors[1], VectorInput)
    assert processors[1].normalizer is not None


def test_oracle_position_state_rejects_wrong_sensor():
    with pytest.raises(UnityTrainerException, match="exactly one"):
        ModelUtils.create_input_processors(
            [vector_spec("VectorSensor_size17", 17)],
            512,
            EncoderType.SIMPLE,
            128,
            use_oracle_position_state=True,
        )

    with pytest.raises(UnityTrainerException, match="must have shape"):
        ModelUtils.create_input_processors(
            [vector_spec("01_OraclePositionState", 11)],
            512,
            EncoderType.SIMPLE,
            128,
            use_oracle_position_state=True,
        )


def test_oracle_position_state_exports_with_onnx_opset_9(tmp_path):
    output_path = tmp_path / "oracle_position_state.onnx"
    export_options = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_options["dynamo"] = False
    torch.onnx.export(
        OraclePositionStateInput(),
        torch.zeros((1, 12)),
        str(output_path),
        opset_version=9,
        input_names=["positions"],
        output_names=["padded_positions"],
        dynamic_axes={"positions": {0: "batch"}, "padded_positions": {0: "batch"}},
        **export_options,
    )
    assert output_path.is_file()


def test_oracle_ego_state28_preserves_geometry_and_zero_pads():
    processor = OracleEgoState28Input(normalize=False)
    geometry = torch.linspace(-1.0, 1.0, 12).reshape(1, 12)
    output = processor(geometry)

    assert output.shape == (1, 512)
    assert torch.equal(output[:, :12], geometry)
    assert torch.count_nonzero(output[:, 12:]) == 0
    assert sum(parameter.numel() for parameter in processor.parameters()) == 0


def test_oracle_ego_state28_processor_contract():
    specs = [
        vector_spec("01_OracleEgoState28", 12),
        vector_spec("VectorSensor_size16", 16),
    ]
    processors, sizes = ModelUtils.create_input_processors(
        specs,
        h_size=512,
        vis_encode_type=EncoderType.SIMPLE,
        attention_embedding_size=128,
        normalize=True,
        use_oracle_ego_state28=True,
    )

    assert sizes == [512, 16]
    assert isinstance(processors[0], OracleEgoState28Input)
    assert isinstance(processors[1], VectorInput)
    assert processors[0].normalizer is not None
    assert processors[1].normalizer is not None


def test_oracle_ego_state28_rejects_wrong_contract():
    with pytest.raises(UnityTrainerException, match="exactly one"):
        ModelUtils.create_input_processors(
            [vector_spec("VectorSensor_size16", 16)],
            512,
            EncoderType.SIMPLE,
            128,
            use_oracle_ego_state28=True,
        )

    with pytest.raises(UnityTrainerException, match="must have shape"):
        ModelUtils.create_input_processors(
            [vector_spec("01_OracleEgoState28", 11)],
            512,
            EncoderType.SIMPLE,
            128,
            use_oracle_ego_state28=True,
        )


def test_compact_oracle_ego_state28_keeps_total_input_at_28():
    specs = [
        vector_spec("01_OracleEgoState28", 12),
        vector_spec("VectorSensor_size16", 16),
    ]
    processors, sizes = ModelUtils.create_input_processors(
        specs,
        h_size=512,
        vis_encode_type=EncoderType.SIMPLE,
        attention_embedding_size=128,
        normalize=True,
        use_oracle_ego_state28_compact=True,
    )

    assert sizes == [12, 16]
    assert sum(sizes) == 28
    assert isinstance(processors[0], OracleEgoState28CompactInput)
    assert processors[0].normalizer is not None
    assert isinstance(processors[1], VectorInput)


def test_cnn_exact_state28_processors_keep_raw_contract_compact():
    processors, sizes = ModelUtils.create_input_processors(
        [visual_spec(), vector_spec("VectorSensor_size17", 17)],
        h_size=512,
        vis_encode_type=EncoderType.RESNET,
        attention_embedding_size=128,
        normalize=True,
        use_cnn_exact_state28=True,
    )

    assert sizes == [12, 17]
    assert isinstance(processors[1], VectorInput)
    assert processors[1].normalizer is None


def test_cnn_exact_state28_fusion_matches_legacy_order_for_both_agents():
    # Grid coordinates are selected so the inverse camera transform produces
    # simple, distinct values for every canonical object.
    positions = torch.tensor(
        [[0.10, 0.20, 0.70, 0.80, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]],
        dtype=torch.float32,
    )
    manual_agent1 = torch.tensor(
        [[
            0.1, -0.2, 0.3, -0.4,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            -1.0,
            1.0, 0.0, 1.0, 0.0,
        ]],
        dtype=torch.float32,
    )
    manual_agent2 = manual_agent1.clone()
    manual_agent2[:, 0:4] = torch.tensor([[0.3, -0.4, 0.1, -0.2]])
    manual_agent2[:, 4:12] = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]]
    )
    manual_agent2[:, 12] = 1.0

    world = -0.5390625 + positions * 1.078125
    agent1, agent2 = world[:, 0:2], world[:, 2:4]
    onion, dish = world[:, 4:6], world[:, 6:8]
    pot, counter = world[:, 8:10], world[:, 10:12]
    expected_agent1 = torch.cat(
        [
            agent1,
            manual_agent1[:, 0:2],
            (agent2 - agent1) / 2.0,
            manual_agent1[:, 2:4],
            manual_agent1[:, 4:8],
            manual_agent1[:, 8:12],
            (pot - agent1) / 2.0,
            manual_agent1[:, 13:17],
            onion,
            dish,
            counter,
        ],
        dim=1,
    )
    expected_agent2 = torch.cat(
        [
            agent2,
            manual_agent2[:, 0:2],
            (agent1 - agent2) / 2.0,
            manual_agent2[:, 2:4],
            manual_agent2[:, 4:8],
            manual_agent2[:, 8:12],
            (pot - agent2) / 2.0,
            manual_agent2[:, 13:17],
            onion,
            dish,
            counter,
        ],
        dim=1,
    )

    actual_agent1 = ObservationEncoder._fuse_exact_state28(
        positions, manual_agent1
    )
    actual_agent2 = ObservationEncoder._fuse_exact_state28(
        positions, manual_agent2
    )

    assert actual_agent1.shape == (1, 28)
    assert actual_agent2.shape == (1, 28)
    assert torch.allclose(actual_agent1, expected_agent1)
    assert torch.allclose(actual_agent2, expected_agent2)


def test_cnn_exact_state28_rejects_wrong_raw_contract():
    with pytest.raises(UnityTrainerException, match="one visual sensor"):
        ModelUtils.create_input_processors(
            [visual_spec(), vector_spec("VectorSensor_size16", 16)],
            512,
            EncoderType.RESNET,
            128,
            use_cnn_exact_state28=True,
        )
