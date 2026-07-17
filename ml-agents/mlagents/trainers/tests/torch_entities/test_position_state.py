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
