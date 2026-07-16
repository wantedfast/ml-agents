import inspect

import pytest

from mlagents.torch_utils import torch
from mlagents.trainers.exception import UnityTrainerException
from mlagents.trainers.settings import EncoderType
from mlagents.trainers.torch_entities.encoders import (
    OracleGeometryInput,
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


def test_oracle_geometry_preserves_prefix_and_zero_pads():
    processor = OracleGeometryInput()
    geometry = torch.linspace(-1.0, 1.0, 96).reshape(1, 96)
    output = processor(geometry)

    assert output.shape == (1, 512)
    assert torch.equal(output[:, :96], geometry)
    assert torch.count_nonzero(output[:, 96:]) == 0
    assert sum(parameter.numel() for parameter in processor.parameters()) == 0


def test_oracle_geometry_processor_is_named_and_not_normalized():
    specs = [
        vector_spec("01_OracleGeometry", 96),
        vector_spec("VectorSensor_size13", 13),
    ]
    processors, sizes = ModelUtils.create_input_processors(
        specs,
        h_size=512,
        vis_encode_type=EncoderType.SIMPLE,
        attention_embedding_size=128,
        normalize=True,
        use_oracle_navigation_geometry=True,
    )

    assert sizes == [512, 13]
    assert isinstance(processors[0], OracleGeometryInput)
    assert isinstance(processors[1], VectorInput)
    assert processors[1].normalizer is not None


def test_oracle_geometry_rejects_missing_or_wrong_sensor():
    with pytest.raises(UnityTrainerException, match="exactly one"):
        ModelUtils.create_input_processors(
            [vector_spec("VectorSensor_size13", 13)],
            512,
            EncoderType.SIMPLE,
            128,
            use_oracle_navigation_geometry=True,
        )

    with pytest.raises(UnityTrainerException, match="must have shape"):
        ModelUtils.create_input_processors(
            [vector_spec("01_OracleGeometry", 95)],
            512,
            EncoderType.SIMPLE,
            128,
            use_oracle_navigation_geometry=True,
        )


def test_oracle_geometry_exports_with_onnx_opset_9(tmp_path):
    output_path = tmp_path / "oracle_geometry.onnx"
    export_options = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_options["dynamo"] = False
    torch.onnx.export(
        OracleGeometryInput(),
        torch.zeros((1, 96)),
        str(output_path),
        opset_version=9,
        input_names=["geometry"],
        output_names=["padded_geometry"],
        dynamic_axes={"geometry": {0: "batch"}, "padded_geometry": {0: "batch"}},
        **export_options,
    )
    assert output_path.is_file()
