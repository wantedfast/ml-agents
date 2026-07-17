import inspect

import pytest

from mlagents.torch_utils import torch
from mlagents.trainers.exception import UnityTrainerException
from mlagents.trainers.settings import EncoderType, NetworkSettings
from mlagents.trainers.torch_entities.encoders import (
    EnhancedOracleNavigationInput,
    VectorInput,
)
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents.trainers.torch_entities.networks import (
    MultiAgentNetworkBody,
    NetworkBody,
)
from mlagents_envs.base_env import (
    ActionSpec,
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


def test_enhanced_navigation_preserves_prefix_and_zero_pads():
    processor = EnhancedOracleNavigationInput()
    navigation = torch.linspace(-1.0, 1.0, 212).reshape(1, 212)
    output = processor(navigation)

    assert output.shape == (1, 512)
    assert torch.equal(output[:, :212], navigation)
    assert torch.count_nonzero(output[:, 212:]) == 0
    assert sum(parameter.numel() for parameter in processor.parameters()) == 0


def test_enhanced_navigation_processor_is_named_and_not_normalized():
    specs = [
        vector_spec("01_EnhancedOracleNavigation", 212),
        vector_spec("VectorSensor_size13", 13),
    ]
    processors, sizes = ModelUtils.create_input_processors(
        specs,
        h_size=512,
        vis_encode_type=EncoderType.SIMPLE,
        attention_embedding_size=128,
        normalize=True,
        use_enhanced_oracle_navigation=True,
    )

    assert sizes == [512, 13]
    assert isinstance(processors[0], EnhancedOracleNavigationInput)
    assert isinstance(processors[1], VectorInput)
    assert processors[1].normalizer is not None


def test_enhanced_navigation_rejects_missing_or_wrong_sensor():
    with pytest.raises(UnityTrainerException, match="exactly one"):
        ModelUtils.create_input_processors(
            [vector_spec("VectorSensor_size13", 13)],
            512,
            EncoderType.SIMPLE,
            128,
            use_enhanced_oracle_navigation=True,
        )

    with pytest.raises(UnityTrainerException, match="must have shape"):
        ModelUtils.create_input_processors(
            [vector_spec("01_EnhancedOracleNavigation", 211)],
            512,
            EncoderType.SIMPLE,
            128,
            use_enhanced_oracle_navigation=True,
        )


def test_enhanced_navigation_exports_with_onnx_opset_9(tmp_path):
    output_path = tmp_path / "enhanced_oracle_navigation.onnx"
    export_options = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_options["dynamo"] = False
    torch.onnx.export(
        EnhancedOracleNavigationInput(),
        torch.zeros((1, 212)),
        str(output_path),
        opset_version=9,
        input_names=["navigation"],
        output_names=["padded_navigation"],
        dynamic_axes={
            "navigation": {0: "batch"},
            "padded_navigation": {0: "batch"},
        },
        **export_options,
    )
    assert output_path.is_file()


def test_actor_and_poca_critic_use_enhanced_navigation_processor():
    specs = [
        vector_spec("01_EnhancedOracleNavigation", 212),
        vector_spec("VectorSensor_size13", 13),
    ]
    settings = NetworkSettings(
        normalize=True,
        hidden_units=512,
        use_enhanced_oracle_navigation=True,
    )
    actor_body = NetworkBody(specs, settings)
    critic_body = MultiAgentNetworkBody(
        specs,
        settings,
        ActionSpec.create_continuous(2),
    )

    assert isinstance(actor_body.processors[0], EnhancedOracleNavigationInput)
    assert isinstance(critic_body.processors[0], EnhancedOracleNavigationInput)
    assert actor_body.observation_encoder.total_enc_size == 525
    assert critic_body.observation_encoder.total_enc_size == 525
