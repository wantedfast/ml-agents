import hashlib
import os
from typing import List, Tuple

from mlagents.torch_utils import torch, nn
from mlagents.trainers.exception import UnityTrainerException
from mlagents.trainers.torch_entities.encoders import ResNetVisualEncoder


def _visual_encoders(module: nn.Module) -> List[ResNetVisualEncoder]:
    return [
        child
        for child in module.modules()
        if isinstance(child, ResNetVisualEncoder)
    ]


def load_pretrained_visual_encoders(
    actor: nn.Module,
    critic: nn.Module,
    checkpoint_path: str,
    freeze: bool,
) -> Tuple[int, int, str]:
    """Strictly load one encoder checkpoint into every actor/critic visual encoder."""
    if not os.path.isfile(checkpoint_path):
        raise UnityTrainerException(
            f"Pretrained visual encoder checkpoint does not exist: {checkpoint_path}"
        )

    with open(checkpoint_path, "rb") as checkpoint_file:
        digest = hashlib.sha256(checkpoint_file.read()).hexdigest()

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "encoder_state_dict" not in checkpoint:
        raise UnityTrainerException(
            "Pretrained visual checkpoint must contain 'encoder_state_dict'."
        )
    metadata = checkpoint.get("metadata", {})
    expected = {"height": 84, "width": 84, "channels": 3, "output_size": 512}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise UnityTrainerException(
                f"Pretrained visual checkpoint metadata mismatch for {key}: "
                f"expected {value}, got {metadata.get(key)}"
            )

    actor_encoders = _visual_encoders(actor)
    critic_encoders = _visual_encoders(critic)
    if len(actor_encoders) != 1 or len(critic_encoders) != 1:
        raise UnityTrainerException(
            "Frozen visual MA-POCA requires exactly one ResNet visual encoder "
            f"in actor and critic; found actor={len(actor_encoders)}, "
            f"critic={len(critic_encoders)}."
        )

    state_dict = checkpoint["encoder_state_dict"]
    for encoder in actor_encoders + critic_encoders:
        try:
            encoder.load_state_dict(state_dict, strict=True)
        except RuntimeError as error:
            raise UnityTrainerException(
                f"Pretrained visual encoder is incompatible: {error}"
            ) from error
        if freeze:
            encoder.requires_grad_(False)
            encoder.eval()

    frozen_parameters = sum(
        parameter.numel()
        for encoder in actor_encoders + critic_encoders
        for parameter in encoder.parameters()
        if not parameter.requires_grad
    )
    return len(actor_encoders), len(critic_encoders), digest
