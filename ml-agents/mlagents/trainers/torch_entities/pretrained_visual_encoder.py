import hashlib
import os
from typing import List, Optional, Tuple

from mlagents.torch_utils import torch, nn
from mlagents.trainers.exception import UnityTrainerException
from mlagents.trainers.torch_entities.encoders import ResNetVisualEncoder


def _visual_encoders(module: nn.Module) -> List[ResNetVisualEncoder]:
    return [
        child
        for child in module.modules()
        if isinstance(child, ResNetVisualEncoder)
    ]


class NavigationGeometryAdapter(nn.Module):
    """Expose supervised world geometry while preserving the 512-wide interface."""

    def __init__(self):
        super().__init__()
        pairs = 2 * 4
        self.delta = nn.Linear(512, pairs * 2)
        self.direction = nn.Linear(512, pairs * 9)
        self.distance = nn.Linear(512, pairs * 31)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        batch = embedding.shape[0]
        delta = self.delta(embedding).reshape(batch, -1)
        direction = torch.softmax(
            self.direction(embedding).reshape(batch, 8, 9), dim=-1
        ).reshape(batch, -1)
        distance_probabilities = torch.softmax(
            self.distance(embedding).reshape(batch, 8, 31), dim=-1
        )
        distance_classes = torch.arange(
            31, device=embedding.device, dtype=embedding.dtype
        )
        expected_distance = (
            distance_probabilities * distance_classes.reshape(1, 1, 31)
        ).sum(dim=-1) / 30.0
        geometry = torch.cat([delta, direction, expected_distance], dim=1)
        # ML-Agents release_21 exports with ONNX opset 9, whose Pad sizes must
        # be static. The geometry prefix is always 96 values, leaving 416 zeros.
        zeros = embedding[:, :416] * 0.0
        return torch.cat([geometry, zeros], dim=1)


def _remove_navigation_adapter(encoder: ResNetVisualEncoder) -> None:
    handle = getattr(encoder, "_navigation_adapter_hook", None)
    if handle is not None:
        handle.remove()
        delattr(encoder, "_navigation_adapter_hook")
    if hasattr(encoder, "navigation_geometry_adapter"):
        delattr(encoder, "navigation_geometry_adapter")


def _attach_navigation_adapter(
    encoder: ResNetVisualEncoder,
    probe_path: str,
    encoder_checkpoint_sha256: str,
    dataset_manifest_sha256: str,
) -> None:
    if not os.path.isfile(probe_path):
        raise UnityTrainerException(
            f"Navigation probe checkpoint does not exist: {probe_path}"
        )
    payload = torch.load(probe_path, map_location="cpu")
    if not isinstance(payload, dict) or "probe_state_dict" not in payload:
        raise UnityTrainerException(
            "Navigation probe checkpoint must contain 'probe_state_dict'."
        )
    metadata = payload.get("metadata", {})
    expected_metadata = {
        "probe_type": "linear",
        "encoder_checkpoint_sha256": encoder_checkpoint_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "station_order": ["Onion", "Dish", "Pot", "Delivery"],
        "grid_size": 16,
        "embedding_size": 512,
    }
    for key, expected_value in expected_metadata.items():
        if metadata.get(key) != expected_value:
            raise UnityTrainerException(
                f"Navigation probe metadata mismatch for {key}: "
                f"expected {expected_value}, got {metadata.get(key)}"
            )

    adapter = NavigationGeometryAdapter().to(next(encoder.parameters()).device)
    try:
        adapter.load_state_dict(payload["probe_state_dict"], strict=True)
    except RuntimeError as error:
        raise UnityTrainerException(
            f"Navigation probe checkpoint is incompatible: {error}"
        ) from error
    adapter.requires_grad_(False)
    adapter.eval()
    encoder.navigation_geometry_adapter = adapter
    encoder._navigation_adapter_hook = encoder.register_forward_hook(
        lambda module, _inputs, output: module.navigation_geometry_adapter(output)
    )


def load_pretrained_visual_encoders(
    actor: nn.Module,
    critic: nn.Module,
    checkpoint_path: str,
    freeze: bool,
    navigation_probe_path: Optional[str] = None,
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
    expected = {
        "height": 84,
        "width": 84,
        "channels": 3,
        "output_size": 512,
        "encoder_type": "resnet",
        "pretraining_schema": "world_only_v2",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise UnityTrainerException(
                f"Pretrained visual checkpoint metadata mismatch for {key}: "
                f"expected {value}, got {metadata.get(key)}"
            )
    if metadata.get("qualified_for_rl") is not True:
        raise UnityTrainerException(
            "Pretrained visual checkpoint was not qualified for RL."
        )
    dataset_digest = metadata.get("dataset_manifest_sha256")
    if not isinstance(dataset_digest, str) or len(dataset_digest) != 64 or any(
        character not in "0123456789abcdef" for character in dataset_digest.lower()
    ):
        raise UnityTrainerException(
            "Pretrained visual checkpoint has an invalid dataset manifest SHA-256."
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
        _remove_navigation_adapter(encoder)
        try:
            encoder.load_state_dict(state_dict, strict=True)
        except RuntimeError as error:
            raise UnityTrainerException(
                f"Pretrained visual encoder is incompatible: {error}"
            ) from error
        if freeze:
            encoder.requires_grad_(False)
            encoder.eval()
        if navigation_probe_path is not None:
            if not freeze:
                raise UnityTrainerException(
                    "Navigation probe features require a frozen visual encoder"
                )
            _attach_navigation_adapter(
                encoder,
                navigation_probe_path,
                digest,
                dataset_digest,
            )

    frozen_parameters = sum(
        parameter.numel()
        for encoder in actor_encoders + critic_encoders
        for parameter in encoder.parameters()
        if not parameter.requires_grad
    )
    return len(actor_encoders), len(critic_encoders), digest
