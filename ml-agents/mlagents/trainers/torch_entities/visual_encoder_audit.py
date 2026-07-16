import json
from pathlib import Path
from typing import Dict

from mlagents.torch_utils import torch
from mlagents.trainers.exception import UnityTrainerException
from mlagents.trainers.torch_entities.encoders import ResNetVisualEncoder
from mlagents.trainers.torch_entities.pretrained_visual_encoder import (
    _visual_encoders,
)


def _difference(left: torch.Tensor, right: torch.Tensor) -> Dict[str, float]:
    difference = (left - right).abs()
    return {
        "max_abs": float(difference.max().cpu()),
        "mean_abs": float(difference.mean().cpu()),
    }


@torch.no_grad()
def audit_visual_encoder_batch(
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    visual_observation: torch.Tensor,
    checkpoint_path: str,
    output_path: str,
) -> Dict[str, object]:
    """Compare live actor/critic embeddings with an independent reference encoder."""
    actor_encoders = _visual_encoders(actor)
    critic_encoders = _visual_encoders(critic)
    if len(actor_encoders) != 1 or len(critic_encoders) != 1:
        raise UnityTrainerException(
            "Visual audit requires exactly one actor and one critic ResNet encoder"
        )
    checkpoint = torch.load(checkpoint_path, map_location=visual_observation.device)
    metadata = checkpoint.get("metadata", {})
    reference = ResNetVisualEncoder(
        metadata.get("height", 0),
        metadata.get("width", 0),
        metadata.get("channels", 0),
        metadata.get("output_size", 0),
    ).to(visual_observation.device)
    reference.load_state_dict(checkpoint["encoder_state_dict"], strict=True)
    reference.eval()

    sample = visual_observation[: min(64, len(visual_observation))]
    actor_embedding = actor_encoders[0](sample)
    critic_embedding = critic_encoders[0](sample)
    reference_embedding = reference(sample)
    report: Dict[str, object] = {
        "purpose": "live_trainer_visual_integration_audit",
        "input_shape": list(sample.shape),
        "input_min": float(sample.min().cpu()),
        "input_max": float(sample.max().cpu()),
        "input_mean": float(sample.mean().cpu()),
        "embedding_shape": list(reference_embedding.shape),
        "actor_vs_reference": _difference(actor_embedding, reference_embedding),
        "critic_vs_reference": _difference(critic_embedding, reference_embedding),
        "actor_vs_critic": _difference(actor_embedding, critic_embedding),
        "actor_frozen": all(
            not parameter.requires_grad for parameter in actor_encoders[0].parameters()
        ),
        "critic_frozen": all(
            not parameter.requires_grad for parameter in critic_encoders[0].parameters()
        ),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report
