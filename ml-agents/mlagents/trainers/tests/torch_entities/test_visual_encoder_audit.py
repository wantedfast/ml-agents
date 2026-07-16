import json

from mlagents.torch_utils import torch
from mlagents.trainers.torch_entities.encoders import ResNetVisualEncoder
from mlagents.trainers.torch_entities.visual_encoder_audit import (
    audit_visual_encoder_batch,
)


class EncoderOwner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNetVisualEncoder(84, 84, 3, 512)


def test_live_audit_matches_actor_critic_and_reference(tmp_path):
    actor = EncoderOwner()
    critic = EncoderOwner()
    critic.load_state_dict(actor.state_dict())
    actor.requires_grad_(False)
    critic.requires_grad_(False)
    checkpoint = tmp_path / "encoder.pt"
    torch.save(
        {
            "encoder_state_dict": actor.encoder.state_dict(),
            "metadata": {
                "height": 84,
                "width": 84,
                "channels": 3,
                "output_size": 512,
            },
        },
        checkpoint,
    )
    output = tmp_path / "audit.json"

    report = audit_visual_encoder_batch(
        actor, critic, torch.rand(2, 3, 84, 84), str(checkpoint), str(output)
    )

    assert report["actor_vs_reference"]["max_abs"] == 0.0
    assert report["critic_vs_reference"]["max_abs"] == 0.0
    assert report["actor_vs_critic"]["max_abs"] == 0.0
    assert report["actor_frozen"] is True
    assert report["critic_frozen"] is True
    assert json.loads(output.read_text()) == report
