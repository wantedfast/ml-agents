import json
import argparse
from mlagents.trainers.demo_loader import load_demonstration
import numpy as np

def demo_to_json(demo_path: str, json_out: str):
    behavior_spec, info_action_pairs, total_steps = load_demonstration(demo_path)

    data = []
    episode_idx = 0
    current_episode = []

    for pair in info_action_pairs:
        agent_info = pair.agent_info
        action_info = pair.action_info

        record = {
            "agent_id": int(agent_info.id),
            "reward": float(agent_info.reward),
            "done": bool(agent_info.done),
            "observations": [],
            "actions": None,
        }

        for i, obs_bytes in enumerate(agent_info.observations):
            obs_shape = behavior_spec.observation_specs[i].shape
            obs_array = np.array(obs_bytes.float_data.data, dtype=np.float32)
            obs_array = obs_array.reshape(obs_shape)

            if len(obs_shape) == 1:
                record["observations"].append({
                    "type": "vector",
                    "data": obs_array.tolist()
                })
            else:
                # Image / visual obs. Disclaimer: haven't tested it
                record["observations"].append({
                    "type": "visual",
                    "shape": list(obs_shape),
                    "data": obs_array.tolist()
                })

        if len(action_info.continuous_actions) > 0:
            record["actions"] = list(action_info.continuous_actions)
        elif len(action_info.discrete_actions) > 0:
            record["actions"] = list(action_info.discrete_actions)
        elif len(action_info.vector_actions_deprecated) > 0:
            record["actions"] = list(action_info.vector_actions_deprecated)

        current_episode.append(record)

        if agent_info.done:
            data.append({
                "episode": episode_idx,
                "steps": current_episode
            })
            episode_idx += 1
            current_episode = []

    if current_episode:
        data.append({
            "episode": episode_idx,
            "steps": current_episode
        })

    with open(json_out, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(data)} episodes to {json_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert ML-Agents .demo to JSON"
    )
    parser.add_argument(
        "demo_file", type=str,
        help="Path to the .demo file"
    )
    parser.add_argument(
        "output_json", type=str,
        help="Path to save output JSON"
    )
    args = parser.parse_args()
    demo_to_json(args.demo_file, args.output_json)
