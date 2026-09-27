import numpy as np
import torch

from experiments.RQ2.guidance import make_guidance_rewards, root_fingerprint
from verl import DataProto


def _batch():
    return DataProto.from_dict(
        tensors={
            "responses": torch.tensor([[4, 5, 0], [7, 0, 0]]),
            "response_mask": torch.tensor([[1, 1, 0], [1, 0, 0]]),
            "values": torch.tensor([[0.1, 0.8, 0.0], [0.3, 0.0, 0.0]]),
            "true_token_level_rewards": torch.tensor(
                [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
            ),
        },
        non_tensors={"sampling_seed": np.array([11, 12], dtype=np.int64)},
    )


def test_guidance_rewards_share_roots_but_use_different_terminal_signal():
    batch = _batch()
    reward_guidance = make_guidance_rewards(batch, "reward")
    value_guidance = make_guidance_rewards(batch, "value")

    assert torch.equal(reward_guidance, batch.batch["true_token_level_rewards"])
    assert torch.equal(
        value_guidance,
        torch.tensor([[0.0, 0.8, 0.0], [0.3, 0.0, 0.0]]),
    )
    reward_guidance[0, 1] = -1
    assert batch.batch["true_token_level_rewards"][0, 1] == 1


def test_root_fingerprint_covers_responses_masks_and_sampling_seeds():
    batch = _batch()
    original = root_fingerprint(batch)
    assert original == root_fingerprint(_batch())

    changed_response = _batch()
    changed_response.batch["responses"][0, 0] += 1
    assert root_fingerprint(changed_response) != original

    changed_seed = _batch()
    changed_seed.non_tensor_batch["sampling_seed"][0] += 1
    assert root_fingerprint(changed_seed) != original
