"""A LeRobot policy: one action chunk of ``predict_action_chunk``, as ``lerobot-record`` runs it.

    atenviz capture examples/lerobot_policy.py:build -a checkpoint=lerobot/smolvla_base
    atenviz capture examples/lerobot_policy.py:build -a checkpoint=lerobot/pi0_base -a dtype=bfloat16

The other ``-a KEY=VALUE`` arguments set fields of the policy config, for example ``dtype=bfloat16``
(pi0, pi05) or ``num_inference_steps=5``.

The function builds the policy from the checkpoint config on the meta device. It downloads only
``config.json``, and for SmolVLA also the safetensors header, and it allocates no weight memory. When
LeRobot loads a checkpoint, it copies the weights into the parameters of the policy
(``load_state_dict``). Thus the graph has the dtypes of the policy. SmolVLA downloads SmolVLM2 when
``load_vlm_weights`` is set. Thus the function builds SmolVLA from the SmolVLM2 config and gives its
parameters the dtypes of the checkpoint.

lerobot 0.6.1 pins torch<2.12. Install it in the same environment without its dependencies, with the
transformers version that it was tested with:

    uv pip install --no-deps lerobot==0.6.1
    uv pip install transformers==5.5.4 draccus einops num2words accelerate gymnasium opencv-python-headless
"""

import torch
from huggingface_hub import get_safetensors_metadata
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.policies.factory import get_policy_class
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


class PolicyChunk(torch.nn.Module):
    """Run ``predict_action_chunk`` as a call of the policy, so that the graph has the policy as a module."""

    def __init__(self, policy):
        super().__init__()
        policy.forward = policy.predict_action_chunk
        self.policy = policy

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.policy(dict(batch))


def build(checkpoint: str = "lerobot/smolvla_base", **config_fields):
    config = PreTrainedConfig.from_pretrained(checkpoint)
    for key, value in config_fields.items():
        if not hasattr(config, key):
            raise ValueError(f"{type(config).__name__} has no field {key!r}")
        setattr(config, key, value)
    config.device = "meta"
    stored_dtypes = getattr(config, "load_vlm_weights", False)
    if stored_dtypes:
        config.load_vlm_weights = False
    with torch.device("meta"):
        policy = get_policy_class(config.type)(config).eval()
    if stored_dtypes:
        stored = {k: t.dtype for f in get_safetensors_metadata(checkpoint).files_metadata.values()
                  for k, t in f.tensors.items()}
        for name, param in policy.named_parameters():
            if name in stored:
                param.data = param.data.to(DTYPES[stored[name]])

    # Example observations: images in [0, 1], a state vector, and a prompt of the tokenizer length.
    batch = {key: torch.rand(1, *f.shape, device="meta") for key, f in config.input_features.items()
             if f.type in (FeatureType.VISUAL, FeatureType.STATE, FeatureType.ENV)}
    if hasattr(config, "tokenizer_max_length"):
        batch[OBS_LANGUAGE_TOKENS] = torch.zeros(1, config.tokenizer_max_length, dtype=torch.long, device="meta")
        batch[OBS_LANGUAGE_ATTENTION_MASK] = torch.ones(1, config.tokenizer_max_length, dtype=torch.bool, device="meta")
    return PolicyChunk(policy), (batch,)
