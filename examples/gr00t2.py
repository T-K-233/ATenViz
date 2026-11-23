"""GR00T 2 (https://research.nvidia.com/labs/gear/gr00t-2/) on the Unitree G1 and the YAM arms.

GR00T 2 is System 1 and SONIC is System 0. GR00T 2 accepts an optional System 2 planner (any VLM
agent), which has no graph here.

- ``build``: one served request of ``GrootN2.predict``, from NVIDIA's code in Isaac-GR00T.
- ``build_sonic``: SONIC's decoder MLP, which turns a motion token and the robot state into 29 joint
  targets every 20 ms. ``size="full"`` is the size of SONIC's real-world runs, ``"release"`` the
  open-source default.

    git clone -b codex/n2-action-expert https://github.com/NVIDIA/Isaac-GR00T  # commit 7e492ad
    uv run --with peft --with einops --with tyro atenviz capture examples/gr00t2.py:build \
        -a repo=Isaac-GR00T -a variant=turbo_g1 -o out/gr00t2_turbo_g1 \
        --name "GR00T 2 Turbo, G1 (2 video + 4 action steps)"
    uv run --with peft --with einops --with tyro atenviz capture examples/gr00t2.py:build \
        -a repo=Isaac-GR00T -a variant=core_yam -o out/gr00t2_core_yam \
        --name "GR00T 2, YAM (2 video + 4 action steps)"
    atenviz capture examples/gr00t2.py:build_sonic -a workloads=PATH \
        -o out/gr00t2_sonic --name "GR00T 2 SONIC decoder (full, 37.4 M parameters)"

``Gr00tN2Policy`` serves the request with a cached prompt. The Wan VAE encodes the history frames
of each camera. Then the DiT denoises the future video latents and the action chunk together. Each
of the ``video_steps`` joint steps captures the K/V of the video and the state, and
``action_steps / video_steps - 1`` action-only steps reuse this K/V (``kv_cache_cond``).

- ``turbo_g1``: GR00T 2 Turbo on the G1. Wan2.2 VAE, 3 cameras, 80 actions, 921 tokens.
- ``core_yam``: GR00T 2 on the YAM arms. Wan2.1 VAE, 4 cameras, 48 actions, 4529 tokens.

``build`` makes the model of each variant on the meta device, with bf16 weights as in the
checkpoints. The capture differs from the CUDA run in these points:

- The attention runs as SDPA. The masked joint attention uses the dense token mask of its
  FlexAttention BlockMask, as the TensorRT export of GR00T 2 does. Thus the FLOPs are the FLOPs of
  dense attention.
- The capture removes the autocast regions, because ``torch.export`` puts each region into one box
  without modules. All the weights are bf16, so the dtypes of the ops do not change.
- The timestep embeddings and the AdaLN modulations depend only on the fixed noise schedule. Thus
  they are constant-foldable, and their FLOPs are in ``const_flops``, not in the totals.

With the constant-foldable FLOPs, the totals agree with ``FlopCounterMode`` on a Jetson Thor:
12.94 TFLOP for turbo_g1 and 52.82 TFLOP for core_yam.

``build_sonic`` imports SONIC from ``sonic/sonic.py`` in the directory ``workloads``, outside this
repo. This model is a reconstruction from SONIC's release (github.com/NVlabs/GR00T-WholeBodyControl)
on the meta device, with FP32 weights.
"""
import contextlib
import sys
import types
from pathlib import Path
from unittest import mock

import torch

VARIANTS = {  # the layouts of the GR00T 2 whitepaper (Tab. 27)
    "turbo_g1": dict(vae="wan22", cameras=3, horizon=80),
    "core_yam": dict(vae="wan21", cameras=4, horizon=48),
}
HISTORY, FRAMES = 5, 13  # for each camera: 5 history and 8 future frames, which are 2 + 2 latent frames


class Predict(torch.nn.Module):
    def __init__(self, model, data: dict, steps: dict):
        super().__init__()
        self.model = model
        self.data = data
        self.steps = steps

    def forward(self, images, state, prompt_embs):
        from groot.core.training.models.dream_cosmos.attention_utils.flex_attention import torch_attention_op
        from groot.core.training.models.n2.components import sequence_packing, transformer_cosmos
        from groot.core.training.models.n2.components.kv_cache import CondKVCache

        # FlexAttention does not run on the meta device.
        def dense_mask(mask_mod, B, H, Q_LEN, KV_LEN, BLOCK_SIZE, device):
            b = torch.arange(B, device=device)[:, None, None]
            q, kv = torch.arange(Q_LEN, device=device)[None, :, None], torch.arange(KV_LEN, device=device)[None, None, :]
            return types.SimpleNamespace(dense=mask_mod(b, None, q, kv)[:, None])

        def masked_sdpa(q, k, v, attn_mask, pad, flatten_heads, compile_attention):
            mask = attn_mask.dense[:, :, : q.shape[1], : k.shape[1]]  # without the block padding
            return torch_attention_op(q, k, v, mask, flatten_heads)

        horizon = self.model.config.action_cfg["action_horizon"]

        def slice_cache(cache, modality_ids, token_valid_mask):
            """Keep the K/V of the video and the state, without the data-dependent ``nonzero`` of
            ``_slice_cache_to_static_prefix``. ConcatPacker puts the action tokens last."""
            sliced = CondKVCache()
            keep = modality_ids.shape[1] - horizon
            sliced.key_cache = [k[:, :, :keep].clone() for k in cache.key_cache]
            sliced.value_cache = [v[:, :, :keep].clone() for v in cache.value_cache]
            return sliced

        # predict draws the initial noise from a seeded generator, and the meta device has none.
        with (mock.patch.object(torch, "Generator", lambda device: mock.Mock(manual_seed=lambda seed: None)),
              mock.patch.object(torch.amp, "autocast", lambda dtype, device_type: contextlib.nullcontext()),
              mock.patch.object(sequence_packing, "create_block_mask", dense_mask),
              mock.patch.object(transformer_cosmos, "flex_attention_op", masked_sdpa),
              mock.patch.object(transformer_cosmos, "flash_attn_func_v23",
                                lambda q, k, v, flatten: torch_attention_op(q, k, v, None, flatten)),
              mock.patch.object(self.model, "_slice_cache_to_static_prefix", slice_cache)):
            data = {**self.data, "images": images, "state": state}
            return self.model.predict(data, prompt_embs=prompt_embs, **self.steps)["action_pred"]


def build(repo: str, variant: str = "turbo_g1", video_steps: int = 2, action_steps: int = 4):
    sys.path.insert(0, repo)
    # These packages import the training and data pipelines of GR00T, or all the model configs.
    # Register them without their __init__ files, so that only the files of the ``n2`` packages load.
    for package in ("training.configs.model", "training.models", "training.models.n2",
                    "training.models.n2.components"):
        module = types.ModuleType(f"groot.core.{package}")
        module.__path__ = [str(Path(repo, "groot", "core", *package.split(".")))]
        sys.modules[module.__name__] = module
    sys.modules["groot.core.training.configs.model"].register_model_config = lambda name, config: None
    from groot.core.training.configs.model.n2 import GrootN2Config
    from groot.core.training.models.n2 import model as n2_model

    layout = VARIANTS[variant]
    action_cfg = {
        "max_action_dim": 128, "max_state_dim": 128, "action_horizon": layout["horizon"],
        "max_action_horizon": layout["horizon"], "num_embodiments": 32, "use_temporal_pos_emb": True,
        "attention_mask_pattern": [("bidirectional_video_action_full", 1.0)],
    }
    config = GrootN2Config(
        model_dtype="bfloat16", model_size="2B", vae_type=layout["vae"], tiled=False, image_target_size=[224, 320],
        cond_timestep=0.1, view_embed_dim=7, max_n_cameras=layout["cameras"], video_scheduler_type="beta",
        video_noise_beta_alpha=7.0, video_noise_beta_beta=1.0, video_noise_s=0.999, action_cfg=action_cfg,
        rope_fp32=True, text_encoder_cfg={}, vae_cfg={},
    )
    # The prompt is cached, so the request does not run the text encoder (Cosmos-Reason1-7B).
    with mock.patch.object(n2_model, "CosmosTextEncoder", torch.nn.Module), torch.device("meta"):
        model = n2_model.GrootN2(config).eval()
    for param in model.parameters():
        param.data = param.data.to(torch.bfloat16)
    if layout["vae"] == "wan21":  # WanVideoVAE makes its latent statistics on the GPU
        vae = model.vae
        vae.mean, vae.std = vae.mean.to("meta"), vae.std.to("meta")
        vae.scale = [vae.mean, 1.0 / vae.std]

    # The output of GrootN2Processor for one observation. The policy gives only the history frames
    # (Gr00tPolicy._load_modality_configs), and the model generates the latents of the future frames.
    data = {
        "num_cond_frames": HISTORY,
        "num_total_frames": FRAMES,
        "video_padded_mask": torch.zeros(1, FRAMES * layout["cameras"], device="meta"),  # no padded cameras
        "state_keep": torch.ones(1, device="meta"),
        "embodiment_id": torch.zeros(1, dtype=torch.long, device="meta"),
    }
    images = torch.zeros(1, HISTORY * layout["cameras"], *config.image_target_size, 3, dtype=torch.uint8, device="meta")
    state = torch.zeros(1, 1, action_cfg["max_state_dim"], device="meta")
    # The hidden states of the 28 layers of Cosmos-Reason1-7B for 512 tokens, concatenated.
    prompt_embs = torch.zeros(1, 512, 28 * 3584, dtype=torch.bfloat16, device="meta")
    steps = {"num_video_steps": video_steps, "num_action_steps": action_steps, "kv_cache_cond": True}
    return Predict(model, data, steps), (images, state, prompt_embs)


def build_sonic(workloads: str, size: str = "full"):
    sys.path.insert(0, str(Path(workloads, "sonic")))
    import sonic

    return sonic.model(size, device="meta"), (sonic.example().to("meta"),)
