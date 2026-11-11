"""The three models of HALO (https://github.com/apple-aiml-research/ml-halo), as the robot runs them.

- ``build_generator``: the motion generator. ``MotionGenerator.predict`` turns one RGB frame and an
  instruction into a 150-frame whole-body plan: Qwen3.5-2B, then 15 flow-matching steps of a DiT.
- ``build_tracker``: the tracking policy of the G1, the student that the training exports to ONNX. It
  gives 29 joint targets at 50 Hz.
- ``build_orchestrator``: the planner. One prefill of Qwen3.6-35B-A3B over the planning prompt and one
  camera image, which gives the first output token.

    git clone https://github.com/apple-aiml-research/ml-halo
    atenviz capture examples/halo.py:build_generator -a repo=ml-halo \
        -o out/halo_generator --name "HALO generator (Qwen3.5-2B + DiT, 15 steps)"
    atenviz capture examples/halo.py:build_orchestrator -a repo=ml-halo \
        -o out/halo_orchestrator --name "HALO orchestrator (Qwen3.6-35B-A3B 4-bit prefill)"
    uv run --with "rsl-rl-lib>=5.5.1,<5.6" atenviz capture examples/halo.py:build_tracker -a repo=ml-halo \
        -o out/halo_tracker_depth --name "HALO depth tracker (student policy, as exported to ONNX)"
    uv run --with "rsl-rl-lib>=5.5.1,<5.6" atenviz capture examples/halo.py:build_tracker -a repo=ml-halo \
        -a depth=False -o out/halo_tracker_proprio --name "HALO proprio tracker (student policy, as exported to ONNX)"

ml-halo ships no weights, and the capture needs none: the functions build the models from the configs
of ml-halo and Hugging Face on the meta device. The dtypes are the dtypes that HALO runs. The
generator has Qwen3.5-2B in bf16 and the DiT in fp32, and the tracker is fp32. The orchestrator has
the weights of mlx-community/Qwen3.6-35B-A3B-4bit, which HALO runs with MLX. See
``build_orchestrator``.
"""

import sys
from pathlib import Path
from unittest import mock

import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
from transformers.vision_utils import (
    get_vision_attention_seqlens,
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
)

from atenviz import QuantizedLinear, quantize


def _qwen_constants(vlm, tokens: dict, grid: torch.Tensor) -> dict:
    """Return the inputs of a Qwen3.5 forward that depend only on the prompt and the image size.

    torch.export cannot trace how transformers makes them (``.tolist()``, ``repeat_interleave``), but
    transformers accepts them precomputed. They are computed on the CPU and moved to the meta device.
    """
    visual = vlm.model.visual
    indices, weights = get_vision_interpolation_indices_and_weights(
        grid, num_grid_per_side=visual.num_grid_per_side, mode=visual.interpolation_mode,
        align_corners=visual.interpolation_align_corners, spatial_merge_size=visual.spatial_merge_size)
    cu_seqlens, max_seqlen = get_vision_attention_seqlens(grid, visual.config)
    vision = {"interp_indices": indices, "interp_weights": weights, "cu_seqlens": cu_seqlens,
              "position_ids": get_vision_position_ids(grid, visual.spatial_merge_size), "max_seqlen": max_seqlen}
    positions = vlm.model.compute_3d_position_ids(
        input_ids=tokens["input_ids"], inputs_embeds=None, image_grid_thw=grid,
        attention_mask=tokens["attention_mask"], mm_token_type_ids=tokens["mm_token_type_ids"])
    meta = {k: v.to("meta") if isinstance(v, torch.Tensor) else v for k, v in vision.items()}
    return {"grid": grid.to("meta"), "vision": meta, "positions": positions.to("meta"),
            "tokens": {k: v.to("meta") for k, v in tokens.items()}}


def _qwen_forward(vlm, pixel_values: torch.Tensor, constants: dict, **kwargs):
    """Run the vision tower and the language model of Qwen3.5 over one image and the prompt."""
    image = vlm.model.visual(pixel_values, grid_thw=constants["grid"], **constants["vision"])
    image.pooler_output = (image.pooler_output,)  # get_image_features splits it by image: one image here
    return vlm.model(**constants["tokens"], position_ids=constants["positions"],
                     mm_encoder_outputs={"image": image}, use_cache=False, **kwargs)


class GeneratorPlan(torch.nn.Module):
    """``MotionGenerator.predict`` as ``halo_serve`` calls it: one RGB frame and the first frames of the
    last plan, in the normalized space, give the next plan."""

    def __init__(self, generator, instruction: str):
        super().__init__()
        self.generator = generator
        self.instruction = instruction

    def forward(self, rgb: torch.Tensor, prev_chunk: torch.Tensor) -> torch.Tensor:
        return self.generator.predict(rgb, self.instruction, prev_chunk)


def build_generator(repo: str, config: str = "qwen3_5_2b_dit_vision_frozen.yaml",
                    instruction: str = "sit on the sofa"):
    sys.path.insert(0, str(Path(repo, "generator")))
    from configs.config import load_config
    from models.motion_generator import MotionGenerator, model_config_from

    full = load_config(str(Path(repo, "generator", "configs", config)))

    def from_config(name, torch_dtype):
        return AutoModelForImageTextToText.from_config(AutoConfig.from_pretrained(name), dtype=torch_dtype)

    with mock.patch.object(AutoModelForImageTextToText, "from_pretrained", from_config), torch.device("meta"):
        generator = MotionGenerator(model_config_from(full)).eval()

    # VLMEncoder.forward with gpu_preprocess for one frame. The original reads the token spans back
    # from the device, which torch.export cannot trace. The instruction and the image size fix them.
    encoder = generator.vlm_encoder
    height, width = full.data.image_size
    prompt = encoder._prompt(instruction, height, width)
    constants = _qwen_constants(encoder.model, {k: v[None] for k, v in prompt["seq"].items()},
                                torch.tensor([prompt["grid"]]))
    start, end = prompt["span"]  # the image tokens; end is the index of <|vision_end|>

    def encode(images, texts):
        assert texts == [instruction]
        u8 = images.clamp(0, 1).mul(255).to(torch.uint8)
        pixels = encoder.processor.image_processor(images=list(u8.unbind(0)), return_tensors="pt")["pixel_values"]
        # The language model without its head, as the MLX encoder of halo_serve runs it.
        hidden = _qwen_forward(encoder.model, pixels.to(encoder.dtype), constants,
                               output_hidden_states=True).hidden_states[-1]
        text = hidden[:, end + 1:].float()
        return hidden[:, start:end].float(), text, torch.zeros(text.shape[:2], dtype=torch.bool, device=text.device)

    encoder.forward = encode
    rgb = torch.rand(3, height, width, device="meta")
    prev_chunk = torch.zeros(generator.config.prev_chunk_frames, generator.config.output_dim, device="meta")
    return GeneratorPlan(generator, instruction), (rgb, prev_chunk)


def build_tracker(repo: str, depth: bool = True):
    sys.path.insert(0, str(Path(repo, "tracker", "training", "source", "halo_tracker")))
    from halo_tracker.rsl_rl_ext import HaloCNNModel
    from tensordict import TensorDict

    # The observation groups of the student (tracker/README.md) and its encoders and trunk
    # (_scene_student in rsl_rl_distill_cfg.py, which needs Isaac Lab to import).
    shapes = {"policy_state": (434,), "policy_temporal": (30, 166)}
    cnn_cfg = {"policy_temporal": {"output_channels": [32, 64, 128], "kernel_size": [(3, 1), (3, 1), (1, 1)],
                                   "stride": [(2, 1), (2, 1), (1, 1)], "padding": "none", "activation": "elu",
                                   "flatten": True}}
    if depth:
        shapes["student_depth"] = (1, 48, 84)
        cnn_cfg["student_depth"] = {"output_channels": [16, 32, 64, 128], "kernel_size": 3, "stride": 2,
                                    "padding": "none", "activation": "elu", "flatten": True}
    obs = TensorDict({k: torch.zeros(1, *s) for k, s in shapes.items()}, batch_size=[1])
    student = HaloCNNModel(
        obs, {"student": list(shapes)}, "student", 29, hidden_dims=[512], activation="gelu",
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1e-6, "std_type": "scalar"},
        cnn_cfg=cnn_cfg, mlp_arch="residual", residual_width=512, residual_blocks=2, residual_expansion=4,
        orthogonal_init=True)
    policy = student.as_onnx().eval()
    return policy, policy.get_dummy_inputs()


class QuantizedRouter(QuantizedLinear):
    """``Qwen3_5MoeTopKRouter`` with a quantized weight."""

    def __init__(self, router, bits: int, group_size: int):
        super().__init__(router.weight, bits, group_size)
        self.top_k = router.top_k

    def forward(self, hidden: torch.Tensor):
        logits = super().forward(hidden.reshape(-1, self.weight.shape[-1]))
        scores, indices = logits.softmax(-1, dtype=torch.float).topk(self.top_k)
        return logits, (scores / scores.sum(-1, keepdim=True)).to(logits.dtype), indices


class QuantizedExperts(torch.nn.Module):
    """``Qwen3_5MoeExperts`` with quantized weights: ``grouped_mm_experts_forward`` of transformers
    with ``atenviz.quantized_mm`` for ``torch._grouped_mm``."""

    def __init__(self, experts, bits: int, group_size: int):
        super().__init__()
        self.num_experts, self.act_fn = experts.num_experts, experts.act_fn
        self.gate_up_proj = QuantizedLinear(experts.gate_up_proj, bits, group_size)
        self.down_proj = QuantizedLinear(experts.down_proj, bits, group_size)

    def forward(self, hidden: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
        expert_ids, perm = top_k_index.flatten().sort()  # the rows of each expert, one after the other
        counts = torch.histc(expert_ids.int(), bins=self.num_experts, min=0, max=self.num_experts - 1)
        offs = counts.cumsum(0, dtype=torch.int32)
        gate, up = self.gate_up_proj(hidden[perm // top_k_index.shape[-1]], offs).chunk(2, dim=-1)
        out = self.down_proj(self.act_fn(gate) * up, offs) * top_k_weights.flatten()[perm, None]
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(perm.shape[0], device=perm.device)
        return out[inv_perm].view(*top_k_index.shape, -1).sum(1).to(hidden.dtype)


class OrchestratorPrefill(torch.nn.Module):
    """The first decoding step of the planner: the prompt and the image give the logits of the first
    output token."""

    def __init__(self, vlm, constants: dict):
        super().__init__()
        self.vlm = vlm
        self.constants = constants

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden = _qwen_forward(self.vlm, pixel_values, self.constants).last_hidden_state
        return self.vlm.lm_head(hidden[:, -1:])


def build_orchestrator(repo: str, model: str = "Qwen/Qwen3.6-35B-A3B", camera: tuple[int, int] = (640, 480),
                       request: str = "OPERATOR INSTRUCTION: sit on the sofa"):
    sys.path.insert(0, str(Path(repo, "orchestrator")))
    from halo_orchestrator import prompts, registry
    from halo_orchestrator.interface import DEFAULT_MAX_PX

    # The planning call (Orchestrator.deliberate) for the skills of skills.yaml, with the camera frame
    # scaled down as VLMInterface sends it.
    skills = registry.load()
    system = prompts.build_t1_system_prompt(registry.names(skills), reg=skills)
    image = Image.new("RGB", camera)
    image.thumbnail(DEFAULT_MAX_PX)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": [{"type": "text", "text": request}, {"type": "image"}]}]
    processor = AutoProcessor.from_pretrained(model)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = processor(text=[text], images=[image], return_tensors="pt")

    with torch.device("meta"):
        vlm = AutoModelForImageTextToText.from_config(AutoConfig.from_pretrained(model), dtype=torch.bfloat16).eval()
    # The weights of mlx-community/Qwen3.6-35B-A3B-4bit: the language model has 4-bit weights with a
    # bf16 scale and bias for each group of 64, except for the 8-bit routers. The vision tower is bf16.
    language_model = vlm.model.language_model
    for layer in language_model.layers:
        layer.mlp.gate = QuantizedRouter(layer.mlp.gate, 8, 64)
        layer.mlp.shared_expert_gate = QuantizedLinear(layer.mlp.shared_expert_gate.weight, 8, 64)
        layer.mlp.experts = QuantizedExperts(layer.mlp.experts, 4, 64)
    quantize(language_model, bits=4, group_size=64)
    vlm.lm_head = QuantizedLinear(vlm.lm_head.weight, 4, 64)
    tokens = {k: inputs[k] for k in ("input_ids", "attention_mask", "mm_token_type_ids")}
    constants = _qwen_constants(vlm, tokens, inputs["image_grid_thw"])
    pixels = torch.zeros(inputs["pixel_values"].shape, dtype=torch.bfloat16, device="meta")
    return OrchestratorPrefill(vlm, constants), (pixels,)
