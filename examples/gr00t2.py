"""The GR00T 2 stack on the Unitree G1, as NVIDIA's whitepaper describes it: GR00T 2 Turbo is System 1 and
SONIC is System 0. GR00T 2 accepts an optional System 2 planner (any VLM agent), which has no graph here.

- ``build_turbo``: one call of Turbo's transformer (Cosmos-Predict2.5-2B): a joint denoising pass over
  921 video, state and action tokens, then an action-only pass over the 80 action tokens. This is the
  deployed schedule of one video and two action steps (1v2a).
- ``build_turbo_vae``: Turbo's Wan 2.2 VAE encoder over five frames from each of the three cameras.
- ``build_sonic``: SONIC's decoder MLP, which turns a motion token and the robot state into 29 joint
  targets every 20 ms. ``size="full"`` is the size of SONIC's real-world runs, ``"release"`` the
  open-source default.

NVIDIA has not released Turbo's code. The models are a reconstruction from the GR00T 2 whitepaper, the
public Cosmos-Predict2.5-2B and Wan 2.2 VAE, and SONIC's release (github.com/NVlabs/GR00T-WholeBodyControl),
on the meta device. ``workloads`` is the directory that holds ``gr00t2/wam.py``, ``gr00t2/vae.py`` and
``sonic/sonic.py``. The weights are FP32: the capture does not apply the deployed FP8 and BF16.

    atenviz capture examples/gr00t2.py:build_turbo -a workloads=PATH \\
        -o out/gr00t2_turbo --name "GR00T 2 Turbo transformer (1v2a, 921 tokens)"
    atenviz capture examples/gr00t2.py:build_turbo_vae -a workloads=PATH \\
        -o out/gr00t2_turbo_vae --name "GR00T 2 Turbo VAE encoder (Wan 2.2, 3 cameras x 5 frames)"
    atenviz capture examples/gr00t2.py:build_sonic -a workloads=PATH \\
        -o out/gr00t2_sonic --name "GR00T 2 SONIC decoder (full, 37.4 M parameters)"
"""

import sys
from pathlib import Path


def build_turbo(workloads: str):
    sys.path.insert(0, str(Path(workloads, "gr00t2")))
    import wam

    model = wam.build("turbo", "full", device="meta")
    return model, model.inputs(device="meta")


def build_turbo_vae(workloads: str):
    sys.path.insert(0, str(Path(workloads, "gr00t2")))
    import vae

    return vae.build("turbo", "full", device="meta"), (vae.example("full", "meta"),)


def build_sonic(workloads: str, size: str = "full"):
    sys.path.insert(0, str(Path(workloads, "sonic")))
    import sonic

    return sonic.model(size, device="meta"), (sonic.example().to("meta"),)
