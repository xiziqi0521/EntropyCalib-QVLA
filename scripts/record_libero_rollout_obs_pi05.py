#!/usr/bin/env python
"""Record LIBERO calibration observations from REAL policy rollouts.

Unlike `Omega-QVLA/tools/record_libero_obs_for_pi05.py` -- which only
records each task's post-settle INITIAL state, reached via zero/no-op
actions, cycled round-robin across tasks -- this script loads the actual,
UNQUANTIZED FP16 pi0.5 policy being calibrated and lets it really attempt
each task (same action-chunking / replan_steps cadence as real eval,
`examples.Libero.eval.run_libero_eval`), then samples observations from
PROPORTIONAL POSITIONS along each resulting trajectory (e.g. 20/40/60/80%
through the episode), not just t=0.

Why this exists: the "blind" and entropy-guided calibration pools built so
far are both drawn from a pool of pure task-opening static frames (see
MECHANISM_CN.md section 2) -- neither ever sees a real mid-task state
(arm mid-reach, object mid-grasp, near-goal, etc.), and the entropy signal
computed over that pool measures "how consistent is the policy's very
first reaction to this static opening frame", not "how confident is the
policy at this point IN THE TASK". This script fixes the input side of
that gap; the entropy-scoring/HDBSCAN/selection code downstream (
compute_entropy_labels_pi05.py, select_entropy_calibration_obs.py) is
unchanged, since the output format below matches
record_libero_obs_for_pi05.py's obs pool exactly.

Why FP16/unquantized: using the not-yet-calibrated quantized model to
decide which observations to calibrate on would be circular -- a policy
whose behavior is already skewed by quantization error could pick
calibration data that reinforces its own errors instead of correcting them.

Neither Omega-QVLA-official-baseline, Omega-QVLA, nor openpi is modified.
`load_pi05_policy` is imported read-only from Omega-QVLA-official-baseline's
`tools/build_pi05_a2lite_gptq_perstep.py` (unquantized loader, confirmed by
reading its source: it only calls policy_config.create_trained_policy and
sets eval()/requires_grad_(False), no GPTQ/RTN/DuQuant wrapping happens
there -- that only happens in scripts/openpi_inference_service.py's
_apply_quantization_if_requested, which this script never calls).
get_libero_env/get_libero_image/get_libero_dummy_action/quat2axisangle are
imported read-only from examples.Libero.eval.utils (the same helpers
run_libero_eval.py itself uses), so the observation format matches real
eval byte-for-byte.

Run with the SAME python + PYTHONPATH combo as run_libero_eval.py (this
needs both torch/openpi AND libero/robosuite in one process):

    cd /private/xzq/Omega-QVLA-official-baseline && \\
    PYTHONPATH=/private/xzq/openpi/third_party/libero \\
    /private/xzq/openpi/.venv/bin/python \\
        /private/xzq/EntropyCalib-QVLA/scripts/record_libero_rollout_obs_pi05.py \\
        --task-suite-name libero_object \\
        --checkpoint /private/xzq/openpi/checkpoints/pi05_libero/pi05_independent_pytorch_l20_seed42/30000 \\
        --num-tasks 10 --episodes-per-task 5 \\
        --sample-fractions 0.2 0.4 0.6 0.8 \\
        --output cache/object/obs_pool_rollout.pt

For a quick validation run (fast, small): --num-tasks 1 --episodes-per-task 2
--max-steps-cap 60.
"""
from __future__ import annotations

import argparse
import collections
import copy
import sys
from pathlib import Path

import numpy as np
import torch

OFFICIAL_BASELINE_ROOT = "/private/xzq/Omega-QVLA-official-baseline"
sys.path.insert(0, OFFICIAL_BASELINE_ROOT)

from libero.libero import benchmark  # noqa: E402
from examples.Libero.eval.utils import (  # noqa: E402
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
)
from tools.build_pi05_a2lite_gptq_perstep import load_pi05_policy  # noqa: E402

# openpi's "openpi" max-step profile, matching what our eval pipeline
# actually uses (run_suite_full_eval.sh passes --max-steps-profile openpi).
MAX_STEPS_OPENPI_PROFILE = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def build_calibration_element(obs, lang: str, resize_size: int) -> dict:
    from openpi_client import image_tools

    img, wrist_img = get_libero_image(obs)
    img = np.ascontiguousarray(img)
    wrist_img = np.ascontiguousarray(wrist_img)
    img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(img, resize_size, resize_size)
    )
    wrist_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_img, resize_size, resize_size)
    )
    state = np.concatenate(
        (
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            quat2axisangle(obs["robot0_eef_quat"]).astype(np.float32),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        )
    )
    return {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": state,
        "prompt": str(lang),
    }


def rollout_episode(policy, env, task, *, replan_steps: int, num_steps_wait: int,
                     resize_size: int, max_steps: int, init_state):
    """Run one real episode with the real policy in the loop. Buffers a
    deep copy of the raw LIBERO obs dict at every post-settle step (robosuite
    reuses internal arrays across steps, so a shallow copy would leave every
    buffered entry pointing at the same, final, mutated array).

    Returns (buffer, succeeded).
    """
    obs = env.set_init_state(init_state)
    action_plan: collections.deque = collections.deque()
    buffer = []
    t = 0
    done = False
    while t < max_steps + num_steps_wait:
        if t < num_steps_wait:
            obs, _, done, _ = env.step(get_libero_dummy_action())
            t += 1
            continue
        buffer.append(copy.deepcopy(obs))
        if not action_plan:
            element = build_calibration_element(obs, task.language, resize_size)
            with torch.no_grad():
                action_chunk = policy.infer(element)["actions"]
            if len(action_chunk) < replan_steps:
                raise ValueError(
                    f"policy returned {len(action_chunk)} actions, "
                    f"but replan_steps={replan_steps}"
                )
            action_plan.extend(action_chunk[:replan_steps])
        action = np.asarray(action_plan.popleft(), dtype=np.float32)
        obs, _, done, _ = env.step(action.tolist())
        if done:
            break
        t += 1
    return buffer, done


def sample_at_fractions(buffer: list, fractions: list[float]) -> list:
    if not buffer:
        return []
    n = len(buffer)
    idxs = sorted({min(n - 1, max(0, int(round(f * (n - 1))))) for f in fractions})
    return [buffer[i] for i in idxs]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task-suite-name", required=True,
                    help="libero_spatial | libero_object | libero_goal | libero_10 | libero_90")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-config", default="pi05_libero")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-tasks", type=int, default=None,
                    help="How many tasks (task_id 0..N-1) to roll out; default = all tasks in suite")
    p.add_argument("--episodes-per-task", type=int, default=5)
    p.add_argument("--sample-fractions", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8])
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--num-steps-wait", type=int, default=10)
    p.add_argument("--resize-size", type=int, default=224)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--max-steps-cap", type=int, default=None,
                    help="Hard cap on steps/episode, overriding the openpi profile -- use a small "
                         "value (e.g. 60) for fast validation runs")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    print(f"[record_rollout_obs] loading FP16 policy from {args.checkpoint}", flush=True)
    policy, _model = load_pi05_policy(args.checkpoint, args.data_config, args.device)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks = args.num_tasks or task_suite.n_tasks
    max_steps = args.max_steps_cap or MAX_STEPS_OPENPI_PROFILE[args.task_suite_name]

    samples = []
    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, resolution=args.resolution)

        for ep in range(args.episodes_per_task):
            env.reset()
            init_state = initial_states[ep % len(initial_states)]
            buffer, done = rollout_episode(
                policy, env, task,
                replan_steps=args.replan_steps, num_steps_wait=args.num_steps_wait,
                resize_size=args.resize_size, max_steps=max_steps, init_state=init_state,
            )
            picked = sample_at_fractions(buffer, args.sample_fractions)
            for raw_obs in picked:
                samples.append(build_calibration_element(raw_obs, task_description, args.resize_size))
            print(f"[record_rollout_obs] task={task_id} ({task_description!r}) ep={ep} "
                  f"trajectory_steps={len(buffer)} success={done} picked={len(picked)} "
                  f"total_samples_so_far={len(samples)}", flush=True)

        if hasattr(env, "close"):
            env.close()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(samples, args.output)
    print(f"[record_rollout_obs] wrote {args.output} ({len(samples)} entries)", flush=True)


if __name__ == "__main__":
    main()
