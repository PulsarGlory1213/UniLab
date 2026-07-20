"""Measure deterministic LEAP in-hand rotation over fixed time windows."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from rsl_rl.utils import resolve_callable
from tensordict import TensorDict

from unilab.envs.manipulation.leap_inhand.rotation import compute_ball_angvel
from unilab.training import BackendAdapter, create_env, ensure_registries

ROOT_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", type=Path)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--action-gain", type=float, default=1.0)
    parser.add_argument("--target-mode")
    parser.add_argument("--action-scale", type=float)
    parser.add_argument("--target-offset-limit", type=float, default=0.35)
    parser.add_argument("--output-video", type=Path)
    args = parser.parse_args()

    overrides = ["task=leap_inhand/motrix"]
    if args.target_mode is not None:
        overrides.append(f"env.control_config.target_mode={args.target_mode}")
    if args.action_scale is not None:
        overrides.append(f"env.control_config.action_scale={args.action_scale}")
    if args.target_mode == "bounded_incremental":
        overrides.append(
            f"+env.control_config.target_offset_limit={args.target_offset_limit}"
        )
    with initialize_config_dir(config_dir=str(ROOT_DIR / "conf" / "appo"), version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)

    ensure_registries()
    env_cfg = BackendAdapter(cfg, root_dir=ROOT_DIR, algo_name="appo").build_task_env_cfg_override()
    env = create_env(cfg, num_envs=1, env_cfg_override=env_cfg)
    if env.state is None:
        env.init_state()

    obs, _ = env.reset(np.array([0], dtype=np.int32))
    obs_array = np.asarray(obs["obs"], dtype=np.float32)
    if args.policy.suffix == ".onnx":
        session = ort.InferenceSession(str(args.policy), providers=["CPUExecutionProvider"])

        def infer(observation: np.ndarray) -> np.ndarray:
            return session.run(None, {"obs": observation})[0].astype(np.float32)

    elif args.policy.suffix == ".pt":
        rl_cfg = OmegaConf.to_container(cfg.algo, resolve=True)
        assert isinstance(rl_cfg, dict)
        actor_cfg = deepcopy(rl_cfg["actor"])
        actor_cls = resolve_callable(actor_cfg.pop("class_name"))
        actor_cfg.pop("num_actions", None)
        example = torch.zeros((1, obs_array.shape[1]), dtype=torch.float32)
        actor = actor_cls(
            TensorDict({"policy": example}, batch_size=1),
            rl_cfg["obs_groups"],
            "actor",
            env.action_space.shape[0],
            **actor_cfg,
        )
        checkpoint = torch.load(args.policy, map_location="cpu", weights_only=True)
        actor.load_state_dict(checkpoint["actor"])
        actor.eval()

        def infer(observation: np.ndarray) -> np.ndarray:
            with torch.inference_mode():
                return actor.mlp(torch.from_numpy(observation)).numpy().astype(np.float32)

    else:
        raise ValueError(f"Unsupported policy file: {args.policy}")
    previous_quat = np.asarray(env.get_ball_quat(), dtype=np.float32).copy()
    ctrl_dt = float(cfg.env.ctrl_dt)
    step_count = round(args.seconds / ctrl_dt)
    angular_velocity_z: list[float] = []

    for _ in range(step_count):
        action = infer(obs_array) * args.action_gain
        transition = env.step(action)
        obs_array = np.asarray(transition.obs["obs"], dtype=np.float32)
        quat = np.asarray(env.get_ball_quat(), dtype=np.float32)
        angular_velocity = compute_ball_angvel(quat, previous_quat, ctrl_dt)
        angular_velocity_z.append(float(angular_velocity[0, 2]))
        previous_quat = quat.copy()

    per_second = []
    steps_per_second = round(1.0 / ctrl_dt)
    for start in range(0, step_count, steps_per_second):
        angle = sum(angular_velocity_z[start : start + steps_per_second]) * ctrl_dt
        per_second.append(angle)

    print("per_second_rad:", " ".join(f"{value:+.4f}" for value in per_second))
    print(f"total_rad: {sum(per_second):+.4f}")
    print(f"last_4s_rad: {sum(per_second[-4:]):+.4f}")

    if args.output_video is not None:
        env.run_playback_mode(
            play_render_mode="record",
            play_steps=step_count,
            output_video=str(args.output_video),
            render_spacing=float(cfg.training.render_spacing),
            initialize=lambda: np.asarray(
                env.reset(np.array([0], dtype=np.int32))[0]["obs"], dtype=np.float32
            ),
            step=lambda observation: np.asarray(
                env.step(infer(observation) * args.action_gain).obs["obs"],
                dtype=np.float32,
            ),
            camera_kwargs={
                "render_width": int(cfg.training.render_width),
                "render_height": int(cfg.training.render_height),
                "cam_distance": float(cfg.training.cam_distance),
                "cam_lookat": list(cfg.training.cam_lookat),
                "cam_elevation": float(cfg.training.cam_elevation),
                "cam_azimuth": float(cfg.training.cam_azimuth),
            },
        )


if __name__ == "__main__":
    main()
