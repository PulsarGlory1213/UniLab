"""Preview the configured LEAP ball-catch initial and final reference poses."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from unilab.training import BackendAdapter, create_env, ensure_registries


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview initial_pose, catch_pose, or both using the APPO Motrix YAML."
    )
    parser.add_argument("--pose", choices=("initial", "catch", "both"), default="both")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.num_envs < 1:
        raise SystemExit("--num-envs must be at least 1")
    if args.pose == "both" and args.num_envs < 2:
        raise SystemExit("--pose both requires --num-envs 2 or greater")

    ensure_registries()
    config_dir = ROOT_DIR / "conf" / "appo"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "task=leap_ball_catch/motrix",
                f"env.pose_preview={args.pose}",
                f"training.play_env_num={args.num_envs}",
            ],
        )

    env_cfg_override = BackendAdapter(
        cfg, root_dir=ROOT_DIR, algo_name="appo"
    ).build_task_env_cfg_override()
    env = create_env(cfg, num_envs=args.num_envs, env_cfg_override=env_cfg_override)
    actions = np.zeros((args.num_envs, env.action_space.shape[0]), dtype=np.float32)

    print(
        f"[pose-preview] pose={args.pose}, num_envs={args.num_envs}. "
        "For both: first half=initial_pose, second half=catch_pose."
    )
    try:
        env.run_playback_mode(
            play_render_mode="interactive",
            play_steps=args.steps,
            output_video=None,
            render_spacing=float(cfg.training.render_spacing),
            initialize=lambda: env.reset(np.arange(args.num_envs, dtype=np.int32))[0]["obs"],
            step=lambda _obs: env.step(actions).obs["obs"],
            camera_kwargs={
                "render_width": int(cfg.training.render_width),
                "render_height": int(cfg.training.render_height),
                "cam_distance": float(cfg.training.cam_distance),
                "cam_lookat": list(cfg.training.cam_lookat),
                "cam_elevation": float(cfg.training.cam_elevation),
                "cam_azimuth": float(cfg.training.cam_azimuth),
            },
        )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
