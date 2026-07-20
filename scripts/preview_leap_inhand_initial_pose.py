"""Preview the LEAP Allegro-mapped home pose without loading a grasp cache."""

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--output-video", type=Path, default=None)
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run gravity/contact dynamics instead of freezing the authored pose.",
    )
    args = parser.parse_args()
    if args.num_envs < 1:
        raise SystemExit("--num-envs must be at least 1")

    ensure_registries()
    with initialize_config_dir(config_dir=str(ROOT_DIR / "conf" / "appo"), version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "task=leap_inhand/motrix",
                "env.gen_grasp=true",
                "env.domain_rand.joint_noise=0.0",
                "env.domain_rand.ball_vel_noise=0.0",
                f"training.play_env_num={args.num_envs}",
            ],
        )

    env_cfg_override = BackendAdapter(
        cfg, root_dir=ROOT_DIR, algo_name="appo"
    ).build_task_env_cfg_override()
    env = create_env(cfg, num_envs=args.num_envs, env_cfg_override=env_cfg_override)
    env_ids = np.arange(args.num_envs, dtype=np.int32)
    actions = np.zeros((args.num_envs, env.action_space.shape[0]), dtype=np.float32)

    def initialize() -> np.ndarray:
        return env.reset(env_ids)[0]["obs"]

    def step(obs: np.ndarray) -> np.ndarray:
        del obs
        if args.simulate:
            # Zero residual actions keep the authored joint targets while physics runs.
            return env.step(actions).obs["obs"]
        # Static mode resets every frame so only the authored geometry is shown.
        return env.reset(env_ids)[0]["obs"]

    try:
        env.run_playback_mode(
            play_render_mode="record" if args.output_video is not None else "interactive",
            play_steps=args.steps,
            output_video=str(args.output_video) if args.output_video is not None else None,
            render_spacing=float(cfg.training.render_spacing),
            initialize=initialize,
            step=step,
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
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
