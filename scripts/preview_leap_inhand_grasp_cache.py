"""Preview sequential batches from the LEAP in-hand grasp cache in Motrix."""

from __future__ import annotations

import argparse
import sys
import time
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
        description="Preview LEAP grasp-cache entries sequentially in Motrix."
    )
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seconds-per-batch", type=float, default=4.0)
    parser.add_argument("--steps", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.num_envs < 1:
        raise SystemExit("--num-envs must be at least 1")
    if args.start_index < 0:
        raise SystemExit("--start-index must be non-negative")
    if args.seconds_per_batch <= 0.0:
        raise SystemExit("--seconds-per-batch must be positive")

    ensure_registries()
    config_dir = ROOT_DIR / "conf" / "appo"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "task=leap_inhand/motrix",
                f"training.play_env_num={args.num_envs}",
                "env.grasp_cache_sample_mode=sequential",
                f"env.grasp_cache_start_index={args.start_index}",
                "env.domain_rand.ball_vel_noise=0.0",
            ],
        )

    env_cfg_override = BackendAdapter(
        cfg, root_dir=ROOT_DIR, algo_name="appo"
    ).build_task_env_cfg_override()
    cache_path = ROOT_DIR / "src" / "unilab" / "assets" / str(cfg.env.grasp_cache_path)
    if not cache_path.exists():
        raise SystemExit(f"grasp cache does not exist: {cache_path}")
    cache_size = int(np.load(cache_path, mmap_mode="r").shape[0])
    env = create_env(cfg, num_envs=args.num_envs, env_cfg_override=env_cfg_override)
    env_ids = np.arange(args.num_envs, dtype=np.int32)
    actions = np.zeros((args.num_envs, env.action_space.shape[0]), dtype=np.float32)
    env.set_autoreset(False)
    next_switch_time = time.monotonic() + args.seconds_per_batch
    batch_start = args.start_index

    def initialize() -> np.ndarray:
        nonlocal batch_start
        wrapped_start = batch_start % cache_size
        wrapped_end = (batch_start + args.num_envs - 1) % cache_size
        print(
            f"[grasp-cache-preview] showing cache indices "
            f"{wrapped_start}..{wrapped_end}"
        )
        batch_start += args.num_envs
        return env.reset(env_ids)[0]["obs"]

    def step(obs: np.ndarray) -> np.ndarray:
        del obs
        nonlocal next_switch_time
        if time.monotonic() >= next_switch_time:
            next_switch_time = time.monotonic() + args.seconds_per_batch
            return initialize()
        return env.step(actions).obs["obs"]

    print(
        "[grasp-cache-preview] zero actions hold each cached joint pose; "
        f"a new batch appears every {args.seconds_per_batch:g}s."
    )
    try:
        env.run_playback_mode(
            play_render_mode="interactive",
            play_steps=args.steps,
            output_video=None,
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
