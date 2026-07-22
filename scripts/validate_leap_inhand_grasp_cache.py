"""Replay every LEAP grasp-cache entry and reject unstable grasps."""

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

from unilab.training import BackendAdapter, create_env, ensure_registries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="caches/leap_hand_allegro_style_20k.npy")
    parser.add_argument("--sim", choices=("mujoco", "motrix"), default="mujoco")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--warmup-seconds", type=float, default=0.5)
    parser.add_argument("--output-passing-cache")
    parser.add_argument("--required-count", type=int)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.seconds <= 0.0:
        raise SystemExit("--batch-size and --seconds must be positive")
    if not 0.0 <= args.warmup_seconds < args.seconds:
        raise SystemExit("--warmup-seconds must be in [0, --seconds)")

    cache_path = ROOT_DIR / "src" / "unilab" / "assets" / args.cache
    cache = np.load(cache_path, mmap_mode="r")
    if cache.ndim != 2 or cache.shape[1] != 23:
        raise SystemExit(f"invalid cache shape: {cache.shape}")
    if cache.shape[0] % args.batch_size != 0:
        raise SystemExit(
            f"cache size {cache.shape[0]} must be divisible by batch size {args.batch_size}"
        )

    ensure_registries()
    with initialize_config_dir(config_dir=str(ROOT_DIR / "conf" / "appo"), version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"task=leap_inhand/{args.sim}",
                f"env.grasp_cache_path={args.cache}",
                "env.use_grasp_cache=true",
                "+env.grasp_cache_sample_mode=sequential",
                "+env.grasp_cache_start_index=0",
                "env.domain_rand.joint_noise=0.0",
                "env.domain_rand.ball_vel_noise=0.0",
            ],
        )
    env_cfg = BackendAdapter(cfg, root_dir=ROOT_DIR, algo_name="appo").build_task_env_cfg_override()
    env = create_env(cfg, num_envs=args.batch_size, env_cfg_override=env_cfg)
    env.set_autoreset(False)
    env_ids = np.arange(args.batch_size, dtype=np.int32)
    actions = np.zeros((args.batch_size, env.action_space.shape[0]), dtype=np.float32)
    steps = int(round(args.seconds / float(cfg.env.ctrl_dt)))
    warmup_steps = int(round(args.warmup_seconds / float(cfg.env.ctrl_dt)))
    reset_z = float(cfg.reward.reset_z_threshold)
    failure_indices: set[int] = set()
    drop_failure_indices: set[int] = set()
    contact_quality_failure_indices: set[int] = set()
    min_ball_z = float("inf")
    min_contacts = 4
    max_fingertip_distance = 0.0
    palm_contact_count = 0

    try:
        for batch_start in range(0, cache.shape[0], args.batch_size):
            env.reset(env_ids)
            for step in range(steps):
                state = env.step(actions)
                ball_pos = env.get_ball_pos()
                min_ball_z = min(min_ball_z, float(np.min(ball_pos[:, 2])))
                dropped = np.asarray(state.terminated, dtype=bool) | (ball_pos[:, 2] <= reset_z)
                invalid = dropped.copy()
                if np.any(dropped):
                    local = np.flatnonzero(dropped)
                    drop_failure_indices.update((batch_start + local).tolist())
                if step >= warmup_steps:
                    contacts = np.stack(
                        [
                            env._contact_flag(env.get_sensor_data(name))
                            for name in env._CONTACT_SENSORS
                        ],
                        axis=1,
                    )
                    contact_count = np.sum(contacts, axis=1)
                    min_contacts = min(min_contacts, int(np.min(contact_count)))
                    palm = env._contact_flag(env.get_sensor_data(env._PALM_CONTACT_SENSOR)).astype(
                        bool
                    )
                    palm_contact_count += int(np.count_nonzero(palm))
                    fingertip_distance = np.linalg.norm(
                        env.get_fingertip_pos() - ball_pos[:, None, :], axis=2
                    )
                    max_fingertip_distance = max(
                        max_fingertip_distance, float(np.max(fingertip_distance))
                    )
                    contact_quality_invalid = contact_count < 2
                    contact_quality_invalid |= palm
                    contact_quality_invalid |= np.any(fingertip_distance >= 0.1, axis=1)
                    invalid |= contact_quality_invalid
                    if np.any(contact_quality_invalid):
                        local = np.flatnonzero(contact_quality_invalid)
                        contact_quality_failure_indices.update((batch_start + local).tolist())
                if np.any(invalid):
                    local = np.flatnonzero(invalid)
                    failure_indices.update((batch_start + local).tolist())
            print(
                f"validated {batch_start + args.batch_size}/{cache.shape[0]} "
                f"failures={len(failure_indices)}"
            )
    finally:
        env.close()

    print(f"cache={cache_path}")
    print(f"shape={cache.shape} duration={args.seconds:.2f}s")
    print(
        f"failures={len(failure_indices)} "
        f"drop_failures={len(drop_failure_indices)} "
        f"contact_quality_failures={len(contact_quality_failure_indices)} "
        f"min_ball_z={min_ball_z:.6f} reset_z={reset_z:.6f}"
    )
    print(
        f"min_contacts={min_contacts} palm_contact_samples={palm_contact_count} "
        f"max_fingertip_distance={max_fingertip_distance:.6f}"
    )
    passing_count = int(cache.shape[0] - len(failure_indices))
    print(f"passing={passing_count}/{cache.shape[0]}")
    if args.output_passing_cache is not None:
        required_count = args.required_count or passing_count
        if required_count <= 0:
            raise SystemExit("--required-count must be positive")
        if passing_count < required_count:
            print(f"cannot write {required_count} states: only {passing_count} passed")
            return 1
        passing_mask = np.ones(cache.shape[0], dtype=bool)
        if failure_indices:
            passing_mask[np.asarray(sorted(failure_indices), dtype=np.int64)] = False
        output_path = ROOT_DIR / "src" / "unilab" / "assets" / args.output_passing_cache
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(
            output_path,
            np.asarray(cache[passing_mask][:required_count], dtype=np.float32),
        )
        print(f"wrote_passing_cache={output_path} shape=({required_count}, 23)")
    if failure_indices:
        print(f"first_failure_indices={sorted(failure_indices)[:20]}")
        return 0 if args.output_passing_cache is not None else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
