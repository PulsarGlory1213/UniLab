"""Diagnose static stability and local rotational manipulability of LEAP home pose."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir

from unilab.envs.manipulation.leap_inhand.rotation import compute_ball_angvel
from unilab.training import BackendAdapter, create_env, ensure_registries

ROOT_DIR = Path(__file__).resolve().parents[1]
FINGER_NAMES = ("ff", "mf", "rf", "th")


def _contacts(env: object) -> np.ndarray:
    return np.stack(
        [env._contact_flag(env.get_sensor_data(name)) for name in env._CONTACT_SENSORS],
        axis=1,
    )


def _palm_contact(env: object) -> np.ndarray:
    return env._contact_flag(env.get_sensor_data(env._PALM_CONTACT_SENSOR))


def _make_env(num_envs: int, *, action_scale: float):
    ensure_registries()
    with initialize_config_dir(config_dir=str(ROOT_DIR / "conf" / "appo"), version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "task=leap_inhand/mujoco",
                "env.gen_grasp=true",
                "env.use_grasp_cache=false",
                "env.domain_rand.joint_noise=0.0",
                "env.domain_rand.ball_vel_noise=0.0",
                "env.control_config.target_mode=default_offset",
                f"env.control_config.action_scale={action_scale}",
                f"training.play_env_num={num_envs}",
            ],
        )
    env_cfg = BackendAdapter(cfg, root_dir=ROOT_DIR, algo_name="appo").build_task_env_cfg_override()
    env = create_env(cfg, num_envs=num_envs, env_cfg_override=env_cfg)
    env.set_autoreset(False)
    return cfg, env


def _static_diagnostic(seconds: float) -> dict[str, object]:
    cfg, env = _make_env(1, action_scale=0.01)
    env_ids = np.array([0], dtype=np.int32)
    zero_action = np.zeros((1, env.action_space.shape[0]), dtype=np.float32)
    env.reset(env_ids)
    initial_ball_pos = np.asarray(env.get_ball_pos(), dtype=np.float64).copy()
    minimum_contacts = 4
    contact_sum = np.zeros(4, dtype=np.float64)
    palm_samples = 0
    max_drift = 0.0
    min_ball_z = float(initial_ball_pos[0, 2])
    terminated = False
    steps = round(seconds / float(cfg.env.ctrl_dt))
    try:
        for _ in range(steps):
            transition = env.step(zero_action)
            ball_pos = np.asarray(env.get_ball_pos(), dtype=np.float64)
            contacts = _contacts(env)
            contact_sum += contacts[0]
            minimum_contacts = min(minimum_contacts, int(np.sum(contacts[0])))
            palm_samples += int(_palm_contact(env)[0])
            max_drift = max(max_drift, float(np.linalg.norm(ball_pos[0] - initial_ball_pos[0])))
            min_ball_z = min(min_ball_z, float(ball_pos[0, 2]))
            terminated |= bool(transition.terminated[0])

        qpos = np.asarray(env.get_hand_dof_pos()[0], dtype=np.float64)
        lower = np.asarray(env._ctrl_lower, dtype=np.float64)
        upper = np.asarray(env._ctrl_upper, dtype=np.float64)
        normalized_margin = np.minimum(qpos - lower, upper - qpos) / (upper - lower)
        fingertip_distance = np.linalg.norm(
            np.asarray(env.get_fingertip_pos()[0], dtype=np.float64)
            - np.asarray(env.get_ball_pos()[0], dtype=np.float64),
            axis=1,
        )
        return {
            "initial_ball_pos": initial_ball_pos[0],
            "final_ball_pos": np.asarray(env.get_ball_pos()[0], dtype=np.float64),
            "max_drift": max_drift,
            "min_ball_z": min_ball_z,
            "minimum_contacts": minimum_contacts,
            "contact_fraction": contact_sum / steps,
            "palm_fraction": palm_samples / steps,
            "terminated": terminated,
            "joint_margin": normalized_margin,
            "fingertip_distance": fingertip_distance,
        }
    finally:
        env.close()


def _candidate_actions(num_candidates: int, seed: int) -> np.ndarray:
    if num_candidates < 32:
        raise ValueError("--candidates must be at least 32")
    rng = np.random.default_rng(seed)
    actions = rng.normal(size=(num_candidates, 16))
    actions /= np.maximum(np.max(np.abs(actions), axis=1, keepdims=True), 1e-8)
    actions[:32] = 0.0
    for joint_index in range(16):
        actions[2 * joint_index, joint_index] = 1.0
        actions[2 * joint_index + 1, joint_index] = -1.0
    return actions.astype(np.float32)


def _probe_scale(
    *,
    scale: float,
    candidates: int,
    seed: int,
    settle_seconds: float,
    ramp_seconds: float,
    hold_seconds: float,
) -> dict[str, object]:
    cfg, env = _make_env(candidates, action_scale=scale)
    env_ids = np.arange(candidates, dtype=np.int32)
    candidate_actions = _candidate_actions(candidates, seed)
    zero_actions = np.zeros_like(candidate_actions)
    ctrl_dt = float(cfg.env.ctrl_dt)
    env.reset(env_ids)
    try:
        for _ in range(round(settle_seconds / ctrl_dt)):
            env.step(zero_actions)

        start_pos = np.asarray(env.get_ball_pos(), dtype=np.float64).copy()
        previous_quat = np.asarray(env.get_ball_quat(), dtype=np.float32).copy()
        angle = np.zeros((candidates, 3), dtype=np.float64)
        max_drift = np.zeros(candidates, dtype=np.float64)
        support_steps = np.zeros(candidates, dtype=np.int32)
        palm_steps = np.zeros(candidates, dtype=np.int32)
        min_contacts = np.full(candidates, 4, dtype=np.int32)
        terminated = np.zeros(candidates, dtype=bool)
        ramp_steps = max(1, round(ramp_seconds / ctrl_dt))
        hold_steps = max(0, round(hold_seconds / ctrl_dt))
        total_steps = ramp_steps + hold_steps

        for step_index in range(total_steps):
            gain = min(1.0, (step_index + 1) / ramp_steps)
            transition = env.step(candidate_actions * gain)
            ball_pos = np.asarray(env.get_ball_pos(), dtype=np.float64)
            quat = np.asarray(env.get_ball_quat(), dtype=np.float32)
            angle += compute_ball_angvel(quat, previous_quat, ctrl_dt) * ctrl_dt
            previous_quat = quat.copy()
            drift = np.linalg.norm(ball_pos - start_pos, axis=1)
            max_drift = np.maximum(max_drift, drift)
            contacts = _contacts(env)
            contact_count = np.sum(contacts, axis=1).astype(np.int32)
            support_steps += contact_count >= 2
            min_contacts = np.minimum(min_contacts, contact_count)
            palm_steps += _palm_contact(env).astype(np.int32)
            terminated |= np.asarray(transition.terminated, dtype=bool)

        final_pos = np.asarray(env.get_ball_pos(), dtype=np.float64)
        support_fraction = support_steps / total_steps
        palm_fraction = palm_steps / total_steps
        valid = (
            (~terminated)
            & (max_drift <= 0.003)
            & (support_fraction >= 0.90)
            & (palm_fraction == 0.0)
        )
        score = angle[:, 2] - 10.0 * max_drift - 0.25 * np.linalg.norm(angle[:, :2], axis=1)
        score[~valid] = -np.inf
        ranked = np.argsort(score)[::-1]
        top = [index for index in ranked[:10] if np.isfinite(score[index])]
        return {
            "scale": scale,
            "actions": candidate_actions,
            "angle": angle,
            "max_drift": max_drift,
            "final_drift": np.linalg.norm(final_pos - start_pos, axis=1),
            "support_fraction": support_fraction,
            "palm_fraction": palm_fraction,
            "min_contacts": min_contacts,
            "terminated": terminated,
            "valid": valid,
            "score": score,
            "top": top,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-seconds", type=float, default=10.0)
    parser.add_argument("--candidates", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument("--ramp-seconds", type=float, default=0.30)
    parser.add_argument("--hold-seconds", type=float, default=0.30)
    parser.add_argument("--scales", type=float, nargs="+", default=[0.01, 0.02, 0.03])
    args = parser.parse_args()

    static = _static_diagnostic(args.static_seconds)
    print("STATIC")
    print(" initial_ball_pos:", np.round(static["initial_ball_pos"], 6).tolist())
    print(" final_ball_pos:  ", np.round(static["final_ball_pos"], 6).tolist())
    print(
        f" max_drift_m: {static['max_drift']:.6f} min_ball_z: {static['min_ball_z']:.6f} "
        f"min_contacts: {static['minimum_contacts']} terminated: {static['terminated']}"
    )
    print(" contact_fraction [ff,mf,rf,th]:", np.round(static["contact_fraction"], 4).tolist())
    print(f" palm_fraction: {static['palm_fraction']:.4f}")
    print(" fingertip_distance_m:", np.round(static["fingertip_distance"], 6).tolist())
    print(" normalized_joint_margin:", np.round(static["joint_margin"], 4).tolist())
    print(f" minimum_joint_margin: {np.min(static['joint_margin']):.4f}")

    for scale in args.scales:
        result = _probe_scale(
            scale=scale,
            candidates=args.candidates,
            seed=args.seed,
            settle_seconds=args.settle_seconds,
            ramp_seconds=args.ramp_seconds,
            hold_seconds=args.hold_seconds,
        )
        valid_count = int(np.count_nonzero(result["valid"]))
        positive_valid = int(np.count_nonzero(result["valid"] & (result["angle"][:, 2] > 0.0)))
        print(f"\nLOCAL scale_rad={scale:.3f}")
        print(
            f" valid: {valid_count}/{args.candidates} positive_valid: "
            f"{positive_valid}/{args.candidates}"
        )
        if not result["top"]:
            print(" no candidate met drift/contact/palm constraints")
            continue
        for rank, index in enumerate(result["top"][:5], start=1):
            action = result["actions"][index]
            active = [(joint, round(float(value), 3)) for joint, value in enumerate(action) if abs(value) >= 0.2]
            print(
                f" #{rank} idx={index} z_angle={result['angle'][index, 2]:+.5f} "
                f"off_axis={np.linalg.norm(result['angle'][index, :2]):.5f} "
                f"max_drift_m={result['max_drift'][index]:.6f} "
                f"support={result['support_fraction'][index]:.3f} "
                f"min_contacts={result['min_contacts'][index]} active={active}"
            )


if __name__ == "__main__":
    main()
