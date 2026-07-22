"""Optimize the LEAP seed-0 index-finger contact around the authored pose."""

from __future__ import annotations

import argparse
import heapq
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from search_leap_inhand_fingertip_pose import (
    LEAP_TIP_GEOMS,
    _geom_ids,
    _load_home,
    _load_leap_model,
    _name_id,
    _surface_gaps,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_HAND = np.array(
    [
        0.675558466,
        -0.311245401,
        1.28696011,
        0.481546183,
        1.1454015,
        0.0429715889,
        0.0624296905,
        0.440274799,
        1.60076238,
        -0.184116801,
        0.00624721579,
        0.743162204,
        1.40099437,
        1.32970657,
        0.528728303,
        0.312199124,
    ],
    dtype=np.float64,
)
MIDDLE_PREPOSITION_DIRECTION = np.array(
    [1.0, 0.9785605, 0.34891954, 0.8648466], dtype=np.float64
)
BALL_POS = np.array([-0.0225964729, 0.021169898, 0.639558165], dtype=np.float64)
BALL_QUAT = np.array(
    [0.999573468, -0.0156558521, -0.0229829881, 0.00891954799], dtype=np.float64
)
CONTACT_SENSOR_NAMES = (
    "leap_ff_contact",
    "leap_mf_contact",
    "leap_rf_contact",
    "leap_th_contact",
    "leap_rotation_palm_contact",
)
PAD_SITE_NAMES = (
    ("ff_pad_center", "ff_pad_normal"),
    ("mf_pad_center", "mf_pad_normal"),
    ("rf_pad_center", "rf_pad_normal"),
    ("th_pad_center", "th_pad_normal"),
)


@dataclass(frozen=True)
class CandidateResult:
    score: float
    hand_qpos: np.ndarray
    pad_alignment: float
    contact_fraction: np.ndarray
    max_ball_drift: float
    final_ball_drift: float
    max_normal_force: float
    max_penetration: float
    min_joint_margin: float


def _sensor_values(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    values = []
    for name in CONTACT_SENSOR_NAMES:
        sensor_id = _name_id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        values.append(float(data.sensordata[int(model.sensor_adr[sensor_id])]))
    return np.asarray(values, dtype=np.float64)


def _pad_alignment(
    model: mujoco.MjModel, data: mujoco.MjData, finger_index: int
) -> float:
    center_name, normal_name = PAD_SITE_NAMES[finger_index]
    center_id = _name_id(model, mujoco.mjtObj.mjOBJ_SITE, center_name)
    normal_id = _name_id(model, mujoco.mjtObj.mjOBJ_SITE, normal_name)
    center = np.asarray(data.site_xpos[center_id], dtype=np.float64)
    normal = np.asarray(data.site_xpos[normal_id], dtype=np.float64) - center
    normal /= max(float(np.linalg.norm(normal)), 1e-9)
    ball_ray = np.asarray(data.qpos[16:19], dtype=np.float64) - center
    ball_ray /= max(float(np.linalg.norm(ball_ray)), 1e-9)
    return float(normal @ ball_ray)


def _joint_limits(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    lower = np.empty(16, dtype=np.float64)
    upper = np.empty(16, dtype=np.float64)
    for joint_id in range(16):
        lower[joint_id], upper[joint_id] = model.jnt_range[joint_id]
    return lower, upper


def _set_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    home_key_id: int,
    hand_qpos: np.ndarray,
) -> None:
    mujoco.mj_resetDataKeyframe(model, data, home_key_id)
    data.qpos[:16] = hand_qpos
    data.qpos[16:19] = BALL_POS
    data.qpos[19:23] = BALL_QUAT
    data.qvel[:] = 0.0
    data.ctrl[:16] = hand_qpos
    mujoco.mj_forward(model, data)


def _geometry_score(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    home_key_id: int,
    hand_qpos: np.ndarray,
    tip_geom_ids: np.ndarray,
    ball_geom_id: int,
) -> float:
    _set_pose(model, data, home_key_id, hand_qpos)
    ff_gap = float(_surface_gaps(model, data, tip_geom_ids[:1], ball_geom_id)[0])
    alignment = _pad_alignment(model, data, 0)

    if ff_gap < -0.0015 or ff_gap > 0.0035:
        return np.inf

    non_tip_penetration = 0.0
    self_penetration = 0.0
    tip_ids = {int(value) for value in tip_geom_ids}
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        distance = float(contact.dist)
        if distance >= -0.0005:
            continue
        first, second = int(contact.geom1), int(contact.geom2)
        pair = {first, second}
        if ball_geom_id in pair:
            other = second if first == ball_geom_id else first
            if other not in tip_ids:
                non_tip_penetration += (-distance - 0.0005) ** 2
        elif first != 0 and second != 0:
            self_penetration += (-distance - 0.0005) ** 2

    pose_delta = (hand_qpos[:4] - DEFAULT_HAND[:4]) / 0.30
    return float(
        1500.0 * (ff_gap / 0.0015) ** 2
        + 80.0 * max(0.70 - alignment, 0.0) ** 2
        + 5.0 * max(-alignment, 0.0) ** 2
        + 0.15 * np.sum(np.square(pose_delta))
        + 2.0e7 * non_tip_penetration
        + 2.0e7 * self_penetration
    )


def _dynamic_validate(
    model: mujoco.MjModel,
    home_key_id: int,
    hand_qpos: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    seconds: float,
) -> CandidateResult:
    data = mujoco.MjData(model)
    _set_pose(model, data, home_key_id, hand_qpos)
    initial_ball_pos = data.qpos[16:19].copy()
    steps = int(np.ceil(seconds / model.opt.timestep))
    contact_sum = np.zeros(5, dtype=np.float64)
    max_ball_drift = 0.0
    max_normal_force = 0.0
    max_penetration = 0.0
    ball_geom_id = _name_id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball")
    ff_geom_id = _name_id(model, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_collision")

    for _ in range(steps):
        mujoco.mj_step(model, data)
        contact_sum += _sensor_values(model, data) > 0.5
        max_ball_drift = max(
            max_ball_drift,
            float(np.linalg.norm(data.qpos[16:19] - initial_ball_pos)),
        )
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            if {int(contact.geom1), int(contact.geom2)} != {ff_geom_id, ball_geom_id}:
                continue
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(model, data, contact_index, force)
            max_normal_force = max(max_normal_force, abs(float(force[0])))
            max_penetration = max(max_penetration, max(-float(contact.dist), 0.0))

    contact_fraction = contact_sum / steps
    alignment = _pad_alignment(model, data, 0)
    final_ball_drift = float(np.linalg.norm(data.qpos[16:19] - initial_ball_pos))
    joint_margin = np.minimum(hand_qpos - lower, upper - hand_qpos) / (upper - lower)
    min_joint_margin = float(np.min(joint_margin))
    invalid = (
        contact_fraction[0] < 0.95
        or contact_fraction[1] > 0.01
        or contact_fraction[2] < 0.95
        or contact_fraction[3] < 0.95
        or contact_fraction[4] > 0.0
        or alignment < 0.60
        or max_ball_drift > 0.001
        or min_joint_margin < 0.20
    )
    score = (
        100.0 * max(0.70 - alignment, 0.0)
        + 5000.0 * max_ball_drift
        + 0.5 * max_normal_force
        + 10000.0 * max_penetration
        + 0.2 * float(np.linalg.norm(hand_qpos[:4] - DEFAULT_HAND[:4]))
    )
    if invalid:
        score += 1.0e6
    return CandidateResult(
        score=score,
        hand_qpos=hand_qpos.copy(),
        pad_alignment=alignment,
        contact_fraction=contact_fraction,
        max_ball_drift=max_ball_drift,
        final_ball_drift=final_ball_drift,
        max_normal_force=max_normal_force,
        max_penetration=max_penetration,
        min_joint_margin=min_joint_margin,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--physics-candidates", type=int, default=64)
    parser.add_argument("--validation-seconds", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT_DIR / "artifacts" / "leap_seed0_optimized.npy",
    )
    args = parser.parse_args()

    model = _load_leap_model()
    data = _load_home(model)
    home_key_id = _name_id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    ball_geom_id = _name_id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball")
    tip_geom_ids = _geom_ids(model, LEAP_TIP_GEOMS)
    lower, upper = _joint_limits(model)

    base = DEFAULT_HAND.copy()
    base[4:8] += 0.008 * MIDDLE_PREPOSITION_DIRECTION
    rng = np.random.default_rng(args.seed)
    low = np.maximum(lower[:4], DEFAULT_HAND[:4] - np.array([0.45, 0.35, 0.45, 0.40]))
    high = np.minimum(upper[:4], DEFAULT_HAND[:4] + np.array([0.45, 0.35, 0.45, 0.40]))

    heap: list[tuple[float, int, np.ndarray]] = []
    keep = max(args.physics_candidates * 4, 128)
    for index in range(args.samples):
        hand = base.copy()
        hand[:4] = rng.uniform(low, high)
        score = _geometry_score(
            model, data, home_key_id, hand, tip_geom_ids, ball_geom_id
        )
        if not np.isfinite(score):
            continue
        entry = (-score, index, hand)
        if len(heap) < keep:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)

    geometry_candidates = [item[2] for item in sorted(heap, reverse=True)]
    print(f"geometry_candidates={len(geometry_candidates)}")
    results = [
        _dynamic_validate(
            model,
            home_key_id,
            hand,
            lower,
            upper,
            args.validation_seconds,
        )
        for hand in geometry_candidates[: args.physics_candidates]
    ]
    results.sort(key=lambda result: result.score)
    if not results or results[0].score >= 1.0e6:
        for result in results[:10]:
            print(
                f"rejected score={result.score:.3f} alignment={result.pad_alignment:.3f} "
                f"contacts={np.round(result.contact_fraction, 3).tolist()} "
                f"drift_mm={1000.0 * result.max_ball_drift:.3f} "
                f"force_n={result.max_normal_force:.3f} q={np.round(result.hand_qpos[:4], 6).tolist()}"
            )
        raise SystemExit("No candidate passed the seed-0 validation contract")

    best = results[0]
    row = np.concatenate((best.hand_qpos, BALL_POS, BALL_QUAT)).astype(np.float64)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, row[None, :])
    print("output:", args.output.resolve())
    print("hand_qpos:", np.round(best.hand_qpos, 9).tolist())
    print(f"ff_pad_alignment: {best.pad_alignment:.6f}")
    print("contact_fraction [ff,mf,rf,th,palm]:", np.round(best.contact_fraction, 4).tolist())
    print(f"max_ball_drift_mm: {1000.0 * best.max_ball_drift:.6f}")
    print(f"final_ball_drift_mm: {1000.0 * best.final_ball_drift:.6f}")
    print(f"max_ff_normal_force_n: {best.max_normal_force:.6f}")
    print(f"max_ff_penetration_mm: {1000.0 * best.max_penetration:.6f}")
    print(f"minimum_joint_margin: {best.min_joint_margin:.6f}")


if __name__ == "__main__":
    main()
