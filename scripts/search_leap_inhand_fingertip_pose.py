"""Search a neutral LEAP in-hand pose with open fingers and natural contact geometry."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from unilab.base.backend.mujoco.xml import materialize_scene_fragments

ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "src" / "unilab" / "assets" / "robots" / "leap_hand"
TIP_GEOMS = (
    "fingertip_collision",
    "fingertip_2_collision",
    "fingertip_3_collision",
    "thumb_fingertip_collision",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=200_000)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--four-side-layout",
        "--allegro-layout",
        dest="four_side_layout",
        action="store_true",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    materialized = Path(
        materialize_scene_fragments(
            str(ASSET_DIR / "leap_hand.xml"),
            fragment_files=[str(ASSET_DIR / "scene.xml")],
        )
    )
    try:
        model = mujoco.MjModel.from_xml_path(str(materialized))
    finally:
        materialized.unlink(missing_ok=True)
    data = mujoco.MjData(model)
    tip_geom_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in TIP_GEOMS]
    )
    ball_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball")
    palm_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "palm_lower_collision"
    )
    model_lower = model.jnt_range[:16, 0].copy()
    model_upper = model.jnt_range[:16, 1].copy()
    # Keep the three main fingers in a mostly straight family.
    # In each 4-DoF main-finger slice, local indices 2 and 3 are the
    # distal flexion joints. Their upper bounds are reduced so the
    # optimizer cannot curl the fingertips deeply around the ball.
    lower = np.array(
        [0.35, -0.55, 0.00, -0.05] * 3 + [0.55, 0.05, 0.05, 0.05],
        dtype=np.float64,
    )
    upper = np.array(
        [1.80, 0.55, 1.00, 0.80] * 3 + [1.95, 1.55, 1.55, 1.55],
        dtype=np.float64,
    )
    lower = np.maximum(lower, model_lower)
    upper = np.minimum(upper, model_upper)
    ball = np.array(
        [-0.013583, -0.002142, 0.633196],
        dtype=np.float64,
    )
    # Seed from a natural flexion pattern, not the former palm-catch cache.
    center = np.array(
        [
            0.710557, 0.720452, 0.720249, -0.330859,
            0.789181, 0.740685, 0.601233, 0.543166,
            1.208433, -0.034896, 0.719992, 0.083205,
            1.192924, 0.939653, 1.205634, 0.243515,
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(args.seed)
    if args.four_side_layout:
        # Build a useful starting point one finger at a time.  This stage only
        # asks each fingertip mesh to stay near the ball without prescribing
        # which surface of the fingertip must touch or which exact direction
        # around the ball the finger must occupy.  The full-hand objective below
        # later spreads the four fingertips naturally.
        solved = np.clip(center, lower, upper)
        finger_slices = (
            slice(0, 4),
            slice(4, 8),
            slice(8, 12),
            slice(12, 16),
        )

        for finger_index, finger_slice in enumerate(finger_slices):
            tip_geom_id = int(tip_geom_ids[finger_index])

            def finger_score(finger_qpos: np.ndarray) -> float:
                candidate = solved.copy()
                candidate[finger_slice] = finger_qpos

                data.qpos[:16] = candidate
                data.qpos[16:19] = ball
                data.qpos[19:23] = (1.0, 0.0, 0.0, 0.0)
                mujoco.mj_forward(model, data)

                # Use the real fingertip collision mesh.  A small positive gap
                # is acceptable because only two fingers must actually contact
                # the ball in the final pose.
                tip_gap = mujoco.mj_geomDistance(
                    model,
                    data,
                    tip_geom_id,
                    int(ball_geom_id),
                    0.2,
                    None,
                )
                near_error = np.square(
                    max(tip_gap - 0.008, 0.0) / 0.015
                )
                penetration_error = np.square(
                    max(-tip_gap - 0.0015, 0.0) / 0.003
                )

                # Prefer a fairly open main-finger posture, but keep this soft so
                # MuJoCo and the optimizer can choose fingertip, pad, or side
                # contact naturally.
                straight_error = 0.0
                if finger_index < 3:
                    distal = finger_qpos[2:4]
                    straight_target = np.array(
                        [0.20, 0.10],
                        dtype=np.float64,
                    )
                    straight_scale = np.array(
                        [0.45, 0.35],
                        dtype=np.float64,
                    )
                    straight_error = np.square(
                        (distal - straight_target) / straight_scale
                    ).sum()

                return float(
                    8.0 * near_error
                    + 2.0 * straight_error
                    + 12.0 * penetration_error
                )

            finger_lower = lower[finger_slice]
            finger_upper = upper[finger_slice]
            finger_best = solved[finger_slice].copy()
            finger_best_score = finger_score(finger_best)
            finger_sigma = 0.35 * (finger_upper - finger_lower)
            finger_rng = np.random.default_rng(args.seed + finger_index)

            for ik_round in range(8):
                if ik_round == 0:
                    candidates = finger_rng.uniform(
                        finger_lower,
                        finger_upper,
                        size=(8000, 4),
                    )
                else:
                    candidates = (
                        finger_best
                        + finger_rng.normal(size=(8000, 4)) * finger_sigma
                    )
                    candidates = np.clip(
                        candidates,
                        finger_lower,
                        finger_upper,
                    )

                for candidate in candidates:
                    candidate_score = finger_score(candidate)
                    if candidate_score < finger_best_score:
                        finger_best_score = candidate_score
                        finger_best = candidate.copy()

                finger_sigma *= 0.48

            solved[finger_slice] = finger_best
            print(
                f"finger={finger_index} ik_score={finger_best_score:.6f} "
                f"qpos={np.round(finger_best, 6).tolist()}"
            )

        center = solved

    def score(qpos: np.ndarray) -> tuple[float, np.ndarray]:
        data.qpos[:16] = qpos
        data.qpos[16:19] = ball
        data.qpos[19:23] = (1.0, 0.0, 0.0, 0.0)
        mujoco.mj_forward(model, data)

        tips = data.geom_xpos[tip_geom_ids].copy()

        # Real ball-to-fingertip-mesh distance:
        # positive = separated, zero = touching, negative = penetrating.
        tip_ball_distance = np.array(
            [
                mujoco.mj_geomDistance(
                    model,
                    data,
                    int(tip_id),
                    int(ball_geom_id),
                    0.2,
                    None,
                )
                for tip_id in tip_geom_ids
            ],
            dtype=np.float64,
        )

        # Keep all four fingertips available near the ball.  Contact orientation
        # is deliberately unconstrained: pad, tip, and side contact are all valid.
        all_four_near = np.square(
            np.maximum(tip_ball_distance - 0.012, 0.0) / 0.015
        ).sum()

        # Require whichever two fingertips happen to be closest to reach contact.
        # No specific finger pair is selected.
        positive_contact_gap = np.maximum(
            tip_ball_distance - 0.0015,
            0.0,
        )
        closest_two_gaps = np.sort(positive_contact_gap)[:2]
        contact_gap = np.square(closest_two_gaps / 0.008).sum()

        contact_penetration = np.square(
            np.maximum(-tip_ball_distance - 0.0015, 0.0) / 0.003
        ).sum()

        palm_distance = mujoco.mj_geomDistance(
            model,
            data,
            int(palm_geom_id),
            int(ball_geom_id),
            0.2,
            None,
        )
        palm_penalty = np.square(
            max(0.004 - palm_distance, 0.0) / 0.004
        )

        # Spread the four fingertip bodies without assigning any finger to a
        # fixed compass direction around the ball.
        separation = 0.0
        for first in range(4):
            for second in range(first + 1, 4):
                gap = np.linalg.norm(tips[first] - tips[second])
                separation += float(
                    np.square(max(0.028 - gap, 0.0) / 0.012)
                )

        # Soft preference for the visually open, nearly straight main fingers
        # from the reference pose.  It is not a hard contact prescription.
        main_distal_indices = np.array(
            [2, 3, 6, 7, 10, 11],
            dtype=np.int32,
        )
        straight_target = np.array(
            [0.20, 0.10, 0.20, 0.10, 0.20, 0.10],
            dtype=np.float64,
        )
        straight_scale = np.array(
            [0.45, 0.35, 0.45, 0.35, 0.45, 0.35],
            dtype=np.float64,
        )
        straight_penalty = np.square(
            (qpos[main_distal_indices] - straight_target) / straight_scale
        ).sum()

        normalized = (qpos - lower) / np.maximum(upper - lower, 1e-9)
        limit = np.square(
            np.maximum(np.abs(normalized - 0.5) - 0.46, 0.0) / 0.04
        ).sum()
        spread = 0.15 * np.square(qpos[[1, 5, 9]] / 0.65).sum()

        # A mild thumb-opposition prior keeps the hand useful for in-hand
        # manipulation while still allowing any finger pair and contact surface.
        main_side = np.mean(tips[:3] - ball, axis=0)
        thumb_side = tips[3] - ball
        main_side /= max(np.linalg.norm(main_side), 1e-9)
        thumb_side /= max(np.linalg.norm(thumb_side), 1e-9)
        opposition = np.square(
            max(np.dot(main_side, thumb_side) + 0.25, 0.0) / 0.25
        )

        # Reject solutions that push any hand link deeply through the ball.
        penetration = 0.0
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            geom_names = {
                mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1
                ),
                mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2
                ),
            }
            if "ball" in geom_names and contact.dist < -0.002:
                penetration += float(
                    np.square((-contact.dist - 0.002) / 0.003)
                )

        score_value = (
            4.0 * all_four_near
            + separation
            + limit
            + spread
            + 2.0 * straight_penalty
            + 2.0 * opposition
            + 10.0 * contact_gap
            + 10.0 * contact_penetration
            + 12.0 * palm_penalty
            + 8.0 * penetration
        )

        return float(score_value), tip_ball_distance

    best_score, best_tip_gap = score(center)
    best = center.copy()
    sigma = np.array([0.55, 0.50, 0.65, 0.55] * 3 + [0.55, 0.65, 0.65, 0.65])
    per_round = max(1, args.samples // max(args.rounds, 1))
    for round_index in range(args.rounds):
        if round_index == 0:
            candidates = rng.uniform(lower, upper, size=(per_round, 16))
        else:
            candidates = best + rng.normal(size=(per_round, 16)) * sigma
            candidates = np.clip(candidates, lower, upper)
        for candidate in candidates:
            candidate_score, tip_gap = score(candidate)
            if candidate_score < best_score:
                best_score = candidate_score
                best = candidate.copy()
                best_tip_gap = tip_gap
        sigma *= 0.58
        print(
            f"round={round_index + 1} score={best_score:.6f} "
            f"tip_gap={np.round(best_tip_gap, 5).tolist()}"
        )
    print("pose:", np.round(best, 6).tolist())
    data.qpos[:16] = best
    data.qpos[16:19] = ball
    data.qpos[19:23] = (1.0, 0.0, 0.0, 0.0)
    data.ctrl[:] = best
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    for _ in range(750):
        mujoco.mj_step(model, data)
    contact_values = []
    for name in (
        "leap_ff_contact",
        "leap_mf_contact",
        "leap_rf_contact",
        "leap_th_contact",
        "leap_rotation_palm_contact",
    ):
        sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        address = int(model.sensor_adr[sensor_id])
        contact_values.append(float(data.sensordata[address]))
    print(
        "settled_pose:",
        np.round(data.qpos[:16], 6).tolist(),
    )

    print(
        "settled_ball:",
        np.round(data.qpos[16:19], 6).tolist(),
    )

    print(
        "settled_ball_quat:",
        np.round(data.qpos[19:23], 6).tolist(),
    )

    print(
        "settled_contacts [ff,mf,rf,th,palm]:",
        contact_values,
    )

    settled_tip_gap = np.array(
        [
            mujoco.mj_geomDistance(
                model,
                data,
                int(tip_id),
                int(ball_geom_id),
                0.2,
                None,
            )
            for tip_id in tip_geom_ids
        ],
        dtype=np.float64,
    )
    print(
        "settled_tip_gap:",
        np.round(settled_tip_gap, 6).tolist(),
    )
    print(
        "settled_main_distal:",
        np.round(
            data.qpos[[2, 3, 6, 7, 10, 11]],
            6,
        ).tolist(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())