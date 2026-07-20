"""Search a LEAP pose whose four finger pads oppose a high rotation ball."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from unilab.base.backend.mujoco.xml import materialize_scene_fragments

ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "src" / "unilab" / "assets" / "robots" / "leap_hand"
PAD_SITES = ("ff_pad_center", "mf_pad_center", "rf_pad_center", "th_pad_center")
NORMAL_SITES = ("ff_pad_normal", "mf_pad_normal", "rf_pad_normal", "th_pad_normal")
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
    pad_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in PAD_SITES]
    )
    normal_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in NORMAL_SITES]
    )
    tip_geom_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in TIP_GEOMS]
    )
    ball_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball")
    palm_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "palm_lower_collision"
    )
    model_lower = model.jnt_range[:16, 0].copy()
    model_upper = model.jnt_range[:16, 1].copy()
    # Stay in a human-readable inward-curl family. In particular, do not let
    # the optimizer solve contact by hyperextending a distal joint.
    lower = np.array(
        [0.35, -0.55, 0.15, 0.05] * 3 + [0.55, 0.05, 0.05, 0.05],
        dtype=np.float64,
    )
    upper = np.array(
        [1.80, 0.55, 1.70, 1.55] * 3 + [1.95, 1.55, 1.55, 1.55],
        dtype=np.float64,
    )
    lower = np.maximum(lower, model_lower)
    upper = np.minimum(upper, model_upper)
    ball = np.array([-0.013583, -0.002142, 0.633196],dtype=np.float64,)
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
    layout_directions: np.ndarray | None = None
    if args.four_side_layout:
        # Allegro's authored home pose distributes the three main fingertips
        # across one side of the ball and opposes them with the thumb. Solve
        # LEAP's independent finger chains against the actual pad sites.
        four_side_directions = np.array(
            [
                # 食指：球的左前方
                [-0.68, -0.39, -0.62],

                # 中指：球的前下方
                [0.00, -0.78, -0.62],

                # 無名指：球的右前方
                [0.68, -0.39, -0.62],

                # 拇指：從另一個方向包住球
                [0.00, 0.78, -0.62],
            ],
            dtype=np.float64,
        )

        four_side_directions /= np.linalg.norm(
            four_side_directions,
            axis=1,
            keepdims=True,
        )

        layout_directions = four_side_directions

        pad_radii = np.array(
            [0.039, 0.039, 0.039, 0.040],
            dtype=np.float64,
        )

        pad_targets = (
            ball[None, :]
            + pad_radii[:, None] * four_side_directions
        )
        solved = np.clip(center, lower, upper)
        finger_slices = (slice(0, 4), slice(4, 8), slice(8, 12), slice(12, 16))
        for finger_index, finger_slice in enumerate(finger_slices):
            target = pad_targets[finger_index]

            def finger_score(finger_qpos: np.ndarray) -> float:
                candidate = solved.copy()
                candidate[finger_slice] = finger_qpos
                data.qpos[:16] = candidate
                data.qpos[16:19] = ball
                data.qpos[19:23] = (1.0, 0.0, 0.0, 0.0)
                mujoco.mj_forward(model, data)
                pad = data.site_xpos[pad_ids[finger_index]]
                normal = data.site_xpos[normal_ids[finger_index]] - pad
                normal /= max(np.linalg.norm(normal), 1e-9)
                target_normal = -allegro_directions[finger_index]
                position_error = np.sum(np.square((pad - target) / 0.003))
                normal_error = np.square(
                    max(0.75 - np.dot(normal, target_normal), 0.0) / 0.15
                )
                return float(position_error + 2.0 * normal_error)

            finger_lower = lower[finger_slice]
            finger_upper = upper[finger_slice]
            finger_best = solved[finger_slice].copy()
            finger_best_score = finger_score(finger_best)
            finger_sigma = 0.35 * (finger_upper - finger_lower)
            finger_rng = np.random.default_rng(args.seed + finger_index)
            for ik_round in range(8):
                if ik_round == 0:
                    candidates = finger_rng.uniform(finger_lower, finger_upper, size=(8000, 4))
                else:
                    candidates = finger_best + finger_rng.normal(size=(8000, 4)) * finger_sigma
                    candidates = np.clip(candidates, finger_lower, finger_upper)
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

    def score(qpos: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        data.qpos[:16] = qpos
        data.qpos[16:19] = ball
        data.qpos[19:23] = (1.0, 0.0, 0.0, 0.0)
        mujoco.mj_forward(model, data)
        pads = data.site_xpos[pad_ids].copy()
        normals = data.site_xpos[normal_ids] - pads
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
        to_ball = ball - pads
        distances = np.linalg.norm(to_ball, axis=1)
        directions = to_ball / np.maximum(distances[:, None], 1e-9)
        alignment = np.sum(normals * directions, axis=1)
        layout_penalty = 0.0

        if layout_directions is not None:
            pad_directions = pads - ball[None, :]
            pad_directions /= np.maximum(
                np.linalg.norm(
                    pad_directions,
                    axis=1,
                    keepdims=True,
                ),
                1e-9,
            )
        
            layout_cosine = np.sum(
                pad_directions * layout_directions,
                axis=1,
            )
        
            # 每根手指必須留在指定的球面區域，
            # 避免最後又全部擠到同一側。
            layout_penalty = np.square(
                np.maximum(0.82 - layout_cosine, 0.0) / 0.18
            ).sum()
        pad_targets = np.array([0.036, 0.039, 0.050, 0.040], dtype=np.float64)
        alignment_targets = np.array([0.20, 0.20, 0.00, 0.20], dtype=np.float64)
        surface = np.square((distances - pad_targets) / 0.010).sum()
        facing = np.square(np.minimum(alignment - alignment_targets, 0.0) / 0.20).sum()

        tip_ball_distance = np.array(
            [
                mujoco.mj_geomDistance(
                    model, data, int(tip_id), int(ball_geom_id), 0.2, None
                )
                for tip_id in tip_geom_ids
            ]
        )
        positive_contact_gap = np.maximum(
        tip_ball_distance - 0.0015,
        0.0,
        )

        # 只要求距離球最近的兩根手指真正接觸。
        # 另外兩根仍由 pad surface 與 layout penalty
        # 約束在球的四周。
        closest_two_gaps = np.sort(positive_contact_gap)[:2]

        contact_gap = np.square(
            closest_two_gaps / 0.008
        ).sum()
        contact_penetration = np.square(
            np.maximum(-tip_ball_distance - 0.002, 0.0) / 0.003
        ).sum()
        palm_distance = mujoco.mj_geomDistance(
            model, data, int(palm_geom_id), int(ball_geom_id), 0.2, None
        )
        palm_penalty = np.square(max(0.004 - palm_distance, 0.0) / 0.004)
        # Keep the three main fingertips fanned instead of stacked together.
        # 四個 pad 都不能擠在一起。
        separation = 0.0

        for first in range(4):
            for second in range(first + 1, 4):
                gap = np.linalg.norm(
                    pads[first] - pads[second]
                )

                separation += float(
                    np.square(
                        max(0.028 - gap, 0.0) / 0.012
                    )
                )
        # Avoid joint-limit tricks and sideways main-finger spread.
        normalized = (qpos - lower) / np.maximum(upper - lower, 1e-9)
        limit = np.square(np.maximum(np.abs(normalized - 0.5) - 0.46, 0.0) / 0.04).sum()
        spread = 0.15 * np.square(qpos[[1, 5, 9]] / 0.65).sum()
        main_side = np.mean(pads[:3] - ball, axis=0)
        thumb_side = pads[3] - ball
        main_side /= max(np.linalg.norm(main_side), 1e-9)
        thumb_side /= max(np.linalg.norm(thumb_side), 1e-9)
        opposition = np.square(max(np.dot(main_side, thumb_side) + 0.25, 0.0) / 0.25)

        # Reject solutions that reach the pad target by pushing another link through the ball.
        penetration = 0.0
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            geom_names = {
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1),
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2),
            }
            if "ball" in geom_names and contact.dist < -0.002:
                penetration += float(np.square((-contact.dist - 0.002) / 0.003))
        score_value = (
            surface
            + 12.0 * facing
            + 8.0 * layout_penalty
            + separation
            + limit
            + spread
            + 2.0 * opposition
            + 10.0 * contact_gap
            + 10.0 * contact_penetration
            + 12.0 * palm_penalty
        )
        return float(score_value + 8.0 * penetration), distances, alignment

    best_score, best_dist, best_alignment = score(center)
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
            candidate_score, distances, alignment = score(candidate)
            if candidate_score < best_score:
                best_score = candidate_score
                best = candidate.copy()
                best_dist = distances
                best_alignment = alignment
        sigma *= 0.58
        print(
            f"round={round_index + 1} score={best_score:.6f} "
            f"dist={np.round(best_dist, 5).tolist()} "
            f"align={np.round(best_alignment, 4).tolist()}"
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
