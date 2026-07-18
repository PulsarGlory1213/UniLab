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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=200_000)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
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
    ball = np.array([-0.01, 0.045, 0.6245])
    radius = 0.0335
    # Seed from a natural flexion pattern, not the former palm-catch cache.
    center = np.array(
        [
            1.244, 0.082, 0.265, 0.298,
            1.096, 0.005, 0.080, 0.150,
            1.337, 0.029, 0.285, 0.317,
            1.104, 1.163, 0.953, -0.138,
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(args.seed)

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
        surface = np.square((distances - radius) / 0.004).sum()
        facing = np.square(np.minimum(alignment - 0.72, 0.0) / 0.18).sum()
        # Keep the three main fingertips fanned instead of stacked together.
        separation = 0.0
        for first in range(3):
            for second in range(first + 1, 3):
                gap = np.linalg.norm(pads[first] - pads[second])
                separation += float(np.square(max(0.035 - gap, 0.0) / 0.015))
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
        score_value = surface + 4.0 * facing + separation + limit + spread + 3.0 * opposition
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
    print("settled_ball:", np.round(data.qpos[16:19], 6).tolist())
    print("settled_contacts [ff,mf,rf,th,palm]:", contact_values)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
